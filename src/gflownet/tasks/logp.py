import socket
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward
from gflownet.proxy.mpnn import GraphTransformer, load_mpnn_to_gflow, mol2graph

MPNN_PROXY = "../proxy/model.pth"


class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy mpnn model that outputs LogP of a molecule."""

    def __init__(
        self,
        cfg: Config,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        # Model seems to be loaded correctly
        model = load_mpnn_to_gflow(saved_model_path=MPNN_PROXY)
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        preds = (
            self.models["seh"](batch["x"], batch["edge_index"], batch["edge_attr"], batch["batch"])
            .reshape((-1,))
            .data.cpu()
            / 8
        )
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        # Custom mol2graph works
        graphs = [mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


SOME_MOLS = ["CCC", "CCCC"]


class LittleSEHDataset(Dataset):
    """Note: this dataset isn't used by default, but turning it on showcases some features of this codebase.

    To turn on, self `cfg.algo.num_from_dataset > 0`"""

    def __init__(self, smis) -> None:
        super().__init__()
        self.props: ObjectProperties
        self.mols: List[Graph] = []
        self.smis = smis

    def setup(self, task: SEHTask, ctx: FragMolBuildingEnvContext) -> None:
        # This works
        rdmols = [Chem.MolFromSmiles(i) for i in SOME_MOLS]
        # mols represent the molecule objects as states in the env
        self.mols = [ctx.obj_to_graph(i) for i in rdmols]
        # Remove indexing here
        # props are the rewards
        # this is a tuple
        self.props, self.valid_props = task.compute_obj_properties(rdmols)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, index):
        return self.mols[index], self.props[index]


class SEHFragTrainer(StandardOnlineTrainer):
    task: SEHTask
    training_data: LittleSEHDataset

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        # For my laptop
        cfg.num_workers = 0
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 64
        cfg.num_validation_gen_steps = 10
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        self.task = SEHTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_data(self):
        super().setup_data()
        if self.cfg.task.seh.reduced_frag:
            # The examples don't work with the 18 frags
            self.training_data = LittleSEHDataset([])
        else:
            self.training_data = LittleSEHDataset(SOME_MOLS)

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)


def main():
    """Example of how this model can be run."""
    import datetime

    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = f"./logs/debug_run_seh_frag_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.num_training_steps = 1_00
    config.validate_every = 20
    config.num_final_gen_steps = 10
    config.num_workers = 1
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64.0]

    trial = SEHFragTrainer(config)
    trial.run()

    # task = SEHTask(config)
    # preds, val = task.compute_obj_properties(SOME_MOLS)
    # print(preds)

    # data = LittleSEHDataset(SOME_MOLS)
    # data.setup(
    #     task=SEHTask(config),
    #     ctx=FragMolBuildingEnvContext(
    #         max_frags=9,
    #         num_cond_dim=TemperatureConditional(config).encoding_size(),
    #         fragments=bengio2021flow.FRAGMENTS,
    #     ),
    # )


if __name__ == "__main__":
    main()
