import socket
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
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
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, GraphBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward
from gflownet.proxy.mol_utils import smiles2graph
from gflownet.proxy.model import load_proxy_to_gflow, load_proxy_to_gflow_sol
from gflownet.algo.trajectory_balance import TBVariant, Backward
from gflownet.utils import sascore


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


def mol2sas(mols: list[RDMol], is_valid: list[bool], default=10):
    sas = torch.tensor([safe(sascore.calculateScore, i, default) for i, v in zip(mols, is_valid) if v])
    sas = (10 - sas) / 9  # Turn into a [0-1] reward
    return sas


class TrajectoryBalanceTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        cfg: Config,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.min_logp = -13.71
        self.max_logp = 2.41
        self.width = self.max_logp - self.min_logp

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar"""
        flat_r = 1 - ((y - self.min_logp) / self.width)
        return ObjectProperties(flat_r)

    def _load_task_models(self):
        # TODO: Here will need to load my model
        model = load_proxy_to_gflow_sol("../proxy/best_model_sol.pt")
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"logp": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["logp"].device if hasattr(self.models["logp"], "device") else get_worker_device())
        preds = self.models["logp"](batch["x"], batch["edge_index"], batch["edge_attr"], batch["batch"])
        preds[preds.isnan()] = 0
        preds = self.reward_transform(preds).reshape((-1,)).data.cpu()
        return preds.clip(1e-4, 2).reshape((-1,))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [smiles2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds_sol = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        preds_sas = mol2sas(mols, is_valid).reshape(-1, 1)
        preds = preds_sol * preds_sas
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class SEHFragTrainer(StandardOnlineTrainer):
    task: TrajectoryBalanceTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64

        cfg.num_training_steps = 5000
        cfg.validate_every = 250
        # Need to output a lot of molecules to get a good estimate of the reward
        cfg.num_final_gen_steps = 100

        cfg.algo.method = "TB"
        cfg.algo.tb.variant = TBVariant.TB
        cfg.algo.max_nodes = 6
        cfg.algo.sampling_tau = 0.1  # ??
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.05
        cfg.algo.train_det_after = 3000
        cfg.algo.valid_random_action_prob = 0.05
        cfg.algo.valid_num_from_policy = 64
        cfg.algo.tb.Z_learning_rate = 0.001
        cfg.num_validation_gen_steps = 10

        # b where R^b where b is constant as in the first paper
        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [64]
        cfg.cond.temperature.num_thermometer_dim = 1

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg.print_every = 1

        cfg.overwrite_existing_exp = True

        cfg.model.num_emb = 64
        cfg.model.num_layers = 4
        cfg.model.graph_transformer.num_heads = 4
        cfg.model.graph_transformer.num_mlp_layers = 2

    def setup_task(self):
        self.task = TrajectoryBalanceTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_data(self):
        super().setup_data()

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()


def main():
    """Example of how this model can be run."""

    config = init_empty(Config())
    config.log_dir = "./logs/mo_beta_exp(64)"
    seed = 42
    import random
    import wandb

    wandb.login()
    wandb.init(project="all_runs_mo", config=config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    trial = SEHFragTrainer(config)
    trial.run()
    wandb.finish()


if __name__ == "__main__":
    main()
