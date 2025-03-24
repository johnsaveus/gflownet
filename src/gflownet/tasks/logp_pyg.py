import os
import yaml
import socket
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from lightning import pytorch as pl
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit import Chem
from torch import Tensor
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.utils.yaml_utils import yml2cfg
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward
from gflownet.proxy_pyg.make_datasets import mol2graph
from gflownet.proxy_pyg.model import load_mpnn_to_gflow
import torch_geometric.data as gd
from torch_geometric.data import Data
from gflownet.proxy_pyg.model import CombinedRepresentation


class LogPTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy mpnn model that outputs LogP of a molecule."""

    def __init__(
        self,
        cfg: Config,
        rew_tran: str = "0-10",
        y_min: float = -5.08,
        y_max: float = 11.29,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        self.cfg = cfg
        self.rew_tran = rew_tran
        self.y_min = y_min
        self.y_max = y_max
        self.width = y_max - y_min
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        if self.rew_tran == "exp":
            flat_r = np.exp(-(y - self.y_min) / self.width)
        elif self.rew_tran == "unit":
            flat_r = (y - self.y_min) / self.width
        elif self.rew_tran == "0-10":
            flat_r = ((y - self.y_min) / self.width) * 10
        return flat_r

    def _load_task_models(self):
        # Model seems to be loaded correctly
        ppath = "/home/john-savvas/Documents/gflownet/src/gflownet/proxy_pyg/best_model.pt"
        ckp = torch.load(ppath)
        model = CombinedRepresentation(fp_dim=167, graph_in_channels=24, hidden_dim=32, output_dim=1)
        model.load_state_dict(ckp["model_state_dict"])
        # model = load_mpnn_to_gflow(saved_model_path=ppath)
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
        preds = self.models["logp"](batch["fp"], batch["x"], batch["edge_index"], batch["batch"]).data.cpu()
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 20).reshape((-1,))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        # Custom mol2graph works
        graphs = [mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class LogPTrainer(StandardOnlineTrainer):
    task: LogPTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()

    def setup_task(self):
        self.task = LogPTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()


def main():
    """Example of how this model can be run."""
    import wandb

    # Need to init and GFNTrainer will automatically log to wandb
    # wandb.login()
    # wandb.init(project="gflow_test")
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logp.yaml")
    config = yml2cfg(file_path)
    trial = LogPTrainer(config)
    trial.run()
    # Read SQL
    # from gflownet.utils.sqlite_log import read_all_results

    # results = read_all_results(config.log_dir + "/valid")
    # wandb.finish()

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
