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
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward
from gflownet.proxy_chemprop.mpnn_pipeline import load_model
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop import data

CKP_PATH = "../proxy_chemprop/checkpoints/best-epoch=84-val_loss=0.06.ckpt"


class LogPTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy mpnn model that outputs LogP of a molecule."""

    def __init__(
        self,
        cfg: Config,
        rew_tran: str = "0-10",
        y_min: float = -5.08,
        y_max: float = 11.29,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
        mpnn_ckp_path: str = CKP_PATH,
    ) -> None:
        self.cfg = cfg
        self.rew_tran = rew_tran
        self.y_min = y_min
        self.y_max = y_max
        self.width = y_max - y_min
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.mpnn_ckp_path = mpnn_ckp_path
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        if self.rew_tran == "exp":
            flat_r = np.exp(-(y - self.y_min) / self.width)
        elif self.rew_tran == "unit":
            flat_r = (y - self.y_min) / self.width
        elif self.rew_tran == "0-10":
            flat_r = ((y - self.y_min) / self.width) * 10
        return flat_r

    def inv_transform(self, rp: Tensor) -> float:
        if self.rew_tran == "exp":
            inv_r = -np.log(rp) * self.width + self.y_min
        elif self.rew_tran == "unit":
            inv_r = (1 - rp) * self.width + self.y_min
        elif self.rew_tran == "0-10":
            inv_r = (rp / 10) * self.width + self.y_min
        return inv_r

    def _load_task_models(self):
        # Load MPNN Model from Chemprop with best ckp
        model = load_model(self.mpnn_ckp_path)  # .to(get_worker_device())
        model = self._wrap_model(model)
        return {"logp": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, loader) -> Tensor:
        trainer = pl.Trainer(logger=None, enable_progress_bar=True, accelerator="cpu", devices=1)
        preds = trainer.predict(self.models["logp"], loader)[0].reshape((-1)).data.cpu()
        preds[preds.isnan()] = 0
        preds = (
            self.reward_transform(preds)
            .clip(1e-4, 20)
            .reshape(
                -1,
            )
        )
        return preds

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        ####
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        test_data = [data.MoleculeDatapoint.from_smi(smile) for smile in smiles]
        test_dset = data.MoleculeDataset(test_data, featurizer=self.featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        ###
        is_valid = torch.tensor([i is not None for i in test_dset]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid
        preds = self.compute_reward_from_graph(test_loader).reshape((-1, 1))
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
        from rdkit.Chem.rdchem import ChiralType

        self.ctx = MolBuildingEnvContext(
            atoms=["C", "O"],  #
            chiral_types=[ChiralType.CHI_UNSPECIFIED],
            expl_H_range=[0, 1],
            charges=[-1, 0, 1],
            allow_explicitly_aromatic=False,
            allow_5_valence_nitrogen=False,
            num_cond_dim=self.task.num_cond_dim,
        )

    def setup(self):
        super().setup()


def main():
    """Example of how this model can be run."""
    import wandb

    # Need to init and GFNTrainer will automatically log to wandb
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logp.yaml")
    config = yml2cfg(file_path)
    wandb.login()
    wandb.init(project="gflow_mol_building", config=config)
    trial = LogPTrainer(config)
    trial.run()
    # Read SQL
    # from gflownet.utils.sqlite_log import read_all_results

    # results = read_all_results(config.log_dir + "/valid")
    wandb.finish()
    # Need to decide what to log in wandb
    # 1) online_loss which is the same as tb_loss
    # 2) sampled_reward_avg
    # 3) smiles and reward on every validation. Eventually we want to see the distribution over time


if __name__ == "__main__":
    main()
