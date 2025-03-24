from typing import Any
from pathlib import Path
import joblib
import pandas as pd
import torch
from gflownet.proxy_chemprop.mpnn_pipeline import load_model
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.models import bengio2021flow
from omegaconf import OmegaConf
from rdkit import Chem
from chemprop import data
from lightning import pytorch as pl
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from gflownet.algo.trajectory_balance import TrajectoryBalance
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse


class ModelService:
    def __init__(self, cfg):
        self.current_dir = Path(".").resolve()
        cfg = Path("gflownet/src/tasks/logs/run/config.yaml")
        self.cfg = OmegaConf.load(cfg)
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        self._load_proxy()
        self._load_sampling()
        self._load_algo()

    def _load_proxy(self):
        proxy_dir = Path("src/gflownet/proxy_chemprop/checkpoints/best-epoch=84-val_loss=0.06.ckpt")
        self.proxy = load_model(proxy_dir)

    def _load_sampling(self):
        model_dir = Path("src/gflownet/tasks/logs/run/model_state.pt")
        self.temp_cond = TemperatureConditional(self.cfg)
        num_cond_dim = self.temp_cond.encoding_size()
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS,
        )
        # Load GFN Model
        model = GraphTransformerGFN(
            env_ctx=self.ctx,
            cfg=self.cfg,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
        )
        model.load_state_dict((torch.load(model_dir)["sampling_model_state_dict"][0]))
        model.eval()
        self.sampling_model = model

    def _load_algo(self):
        env = GraphBuildingEnv()
        self.algo = TrajectoryBalance(env, self.ctx, self.cfg)

    def get_smiles(self, n):
        cond_info = self.temp_cond.sample(n)["encoding"]
        samples = self.algo.create_training_data_from_own_samples(model=self.sampling_model, n=n, cond_info=cond_info)
        trajectories = [sample["traj"] for sample in samples]
        rdkit_mols = [self.ctx.graph_to_obj(traj[-1][0]) for traj in trajectories]
        smiles = [Chem.MolToSmiles(mol) for mol in rdkit_mols]
        return smiles

    def predict(self, request) -> PredictionResponse:
        num = request.dataset.input[0]["numGenerations"]
        if num > 50:
            print("Too many samples. Please enter a number less than 10")
        else:
            smiles = self.get_smiles(num)
            test_data = [data.MoleculeDatapoint.from_smi(smile) for smile in smiles]
            test_dset = data.MoleculeDataset(test_data, featurizer=self.featurizer)
            test_loader = data.build_dataloader(test_dset, shuffle=False)
            trainer = pl.Trainer(logger=None, enable_progress_bar=True, accelerator="cpu", devices=1)
            preds = trainer.predict(self.proxy, test_loader)[0]

        result = [
            {
                "smiles": smile,
                "prediction": round(float(pred), 3),
                "jaqpotMetadata": {"jaqpotRowId": request.dataset.input[0]["jaqpotRowId"]},
            }
            for pred, smile in zip(preds, smiles)
        ]
        return PredictionResponse(predictions=result)


if __name__ == "__main__":
    model = ModelService("config.yaml")
    print(model.predict())
