import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from gflownet.proxy_chemprop.mpnn_pipeline import load_model
from omegaconf import OmegaConf
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.utils.conditioning import TemperatureConditional
from pathlib import Path
from lightning import pytorch as pl
from gflownet.models import bengio2021flow
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop import data

current_dir = Path(".").resolve()
run_name = "run"
proxy_ckp = "best-epoch=84-val_loss=0.06.ckpt"
yaml_dir = current_dir / "src" / "gflownet" / "tasks" / "logs" / run_name / "config.yaml"
model_dir = current_dir / "src" / "gflownet" / "tasks" / "logs" / run_name / "model_state.pt"
proxy_dir = current_dir / "src" / "gflownet" / "proxy_chemprop" / "checkpoints" / proxy_ckp
cfg = OmegaConf.load(yaml_dir)
# Load env
env = GraphBuildingEnv()
temp_cond = TemperatureConditional(cfg)
num_cond_dim = temp_cond.encoding_size()
ctx = FragMolBuildingEnvContext(
    max_frags=cfg.algo.max_nodes,
    num_cond_dim=num_cond_dim,
    fragments=bengio2021flow.FRAGMENTS,
)
# Load GFN Model
model = GraphTransformerGFN(
    env_ctx=ctx,
    cfg=cfg,
    num_graph_out=cfg.algo.tb.do_predict_n + 1,
    do_bck=cfg.algo.tb.do_parameterize_p_b,
)
model.load_state_dict((torch.load(model_dir)["sampling_model_state_dict"][0]))
model.eval()
# Load Algo
algo = TrajectoryBalance(env, ctx, cfg)
# Load cond_info
np.random.seed(42)
torch.manual_seed(42)
cond_info = temp_cond.sample(10)["encoding"]
samples = algo.create_training_data_from_own_samples(model=model, n=10, cond_info=cond_info)
trajectories = [sample["traj"] for sample in samples]
# valid = [sample["is_valid"] for sample in samples]
rdkit_mols = [ctx.graph_to_obj(traj[-1][0]) for traj in trajectories]
# Calc reward
# TODO: Automate this
model = load_model(proxy_dir)
featurizer = SimpleMoleculeMolGraphFeaturizer()
smiles = [Chem.MolToSmiles(mol) for mol in rdkit_mols]
test_data = [data.MoleculeDatapoint.from_smi(smile) for smile in smiles]
test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, shuffle=False)
trainer = pl.Trainer(logger=None, enable_progress_bar=True, accelerator="cpu", devices=1)
preds = trainer.predict(model, test_loader)[0]
print(preds)
img = Draw.MolToImageFile(rdkit_mols[1], "test.png")
# img = Draw.MolsToGridImage(
#     [Chem.MolFromSmiles(mol) for mol in results["smiles"]],
#     molsPerRow=5,
#     legends=[rew for rew in results["r"]],
#     subImgSie=(200, 200),
# )
