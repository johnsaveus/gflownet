import random
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem
import torch_geometric.data as gd
from gflownet.proxy.mol_utils import smiles2graph
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.flow_matching import FlowMatching
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from utils import load_proxy_sol
# import warnings
# from rdkit import RDLogger

# RDLogger.DisableLog("rdApp.*")
# warnings.filterwarnings("ignore", category=DeprecationWarning)


# ids = ["fm_experiment(lr=1e-4)", "fm_experiment(lr=5e-4)", "fm_experiment(lr=1e-3)"]
# ids = ["db_lr(1e-4)", "db_lr(5e-4)", "db_lr(1e-3)"]
# ids = ["tb_lr(1e-4)", "tb_lr(5e-4)", "tb_lr(1e-3)"]
# ids = ["subtb_lr(1e-4)", "subtb_lr(5e-4)", "subtb_lr(1e-3)"]
ids = ["fm_experiment(lr=5e-4)", "db_lr(1e-3)", "tb_lr(1e-3)", "subtb_lr(5e-4)"]
current_dir = Path.cwd().parent
proxy_model = load_proxy_sol()
df_ids = []
# Iterate over each algo
for i, id in enumerate(ids):
    yaml_dir = current_dir / "src" / "gflownet" / "tasks" / "logs" / id / "config.yaml"
    model_dir = current_dir / "src" / "gflownet" / "tasks" / "logs" / id / "model_state.pt"
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
    model.load_state_dict((torch.load(model_dir)["models_state_dict"][0]))
    model.eval()
    # Load Algo
    if cfg.algo.method == "TB":
        algo = TrajectoryBalance(env, ctx, cfg)
    else:
        algo = FlowMatching(env, ctx, cfg)
    # Load cond_info
    total_smiles = []
    total_preds = []
    total_policy = []
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_gen = 100
    min_sol = -13.71
    max_sol = 2.41
    for _ in range(250):
        # Sample from the GFlowNet
        cond_info = temp_cond.sample(num_gen)["encoding"]
        samples = algo.create_training_data_from_own_samples(model=model, n=num_gen, cond_info=cond_info)
        trajectories = [sample["traj"] for sample in samples]
        rdkit_mols = [ctx.graph_to_obj(traj[-1][0]) for traj in trajectories]
        smiles = [Chem.MolToSmiles(mol) for mol in rdkit_mols]
        # Get preds
        graphs = [smiles2graph(Chem.MolFromSmiles(smile)) for smile in smiles]
        batch = gd.Batch.from_data_list([g for g in graphs if g is not None])
        preds = (
            proxy_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(dim=-1).cpu().detach().numpy()
        )
        scaled_preds = list(1 - ((preds - min_sol) / (max_sol - min_sol)))
        policy = [p["fwd_logprob"].item() for p in samples]
        # Add batch preds
        total_preds.extend(scaled_preds)
        total_smiles.extend(smiles)
        total_policy.extend(policy)
    df_all = [pd.DataFrame({"smiles": total_smiles, "reward": total_preds, "policy": total_policy})]
    df_ids.append(df_all)
with open("inference_list_all.pkl", "wb") as file:
    pickle.dump(df_ids, file)
