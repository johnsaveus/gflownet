from pathlib import Path
import yaml
import torch
from gflownet.proxy.model import GraphAttention


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def scale_rew(sol):
    """Scale the solubility values to be between 0 and 1."""
    min_sol = -13.71
    max_sol = 2.41
    return [(1 - (s - min_sol) / (max_sol - min_sol)) for s in sol]


def create_dir(name):
    """Create directory if it doesn't exist"""
    path = Path.cwd() / name
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def obtain_run(run_id):
    "Get run directory"
    repo_dir = Path.cwd().parent
    run = repo_dir / "src" / "gflownet" / "tasks" / "logs" / run_id
    return run


def get_config(run_id):
    "Get yaml from run"
    run_path = obtain_run(run_id)
    config_path = run_path / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def infer_model(model, loader):
    "Get molecule proxty preds"
    preds = []
    for data in loader:
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.extend([pr for pr in pred.squeeze(dim=-1).cpu().numpy()])
    return preds


def load_proxy():
    """Load the LogP prediction model"""
    proxy_path = Path.cwd().parent / "src" / "gflownet" / "proxy" / "best_model.pt"
    proxy_path = proxy_path.resolve()
    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=3,
        gnn_channels=128,
        heads=8,
        dropout_proba=0.2,
        gnn_norm=False,
        mlp_layers=2,
        mlp_channels=128,
        mlp_norm=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(proxy_path, map_location=DEVICE))
    model.eval()
    return model


def load_proxy_sol():
    "Load the solubility prediction model"
    proxy_path = Path.cwd().parent / "src" / "gflownet" / "proxy" / "best_model_sol.pt"
    proxy_path = proxy_path.resolve()
    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=2,
        gnn_channels=64,
        heads=8,
        dropout_proba=0.2,
        gnn_norm=False,
        mlp_layers=2,
        mlp_channels=64,
        mlp_norm=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(proxy_path, map_location=DEVICE))
    model.eval()
    return model
