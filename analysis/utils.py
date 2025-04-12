import torch
import yaml
from pathlib import Path
from gflownet.proxy.model import GraphAttention
from gflownet.utils.sqlite_log import read_all_results


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def obtain_run(id):
    repo_dir = Path.cwd().parent
    run = repo_dir / "src" / "gflownet" / "tasks" / "logs" / id
    return run


def get_config(id):
    run_path = obtain_run(id)
    config_path = run_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def infer_model(model, loader):
    preds = []
    for data in loader:
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.extend([pr for pr in pred.squeeze(dim=-1).cpu().numpy()])
    return preds


def load_proxy():
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
