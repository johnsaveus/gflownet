import torch
import random
import os
import json
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from gflownet.proxy.model import GraphAttention
from gflownet.proxy.mol_utils import smiles2graph, random_split
from gflownet.proxy.train_utils import build_loaders, infer_model, get_metrics, plot_results

if __name__ == "__main__":
    # ------------  Hyperparameters for tuning
    def parse_args():
        argparser = argparse.ArgumentParser(description="GNN for KOW prediction")
        argparser.add_argument("--learning_rate", type=float, default=0.001)
        argparser.add_argument("--batch_size", type=int, default=64)
        argparser.add_argument("--gnn_layers", type=int, default=2)
        argparser.add_argument("--gnn_channels", type=int, default=64)
        argparser.add_argument("--heads", type=int, default=8)
        argparser.add_argument("--mlp_layers", type=int, default=2)
        argparser.add_argument("--dropout_proba", type=float, default=0.2)
        return argparser.parse_args()

    args = parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # ------------  Load data
    data_url = r"https://raw.githubusercontent.com/CesareWang/Predictors-for-15-Environmental-Endpoints/main/predictors/data/SW.csv"
    kow_data = pd.read_csv(data_url, index_col=0)
    kow_data.rename(columns={"active": "logKOW"}, inplace=True)

    # ------------ Create data splits randomly
    train_id, valid_id, test_id = random_split(frac_train=0.80, smiles=kow_data)
    train_data = kow_data.loc[train_id]
    valid_data = kow_data.loc[valid_id]
    test_data = kow_data.loc[test_id]

    # ------------ Create GNN datasets
    train_dataset = [
        smiles2graph(Chem.MolFromSmiles(smiles), y) for smiles, y in zip(train_data["smiles"], train_data["logKOW"])
    ]
    valid_dataset = [
        smiles2graph(Chem.MolFromSmiles(smiles), y) for smiles, y in zip(valid_data["smiles"], valid_data["logKOW"])
    ]
    test_dataset = [
        smiles2graph(Chem.MolFromSmiles(smiles), y) for smiles, y in zip(test_data["smiles"], test_data["logKOW"])
    ]
    # ------------ Loaders
    tr_loader, val_loader, te_loader = build_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=args.batch_size
    )
    # ------------- Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=args.gnn_layers,
        gnn_channels=args.gnn_channels,
        heads=args.heads,
        dropout_proba=args.dropout_proba,
        gnn_norm=False,
        mlp_layers=args.mlp_layers,
        mlp_channels=args.gnn_channels,
        mlp_norm=False,
    ).to(device)

    model.load_state_dict(torch.load("best_model_sol.pt", map_location=torch.device("cpu")))
    model.eval()

    # Train metrics
    all_preds, all_true = infer_model(model, tr_loader, device)
    rmse_train, mae_train, r2_train = get_metrics(all_preds, all_true)

    # Validation metrics
    all_preds, all_true = infer_model(model, val_loader, device)
    rmse_val, mae_val, r2_val = get_metrics(all_preds, all_true)

    # Test metrics
    all_preds, all_true = infer_model(model, te_loader, device)
    rmse_test, mae_test, r2_test = get_metrics(all_preds, all_true)

    if not os.path.exists("results"):
        os.makedirs("results")
    metrics = {
        "train": {"RMSE": rmse_train, "MAE": mae_train, "R2": r2_train},
        "validation": {"RMSE": rmse_val, "MAE": mae_val, "R2": r2_val},
        "test": {"RMSE": rmse_test, "MAE": mae_test, "R2": r2_test},
    }
    with open("results/metrics_sol.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_results(all_preds, all_true)
