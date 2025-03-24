import torch
import random
import pandas as pd
import numpy as np
from rdkit import Chem
from gflownet.proxy.model import GraphAttention
from gflownet.proxy.mol_utils import scaffold_split, smiles2graph
from gflownet.proxy.train_utils import build_loaders, train_epoch, eval_epoch, infer_model, get_metrics, plot_results

if __name__ == "__main__":
    # ------------  Hyperparameters for tuning
    hparams = {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "batch_size": [64, 128, 256],
        "gnn_layers": [2, 3, 4],
        "gnn_channels": [32, 64, 128],
        "heads": [2, 4, 6, 8],
        "mlp_layers": [1, 2, 3],
        "mlp_channels": [32, 64, 128],
        "dropout_proba": [0.1, 0.2, 0.3],
    }
    # ------------  Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # ------------  Load data
    data_url = r"https://raw.githubusercontent.com/CesareWang/Predictors-for-15-Environmental-Endpoints/main/predictors/data/KOW.csv"
    kow_data = pd.read_csv(data_url, index_col=0).iloc[:300]
    kow_data.rename(columns={"active": "logKOW"}, inplace=True)

    # ------------ Create data splits with scaffold
    train_id, valid_id, test_id = scaffold_split(
        frac_train=0.75, frac_valid=0.125, smiles=list(kow_data["smiles"]), include_chirality=True
    )
    train_data = kow_data.iloc[train_id]
    valid_data = kow_data.iloc[valid_id]
    test_data = kow_data.iloc[test_id]

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
    tr_loader, val_loader, te_loader = build_loaders(train_dataset, valid_dataset, test_dataset, batch_size=128)

    # ------------- Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=2,
        gnn_channels=32,
        heads=4,
        dropout_proba=0.3,
        gnn_norm=True,
        mlp_layers=2,
        mlp_channels=32,
        mlp_norm=True,
    ).to(device)

    # --------------- Training configs
    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, verbose=True)

    # --------------- Training
    for epoch in range(epochs):
        train_loss = train_epoch(model, tr_loader, optimizer, scheduler, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Inference
    all_preds, all_true = infer_model(model, te_loader, device)
    rmse, mae, r2 = get_metrics(all_preds, all_true)
    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    plot_results(all_preds, all_true)
