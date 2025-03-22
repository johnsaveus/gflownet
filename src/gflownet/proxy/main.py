import pandas as pd
from rdkit import Chem
from gflownet.proxy.model import GraphAttention
from gflownet.proxy.mol_utils import scaffold_split, smiles2graph
from gflownet.proxy.train_utils import build_loaders

if __name__ == "__main__":
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
    tr_loader, val_loader, te_loader = build_loaders(train_dataset, valid_dataset, test_dataset, batch_size=64)

    # ------------- Build Model

    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=2,
        gnn_channels=16,
        heads=2,
        dropout_proba=0.1,
        gnn_norm=True,
        mlp_layers=2,
        mlp_channels=16,
        mlp_norm=True,
    )
    # --------------- Training
