from rdkit import Chem
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
import os
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Disable Hydrogen warnings
RDLogger.DisableLog("rdApp.*")


def feat_encoding(x, feature_set):
    return [x == s for s in feature_set]


def atom_features(atom):
    atom_feats = torch.tensor(
        feat_encoding(
            atom.GetSymbol(),
            ["C", "Br", "F", "P", "S", "O", "Cl", "I", "N"],
        )
        + feat_encoding(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
            ],
        )
        + feat_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + feat_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        + [atom.GetFormalCharge()]
        + [atom.GetIsAromatic()],
        dtype=torch.float32,
    )
    return atom_feats


class Dataset:
    def __init__(self, smiles_list, y_list):
        self.dataset = []
        for smi, y in zip(smiles_list, y_list):
            # TODO: Check sanitization
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() <= 1:
                continue
            x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

            src, dest = [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                src += [start, end]
                dest += [end, start]
            edge_index = torch.tensor([src, dest], dtype=torch.long)
            # TODO: Maybe change the dims
            maccs = torch.tensor(list(AllChem.GetMACCSKeysFingerprint(mol)), dtype=torch.float).unsqueeze(dim=0)
            # Create graph object
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]), fp=maccs, dtype=torch.float)
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def split_data():
    import pandas as pd

    data = pd.read_csv("KOW.csv").iloc[:500]
    y = list(data["active"])
    smi = list(data["smiles"])
    # Calculate split sizes
    total_size = len(smi)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)

    # Create indices and shuffle them
    indices = list(range(total_size))
    torch.manual_seed(42)  # for reproducibility
    torch.randperm(total_size, out=torch.LongTensor(indices))

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create datasets
    train_dataset = Dataset([smi[i] for i in train_indices], [y[i] for i in train_indices])
    val_dataset = Dataset([smi[i] for i in val_indices], [y[i] for i in val_indices])
    test_dataset = Dataset([smi[i] for i in test_indices], [y[i] for i in test_indices])

    return train_dataset, val_dataset, test_dataset


def get_adjacency_matrix(mol):
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def get_node_features(mol):
    return torch.stack(
        [atom_features(atom) for atom in mol.GetAtoms()],
    )


def get_maccs(mol):
    maccs = torch.tensor(list(AllChem.GetMACCSKeysFingerprint(mol)), dtype=torch.float).unsqueeze(dim=0)
    return maccs


def mol2graph(mol):
    data = Data(fp=get_maccs(mol), x=get_node_features(mol), edge_index=get_adjacency_matrix(mol))
    return data


if __name__ == "__main__":
    data = pd.read_csv("KOW.csv")
    y = list(data["active"])
    smi = list(data["smiles"])
    dataset = Dataset(smi, y)
    print(dataset[20])
