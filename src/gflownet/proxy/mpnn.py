from torch import Tensor
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.data import Data
import torch
from rdkit import Chem


class GraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        heads: List[int],
        fc_dims: List[int],
        dropout_proba: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        use_external: bool = False,
        edge_dim: Optional[int] = None,
    ):
        super(GraphTransformer, self).__init__()

        assert hidden_dims[-1] * heads[-1] == fc_dims[0]
        if not use_external:
            assert fc_dims[-1] == 1
        self.gnn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_proba)

        self.gnn_layers.append(TransformerConv(input_dim, hidden_dims[0], heads[0], edge_dim=edge_dim))

        for ix in range(len(hidden_dims) - 1):
            self.gnn_layers.append(
                TransformerConv(
                    hidden_dims[ix] * heads[ix],
                    hidden_dims[ix + 1],
                    heads[ix + 1],
                    edge_dim=edge_dim,
                )
            )

        for ix in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Linear(fc_dims[ix], fc_dims[ix + 1]))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor], batch: Tensor) -> Tensor:
        for gnn_layer in self.gnn_layers:
            x = self.dropout(self.activation(gnn_layer(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch)
        for fc_layer in self.fc_layers[:-1]:
            x = self.dropout(self.activation(fc_layer(x)))
        x = self.fc_layers[-1](x)
        return x


def load_mpnn_to_gflow(saved_model_path, input_dim=28, edge_dim=7):
    model = GraphTransformer(
        input_dim,
        [32, 32],
        [4, 4],
        [32 * 4, 32, 1],
        edge_dim=edge_dim,
    )
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    model.eval()
    return model


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return torch.tensor(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ["As", "B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si"],
        )
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
            ],
        )
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3])
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        + [atom.GetFormalCharge()]
        + [atom.GetIsAromatic()],
        dtype=torch.float,
    )


def bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        bond.GetStereo(),
    ]

    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOZ", "STEREOE"])

    return bond_feats


def get_node_features(mol):
    return torch.stack(
        [atom_features(atom) for atom in mol.GetAtoms()],
    )


def get_edge_features(mol):
    features = []
    for bond in mol.GetBonds():
        bond_feat = bond_features(bond)
        features.append(bond_feat)  # i->j
        features.append(bond_feat)  # j->i
    return torch.tensor(features, dtype=torch.float)


def get_adjacency_matrix(mol):
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def mol2graph(mol):
    data = Data(x=get_node_features(mol), edge_index=get_adjacency_matrix(mol), edge_attr=get_edge_features(mol))
    return data
