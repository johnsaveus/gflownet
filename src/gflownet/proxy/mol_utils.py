import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


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
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
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


def get_edge_features(mol):
    features = []
    for bond in mol.GetBonds():
        bond_feat = bond_features(bond)
        features.append(bond_feat)  # i->j
        features.append(bond_feat)  # j->i
    return torch.tensor(features, dtype=torch.float)


def smiles2graph(mol, y=None):
    if y is not None:
        data = Data(
            x=get_node_features(mol),
            edge_index=get_adjacency_matrix(mol),
            edge_attr=get_edge_features(mol),
            y=torch.tensor([y]),
        )
    else:
        data = Data(x=get_node_features(mol), edge_index=get_adjacency_matrix(mol), edge_attr=get_edge_features(mol))
    return data


def random_split(frac_train, smiles):
    from sklearn.model_selection import train_test_split

    frac_valid_test = 1 - frac_train
    train, valid_test = train_test_split(smiles, test_size=frac_valid_test, random_state=42)
    valid, test = train_test_split(valid_test, test_size=0.5, random_state=42)

    return train.index, valid.index, test.index


def scaffold_split(frac_train, frac_valid, smiles, include_chirality=False):
    """Taken from deepchem"""

    def _generate_scaffold(smiles, include_chirality):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold

    def _generate_scaffolds(smiles, include_chirality):
        scaffolds = {}
        for ind, sm in enumerate(smiles):
            scaffold = _generate_scaffold(sm, include_chirality=include_chirality)
            if scaffold is not None:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [ind]
                else:
                    scaffolds[scaffold].append(ind)
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        return scaffold_sets

    scaffold_sets = _generate_scaffolds(smiles, include_chirality)

    train_cutoff = frac_train * len(smiles)
    valid_cutoff = (frac_train + frac_valid) * len(smiles)
    train_inds = []
    valid_inds = []
    test_inds = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds
