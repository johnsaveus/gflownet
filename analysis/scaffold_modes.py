import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import MACCSkeys
from gflownet.proxy.mol_utils import random_split
from gflownet.utils.sqlite_log import read_all_results
from utils import *
import json


def get_bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
    return scaffold_smiles


def get_scaffold_modes(smiles_list, rew_list):
    "Find unique scaffolds based on Bemis-Murcko"
    # Return modes
    ppm_threshold = 30
    logS = rew2logS(rew_list)
    ppm = logS2ppm(smiles_list, logS)
    unique_scaffolds = set()
    smiles_seen = set()
    # rew_threshold = 0.417
    scaffold_sum = []
    # For now put minimization
    for idx, smile in enumerate(smiles_list):
        # scaffs = 0
        scaffold = get_bemis_murcko_scaffold(smile)
        if scaffold is not None:
            if scaffold not in unique_scaffolds and ppm[idx] > ppm_threshold and smile not in smiles_seen:
                unique_scaffolds.add(scaffold)
                if idx == 0:
                    scaffold_sum.append(1)
                else:
                    scaffold_sum.append(scaffold_sum[-1] + 1)
            else:
                if idx == 0:
                    scaffold_sum.append(0)
                else:
                    scaffold_sum.append(scaffold_sum[-1])
        else:
            scaffold_sum.append(scaffold_sum[-1])
        smiles_seen.add(smile)
    return scaffold_sum


def main(ids):
    create_dir("results")
    create_dir("results/modes")
    # Create a directory for the plots
    modes = []
    for id in ids:
        run_path = obtain_run(id)
        # Config probably to get the betas
        results = read_all_results(run_path / "train")
        # Gotta rewrite scaffolds#else:
        smiles = results["smi"]
        rewards = results["r"]
        y1 = get_scaffold_modes(smiles, rew_list=rewards)
        modes.append(max(y1) / 320000)
    # Save the modes data to a JSON file
    modes_dict = {id: mode for id, mode in zip(ids, modes)}
    with open("results/modes/scaffold_modes.json", "w") as f:
        json.dump(modes_dict, f)


ids = [
    "fm_experiment(lr=1e-4)",
    "fm_experiment(lr=5e-4)",
    "fm_experiment(lr=1e-3)",
    "db_lr(1e-4)",
    "db_lr(5e-4)",
    "db_lr(1e-3)",
    "tb_lr(1e-4)",
    "tb_lr(5e-4)",
    "tb_lr(1e-3)",
    "subtb_lr(1e-4)",
    "subtb_lr(5e-4)",
    "subtb_lr(1e-3)",
]
main(ids)

# ids = ["db_lr(1e-4)", "db_lr(5e-4)", "db_lr(1e-3)"]
# plot_modes(ids)

# ids = ["tb_lr(1e-4)", "tb_lr(5e-4)", "tb_lr(1e-3)"]
# plot_modes(ids)

# ids = ["subtb_lr(1e-4)", "subtb_lr(5e-4)", "subtb_lr(1e-3)"]
# plot_modes(ids)

# ids = ["fm_experiment(lr=5e-4)", "db_lr(5e-4)", "tb_lr(5e-4)", "subtb_lr(1e-3)"]
# plot_modes(ids)
