import pandas as pd
from gflownet.utils.sqlite_log import read_all_results
from utils import *
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from itertools import combinations
from rdkit import RDLogger
import warnings

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


all_ids = {"fm": "fm_experiment(lr=5e-4)", "db": "db_lr(1e-3)", "tb": "tb_lr(1e-3)", "subtb": "subtb_lr(5e-4)"}

# Set target ID (the one to compare against others)
target_key = "subtb"
target_id = all_ids[target_key]

# Load SMILES from the target experiment
target_path = obtain_run(target_id)
target_smiles = set(read_all_results(target_path / "valid")["smi"])

# Compare with SMILES from the other experiments
overlapping_smiles = set()
for key, exp_id in all_ids.items():
    if key == target_key:
        continue
    path = obtain_run(exp_id)
    smiles_set = set(read_all_results(path / "valid")["smi"])
    overlapping_smiles.update(target_smiles.intersection(smiles_set))

# Print the number of overlapping SMILES
print(f"Number of SMILES in '{target_key}' also found in other experiments: {len(overlapping_smiles)}")
