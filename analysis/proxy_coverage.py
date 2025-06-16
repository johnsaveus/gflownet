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

data_url = r"https://raw.githubusercontent.com/CesareWang/Predictors-for-15-Environmental-Endpoints/main/predictors/data/SW.csv"
sol_data = pd.read_csv(data_url, index_col=0)
sol_smiles = list(sol_data["smiles"])

# ids = ["fm_experiment(lr=5e-4)", "db_lr(1e-3)", "tb_lr(1e-3)", "subtb_lr(5e-4)"]

id = "random_agent"
# id = "fm_experiment(lr=5e-4)"
# id = "db_lr(1e-3)"
# id = "tb_lr(1e-3)"
# id = "subtb_lr(5e-4)"
run_path = obtain_run(id)
train_results = read_all_results(run_path / "train")
train_smiles = train_results["smi"]

val_results = read_all_results(run_path / "valid")
val_smiles = val_results["smi"]


# common_smiles = set(train_smiles) & set(sol_smiles)
# print(f"Number of SMILES in both sets: {len(common_smiles)}")
# print(len(common_smiles) / len(sol_smiles) * 100)

# Find how many SMILES in 'smiles' are also in 'sol_smiles'
common_smiles = set(val_smiles) & set(sol_smiles)
print(f"Number of SMILES in both sets: {len(common_smiles)}")
print(len(common_smiles) / len(sol_smiles) * 100)

# Find how many SMILES in 'val_smiles' are also in 'train_smiles'
common_val_train = set(val_smiles) & set(train_smiles)
print(f"Number of val_smiles in train_smiles: {len(common_val_train)}")
print(len(common_val_train) / len(val_smiles) * 100)
