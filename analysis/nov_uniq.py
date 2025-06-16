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


def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    """Convert SMILES to ECFP4 fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_mean_diversity(smiles_list):
    # Convert SMILES to fingerprints
    fps = [smiles_to_ecfp(sm) for sm in smiles_list]
    fps = [fp for fp in fps if fp is not None]  # Remove invalid

    # Compute pairwise Tanimoto similarities
    similarities = []
    for fp1, fp2 in combinations(fps, 2):
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarities.append(sim)

    mean_similarity = np.mean(similarities)
    mean_diversity = 1 - mean_similarity
    return mean_diversity


def compute_uniqueness_and_novelty(proxy_dataset, generated_samples):
    # Convert to sets for fast lookup and uniqueness checks
    proxy_set = set(proxy_dataset)
    generated_set = set(generated_samples)

    n_generated = len(generated_samples)
    n_unique = len(generated_set)

    # Uniqueness: how many of the generated are unique
    uniqueness = n_unique / n_generated

    # Novelty: how many of the generated are not in proxy set
    n_novel = sum(1 for x in generated_samples if x not in proxy_set)
    novelty = n_novel / n_generated

    return {
        "n_generated": n_generated,
        "n_unique": n_unique,
        "n_novel": n_novel,
        "uniqueness": round(uniqueness * 100, 2),
        "novelty": round(novelty * 100, 2),
    }


ids = ["fm_experiment(lr=1e-4)", "db_lr(1e-4)", "tb_lr(1e-4)", "subtb_lr(1e-4)"]
# ids = ["db_lr(1e-4)", "db_(pb,lr=1e-4)", "tb_lr(1e-4)", "tb_(pb,lr=1e-4)"]
for id in ids:
    print("Processing run:", id)
    run_path = obtain_run(id)
    results = read_all_results(run_path / "train")
    smiles = results["smi"]

    # Run and print
    metrics = compute_uniqueness_and_novelty(sol_smiles, smiles)
    # mean_diversity = compute_mean_diversity(smiles)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    # print(f"Mean Diversity: {mean_diversity:.4f}")
