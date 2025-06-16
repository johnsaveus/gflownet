from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import RDLogger
import warnings

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_smiles_from_txt(file_path):
    with open(file_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list


smiles = load_smiles_from_txt("results/case_studies/herbicides.txt")
# Generate RDKit molecule objects
molecules = [Chem.MolFromSmiles(smile) for smile in smiles]

# Generate ECFP4 fingerprints (Morgan with radius=2)
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in molecules]

# Compute Tanimoto similarity matrix
n = len(fingerprints)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])

# Convert to DataFrame for heatmap
sim_df = pd.DataFrame(similarity_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(sim_df, annot=True, cmap="viridis", square=True)
plt.title("Tanimoto Similarity Heatmap")
plt.tight_layout()
plt.savefig("results/case_studies/tanimoto_similarity_heatmap.png", dpi=300)
