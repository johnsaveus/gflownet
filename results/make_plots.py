from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from gflownet.utils.sqlite_log import read_all_results

y_min = -5.08
y_max = 11.29
current_dir = Path(".").resolve()
# TODO: Do this in an automatic way
run_name = "run"
target_dir = current_dir.parent / "src" / "gflownet" / "tasks" / "logs" / run_name
# train_dir = target_dir / "train"
# val_dir = target_dir / "valid"
final_dir = target_dir / "final"
# Final gen (Maybe generate 10.000 to match dataset)
results = read_all_results(final_dir)
# Load proxy dataset
data_name = "KOW.csv"
proxy_path = current_dir.parent / "src" / "gflownet" / "proxy_chemprop" / "data"
df = pd.read_csv(proxy_path / data_name)
unorml_rew = df["active"].values
norm_rew = ((unorml_rew - y_min) / (y_max - y_min)) * 10


# https://stackoverflow.com/questions/58989973/how-to-smooth-a-probability-distribution-plot-in-python
def rew_hist(reward):
    from scipy.stats import gaussian_kde
    import numpy as np

    kde = gaussian_kde(reward)
    values = np.linspace(reward.min(), reward.max(), 1000)
    probabilities = kde(values)
    return pd.DataFrame(dict(value=values, prob=probabilities))
    # var_range = reward.max() - reward.min()
    # probabilities, values = np.histogram(reward, bins=30, density=True)
    # return pd.DataFrame(dict(value=values[:-1], prob=probabilities))


ax = rew_hist(norm_rew).plot.line(x="value", y="prob", label="Dataset", color="blue")
rew_hist(results["r"]).plot.line(x="value", y="prob", label="GFlownet", color="orange", ax=ax)
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.title("Comparison of Reward Distributions")
plt.grid(linestyle="--", alpha=0.7)
plt.legend()
plt.savefig("plots/dist")


# Check some molecules
img = Draw.MolsToGridImage(
    mols=[Chem.MolFromSmiles(smi) for smi in results["smi"].iloc[:6]],
    molsPerRow=3,
    legends=[f"Reward = {rew:.2f}" for rew in results["r"].iloc[:6]],
)
img.save("plots/mols.png")

# Tanimoto similarity
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs


def generate_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    return fingerprint


smiles = results["smi"]
maccs_fingerprints = [generate_maccs_fingerprint(smile) for smile in smiles]

import numpy as np

# Compute pairwise Tanimoto similarities
n_samples = len(maccs_fingerprints)
similarity_matrix = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(i, n_samples):
        similarity = DataStructs.TanimotoSimilarity(maccs_fingerprints[i], maccs_fingerprints[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Symmetric matrix

# Calculate average pairwise similarity
# Extract upper triangular part of the similarity matrix (excluding diagonal)
upper_triangular = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
# Calculate average and standard deviation
average_similarity = np.mean(upper_triangular)
std_similarity = np.std(upper_triangular)

with open("plots/tanimoto_similarity_results.txt", "w") as file:
    file.write(f"Average pairwise Tanimoto similarity (MACCS): {average_similarity:.4f}\n")
    file.write(f"Standard deviation of pairwise Tanimoto similarity (MACCS): {std_similarity:.4f}\n")

with open("plots/smiles.txt", "w") as file:
    for smiles in results["smi"].iloc[::]:
        file.write(f"{smiles}\n")
