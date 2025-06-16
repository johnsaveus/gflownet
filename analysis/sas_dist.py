import matplotlib.pyplot as plt
from rdkit import Chem
from gflownet.utils.sascore import calculateScore  # or from your own module
import seaborn as sns


def load_smiles_from_txt(file_path):
    with open(file_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list


def compute_sas_scores(smiles_list):
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                score = calculateScore(mol)
                scores.append(score)
            except:
                continue
    return scores


# Example input lists (replace with your data)
list1 = load_smiles_from_txt("results/case_studies/pesticides.txt")
list2 = load_smiles_from_txt("results/case_studies/insecticides.txt")
list3 = load_smiles_from_txt("results/case_studies/herbicides.txt")

# Compute scores
sas1 = compute_sas_scores(list1)
sas2 = compute_sas_scores(list2)
sas3 = compute_sas_scores(list3)

# Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(sas1, label="Pesticides", linewidth=2)
sns.kdeplot(sas2, label="Insecticides", linewidth=2)
sns.kdeplot(sas3, label="Herbicides", linewidth=2)

plt.xlabel("SAS Score", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/case_studies/sas_score_distributions.png", dpi=300)
plt.show()
