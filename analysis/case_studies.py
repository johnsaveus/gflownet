import pickle
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen
import matplotlib.pyplot as plt
from rdkit import Chem
import numpy as np
from gflownet.utils.sascore import calculateScore
from utils import create_dir
from rdkit.Chem import Draw
from rdkit import RDLogger
import warnings

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
pickle_file_path = "inference_list_all.pkl"
create_dir("results/case_studies")
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)


def top_n_sas(smiles, name, n):
    sas_scores = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        sascore = calculateScore(mol)
        sas_scores.append(sascore)

    top_n_indices = np.argsort(sas_scores)[:n]
    top_n_smiles = [smiles[i] for i in top_n_indices]
    top_n_scores = [sas_scores[i] for i in top_n_indices]
    fig, axes = plt.subplots(10, 5, figsize=(10, 20))
    for ax, smile, score in zip(axes.flatten(), top_n_smiles, top_n_scores):
        mol = Chem.MolFromSmiles(smile)
        img = Draw.MolToImage(mol, size=(300, 300))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"SA Score: {score:.2f}", fontsize=12)

    plt.tight_layout()
    plt.savefig("results/case_studies/" + name + "_top10_sas.png")


min_sol = -13.71
max_sol = 2.41
friendly_smiles = []
herbicides = []
pesticides = []
insecticides = []

pesticide_rewards = []
from rdkit.Chem import AllChem, DataStructs
from itertools import combinations


def mean_diversity(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(fp)

    if len(fps) < 2:
        return float("nan")

    dists = []
    for fp1, fp2 in combinations(fps, 2):
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        dists.append(1 - sim)

    return print(np.mean(dists))


for i in data:
    df = i[0]
    rewards = df["reward"].values
    # print(np.mean(rewards))
    smiles = df["smiles"].values
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in smiles]
    # print(f"Mean diversity: {mean_diversity(smiles)}")
    for j in range(len(df)):
        sol = (1 - rewards[j]) * (max_sol - min_sol) + min_sol
        S = np.power(10, sol)
        mol = Chem.MolFromSmiles(smiles[j])
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        ppm = S * mw * 1000
        if ppm < 30:
            friendly_smiles.append(smiles[j])
        # Rules
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rb = Lipinski.NumRotatableBonds(mol)
        logp = Crippen.MolLogP(mol)
        # Aromatic bonds
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        if (
            ppm < 30
            and mw < 500
            and mw > 150
            and logp < 5
            and logp > 0
            and hbd <= 2
            and hba > 1
            and hba < 8
            and rb < 12
        ):
            insecticides.append(smiles[j])

        if ppm < 30 and mw <= 435 and logp <= 6 and hbd <= 2 and hba <= 6 and rb <= 9 and aromatic_bonds <= 17:
            pesticides.append(smiles[j])
            pesticide_rewards.append(rewards[j])
        if ppm < 30 and mw < 500 and mw > 150 and logp <= 3.5 and hbd <= 3 and hba > 2 and hba < 12 and rb < 12:
            herbicides.append(smiles[j])

friendly_smiles = list(set(friendly_smiles))
herbicides = list(set(herbicides))
pesticides = list(set(pesticides))
insecticides = list(set(insecticides))

with open("results/case_studies/friendly_smiles.txt", "w") as f:
    for smile in friendly_smiles:
        f.write(smile + "\n")

with open("results/case_studies/herbicides.txt", "w") as f:
    for smile in herbicides:
        f.write(smile + "\n")

with open("results/case_studies/pesticides.txt", "w") as f:
    for smile in pesticides:
        f.write(smile + "\n")

with open("results/case_studies/insecticides.txt", "w") as f:
    for smile in insecticides:
        f.write(smile + "\n")


top_n_sas(insecticides, "insecticides", 50)
top_n_sas(herbicides, "herbicides", 50)
top_n_sas(pesticides, "pesticides", 50)
