from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import DataStructs
from gflownet.utils.sascore import calculateScore
import matplotlib.pyplot as plt


def load_smiles_from_txt(file_path):
    with open(file_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list


def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)


def tanimoto_distance(fp1, fp2):
    return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)


def most_diverse_subset(smiles_list, top_n=5):
    fps = [(smi, smiles_to_fingerprint(smi)) for smi in smiles_list]
    fps = [(smi, fp) for smi, fp in fps if fp is not None]

    if len(fps) <= top_n:
        return [smi for smi, _ in fps]

    selected = [fps[0]]
    remaining = fps[1:]

    for _ in range(top_n - 1):
        distances = []
        for smi, fp in remaining:
            dists = [tanimoto_distance(fp, sel_fp) for _, sel_fp in selected]
            min_dist = min(dists)
            distances.append((smi, fp, min_dist))

        next_sel = max(distances, key=lambda x: x[2])
        selected.append((next_sel[0], next_sel[1]))
        remaining = [(smi, fp) for smi, fp in remaining if smi != next_sel[0]]

    return [smi for smi, _ in selected]


def plot_grid_with_titles(rows_dict, save_path="diverse_labeled.png"):
    """rows_dict: {"Pesticides": [...], "Herbicides": [...], "Insecticides": [...]}"""
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

    for row_idx, (title, smiles_list) in enumerate(rows_dict.items()):
        for col_idx, smi in enumerate(smiles_list):
            ax = axs[row_idx, col_idx]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                sa = calculateScore(mol)
                img = Draw.MolToImage(mol, size=(200, 200))
                ax.imshow(img)
                ax.set_title(f"SA Score: {sa:.2f}", fontsize=8)
            ax.axis("off")

        # Add row title to the left of the row
        axs[row_idx, 0].annotate(
            title,
            xy=(0, 0.5),
            xycoords="axes fraction",
            fontsize=14,
            ha="right",
            va="center",
            rotation=90,
            xytext=(-axs[row_idx, 0].bbox.width * 0.25, 0),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# Load files
p = load_smiles_from_txt("results/case_studies/pesticides.txt")
i = load_smiles_from_txt("results/case_studies/insecticides.txt")
h = load_smiles_from_txt("results/case_studies/herbicides.txt")

# Get diverse molecules
diverse_p = most_diverse_subset(p, top_n=5)
diverse_h = most_diverse_subset(h, top_n=5)
diverse_i = most_diverse_subset(i, top_n=5)

# Plot with row labels
plot_grid_with_titles({"Pesticides": diverse_p, "Herbicides": diverse_h, "Insecticides": diverse_i})
