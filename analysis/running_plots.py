from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem, DataStructs
from gflownet.proxy.mol_utils import smiles2graph
from gflownet.utils.sqlite_log import read_all_results
from utils import obtain_run, load_proxy, get_config, create_dir
import torch_geometric.data as gd
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


def get_bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    return scaffold_smiles


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return fp


def get_scaffold_plot_data(smiles_list):
    proxy = load_proxy()
    unique_scaffolds = set()
    smiles_mb = [smiles_list[i : i + 64] for i in range(0, len(smiles_list), 64)]
    batch_sum = [sum(len(seg) for seg in smiles_mb[: i + 1]) for i in range(len(smiles_mb))]
    scaffold_sum = []
    # For now put minimization
    logp_threshold = 1
    for idx, sm_list in enumerate(smiles_mb):
        scaffs = 0
        # Take batch preds
        graphs = [smiles2graph(Chem.MolFromSmiles(smile)) for smile in sm_list]
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        preds = proxy(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(dim=-1)
        for j, smile in enumerate(sm_list):
            scaffold = get_bemis_murcko_scaffold(smile)
            if scaffold is not None:
                if scaffold not in unique_scaffolds and preds[j] < logp_threshold:
                    unique_scaffolds.add(scaffold)
                    scaffs += 1
        if idx == 0:
            scaffold_sum.append(scaffs)
        else:
            scaffold_sum.append(scaffs + scaffold_sum[-1])
    return scaffold_sum, batch_sum


def get_tanimoto_plot_data(smiles_list, similarity_threshold=0.7):
    proxy = load_proxy()
    mode_fps = []  # List of fingerprints for identified modes
    smiles_mb = [smiles_list[i : i + 64] for i in range(0, len(smiles_list), 64)]
    batch_sum = [sum(len(seg) for seg in smiles_mb[: i + 1]) for i in range(len(smiles_mb))]
    mode_sum = []
    logp_threshold = 7
    for idx, sm_list in enumerate(smiles_mb):
        new_modes = 0
        # Generate graphs and make predictions
        graphs = [smiles2graph(Chem.MolFromSmiles(smile)) for smile in sm_list]
        batch = gd.Batch.from_data_list([g for g in graphs if g is not None])
        preds = proxy(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(dim=-1)

        for j, smile in enumerate(sm_list):
            fp = get_morgan_fp(smile)
            is_new_mode = all(
                DataStructs.TanimotoSimilarity(fp, existing_fp) < similarity_threshold for existing_fp in mode_fps
            )

            if is_new_mode and preds[j] < logp_threshold:
                mode_fps.append(fp)
                new_modes += 1
        if idx == 0:
            mode_sum.append(new_modes)
        else:
            mode_sum.append(mode_sum[-1] + new_modes)

    return mode_sum, batch_sum


if __name__ == "__main__":
    ids = ["tb_lr_5e-4"]
    proxy = load_proxy()

    results_dir = create_dir("results")
    # Create a directory for the plots
    plots_dir = create_dir("results/plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for id in ids:
        run_path = obtain_run(id)
        # Config probably to get the betas
        config = get_config(id)
        results = read_all_results(run_path / "train")
        # Gotta rewrite scaffolds
        smiles = results["smi"]

        y1, x1 = get_scaffold_plot_data(smiles)
        y2, x2 = get_tanimoto_plot_data(smiles)
        axes[0].plot(x1, y1, marker="o", label=f"Scaffolds ({id})")

        # Plot Tanimoto-based diversity
        axes[1].plot(x2, y2, marker="o", label=f"Tanimoto Modes ({id})")

    # Labels and titles
    axes[0].set_xlabel("Number of Molecules Processed")
    axes[0].set_ylabel("Cumulative Unique Scaffolds")
    axes[0].set_title("Unique Scaffolds vs Molecules")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Number of Molecules Processed")
    axes[1].set_ylabel("Cumulative Unique Modes")
    axes[1].set_title("Tanimoto-Diverse Modes vs Molecules")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / "diversity_comparison_subplots.png")
