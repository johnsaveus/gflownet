import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem, DataStructs
from gflownet.proxy.mol_utils import smiles2graph
from gflownet.utils.sqlite_log import read_all_results
from utils import obtain_run, load_proxy, get_config
import torch_geometric.data as gd
from torch_geometric.loader import DataLoader


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


def top_1000_statistics(smiles):
    graphs = [smiles2graph(smile) for smile in smiles]
    loader = DataLoader(graphs, batch_size=2000, shuffle=False, pin_memory=True)
    proxy = load_proxy()
    all_preds = []
    for data in loader:
        with torch.no_grad():
            pred = proxy(data.x, data.edge_index, data.edge_attr, data.batch)
        pred_list = [pr for pr in pred.squeeze(dim=-1).cpu().numpy()]
        all_preds.extend(pred_list)
    import heapq

    indexed_predictions = list(enumerate(all_preds))
    top_100_with_indices = heapq.nlargest(100, indexed_predictions, key=lambda x: x[1])
    top_100_indices = [idx for idx, value in top_100_with_indices]
    top_100_sm = smiles[top_100_indices]


if __name__ == "__main__":
    ids = ["debug_logp", "min_logp", "min_sol"]
    proxy = load_proxy()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for id in ids:
        run_path = obtain_run(id)
        config = get_config(id)
        results = read_all_results(run_path / "train")
        # Gotta rewrite scaffolds
        smiles = results["smi"][:639]

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
    plt.savefig("diversity_comparison_subplots.png")
    plt.show()
    # total_gen_train = config["algo"]["num_from_policy"] * config["num_training_steps"]
    # I want a dict that for every iter up to total_gen_train, it has 2 keys
    # states_visit_per_mb = config["algo"]["num_from_policy"] * config["algo"]["max_nodes"] * ["num_training_steps"]
