import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
import seaborn as sns
from rdkit import Chem
from gflownet.utils.sqlite_log import read_all_results
from utils import obtain_run, get_config, scale_rew, load_proxy_sol
from gflownet.proxy.mol_utils import smiles2graph
from torch_geometric.loader import DataLoader
import torch


def top_n_statistics(smiles, n):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    graphs = [smiles2graph(mol) for mol in mols]
    loader = DataLoader(graphs, batch_size=2000, shuffle=False, pin_memory=True)
    proxy = load_proxy_sol()
    all_preds = []
    for data in loader:
        with torch.no_grad():
            pred = proxy(data.x, data.edge_index, data.edge_attr, data.batch)
        pred_list = [pr for pr in pred.squeeze(dim=-1).cpu().numpy()]
        all_preds.extend(pred_list)
    indexed_predictions = list(enumerate(all_preds))
    top_100_with_indices = heapq.nsmallest(n, indexed_predictions, key=lambda x: x[1])
    top_100_indices = [idx for idx, _ in top_100_with_indices]
    top_100_smiles = [smiles[i] for i in top_100_indices]
    top_100_preds = [all_preds[i] for i in top_100_indices]
    # top_100_sm = smiles[top_100_indices]
    return all_preds, top_100_smiles, top_100_preds


if __name__ == "__main__":
    data_url = r"https://raw.githubusercontent.com/CesareWang/Predictors-for-15-Environmental-Endpoints/main/predictors/data/SW.csv"
    sol_data = list(pd.read_csv(data_url, index_col=0)["active"])

    ids = ["min_sol"]

    for id in ids:
        run_path = obtain_run(id)
        config = get_config(id)
        results = read_all_results(run_path / "final")
        smiles = list(results["smi"])
        all_preds, top_smi, top_preds = top_n_statistics(smiles, 1000)
        print(np.array(top_preds).mean())
        print(np.array(top_preds).std())
        sns.kdeplot(
            all_preds,
            bw_adjust=0.3,  # Adjust bandwidth for smoothness
            label="scaled reward",
            linewidth=1.5,
        )

    # Add the "proxy dataset" line (black solid line)
    sns.kdeplot(
        sol_data,
        bw_adjust=0.3,
        label="proxy dataset",
        color="black",
        linestyle="-",
        linewidth=1.5,
    )

    # Plot formatting to match your reference
    plt.xlabel(r"$R(x)$", fontsize=16)
    plt.ylabel(r"$\hat{p}(R)$", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("sol_density_plot.png", dpi=300)
