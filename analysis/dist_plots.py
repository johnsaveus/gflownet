"""
This script generates a distribution plot comparing the solubility for
different beta experiments and the proxy dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from torch_geometric.loader import DataLoader
from gflownet.proxy.mol_utils import smiles2graph
from gflownet.utils.sqlite_log import read_all_results
from utils import obtain_run, load_proxy_sol, create_dir, scale_rew, infer_model
from rdkit.Chem import DataStructs, rdMolDescriptors
import numpy as np

# Load data and prepare reference reward distribution
data_url = r"https://raw.githubusercontent.com/CesareWang/Predictors-for-15-Environmental-Endpoints/main/predictors/data/SW.csv"
sol_data = pd.read_csv(data_url, index_col=0)
rew_data = scale_rew(list(sol_data["active"]))

# Experiment identifiers and labels
ids = ["beta_experiment(1)", "beta_experiment(4)", "beta_experiment(8)", "beta_experiment(10)"]
names = ["b = 1", "b = 4", "b = 8", "b = 10"]

# Load proxy predictor
sol_predictor = load_proxy_sol()

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(True, linestyle="--", alpha=0.5)
palette = sns.color_palette("tab10")
linewidth = 1.5
tanimotos = []

# Plot reward distributions and compute average Tanimoto similarity
for i, id in enumerate(ids):
    run_path = obtain_run(id)
    results = read_all_results(run_path / "final")
    smiles = list(results["smi"])
    reward = list(results["r"])

    # Infer proxy predictions (not plotted here but included in logic if needed later)
    graphs = [smiles2graph(Chem.MolFromSmiles(smile)) for smile in smiles]
    graph_loader = DataLoader(graphs, batch_size=1024, shuffle=False, pin_memory=True)
    _ = infer_model(sol_predictor, graph_loader)

    # Plot reward distribution
    sns.kdeplot(
        reward,
        bw_adjust=0.30,
        linestyle="-",
        label=names[i],
        linewidth=linewidth,
        ax=ax,
    )

# Plot reference proxy dataset distribution
sns.kdeplot(
    rew_data,
    bw_adjust=0.30,
    label="proxy dataset",
    color="black",
    linestyle="--",
    linewidth=linewidth,
    ax=ax,
)

# Axis labels and legend
ax.set_title("Reward Distribution - (1 - scaled_solubility)", fontsize=14)
ax.set_xlabel(r"$R(x)$", fontsize=12)
ax.set_ylabel(r"$\hat{p}(R)$", fontsize=12)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

# Save plot
plt.tight_layout()
create_dir("results")
plt.savefig("results/dist_plot_rewards_only.png", dpi=300)
