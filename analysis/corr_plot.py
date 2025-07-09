"""
This script generates a scatter plot comparing the log reward and
the log probability of actions taken by different algorithms.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from rdkit import Chem
import torch_geometric.data as gd
from gflownet.proxy.mol_utils import smiles2graph
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.flow_matching import FlowMatching
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from utils import load_proxy_sol, create_dir
from scipy.stats import pearsonr, spearmanr
import pickle

pickle_file_path = "inference_list_all.pkl"
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
# names = ["fm_experiment(lr=1e-4)", "fm_experiment(lr=5e-4)", "fm_experiment(lr=1e-3)"]
# names = ["tb_lr(1e-4)", "tb_lr(5e-4)", "tb_lr(1e-3)"]
# names = ["db_lr(1e-4)", "db_lr(5e-4)", "db_lr(1e-3)"]
# names = ["subtb_lr(1e-4)", "subtb_lr(5e-4)", "subtb_lr(1e-3)"]
names = ["FM", "DB", "TB", "SubTB"]
markers = ["o", "s", "^", "D"]
cmaps = ["viridis", "plasma", "cividis", "magma"]
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
correlation_results = []
for ix, i in enumerate(data):
    df = i[0]
    rewards = [r**4 for r in df["reward"].values]
    log_rewards = np.log(rewards)
    policy = df["policy"].values
    pearson_corr, pearson_p = pearsonr(log_rewards, policy)
    spearman_corr, spearman_p = spearmanr(log_rewards, policy)

    # quantile_threshold = df["policy"].quantile(0.99)
    # df = df[df["policy"] <= quantile_threshold]

    # df["reward_bin"] = pd.qcut(df["reward"], q=10, labels=False)
    # df["is_high_bin"] = df["reward_bin"] >= 8  # Top 6 bins (i.e., top 60% of rewards)

    # # Use log-sum-exp trick to compute sums in log-space

    # from scipy.special import logsumexp

    # log_p_high = logsumexp(df.loc[df["is_high_bin"], "policy"])
    # log_p_total = logsumexp(df["policy"])

    # Final ratio in normal space
    # p_highest4of10bins = np.exp(log_p_high - log_p_total)

    reward = df["reward"].values
    pi_target = reward / reward.sum()

    # Step 2: Convert model's log probabilities to normalized probabilities
    log_pi_model = df["policy"].values
    pi_model = np.exp(log_pi_model)
    pi_model /= pi_model.sum()  # Normalize to ensure valid probability distribution

    # Step 3: Compute cross-entropy H(π_target, π_model)
    ce = -np.sum(pi_target * np.log(pi_model + 1e-12))
    correlation_results.append(
        f"{names[ix]}: Pearson r = {pearson_corr:.4f}, Pearson p = {pearson_p:.4f}, Spearman r = {spearman_corr:.4f}, spearman p = {spearman_p:.4f}, cross-entropy = {ce:.4f}"
    )
    # ax = axs[ix]
    # slope, intercept = np.polyfit(log_rewards, policy, 1)
    # line = slope * np.array(log_rewards) + intercept
    # ax.scatter(log_rewards, policy, alpha=0.2, marker="o", color=colors[ix])
    # ax.plot(log_rewards, line, label=f"{names[ix]} r = {round(pearson_corr, 2)}", linestyle="-", alpha=0.4)
    # # hb = ax.hexbin(
    # #     total_rew, total_policy, label=f"{names[i]} r = {round(pearson_corr, 2)}", cmap=cmaps[0]
    # # )  # , cmap=cmaps[i])
    # # plt.savefig("results/test.png", dpi=300, bbox_inches="tight")
    # # plt.colorbar(label="Counts")
    # ax.set_title(f"{names[ix]} (r = {round(pearson_corr, 2)})")
    # ax.set_xlabel(r"$\log(R(x)^b)$")
    # ax.set_xlim(-6, 1)
    # ax.set_ylim(-60, -40)
    # ax.set_ylabel(r"$\sum_{t=1}^{n} \log\, P_F(s_t \mid s_{t-1})$")

    ax = axs[ix]
    x = np.array(log_rewards)
    y = np.array(policy)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept

    # Confidence interval (95%)
    n = len(x)
    t_value = stats.t.ppf(1 - 0.025, df=n - 2)
    s_err = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))
    ci = t_value * s_err * np.sqrt(1 / n + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    ci_sorted = ci[sort_idx]

    ax.scatter(x, y, alpha=0.2, marker="o", color=colors[ix])
    ax.plot(x_sorted, y_pred_sorted, label=f"{names[ix]} r = {round(pearson_corr, 2)}", color=colors[ix], alpha=1)
    ax.fill_between(
        x_sorted, y_pred_sorted - ci_sorted, y_pred_sorted + ci_sorted, color=colors[ix], alpha=0.2, label="95% CI"
    )

    ax.set_title(f"{names[ix]} (r = {round(pearson_corr, 2)})")
    ax.set_xlabel(r"$\log(R(x)^b)$")
    ax.set_xlim(-6, 1)
    ax.set_ylim(-60, -40)
    ax.set_ylabel(r"$\sum_{t=1}^{n} \log\, P_F(s_t \mid s_{t-1})$")

with open("results/metrics/correlation_results_all.txt", "w") as f_out:
    for line in correlation_results:
        f_out.write(line + "\n")

plt.savefig(f"results/metrics/ci_corr_plot_all", dpi=300, bbox_inches="tight")
plt.show()
