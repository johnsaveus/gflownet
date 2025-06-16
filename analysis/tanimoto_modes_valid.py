import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import MACCSkeys
from gflownet.proxy.mol_utils import random_split
from gflownet.utils.sqlite_log import read_all_results
from utils import *
import json
from rdkit import RDLogger
import warnings

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def get_tanimoto_modes(smiles_list, rew_list, similarity_threshold, rew_threshold=0.8):
    ppm_threshold = 30
    logS = rew2logS(rew_list)
    ppm = logS2ppm(smiles_list, logS)

    fps = [get_morgan_fp(smi) for smi in smiles_list]
    mode_fps = []
    mode_sum = []
    div_smiles = []
    smiles_seen = set()
    for idx, (fp, smile, rew) in enumerate(zip(fps, smiles_list, rew_list)):
        if idx % 10000 == 0:
            print(f"Processing {idx}")
        if rew <= rew_threshold:
            mode_sum.append(mode_sum[-1] if mode_sum else 0)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, mode_fps)
        is_new_mode = all(sim < similarity_threshold for sim in sims) if mode_fps else True
        if is_new_mode and smile not in smiles_seen:
            div_smiles.append(smile)
            mode_fps.append(fp)
            mode_sum.append(mode_sum[-1] + 1 if mode_sum else 1)
        else:
            mode_sum.append(mode_sum[-1] if mode_sum else 0)
        smiles_seen.add(smile)
    return mode_sum, div_smiles


def main(ids):
    create_dir("results")
    create_dir("results/modes")

    sims = [0.3, 0.4, 0.5]  # → now rows
    reward_thresholds = [0.7, 0.8]  # → now columns
    names = ["Random", "FM", "DB", "TB", "SubTB"]

    # New shape: rows = sim, cols = reward
    fig, axes = plt.subplots(len(sims), len(reward_thresholds), figsize=(14, 12), sharex=True)

    for ix, id in enumerate(ids):
        print(f"Processing {id}")
        run_path = obtain_run(id)
        results = read_all_results(run_path / "valid")
        smiles = results["smi"]
        rewards = results["r"]

        txt_path = f"results/modes/{names[ix]}_mode_valid.txt"

        unique_smiles = set(smiles)
        uniqueness_pct = 100 * len(unique_smiles) / len(smiles)

        with open(txt_path, "w") as f:
            f.write("Similarity_Threshold\tReward_Threshold\tFinal_Mode_Count\n")
            f.write(f"Uniqueness: {uniqueness_pct:.2f}%\n")

            for row_idx, sim_threshold in enumerate(sims):
                for col_idx, rew_thresh in enumerate(reward_thresholds):
                    mode_sum, _ = get_tanimoto_modes(
                        smiles, rew_list=rewards, similarity_threshold=sim_threshold, rew_threshold=rew_thresh
                    )

                    x_vals = np.linspace(1, 320000, len(mode_sum))
                    x_normalized = x_vals / 1e5

                    ax = axes[row_idx][col_idx]
                    ax.plot(x_normalized, mode_sum, label=names[ix])
                    ax.set_title(f"Sim < {sim_threshold}, Rew > {rew_thresh}")
                    ax.grid(True)

                    if col_idx == 0:
                        ax.set_ylabel("Modes Discovered", fontsize=14)
                    if row_idx == len(sims) - 1:
                        ax.set_xlabel("Molecules seen ($\\times10^5$)", fontsize=14)

                    final_count = mode_sum[-1] if mode_sum else 0
                    f.write(f"{sim_threshold}\t{rew_thresh}\t{final_count}\n")

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ids), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/modes/all_valid_and_randomn.png")
    plt.show()


ids = ["random_agent", "fm_experiment(lr=5e-4)", "db_lr(1e-3)", "tb_lr(1e-3)", "subtb_lr(5e-4)"]
# ids = ["random_agent"]

main(ids)
