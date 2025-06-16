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


# def get_tanimoto_modes(smiles_list, rew_list, similarity_threshold):
#     "Finds unique modes based on Tanimoto similarity"
#     # Returns number of modes and diverse smiles
#     mode_fps = []
#     mode_sum = []
#     # For duplicates removal
#     smiles_seen = set()
#     ppm_threshold = 30
#     div_smiles = []
#     # mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
#     logS = rew2logS(rew_list)
#     ppm = logS2ppm(smiles_list, logS)
#     for idx, smile in enumerate(smiles_list):
#         print(idx)
#         fp = get_morgan_fp(smile)
#         is_new_mode = all(
#             DataStructs.BulkTanimotoSimilarity(fp, existing_fp) < similarity_threshold for existing_fp in mode_fps
#         )
#         if is_new_mode and ppm[idx] > ppm_threshold and smile not in smiles_seen:
#             div_smiles.append(smile)
#             mode_fps.append(fp)
#             if idx == 0:
#                 mode_sum.append(1)
#             else:
#                 mode_sum.append(mode_sum[-1] + 1)
#         else:
#             if idx == 0:
#                 mode_sum.append(0)
#             else:
#                 mode_sum.append(mode_sum[-1])
#         smiles_seen.add(smile)
#     return mode_sum, div_smiles


def get_tanimoto_modes(smiles_list, rew_list, similarity_threshold):
    """Returns cumulative number of modes and diverse SMILES based on Tanimoto similarity."""
    ppm_threshold = 30
    logS = rew2logS(rew_list)
    ppm = logS2ppm(smiles_list, logS)

    # Precompute fingerprints
    fps = [get_morgan_fp(smi) for smi in smiles_list]

    mode_fps = []
    mode_sum = []
    div_smiles = []

    for idx, (fp, ppm_val, smi) in enumerate(zip(fps, ppm, smiles_list)):
        if ppm_val < ppm_threshold:
            mode_sum.append(mode_sum[-1] if mode_sum else 0)
            continue
        # Fast vectorized similarity check
        if not mode_fps:
            is_new_mode = True
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, mode_fps)
            is_new_mode = all(sim < similarity_threshold for sim in sims)

        if is_new_mode:
            div_smiles.append(smi)
            mode_fps.append(fp)
            mode_sum.append(mode_sum[-1] + 1 if mode_sum else 1)
        else:
            mode_sum.append(mode_sum[-1] if mode_sum else 0)

    return mode_sum, div_smiles


def main(ids):
    create_dir("results")
    create_dir("results/modes")
    # Create a directory for the plots
    # _, axes = plt.subplots(1, 2, figsize=(18, 6))
    all_modes = {}
    for id in ids:
        run_path = obtain_run(id)
        results = read_all_results(run_path / "train")
        smiles = results["smi"]
        rewards = results["r"]
        modes = []
        sims = [0.3, 0.4, 0.5]
        print(id)
        for sim_threshold in sims:
            print(sim_threshold)
            y, _ = get_tanimoto_modes(smiles, rew_list=rewards, similarity_threshold=sim_threshold)
            modes.append(max(y))
        modes_dict = {f"{sim_threshold}": mode for sim_threshold, mode in zip(sims, modes)}
        all_modes[id] = modes_dict
    with open("results/modes/tanimoto_db(pb).json", "w") as f:
        json.dump(all_modes, f)
        # TODO: Renormalize it when generating all
        # x_ticks = [i / 2 for i in range(1, 7)]  # Generate x-axis ticks as 0.5, 1, 1.5, ..., 10
        # x_values = [int(tick * 1e5) for tick in x_ticks]  # Convert ticks to corresponding trajectory values
        # axes[0].plot(range(len(y1)), y1, label=f"{id}", linestyle="-")
        # # Plot Tanimoto-based diversity
        # axes[1].plot(range(len(y2)), y2, label=f"{id}", linestyle="-")

    # Labels and titles
    # axes[0].set_xlabel("Trajectories Sampled (x 10^5)")
    # axes[0].set_ylabel("Number of modes")
    # axes[0].set_xticks(x_values)
    # axes[0].set_xticklabels(x_ticks)
    # axes[0].legend()
    # axes[0].grid(True)

    # axes[1].set_xlabel("Trajectories Sampled (x 10^5)")
    # axes[1].set_ylabel("Number of modes")
    # axes[1].set_xticks(x_values)
    # axes[1].set_xticklabels(x_ticks)
    # axes[1].legend()
    # axes[1].grid(True)
    # plt.show()
    # plt.tight_layout()
    # plt.savefig("results/modes_all.png")


# ids = ["fm_experiment(lr=1e-4)", "fm_experiment(lr=5e-4)", "fm_experiment(lr=1e-3)"]
# main(ids)

# ids = ["db_lr(1e-4)", "db_lr(5e-4)", "db_lr(1e-3)"]
# main(ids)

# ids = ["tb_lr(1e-4)", "tb_lr(5e-4)", "tb_lr(1e-3)"]
# main(ids)

# ids = ["subtb_lr(1e-4)", "subtb_lr(5e-4)", "subtb_lr(1e-3)"]
# main(ids)

ids = ["db_(pb,lr=1e-4)", "db_(pb,lr=5e-4)", "db_(pb,lr=1e-3)"]
main(ids)

# ids = ["fm_experiment(lr=1e-4)", "db_lr(5e-4)", "tb_lr(5e-4)", "subtb_lr(1e-3)"]
# plot_modes(ids)
