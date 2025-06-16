import json
import matplotlib.pyplot as plt
import numpy as np

# List your JSON files here
json_files = [
    "results/modes/tanimoto_fm.json",
    "results/modes/tanimoto_db.json",
    "results/modes/tanimoto_tb.json",
    "results/modes/tanimoto_subtb.json",
]

# Corresponding experiment keys for each JSON file
target_experiment_keys = [
    "fm_experiment(lr=1e-4)",
    "db_lr(1e-4)",
    "tb_lr(1e-4)",
    "subtb_lr(1e-4)",
]

names = ["FM", "DB", "TB", "SubTB"]
plt.figure(figsize=(10, 6))

for file, exp_key, name in zip(json_files, target_experiment_keys, names):
    with open(file, "r") as f:
        content = json.load(f)

    if exp_key in content:
        thresholds_dict = content[exp_key]

        # Convert and sort thresholds
        sim_thresholds = sorted(thresholds_dict.keys(), key=float)
        sim_thresholds = [float(k) for k in sim_thresholds]
        modes = [thresholds_dict[str(k)] / 1e4 for k in sim_thresholds]
        plt.plot(sim_thresholds[2:], modes[2:], marker="o", label=name)
    else:
        print(f"Experiment key '{exp_key}' not found in {file}")

# Finalize plot
plt.xlabel("Similarity Threshold")
plt.ylabel("Number of Modes (×10⁴)")
# plt.title("Modes vs Similarity Thresholds")
plt.xticks([0.3, 0.4, 0.5])  # Show only specified thresholds
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("results/modes/tanimoto_plot.png")
plt.show()
