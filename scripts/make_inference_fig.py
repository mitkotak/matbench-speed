# /// script
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['xtick.major.size'] = 3.5
matplotlib.rcParams['ytick.major.size'] = 3.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.framealpha'] = 1.0
matplotlib.rcParams['legend.edgecolor'] = 'black'
matplotlib.rcParams['legend.fancybox'] = False
matplotlib.rcParams['legend.fontsize'] = 8

our_model = "Nequix-MP-1.5"
max_atoms = 5000

def _series(df: pd.DataFrame, model_name: str) -> tuple[list[int], list[float]]:
    sub = df[df["model"] == model_name].copy()
    sub = sub[sub["atoms"] != 8]
    sub = sub.sort_values("atoms")
    return sub["atoms"].tolist(), sub["steps_per_day_m"].tolist()


def make_inference_fig(path: str) -> None:
    df = pd.read_csv(path)
    df = df[df["atoms"] < 30000]
    gpu_name = df["gpu"].unique()[0]
    precision = df["precision"].unique()[0]
    df["steps_per_day_m"] = 86_400_000.0 / df["time"] / 1_000_000.0
    # ours_old_x, ours_old_y_millions = _series(df, "Nequix-MP-1")
    ours_x, ours_y_millions = _series(df, "Nequix-MP-1.5")
    # nequip_x, nequip_y_millions = _series(df, "NequIP-MP-L")
    # sevennet_x, sevennet_y_millions = _series(df, "SevenNet-l3i5")
    # esen_x, esen_y_millions = _series(df, "eSEN-30M-MP")
    esen_s_x, esen_s_y_millions = _series(df, "eSEN-6M-OC25")
    mace_x, mace_y_millions = _series(df, "MACE-MP-0")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(ours_x, ours_y_millions, marker="s", linestyle="-", color="tab:blue", label=our_model, markeredgecolor='black')
    # ax.plot(esen_x, esen_y_millions, marker="s", linestyle="-", color="tab:orange", label="eSEN-30M-MP", markeredgecolor='black')
    ax.plot(esen_s_x, esen_s_y_millions, marker="s", linestyle="-", color="tab:red", label="eSEN-6M-OC25", markeredgecolor='black')
    ax.plot(mace_x, mace_y_millions, marker="s", linestyle="-", color="tab:green", label="MACE-MP-0", markeredgecolor='black')
    # ax.plot(nequip_x, nequip_y_millions, marker="s", linestyle="-", color="tab:red", label="NequIP-MP-L", markeredgecolor='black')
    # ax.plot(sevennet_x, sevennet_y_millions, marker="s", linestyle="-", color="tab:cyan", label="SevenNet-l3i5", markeredgecolor='black')

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    unique_x = sorted(set(ours_x) | set(mace_x))
    ax.xaxis.set_major_locator(FixedLocator(unique_x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in unique_x]))
    ax.tick_params(axis="x", labelrotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    # Style grid like the paper
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=6, loc='best')
    legend.get_frame().set_linewidth(0.5)
    
    ax.set_title(f"Benchmark {gpu_name} {precision}", fontsize=10, weight='normal')

    fig.tight_layout()
    fig.savefig(f"./figures/inference_fig_{gpu_name}_{precision}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"./figures/inference_fig_{gpu_name}_{precision}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    make_inference_fig("./data/timing_data_A100.csv")
    make_inference_fig("./data/timing_data_colab.csv")

if __name__ == "__main__":
    main()