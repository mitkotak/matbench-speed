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


def main() -> None:
    df = pd.read_csv("./data/timing_data_A100.csv")
    df = df[df["atoms"] < 5000]
    df["steps_per_day_m"] = 86_400_000.0 / df["time"] / 1_000_000.0
    ours_old_x, ours_old_y_millions = _series(df, "Nequix-MP-1")
    ours_x, ours_y_millions = _series(df, "Nequix-MP-1.5")
    esen_x, esen_y_millions = _series(df, "eSEN-30M-MP")
    esen_s_x, esen_s_y_millions = _series(df, "eSEN-6M-OC25")
    mace_x, mace_y_millions = _series(df, "MACE-MP-0")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(ours_x, ours_y_millions, marker="s", linestyle="-", color="tab:blue", label=our_model, markeredgecolor='black')
    ax.plot(esen_x, esen_y_millions, marker="s", linestyle="-", color="tab:orange", label="eSEN-30M-MP", markeredgecolor='black')
    ax.plot(esen_s_x, esen_s_y_millions, marker="s", linestyle="-", color="tab:red", label="eSEN-6M-OC25", markeredgecolor='black')
    ax.plot(mace_x, mace_y_millions, marker="s", linestyle="-", color="tab:green", label="MACE-MP-0", markeredgecolor='black')

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    unique_x = sorted(set(ours_x) | set(esen_x) | set(mace_x))
    ax.xaxis.set_major_locator(FixedLocator(unique_x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in unique_x]))
    ax.tick_params(axis="x", labelrotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    # Style grid like the paper
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=7, loc='best')
    legend.get_frame().set_linewidth(0.5)
    
    ax.set_title("A100 Benchmark", fontsize=10, weight='normal')

    fig.tight_layout()
    fig.savefig("./figures/inference_fig_A100.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("./figures/inference_fig_A100.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(ours_x, ours_y_millions, marker="s", linestyle="-", color="tab:orange", label=f"{our_model} (kernels)", markeredgecolor='black')
    ax.plot(ours_old_x, ours_old_y_millions, marker="s", linestyle="-", color="tab:blue", label=our_model, markeredgecolor='black')
    ax.plot(esen_x, esen_y_millions, marker="o", linestyle="--", color="gray", label="eSEN-30M-MP", markeredgecolor='black')
    ax.plot(mace_x, mace_y_millions, marker="^", linestyle="--", color="gray", label="MACE-MP-0", markeredgecolor='black')
    
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    unique_x = sorted(set(ours_x) | set(esen_x) | set(mace_x))
    ax.xaxis.set_major_locator(FixedLocator(unique_x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in unique_x]))
    ax.tick_params(axis="x", labelrotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=7, loc='best')
    legend.get_frame().set_linewidth(0.5)
    
    ax.set_title("A100 Benchmark", fontsize=10, weight='normal')

    fig.tight_layout()
    fig.savefig("./figures/inference_fig_A100_old.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("./figures/inference_fig_A100_old.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_rtx = pd.read_csv("./data/timing_data_RTX5080.csv")
    df_rtx = df_rtx[df_rtx["atoms"] < 5000]
    df_rtx["steps_per_day_m"] = 86_400_000.0 / df_rtx["time"] / 1_000_000.0
    ours_x_rtx, ours_y_millions_rtx = _series(df_rtx, "Nequix-MP-1.5")
    esen_x_rtx, esen_y_millions_rtx = _series(df_rtx, "eSEN-30M-MP")
    esen_s_x_rtx, esen_s_y_millions_rtx = _series(df_rtx, "eSEN-6M-OC25")
    mace_x_rtx, mace_y_millions_rtx = _series(df_rtx, "MACE-MP-0")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(ours_x_rtx, ours_y_millions_rtx, marker="s", linestyle="-", color="tab:red", label=f"{our_model} (kernels) - RTX 5080", markeredgecolor='black')
    ax.plot(ours_x, ours_y_millions, marker="s", linestyle="-", color="tab:purple", label=f"{our_model} (kernels) - A100", markeredgecolor='black')
    
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    unique_x = sorted(set(ours_x) | set(esen_x) | set(mace_x))
    ax.xaxis.set_major_locator(FixedLocator(unique_x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in unique_x]))
    ax.tick_params(axis="x", labelrotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=7, loc='best')
    legend.get_frame().set_linewidth(0.5)
    
    ax.set_title("A100 vs RTX 5080", fontsize=10, weight='normal')

    fig.tight_layout()
    fig.savefig("./figures/inference_fig_A100_vs_RTX5080.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("./figures/inference_fig_A100_vs_RTX5080.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(ours_x_rtx, ours_y_millions_rtx, marker="s", linestyle="-", color="tab:blue", label=our_model, markeredgecolor='black')
    ax.plot(esen_x_rtx, esen_y_millions_rtx, marker="s", linestyle="-", color="tab:orange", label="eSEN-30M-MP", markeredgecolor='black')
    ax.plot(esen_s_x_rtx, esen_s_y_millions_rtx, marker="s", linestyle="-", color="tab:red", label="eSEN-6M-OC25", markeredgecolor='black')
    ax.plot(mace_x_rtx, mace_y_millions_rtx, marker="s", linestyle="-", color="tab:green", label="MACE-MP-0", markeredgecolor='black')

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("RTX 5080 Benchmark", fontsize=10, weight='normal')
    
    unique_x = sorted(set(ours_x_rtx) | set(esen_x_rtx) | set(mace_x_rtx))
    ax.xaxis.set_major_locator(FixedLocator(unique_x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in unique_x]))
    ax.tick_params(axis="x", labelrotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=8, loc='best')
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    fig.savefig("./figures/inference_fig_RTX5080.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("./figures/inference_fig_RTX5080.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()