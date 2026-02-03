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


markersize = 2.5
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['xtick.major.size'] = 3.5
matplotlib.rcParams['ytick.major.size'] = 3.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = markersize
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.framealpha'] = 1.0
matplotlib.rcParams['legend.edgecolor'] = 'black'
matplotlib.rcParams['legend.fancybox'] = False
matplotlib.rcParams['legend.fontsize'] = 8

compliant_status = "non-compliant"
atom_name = "H20"

def _series(df: pd.DataFrame, model_name: str) -> tuple[list[int], list[float]]:
    sub = df[df["model"] == model_name].copy()
    sub = sub[sub["num_atoms"] != 8]
    sub = sub.sort_values("num_atoms")
    return sub["num_atoms"].tolist(), sub["steps_per_day_m"].tolist()


def make_inference_fig(path: str) -> None:
    
    atom_name = path.split("_")[3]

    df = pd.read_csv(path)
    gpu_name = df["gpu"].unique()[0]
    precision = df["precision"].unique()[0]
    df["steps_per_day_m"] = 86_400_000.0 / df["time"] / 1_000_000.0
    ours_x, ours_y_millions = _series(df, "Nequix-MP-1.5")
    nequix_x, nequix_y_millions = _series(df, "Nequix-MP-1")
    nequip_x, nequip_y_millions = _series(df, "NequIP-MP-L")
    sevennet_x, sevennet_y_millions = _series(df, "SevenNet-l3i5")
    esen_x, esen_y_millions = _series(df, "eSEN-30M-MP")
    mace_x, mace_y_millions = _series(df, "MACE-MP-0")
    grace_x, grace_y_millions = _series(df, "GRACE-2L-MPtrj")
    
    if not compliant_status == "compliant":
        pet_xl_x, pet_xl_y_millions = _series(df, "PET-OAM-XL")
        pet_s_x, pet_s_y_millions = _series(df, "PET-MAD-S")
        nequip_xl_x, nequip_xl_y_millions = _series(df, "NequIP-MP-XL")
        orb_x, orb_y_millions = _series(df, "Orb-v3-cons-inf-omat")
        esen_s_x, esen_s_y_millions = _series(df, "eSEN-6M-OC25")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(nequix_x, nequix_y_millions, marker="s", markersize=markersize, linestyle="-", color="blue", label="Nequix-MP-1", markeredgecolor='black')
    ax.plot(esen_x, esen_y_millions, marker="s", markersize=markersize, linestyle="-", color="orange", label="eSEN-30M-MP", markeredgecolor='black')
    ax.plot(mace_x, mace_y_millions, marker="s", markersize=markersize, linestyle="-", color="fuchsia", label="MACE-MP-0", markeredgecolor='black')
    ax.plot(nequip_x, nequip_y_millions, marker="s", markersize=markersize, linestyle="-", color="green", label="NequIP-MP-L", markeredgecolor='black')
    ax.plot(sevennet_x, sevennet_y_millions, marker="s", markersize=markersize, linestyle="-", color="red", label="SevenNet-l3i5", markeredgecolor='black')
    ax.plot(grace_x, grace_y_millions, marker="s", markersize=markersize, linestyle="-", color="gray", label="GRACE-2L-MPtrj", markeredgecolor='black')
    
    if compliant_status == "non-compliant":
        ax.plot(pet_s_x, pet_s_y_millions, marker="s", markersize=markersize,
                linestyle="-", color="coral", label="PET-MAD-S", markeredgecolor="black")
        ax.plot(pet_xl_x, pet_xl_y_millions, marker="s", markersize=markersize,
                linestyle="-", color="steelblue", label="PET-OAM-XL", markeredgecolor="black")
        ax.plot(nequip_xl_x, nequip_xl_y_millions, marker="s", markersize=markersize, linestyle="-", color="pink", label="NequIP-MP-XL", markeredgecolor='black')
        ax.plot(esen_s_x, esen_s_y_millions, marker="s", markersize=markersize, linestyle="-", color="purple", label="eSEN-6M-OC25", markeredgecolor='black')
        ax.plot(orb_x, orb_y_millions, marker="s", markersize=markersize, linestyle="-", color="gold", label="Orb-v3-cons-inf-omat", markeredgecolor='black')

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Steps per day (millions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    x_labels = sorted(set(mace_x))
    ax.xaxis.set_major_locator(FixedLocator(x_labels))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in x_labels]))
    ax.tick_params(axis="x", labelrotation=75, labelsize=6)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("center")
    
    # Style grid like the paper
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    legend = ax.legend(fontsize=5, loc='best')
    legend.get_frame().set_linewidth(0.5)
    
    ax.set_title(f"Matbench {compliant_status}, {atom_name} H20 Å \n {gpu_name} {precision}", fontsize=8, weight='normal')

    fig.tight_layout()
    fig.savefig(f"./figures/inference_fig_{compliant_status}_{atom_name}_{gpu_name}_{precision}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"./figures/inference_fig_{compliant_status}_{atom_name}_{gpu_name}_{precision}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
 
    #make_inference_fig(f"./data/timing_data_A100_{atom_name}.csv")
    make_inference_fig(f"./data/timing_data_H100_{atom_name}.csv")
    # make_inference_fig(f"./data/timing_data_T4_{atom_name}.csv")

if __name__ == "__main__":
    main()
