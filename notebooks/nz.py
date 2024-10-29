# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: temps
#     language: python
#     name: temps
# ---

# # FIGURE 5 IN THE PAPER

# ## n(z) distributions

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
import torch
from pathlib import Path

# matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from temps.archive import Archive
from temps.utils import nmad
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule
from temps.plots import plot_nz

eval_methods = False

# ### LOAD DATA

# define here the directory containing the photometric catalogues
parent_dir = Path(
    "/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5"
)
modules_dir = Path("../data/models/")

# +
filename_valid = "euclid_cosmos_DC2_S1_v2.1_valid_matched.fits"

hdu_list = fits.open(parent_dir / filename_valid)
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat["FLAG_PHOT"] == 0]
cat = cat[cat["mu_class_L07"] == 1]
cat = cat[(cat["z_spec_S15"] > 0) | (cat["photo_z_L15"] > 0)]
cat = cat[cat["MAG_VIS"] < 25]

# -

ztarget = [
    cat["z_spec_S15"].values[ii]
    if cat["z_spec_S15"].values[ii] > 0
    else cat["photo_z_L15"].values[ii]
    for ii in range(len(cat))
]
specz_or_photo = [
    0 if cat["z_spec_S15"].values[ii] > 0 else 1 for ii in range(len(cat))
]
ID = cat["ID"]
VISmag = cat["MAG_VIS"]
zsflag = cat["reliable_S15"]

photoz_archive = Archive(path=parent_dir, only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue=cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# ### LOAD TRAINED MODELS AND EVALUATE PDFs AND REDSHIFT

if eval_methods:
    dfs = {}

    for il, lab in enumerate(["z", "L15", "DA"]):

        nn_features = EncoderPhotometry()
        nn_features.load_state_dict(
            torch.load(
                modules_dir / f"modelF_{lab}.pt", map_location=torch.device("cpu")
            )
        )
        nn_z = MeasureZ(num_gauss=6)
        nn_z.load_state_dict(
            torch.load(
                modules_dir / f"modelZ_{lab}.pt", map_location=torch.device("cpu")
            )
        )

        temps_module = TempsModule(nn_features, nn_z)

        z, pz, odds = temps_module.get_pz(input_data=torch.Tensor(col), return_pz=True)
        # Create a DataFrame with the desired columns
        df = pd.DataFrame(
            np.c_[ID, VISmag, z, odds, ztarget, zsflag, specz_or_photo],
            columns=["ID", "VISmag", "z", "odds", "ztarget", "zsflag", "S15_L15_flag"],
        )

        # Calculate additional columns or operations if needed
        df["zwerr"] = (df.z - df.ztarget) / (1 + df.ztarget)

        # Drop any rows with NaN values
        df = df.dropna()

        # Assign the DataFrame to a key in the dictionary
        dfs[lab] = df


# ### LOAD CATALOGUES IF AVAILABLE

if not eval_methods:

    df_zs = pd.read_csv(parent_dir / "predictions_specztraining.csv", header=0)
    df_zsL15 = pd.read_csv(parent_dir / "predictions_speczL15training.csv", header=0)
    df_DA = pd.read_csv(parent_dir / "predictions_speczDAtraining.csv", header=0)

    dfs = {}
    dfs["z"] = df_zs
    dfs["L15"] = df_zsL15
    dfs["DA"] = df_DA

# +
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Create figure and grid specification
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(5, 1, height_ratios=[0.1, 1, 1, 1, 1])

# Upper panel (very thin) with shaded areas
ax1 = plt.subplot(gs[0])
ax1.set_yticks([])

ax1.set_ylabel("Bins", fontsize=10)

# Define the ranges for shaded areas
# z_ranges = [[0.15, 0.35], [0.35, 0.55], [0.55, 0.85], [0.85, 1.05], [1.05, 1.35],
#                 [1.35, 1.55],# [1.55, 1.85], [1.85, 2], [2, 2.5], [2.5, 3], [3, 4]]

z_ranges = [[0.15, 0.5], [0.5, 1], [1, 1.5], [1.5, 2]]  # , [2, 3], [3,4]]#,
# [1.35, 1.55],# [1.55, 1.85], [1.85, 2], [2, 2.5], [2.5, 3], [3, 4]]

colors = [
    "deepskyblue",
    "forestgreen",
    "coral",
    "grey",
    "pink",
    "goldenrod",
    "cyan",
    "seagreen",
    "salmon",
    "steelblue",
    "orange",
]

# Plot shaded areas
x_values = [0, 1, 2]  # Example x values, adjust as needed
for i, (start, end) in enumerate(z_ranges):
    ax1.fill_betweenx(x_values, start, end, color=colors[i], alpha=0.5)

# Middle panel (equally thick)
ax2 = plt.subplot(gs[1])
for i, (start, end) in enumerate(z_ranges):
    dfplot_z = dfs["z"][(dfs["z"]["ztarget"] > start) & (dfs["z"]["ztarget"] < end)]
    ax2.hist(
        dfplot_z.ztarget,
        bins=50,
        color=colors[i],
        histtype="step",
        linestyle="-",
        density=True,
        range=(0, 4),
    )

# Bottom panel (equally thick)
ax3 = plt.subplot(gs[2])
for i, (start, end) in enumerate(z_ranges):
    dfplot_z = dfs["z"][(dfs["z"]["z"] > start) & (dfs["z"]["z"] < end)]
    ax3.hist(
        dfplot_z.ztarget,
        bins=50,
        color=colors[i],
        histtype="step",
        linestyle="-",
        density=True,
        range=(0, 4),
    )

# Bottom panel (equally thick)
ax4 = plt.subplot(gs[3])
for i, (start, end) in enumerate(z_ranges):
    dfplot_z = dfs["L15"][(dfs["L15"]["z"] > start) & (dfs["L15"]["z"] < end)]
    print(len(dfplot_z))
    ax4.hist(
        dfplot_z.ztarget,
        bins=50,
        color=colors[i],
        histtype="step",
        linestyle="-",
        density=True,
        range=(0, 4),
    )

ax5 = plt.subplot(gs[4])
for i, (start, end) in enumerate(z_ranges):
    dfplot_z = dfs["DA"][(dfs["DA"]["z"] > start) & (dfs["DA"]["z"] < end)]
    ax5.hist(
        dfplot_z.ztarget,
        bins=50,
        color=colors[i],
        histtype="step",
        linestyle="-",
        density=True,
        range=(0, 4),
    )

plt.tight_layout()
plt.show()

# -


def plot_nz(df_list, zcuts=[0.1, 0.5, 1, 1.5, 2, 3, 4], save=False):
    # Plot properties
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16

    cmap = plt.get_cmap("Dark2")  # Choose a colormap for coloring lines

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(20, 8), sharex=True)

    for i, df in enumerate(df_list):
        dfplot = df_list[i].copy()  # Assuming df_list contains dataframes
        ax = axs[i]  # Selecting the appropriate subplot

        for iz in range(len(zcuts) - 1):
            dfplot_z = dfplot[
                (dfplot["ztarget"] > zcuts[iz]) & (dfplot["ztarget"] < zcuts[iz + 1])
            ]
            color = cmap(iz)  # Get a different color for each redshift

            zt_mean = np.median(dfplot_z.ztarget.values)
            zp_mean = np.median(dfplot_z.z.values)

            # Plot histogram on the selected subplot
            ax.hist(
                dfplot_z.z,
                bins=50,
                color=color,
                histtype="step",
                linestyle="-",
                density=True,
                range=(0, 4),
            )
            ax.axvline(zt_mean, color=color, linestyle="-", lw=2)
            ax.axvline(zp_mean, color=color, linestyle="--", lw=2)

        ax.set_ylabel(f"Frequency", fontsize=14)
        ax.grid(False)
        ax.set_xlim(0, 3.5)

    axs[-1].set_xlabel(f"$z$", fontsize=18)

    if save:
        plt.savefig(f"nz_hist.pdf", dpi=300, bbox_inches="tight")

    plt.show()


plot_nz(df_list)
