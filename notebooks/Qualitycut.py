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

# # QUALITY CUTS

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import os
import torch
from scipy import stats
from pathlib import Path

# matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from temps.archive import Archive
from temps.utils import nmad, caluclate_eta
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule


# ### LOAD DATA (ONLY SPECZ)

# define here the directory containing the photometric catalogues
parent_dir = Path(
    "/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5"
)
modules_dir = Path("../data/models/")

photoz_archive = Archive(
    path=parent_dir,
    only_zspec=True,
    flags_kept=[
        1.0,
        1.1,
        1.4,
        1.5,
        2,
        2.1,
        2.4,
        2.5,
        3.0,
        3.1,
        3.4,
        3.5,
        4.0,
        9.0,
        9.1,
        9.3,
        9.4,
        9.5,
        11.1,
        11.5,
        12.1,
        12.5,
        13.0,
        13.1,
        13.5,
        14,
    ],
)
(
    f_test_specz,
    ferr_test_specz,
    specz_test,
    VIS_mag_test,
) = photoz_archive.get_testing_data()


# ### LOAD TRAINED MODELS AND EVALUATE PDF OF RANDOM EXAMPLES

# Initialize an empty dictionary to store DataFrames
dfs = {}
pzs = np.zeros(shape=(3, 11016, 1000))
for il, lab in enumerate(["z", "L15", "DA"]):

    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(
        torch.load(modules_dir / f"modelF_{lab}.pt", map_location=torch.device("cpu"))
    )
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(
        torch.load(modules_dir / f"modelZ_{lab}.pt", map_location=torch.device("cpu"))
    )

    temps_module = TempsModule(nn_features, nn_z)

    z, pz, odds = temps_module.get_pz(
        input_data=torch.Tensor(f_test_specz), return_pz=True
    )

    pzs[il] = pz

    # Create a DataFrame with the desired columns
    df = pd.DataFrame(np.c_[z, odds, specz_test], columns=["z", "odds", "ztarget"])

    # Calculate additional columns or operations if needed
    df["zwerr"] = (df.z - df.ztarget) / (1 + df.ztarget)

    # Drop any rows with NaN values
    df = df.dropna()

    # Assign the DataFrame to a key in the dictionary
    dfs[lab] = df


# ### STATS

# +
# odds_test = [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15]
odds_test = np.arange(0, 0.15, 0.01)

df = dfs["DA"].copy()
zgrid = np.linspace(0, 5, 1000)
pz = pzs[2]
# -

diff_matrix = np.abs(df.z.values[:, None] - zgrid[None, :])
idx_peak = np.argmax(pz, 1)
idx = np.argmin(diff_matrix, 1)

odds_cat = np.zeros(shape=(len(odds_test), len(df)))
for ii, odds_ in enumerate(odds_test):
    diff_matrix_upper = np.abs((df.z.values + odds_)[:, None] - zgrid[None, :])
    diff_matrix_lower = np.abs((df.z.values - odds_)[:, None] - zgrid[None, :])

    idx = np.argmin(diff_matrix, 1)
    idx_upper = np.argmin(diff_matrix_upper, 1)
    idx_lower = np.argmin(diff_matrix_lower, 1)

    odds = []
    for jj in range(len(pz)):
        odds.append(pz[jj, idx_lower[jj] : (idx_upper[jj] + 1)].sum())

    odds_cat[ii] = np.array(odds)


odds_df = pd.DataFrame(odds_cat.T, columns=[f"odds_{x}" for x in odds_test])
df = pd.concat([df, odds_df], axis=1)


# ## statistics on ODDS

# +
scatter_odds, eta_odds, xlab_odds, oddsmean = [], [], [], []

for c in complenteness:
    percentile_cutoff = df["odds"].quantile(c)

    df_bin = df[(df.odds > percentile_cutoff)]

    xlab_odds.append((1 - c) * 100)
    oddsmean.append(np.mean(df_bin.odds))
    scatter_odds.append(nmad(df_bin.zwerr))
    eta_odds.append(caluclate_eta(df_bin))
    if np.round(c, 1) == 0.3:
        percentiles_cutoff = [df[f"odds_{col}"].quantile(c) for col in odds_test]
        scatters_odds = [
            nmad(df[df[f"odds_{col}"] > percentile_cutoff].zwerr)
            for (col, percentile_cutoff) in zip(odds_test, percentiles_cutoff)
        ]
        etas_odds = [
            caluclate_eta(df[df[f"odds_{col}"] > percentile_cutoff])
            for (col, percentile_cutoff) in zip(odds_test, percentiles_cutoff)
        ]


# -

df_completeness = pd.DataFrame(
    np.c_[xlab_odds, scatter_odds, eta_odds],
    columns=["completeness", "sigma_odds", "eta_odds"],
)

# ## PLOTS

# +
# Initialize the figure and axis
fig, ax1 = plt.subplots(figsize=(7, 5))

# First plot (Sigma) - using the left y-axis
color = "crimson"
ax1.plot(
    df_completeness.completeness,
    df_completeness.sigma_odds,
    marker=".",
    color=color,
    label=r"NMAD",
    ls="-",
    alpha=0.5,
)


ax1.set_xlabel("Completeness", fontsize=16)
ax1.set_ylabel(r"NMAD [$\Delta z$]", color=color, fontsize=16)
ax1.tick_params(axis="x", labelsize=14)
ax1.tick_params(
    axis="y", which="major", labelsize=14, width=2.5, length=3, labelcolor=color
)
ax1.set_xticks(np.arange(5, 101, 10))

ax2 = ax1.twinx()  # Create another y-axis that shares the same x-axis
color = "navy"
ax2.plot(
    df_completeness.completeness,
    df_completeness.eta_odds,
    marker=".",
    color=color,
    label=r"$\eta$ [%]",
    ls="--",
    alpha=0.5,
)

ax2.set_ylabel(r"$\eta$ [%]", color=color, fontsize=16)

# Adjust notation to allow comparison
ax1.yaxis.get_major_formatter().set_powerlimits(
    (0, 0)
)  # Adjust scientific notation for Sigma
ax2.yaxis.get_major_formatter().set_powerlimits(
    (0, 0)
)  # Adjust scientific notation for Eta
ax2.tick_params(axis="x", labelsize=14)
ax2.tick_params(
    axis="y", which="major", labelsize=14, width=2.5, length=3, labelcolor=color
)

# Final adjustments
fig.tight_layout()
fig.legend(bbox_to_anchor=[-0.18, 0.75, 0.5, 0.2], fontsize=14)
# plt.savefig('Flag_nmad_eta_sigma_comparison.pdf', bbox_inches='tight')
plt.show()


# +
# Initialize the figure and axis
fig, ax1 = plt.subplots(figsize=(7, 5))

# First plot (Sigma) - using the left y-axis
color = "crimson"
ax1.plot(
    odds_test,
    scatters_odds,
    marker=".",
    color=color,
    label=r"NMAD",
    ls="-",
    alpha=0.5,
)


ax1.set_xlabel(r"$\delta z$ (ODDS)", fontsize=16)
ax1.set_ylabel(r"NMAD [$\Delta z$]", color=color, fontsize=16)
ax1.tick_params(axis="x", labelsize=14)
ax1.tick_params(
    axis="y", which="major", labelsize=14, width=2.5, length=3, labelcolor=color
)
ax1.set_xticks(np.arange(0, 0.16, 0.02))

ax2 = ax1.twinx()  # Create another y-axis that shares the same x-axis
color = "navy"
ax2.plot(
    odds_test,
    etas_odds,
    marker=".",
    color=color,
    label=r"$\eta$ [%]",
    ls="--",
    alpha=0.5,
)

ax2.set_ylabel(r"$\eta$ [%]", color=color, fontsize=16)

# Adjust notation to allow comparison
ax1.yaxis.get_major_formatter().set_powerlimits(
    (0, 0)
)  # Adjust scientific notation for Sigma
ax2.yaxis.get_major_formatter().set_powerlimits(
    (0, 0)
)  # Adjust scientific notation for Eta
ax2.tick_params(axis="x", labelsize=14)
ax2.tick_params(
    axis="y", which="major", labelsize=14, width=2.5, length=3, labelcolor=color
)

# Final adjustments
fig.tight_layout()
fig.legend(bbox_to_anchor=[0.10, 0.75, 0.5, 0.2], fontsize=14)
# plt.savefig('ODDS_study.pdf', bbox_inches='tight')
plt.show()

# -
