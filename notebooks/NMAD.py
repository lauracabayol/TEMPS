# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: temps
#     language: python
#     name: temps
# ---

# %% [markdown]
# # FIGURE METRICS

# %% [markdown]
# ## METRICS FOR THE DIFFERENT METHODS ON THE WIDE FIELD SAMPLE

# %% [markdown]
# ### LOAD PYTHON MODULES

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import torch
from pathlib import Path

# %%
# matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


# %%
import temps

# %%
from temps.archive import Archive
from temps.utils import nmad
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule
from temps.plots import plot_photoz


# %%
eval_methods = True

# %% [markdown]
# ### LOAD DATA

# %%

# define here the directory containing the photometric catalogues
parent_dir = Path(
    "/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5"
)
modules_dir = Path("../data/models/")
filename_calib = "euclid_cosmos_DC2_S1_v2.1_calib_clean.fits"
filename_valid = "euclid_cosmos_DC2_S1_v2.1_valid_matched.fits"

# %%
path_file = parent_dir / filename_valid  # Creating the path to the file
hdu_list = fits.open(path_file)
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat["FLAG_PHOT"] == 0]
cat = cat[cat["mu_class_L07"] == 1]
cat = cat[(cat["z_spec_S15"] > 0) | (cat["photo_z_L15"] > 0)]
cat = cat[cat["MAG_VIS"] < 25]


# %%
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

# %%
photoz_archive = Archive(
    path_calib=parent_dir / filename_calib,
    path_valid=parent_dir / filename_valid,
    only_zspec=False,
)
f = photoz_archive._extract_fluxes(catalogue=cat)
col = photoz_archive._to_colors(f)

# %% [markdown]
# ### EVALUATE USING TRAINED MODELS

# %%
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

        z, pz, odds = temps_module.get_pz(
            input_data=torch.Tensor(col), return_pz=True, return_flag=True
        )
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


# %%
dfs["z"]["zwerr"] = (dfs["z"].z - dfs["z"].ztarget) / (1 + dfs["z"].ztarget)
dfs["L15"]["zwerr"] = (dfs["L15"].z - dfs["L15"].ztarget) / (1 + dfs["L15"].ztarget)
dfs["DA"]["zwerr"] = (dfs["DA"].z - dfs["DA"].ztarget) / (1 + dfs["DA"].ztarget)

# %% [markdown]
# ### LOAD CATALOGUES FROM PREVIOUS TRAINING

# %%
if not eval_methods:
    dfs = {}
    dfs["z"] = pd.read_csv(parent_dir / "predictions_specztraining.csv", header=0)
    dfs["L15"] = pd.read_csv(parent_dir / "predictions_speczL15training.csv", header=0)
    dfs["DA"] = pd.read_csv(parent_dir / "predictions_speczDAtraining.csv", header=0)


# %% [markdown]
# ### MAKE PLOT

# %%
df_list = [dfs["z"], dfs["L15"], dfs["DA"]]

# %%
plot_photoz(
    df_list,
    nbins=8,
    xvariable="VISmag",
    metric="nmad",
    type_bin="bin",
    label_list=["zs", "zs+L15", r"TEMPS"],
    save=False,
    samp="L15",
)

# %%
