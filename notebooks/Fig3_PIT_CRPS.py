# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: insight
#     language: python
#     name: insight
# ---

# %% [markdown]
# # FIGURE 3 IN THE PAPER

# %% [markdown]
# ## PIT AND CRPS FOR THE THREE METHODS

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


# %%
#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %%
#insight modules
import sys
sys.path.append('../temps')
#from insight_arch import EncoderPhotometry, MeasureZ
#from insight import Insight_module
from archive import archive 
from utils import nmad
from plots import plot_PIT, plot_crps
from temps_arch import EncoderPhotometry, MeasureZ
from temps import Temps_module


# %% [markdown]
# ### LOAD DATA

# %%
photoz_archive = archive(path = parent_dir,
                         only_zspec=False,
                         flags_kept=[1. , 1.1, 1.4, 1.5, 2,2.1,2.4,2.5,3., 3.1, 3.4, 3.5,  4., 9. , 9.1, 9.3, 9.4, 9.5,11.1, 11.5, 12.1, 12.5, 13. , 13.1, 13.5, 14, ],
                         target_test='L15')
f_test, ferr_test, specz_test ,VIS_mag_test = photoz_archive.get_testing_data()


# %% [markdown]
# ## CREATE PIT; CRPS; SPECTROSCOPIC SAMPLE

# %% [markdown]
# This loads pre-trained models (for the sake of time). You can learn how to train the models in the Tutorial notebook.

# %%
# Initialize an empty dictionary to store DataFrames
crps_dict = {}
pit_dict = {}
for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(torch.load(os.path.join(modules_dir,f'modelZ_{lab}.pt')))
    
    temps = Temps_module(nn_features, nn_z)
    
    
    pit_list = temps.pit(input_data=torch.Tensor(f_test), target_data=torch.Tensor(specz_test))
    crps_list = temps.crps(input_data=torch.Tensor(f_test), target_data=specz_test)

    
    # Assign the DataFrame to a key in the dictionary
    crps_dict[lab] = crps_list
    pit_dict[lab] = pit_list


# %%
plot_PIT(pit_dict['z'],
         pit_dict['L15'],
         pit_dict['DA'], 
         labels=[r'$z_{rm s}$', 'L15', 'TEMPS'], 
         sample='L15',
        save=True)




# %%
plot_crps(crps_dict['z'],
          crps_dict['L15'],
          crps_dict['DA'], 
          labels=[r'$z_{\rm s}$', 'L15', 'TEMPS'],
          sample = 'L15',
         save=True)




# %%
