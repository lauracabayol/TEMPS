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

# # $p(z)$ DISTRIBUTIONS

# ## PIT AND CRPS FOR THE THREE METHODS

# ### LOAD PYTHON MODULES

# %load_ext autoreload
# %autoreload 2

import temps

import pandas as pd
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import torch
from pathlib import Path

#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
from temps.temps import TempsModule
from temps.archive import Archive 
from temps.utils import nmad
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.plots import plot_photoz, plot_PIT, plot_crps


# -

# ### LOAD DATA

# +

#define here the directory containing the photometric catalogues
parent_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
modules_dir = Path('../data/models/')
filename_calib = 'euclid_cosmos_DC2_S1_v2.1_calib_clean.fits'
filename_valid = 'euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'
# -

photoz_archive = Archive(path_calib = parent_dir/filename_calib, 
                         path_valid = parent_dir/filename_valid,
                         only_zspec=False)
f_test, specz_test ,VIS_mag_test = photoz_archive.get_testing_data()


# ## CREATE PIT; CRPS; SPECTROSCOPIC SAMPLE

# This loads pre-trained models (for the sake of time). You can learn how to train the models in the Tutorial notebook.

# Initialize an empty dictionary to store DataFrames
crps_dict = {}
pit_dict = {}
for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(modules_dir / f'modelF_{lab}.pt',map_location=torch.device('cpu')))
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(torch.load(modules_dir / f'modelZ_{lab}.pt',map_location=torch.device('cpu')))
    
    temps_module = TempsModule(nn_features, nn_z)
    
    
    pit_list = temps_module.calculate_pit(input_data=torch.Tensor(f_test), target_data=torch.Tensor(specz_test))
    crps_list = temps_module.calculate_crps(input_data=torch.Tensor(f_test), target_data=specz_test)

    
    # Assign the DataFrame to a key in the dictionary
    crps_dict[lab] = crps_list
    pit_dict[lab] = pit_list


# +
plot_PIT(pit_dict['z'],
         pit_dict['L15'],
         pit_dict['DA'], 
         labels=[r'$z_{rm s}$', 'L15', 'TEMPS'], 
         sample='L15',
        save=True)




# +
plot_crps(crps_dict['z'],
          crps_dict['L15'],
          crps_dict['DA'], 
          labels=[r'$z_{\rm s}$', 'L15', 'TEMPS'],
          sample = 'L15',
         save=True)



# -


