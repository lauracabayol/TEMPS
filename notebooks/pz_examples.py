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

# # $p(z)$ examples

# ## IMPACT OF TEMPS ON CONCRETE P(Z) EXAMPLES

# ### LOAD PYTHON MODULES

# %load_ext autoreload
# %autoreload 2

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

from temps.archive import Archive 
from temps.utils import nmad
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule


# ### LOAD DATA

#define here the directory containing the photometric catalogues
parent_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
modules_dir = Path('../data/models/')

filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'
path_file = parent_dir / filename_valid  # Creating the path to the file
hdu_list = fits.open(path_file)
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat['FLAG_PHOT']==0]
cat = cat[cat['mu_class_L07']==1]
cat = cat[(cat['z_spec_S15'] > 0) | (cat['photo_z_L15'] > 0)]
cat = cat[cat['MAG_VIS']<25]


ztarget = [cat['z_spec_S15'].values[ii] if cat['z_spec_S15'].values[ii]> 0 else cat['photo_z_L15'].values[ii] for ii in range(len(cat))]
specz_or_photo = [0 if cat['z_spec_S15'].values[ii]> 0 else 1 for ii in range(len(cat))]
ID = cat['ID']
VISmag = cat['MAG_VIS']
zsflag = cat['reliable_S15']

photoz_archive = Archive(path = parent_dir,only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue= cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# ### LOAD TRAINED MODELS AND EVALUATE PDF OF RANDOM EXAMPLES

# The notebook 'Tutorial_temps' gives an example of how to train and save models.

# Initialize an empty dictionary to store DataFrames
ii = np.random.randint(0,len(col),1)
pz_dict = {}
for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(modules_dir / f'modelF_{lab}.pt',map_location=torch.device('cpu')))
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(torch.load(modules_dir / f'modelZ_{lab}.pt',map_location=torch.device('cpu')))
    
    temps_module = TempsModule(nn_features, nn_z)
    
    
    z, pz, fodds = temps_module.get_pz(input_data=torch.Tensor(col[ii]),return_pz=True)

    
    # Assign the DataFrame to a key in the dictionary
    pz_dict[lab] = pz


# +
cmap = plt.get_cmap('Dark2') 

plt.plot(np.linspace(0,5,1000),pz_dict['z'][0],label='z', color = cmap(0), ls ='--')
plt.plot(np.linspace(0,5,1000),pz_dict['L15'][0],label='L15', color = cmap(1), ls =':')
plt.plot(np.linspace(0,5,1000),pz_dict['DA'][0],label='TEMPS', color = cmap(2), ls ='-')
plt.axvline(x=np.array(ztarget)[ii][0],ls='-.',color='black')
#plt.xlim(0,2)
plt.legend()

plt.xlabel(r'$z$', fontsize=14)
plt.ylabel('Probability', fontsize=14)
#plt.savefig(f'pz_{ii[0]}.pdf', bbox_inches='tight')
# -


