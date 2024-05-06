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
# # FIGURE 4 IN THE PAPER

# %% [markdown]
# ## IMPACT OF TEMPS ON CONCRETE P(Z) EXAMPLES

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
from temps_arch import EncoderPhotometry, MeasureZ
from temps import Temps_module


# %% [markdown]
# ### LOAD DATA

# %%
#define here the directory containing the photometric catalogues
parent_dir = '/data/astro/scratch2/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5'
modules_dir = '../data/models/'

# %%
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(os.path.join(parent_dir,filename_valid))
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat['FLAG_PHOT']==0]
cat = cat[cat['mu_class_L07']==1]
cat = cat[(cat['z_spec_S15'] > 0) | (cat['photo_z_L15'] > 0)]
cat = cat[cat['MAG_VIS']<25]


# %%
ztarget = [cat['z_spec_S15'].values[ii] if cat['z_spec_S15'].values[ii]> 0 else cat['photo_z_L15'].values[ii] for ii in range(len(cat))]
specz_or_photo = [0 if cat['z_spec_S15'].values[ii]> 0 else 1 for ii in range(len(cat))]
ID = cat['ID']
VISmag = cat['MAG_VIS']
zsflag = cat['reliable_S15']

# %%
photoz_archive = archive(path = parent_dir,only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue= cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# %% [markdown]
# ### LOAD TRAINED MODELS AND EVALUATE PDF OF RANDOM EXAMPLES

# %% [markdown]
# The notebook 'Tutorial_temps' gives an example of how to train and save models.

# %%
# Initialize an empty dictionary to store DataFrames
ii = np.random.randint(0,len(col),1)
pz_dict = {}
for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(torch.load(os.path.join(modules_dir,f'modelZ_{lab}.pt')))
    
    temps = Temps_module(nn_features, nn_z)
    
    
    z,zerr, pz, flag,_ = temps.get_pz(input_data=torch.Tensor(col[ii]),return_pz=True)

    
    # Assign the DataFrame to a key in the dictionary
    pz_dict[lab] = pz


# %%
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

# %%
