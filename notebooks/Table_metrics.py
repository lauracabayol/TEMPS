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

# # TABLE METRICS

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import os
import torch
from scipy import stats
from astropy.io import fits
from astropy.table import Table
from pathlib import Path

#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from temps.archive import Archive 
from temps.utils import nmad, select_cut
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule


# ## LOAD DATA

#define here the directory containing the photometric catalogues
parent_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
modules_dir = Path('../data/models/')

# +
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(parent_dir / filename_valid)
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat['FLAG_PHOT']==0]
cat = cat[cat['mu_class_L07']==1]

cat['SNR_VIS'] = cat.FLUX_VIS / cat.FLUXERR_VIS
# -

cat = cat[cat.SNR_VIS>10]

ztarget = [cat['z_spec_S15'].values[ii] if cat['z_spec_S15'].values[ii]> 0 else cat['photo_z_L15'].values[ii] for ii in range(len(cat))]
specz_or_photo = [0 if cat['z_spec_S15'].values[ii]> 0 else 1 for ii in range(len(cat))]
ID = cat['ID']
VISmag = cat['MAG_VIS']
zsflag = cat['reliable_S15']

cat['ztarget']=ztarget
cat['specz_or_photo']=specz_or_photo

cat = cat[cat.ztarget>0]

# ### EXTRACT PHOTOMETRY

photoz_archive = Archive(path = parent_dir,only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue= cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# ### MEASURE CATALOGUE

# +
# Initialize an empty dictionary to store DataFrames
lab='DA'    
nn_features = EncoderPhotometry()
nn_features.load_state_dict(torch.load(modules_dir / f'modelF_{lab}.pt', map_location=torch.device('cpu')))
nn_z = MeasureZ(num_gauss=6)
nn_z.load_state_dict(torch.load(modules_dir / f'modelZ_{lab}.pt', map_location=torch.device('cpu')))

temps_module = TempsModule(nn_features, nn_z)

z, pz, odds = temps_module.get_pz(input_data=torch.Tensor(col), 
                            return_pz=True)


# Create a DataFrame with the desired columns
df = pd.DataFrame(np.c_[z, odds, cat.ztarget, cat.reliable_S15, cat.specz_or_photo], 
                  columns=['z', 'odds' ,'ztarget','reliable_S15', 'specz_or_photo'])

# Calculate additional columns or operations if needed
df['zwerr'] = (df.z - df.ztarget) / (1 + df.ztarget)

# Drop any rows with NaN values
df = df.dropna()


# -

# ### SPECZ SAMPLE

df_specz = df[(df.reliable_S15==1)&(df.specz_or_photo==0)]

# +
df_selected, cut, dfcuts  = select_cut(df_specz,
                          completenss_lim=None, 
                          nmad_lim=0.055, 
                          outliers_lim=None, 
                          return_df=True)


# -

print(dfcuts.to_latex(float_format="%.3f",
                columns=['Nobj','completeness', 'nmad', 'eta'],
                index=False
               ))

# ### EUCLID SAMPLE

df_euclid = df[(df.z >0.2)&(df.z < 2.6)]

df_euclid

# +
df_selected, cut, dfcuts  = select_cut(df_euclid,
                          completenss_lim=None, 
                          nmad_lim= 0.05, 
                          outliers_lim=None, 
                          return_df=True)


# -

print(dfcuts.to_latex(float_format="%.3f",
                columns=['Nobj','completeness', 'nmad', 'eta'],
                index=False
               ))


