# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: insight
#     language: python
#     name: insight
# ---

# # FIGURE 6 IN THE PAPER

# ## QUALITY CUTS

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import os
import torch
from scipy import stats

#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

#insight modules
import sys
sys.path.append('../temps')
#from insight_arch import EncoderPhotometry, MeasureZ
#from insight import Insight_module
from archive import archive 
from utils import nmad
from temps_arch import EncoderPhotometry, MeasureZ
from temps import Temps_module


# ### LOAD DATA (ONLY SPECZ)

#define here the directory containing the photometric catalogues
parent_dir = '/data/astro/scratch2/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5'
modules_dir = '../data/models/'

photoz_archive = archive(path = parent_dir,only_zspec=True,flags_kept=[1. , 1.1, 1.4, 1.5, 2,2.1,2.4,2.5,3., 3.1, 3.4, 3.5,  4., 9. , 9.1, 9.3, 9.4, 9.5,11.1, 11.5, 12.1, 12.5, 13. , 13.1, 13.5, 14, ])
f_test_specz, ferr_test_specz, specz_test ,VIS_mag_test = photoz_archive.get_testing_data()


# ### LOAD TRAINED MODELS AND EVALUATE PDF OF RANDOM EXAMPLES

# +
# Initialize an empty dictionary to store DataFrames
dfs = {}

for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(torch.load(os.path.join(modules_dir,f'modelZ_{lab}.pt')))
    
    temps = Temps_module(nn_features, nn_z)
    
    z,zerr, pz, flag, odds = temps.get_pz(input_data=torch.Tensor(f_test_specz), 
                                return_pz=True)
    
    
    # Create a DataFrame with the desired columns
    df = pd.DataFrame(np.c_[z, flag, odds, specz_test], 
                      columns=['z','zflag', 'odds' ,'ztarget'])
    
    # Calculate additional columns or operations if needed
    df['zwerr'] = (df.z - df.ztarget) / (1 + df.ztarget)
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Assign the DataFrame to a key in the dictionary
    dfs[lab] = df

# -

# ### STATISTICS BASED ON OUR QUALITY CUT

# +
bin_edges = stats.mstats.mquantiles(df.zflag, np.arange(0,1.01,0.05))
scatter, eta, xlab, xmag, xzs, flagmean = [],[],[], [], [], []

for k in range(len(bin_edges)-1):
    edge_min = bin_edges[k]
    edge_max = bin_edges[k+1]
    
    df_bin = df[(df.zflag > edge_min)]    
    

    xlab.append(np.round(len(df_bin)/len(df),2)*100)
    xzs.append(0.5*(df_bin.ztarget.min()+df_bin.ztarget.max()))
    flagmean.append(np.mean(df_bin.zflag))
    scatter.append(nmad(df_bin.zwerr))
    eta.append(len(df_bin[np.abs(df_bin.zwerr)>0.15])/len(df)*100)


# -

# ### STATISTICS BASED ON ODDS 

# +
bin_edges = stats.mstats.mquantiles(df.odds, np.arange(0,1.01,0.05))
scatter_odds, eta_odds,xlab_odds,  oddsmean = [],[],[], []

for k in range(len(bin_edges)-1):
    edge_min = bin_edges[k]
    edge_max = bin_edges[k+1]
    
    df_bin = df[(df.odds > edge_min)]    
    

    xlab_odds.append(np.round(len(df_bin)/len(df),2)*100)
    oddsmean.append(np.mean(df_bin.zflag))
    scatter_odds.append(nmad(df_bin.zwerr))
    eta_odds.append(len(df_bin[np.abs(df_bin.zwerr)>0.15])/len(df)*100)


# -

# ### PLOTS

# +
plt.plot(xlab_odds,scatter_odds, marker = '.', color ='crimson', label=r'$\theta(\Delta z)$', ls='--', alpha=0.5)
plt.plot(xlab,scatter, marker = '.', color ='navy',label=r'$\xi = \theta(\Delta z)$')


plt.ylabel(r'NMAD [$\Delta z\ /\ (1 + z_{\rm s})$]', fontsize=16)
plt.xlabel('Completeness', fontsize=16)

plt.yticks(fontsize=12)
plt.xticks(np.arange(5,101,10), fontsize=12)
plt.legend(fontsize=14)

plt.savefig('Flag_nmad_zspec.pdf', bbox_inches='tight')
plt.show()

# +
plt.plot(xlab_odds,eta_odds, marker='.', color ='crimson', label=r'$\theta(\Delta z)$', ls='--', alpha=0.5)
plt.plot(xlab,eta, marker='.', color ='navy',label=r'$\xi = \theta(\Delta z)$')

plt.yticks(fontsize=12)
plt.xticks(np.arange(5,101,10), fontsize=12)
plt.ylabel(r'$\eta$ [%]', fontsize=16)
plt.xlabel('Completeness', fontsize=16)
plt.legend()

plt.savefig('Flag_eta_zspec.pdf', bbox_inches='tight')

plt.show()
# -


