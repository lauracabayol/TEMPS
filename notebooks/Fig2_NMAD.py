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
# # FIGURE 2 IN THE PAPER

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

# %%
#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


# %%
#insight modules
import sys
sys.path.append('../insight')
#from insight_arch import EncoderPhotometry, MeasureZ
#from insight import Insight_module
from archive import archive 
from utils import nmad
from plots import plot_photoz


# %%
train_methods=False

# %% [markdown]
# ### LOAD DATA

# %%
#define here the directory containing the photometric catalogues
parent_dir = '/data/astro/scratch2/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5'
modules_dir = '../data/models/'

# %%
#load catalogue and apply cuts

filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(os.path.join(parent_dir,filename_valid))
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat['FLAG_PHOT']==0]
cat = cat[cat['mu_class_L07']==1]
cat = cat[(cat['z_spec_S15'] > 0) | (cat['photo_z_L15'] > 0)]


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
# ### TRAIN METHODS

# %%
if train_methods:
    dfs = {}

    for il, lab in enumerate(['z','L15','DA']):

        nn_features = EncoderPhotometry()
        nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
        nn_z = MeasureZ(num_gauss=6)
        nn_z.load_state_dict(torch.load(os.path.join(modules_dir,f'modelZ_{lab}.pt')))

        insight = Insight_module(nn_features, nn_z)

        z,zerr, pz, flag = insight.get_pz(input_data=torch.Tensor(col), 
                                    return_pz=True)

        # Create a DataFrame with the desired columns
        df = pd.DataFrame(np.c_[ID, VISmag,z, flag, ztarget,zsflag,zerr, specz_or_photo], 
                          columns=['ID','VISmag','z','zflag', 'ztarget','zsflag','zuncert','S15_L15_flag'])

        # Calculate additional columns or operations if needed
        df['zwerr'] = (df.z - df.zs) / (1 + df.zs)

        # Drop any rows with NaN values
        df = df.dropna()

        # Assign the DataFrame to a key in the dictionary
        dfs[lab] = df


# %% [markdown]
# ### LOAD CATALOGUES FROM PREVIOUS TRAINING

# %%
if not train_methods:
    dfs = {}
    dfs['z'] = pd.read_csv(os.path.join(parent_dir, 'predictions_specztraining.csv'), header=0) 
    dfs['L15'] = pd.read_csv(os.path.join(parent_dir, 'predictions_speczL15training.csv'), header=0) 
    dfs['DA'] = pd.read_csv(os.path.join(parent_dir, 'predictions_speczDAtraining.csv'), header=0) 


# %% [markdown]
# ### MAKE PLOT

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_photoz(df_list, nbins, xvariable, metric, type_bin='bin',label_list=None, samp='zs', save=False):
    #plot properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    if xvariable == 'VISmag':
        xvariable_lab = 'VIS'
    if xvariable == 'zs':
        xvariable_lab = r'$z_{\rm s}$'

    bin_edges = stats.mstats.mquantiles(df_list[0][xvariable].values, np.linspace(0.05, 1, nbins))
    cmap = plt.get_cmap('Dark2')  # Choose a colormap for coloring lines
    #plt.figure(figsize=(6, 5))
    ls = ['--',':','-']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ydata_dict = {}

    for i, df in enumerate(df_list):
        ydata, xlab = [], []
        
        label = label_list[i]
        
        if label == 'zs':
            label_lab = r'$z_{\rm s}$'
        if label == 'zs+L15':
            label_lab = r'$z_{\rm s}$+L15'    
        if label == 'TEMPS':
            label_lab = 'TEMPS'   

        for k in range(len(bin_edges)-1):
            edge_min = bin_edges[k]
            edge_max = bin_edges[k+1]

            mean_mag = (edge_max + edge_min) / 2

            if type_bin == 'bin':
                df_plot = df[(df[xvariable] > edge_min) & (df[xvariable] < edge_max)]
            elif type_bin == 'cum':
                df_plot = df[(df[xvariable] < edge_max)]
            else:
                raise ValueError("Only type_bin=='bin' for binned and 'cum' for cumulative are supported")

            xlab.append(mean_mag)
            if metric == 'sig68':
                ydata.append(sigma68(df_plot.zwerr))
            elif metric == 'bias':
                ydata.append(np.mean(df_plot.zwerr))
            elif metric == 'nmad':
                ydata.append(nmad(df_plot.zwerr))
            elif metric == 'outliers':
                ydata.append(len(df_plot[np.abs(df_plot.zwerr) > 0.15]) / len(df_plot)*100)
                
        ydata_dict[f'{i}'] = ydata
        color = cmap(i)  # Get a different color for each dataframe
        ax1.plot(xlab, ydata,marker='.', lw=1, label=label_lab, color=color, ls=ls[i])
        


    ax1.set_ylabel(f'{metric} $[\Delta z]$', fontsize=18)
    #ax1.set_xlabel(f'{xvariable_lab}', fontsize=16)
    ax1.grid(False)
    ax1.legend()
    
    # Plot ratios between lines in the upper panel
    
    ax2.plot(xlab, np.array(ydata_dict['1'])/np.array(ydata_dict['0']), marker='.', color = cmap(1))
    ax2.plot(xlab, np.array(ydata_dict['2'])/np.array(ydata_dict['0']), marker='.', color = cmap(2))
    ax2.set_ylabel(r'Method $X$ / $z_{\rm z}$', fontsize=14)
   

    ax2.set_xlabel(f'{xvariable_lab}', fontsize=16)
    ax2.grid(True)

    
    if save==True:
        plt.savefig(f'{metric}_{xvariable}_{samp}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# %%
dfs['z'] = dfs['z'][(dfs['z'].VISmag<24.5)&(dfs['z'].z<4)&(dfs['z'].zs>0)]
dfs['L15'] = dfs['L15'][(dfs['L15'].VISmag<24.5)&(dfs['L15'].z<4)&(dfs['L15'].zs>0)]
dfs['DA'] = dfs['DA'][(dfs['DA'].VISmag<24.5)&(dfs['DA'].z<4)&(dfs['DA'].zs>0)]

df_list = [dfs['z'],dfs['L15'],dfs['DA'] ]

# %%
plot_photoz(df_list,
            nbins=8, 
            xvariable='VISmag', 
            metric='nmad', 
            type_bin='bin', 
            label_list = ['zs','zs+L15',r'TEMPS'],
            save=True,
            samp='L15'
           )

# %%
plot_photoz(df_list,
            nbins=8, 
            xvariable='VISmag', 
            metric='outliers', 
            type_bin='bin', 
            label_list = ['zs','zs+L15',r'TEMPS'],
            save=True,
            samp='L15'
           )

# %%

# %%
