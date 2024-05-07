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

# # FIGURE COLOURSPACE IN THE PAPER

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import torch

#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
#insight modules
import sys
sys.path.append('../temps')

from archive import archive 
from utils import nmad
from temps_arch import EncoderPhotometry, MeasureZ
from temps import Temps_module
from plots import plot_nz


# -

def estimate_som_map(df, plot_arg='z', nx=40, ny=40):
    """
    Estimate a Self-Organizing Map (SOM) visualization from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame containing data for SOM estimation.
    - plot_arg (str, optional): Column name to be used for plotting. Default is 'z'.
    - nx (int, optional): Number of cells along the X-axis. Default is 40.
    - ny (int, optional): Number of cells along the Y-axis. Default is 40.

    Returns:
    - som_data (numpy.ndarray): Estimated SOM visualization data.
    """
    x_cells = np.arange(0, nx)
    y_cells = np.arange(0, ny)
    index_cell = np.arange(nx * ny)
    cells = np.array(np.meshgrid(x_cells, y_cells)).T.reshape(-1, 2)
    cells = pd.DataFrame(np.c_[cells[:, 0], cells[:, 1], index_cell], columns=['x_cell', 'y_cell', 'cell'])

    if plot_arg == 'count':
        som_vis = df.groupby('cell')['z'].count().reset_index().rename(columns={f'z': 'plot_som'})
    else:
        som_vis = df.groupby('cell')[f'{plot_arg}'].mean().reset_index().rename(columns={f'{plot_arg}': 'plot_som'})

    som_data = som_vis.merge(cells, on='cell')
    som_data = som_data.pivot(index='x_cell', columns='y_cell', values='plot_som')

    return som_data



def plot_som_map(som_data, plot_arg = 'z', vmin=0, vmax=1):
    """
    Plot the Self-Organizing Map (SOM) data.

    Parameters:
    - som_data (numpy.ndarray): The SOM data to be visualized.
    - plot_arg (str, optional): The column name to be plotted. Default is 'z'.
    - vmin (float, optional): Minimum value for color scaling. Default is 0.
    - vmax (float, optional): Maximum value for color scaling. Default is 1.

    Returns:
    None
    """
    plt.imshow(som_data, vmin=vmin, vmax=vmax, cmap='viridis')  # Choose an appropriate colormap
    plt.colorbar(label=f'{plot_arg}')  # Add a colorbar with a label
    plt.xlabel(r'$x$ [pixel]', fontsize=14)  # Add an appropriate X-axis label
    plt.ylabel(r'$y$ [pixel]', fontsize=14)  # Add an appropriate Y-axis label
    plt.show()



# ### LOAD DATA

# +
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(os.path.join(parent_dir,filename_valid))
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat['FLAG_PHOT']==0]
cat = cat[cat['mu_class_L07']==1]
cat = cat[(cat['z_spec_S15'] > 0) | (cat['photo_z_L15'] > 0)]
cat = cat[cat['MAG_VIS']<25]

# -

ztarget = [cat['z_spec_S15'].values[ii] if cat['z_spec_S15'].values[ii]> 0 else cat['photo_z_L15'].values[ii] for ii in range(len(cat))]
specz_or_photo = [0 if cat['z_spec_S15'].values[ii]> 0 else 1 for ii in range(len(cat))]
ID = cat['ID']
VISmag = cat['MAG_VIS']
zsflag = cat['reliable_S15']

photoz_archive = archive(path = parent_dir,only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue= cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# +
dfs = {}

for il, lab in enumerate(['z','L15','DA']):

    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(torch.load(os.path.join(modules_dir,f'modelZ_{lab}.pt')))

    temps = Temps_module(nn_features, nn_z)

    z,zerr ,pz, flag, odds = temps.get_pz(input_data=torch.Tensor(col), 
                                return_pz=True)
    # Create a DataFrame with the desired columns
    df = pd.DataFrame(np.c_[ID, VISmag,z, flag, ztarget,zsflag,zerr, specz_or_photo], 
                      columns=['ID','VISmag','z','zflag', 'ztarget','zsflag','zuncert','S15_L15_flag'])

    # Calculate additional columns or operations if needed
    df['zwerr'] = (df.z - df.ztarget) / (1 + df.ztarget)

    # Drop any rows with NaN values
    df = df.dropna()

    # Assign the DataFrame to a key in the dictionary
    dfs[lab] = df

# -

# ### LOAD TRAINED MODELS AND EVALUATE PDFs AND REDSHIFT

#define here the directory containing the photometric catalogues
parent_dir = '/data/astro/scratch2/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5'
modules_dir = '../data/models/'

df_z = dfs['z']
df_z_DA = dfs['DA']

# ##### LOAD TRAIN SOM ON TRAINING DATA

df_som = pd.read_csv(os.path.join(parent_dir,'som_dataframe.csv'), header = 0, sep =',')
df_z = df_z.merge(df_som, on = 'ID')
df_z_DA = df_z_DA.merge(df_som, on = 'ID')

# ##### APPLY CUTS FOR DIFFERENT SAMPLES

df_zspec = df_z[(df_z.S15_L15_flag==0) & (df_z.zsflag==1)]
df_l15 = df_z[(df_z.ztarget>0)]
df_l15_DA = df_z_DA[(df_z_DA.ztarget>0)]

df_l15_euclid = df_z[(df_z.VISmag <24.5) & (df_z.z > 0.2) & (df_z.z < 2.6)]
df_l15_euclid_cut= df_l15_euclid[df_l15_euclid.zflag>0.033]

df_l15_euclid_da = df_z_DA[(df_z_DA.VISmag <24.5) & (df_z_DA.z > 0.2) & (df_z_DA.z < 2.6)]
df_l15_euclid_cut_da= df_l15_euclid_da[df_l15_euclid_da.zflag>0.018]

# ## MAKE SOM PLOT

from mpl_toolkits.axes_grid1 import make_axes_locatable

# +
fig, axs = plt.subplots(6, 4, figsize=(13, 15), sharex=True, sharey=True, gridspec_kw={'hspace': 0.05, 'wspace': 0.06})

# Plot in the top row (axs[0, i])
#top row, spectroscopic sample
columns = ['ztarget','z','zwerr','count']
titles = [r'$z_{true}$',r'$z$',r'$z_{\rm error}$','Counts']
limits = [[0,4],[0,4],[-0.5,0.5],[0,50]]
for ii in range(4):
    som_data = estimate_som_map(df_zspec, plot_arg=columns[ii], nx=40, ny=40)
    im = axs[0,ii].imshow(som_data, vmin=limits[ii][0], vmax=limits[ii][1], cmap='viridis')  # Choose an appropriate colormap
    axs[0, ii].set_title(f'{titles[ii]}', fontsize=18)
    
    if ii==0:
        axs[0, 0].set_ylabel(r'$y$', fontsize=14)
    elif ii==1:
        cbar_ax = fig.add_axes([0.49, 0.11, 0.01, 0.77])
        fig.colorbar(im, cax=cbar_ax)
    elif ii==2:
        cbar_ax = fig.add_axes([0.685, 0.11, 0.01, 0.77])
        fig.colorbar(im, cax=cbar_ax)
    elif ii==3:
        cbar_ax = fig.add_axes([0.885, 0.11, 0.01, 0.77])
        fig.colorbar(im, cax=cbar_ax)

for jj in range(4):
    som_data = estimate_som_map(df_l15, plot_arg=columns[jj], nx=40, ny=40)
    im = axs[1,jj].imshow(som_data, vmin=limits[jj][0], vmax=limits[jj][1], cmap='viridis')  # Choose an appropriate colormap
    #axs[1, jj].set_title(f'{titles[jj]}', fontsize=14)
    #axs[1, jj].set_xlabel(r'$x$', fontsize=14)
    
    
for kk in range(4):
    som_data = estimate_som_map(df_l15_DA, plot_arg=columns[kk], nx=40, ny=40)
    im = axs[2,kk].imshow(som_data, vmin=limits[kk][0], vmax=limits[kk][1], cmap='viridis')  # Choose an appropriate colormap
    #axs[2, kk].set_title(f'{titles[kk]}', fontsize=14)
    #axs[2, kk].set_xlabel(r'$x$', fontsize=14)
    
for rr in range(4):
    som_data = estimate_som_map(df_l15_euclid_da, plot_arg=columns[rr], nx=40, ny=40)
    im = axs[3,rr].imshow(som_data, vmin=limits[rr][0], vmax=limits[rr][1], cmap='viridis')  # Choose an appropriate colormap
    #axs[3, rr].set_title(f'{titles[rr]}', fontsize=14)
    #axs[3, rr].set_xlabel(r'$x$', fontsize=14)
    
for ll in range(4):
    som_data = estimate_som_map(df_l15_euclid_cut, plot_arg=columns[ll], nx=40, ny=40)
    im = axs[4,ll].imshow(som_data, vmin=limits[ll][0], vmax=limits[ll][1], cmap='viridis')  # Choose an appropriate colormap
    #axs[4, ll].set_title(f'{titles[ll]}', fontsize=14)
    axs[4, ll].set_xlabel(r'$x$', fontsize=14)
    
for ll in range(4):
    som_data = estimate_som_map(df_l15_euclid_cut_da, plot_arg=columns[ll], nx=40, ny=40)
    im = axs[5,ll].imshow(som_data, vmin=limits[ll][0], vmax=limits[ll][1], cmap='viridis')  # Choose an appropriate colormap
    #axs[4, ll].set_title(f'{titles[ll]}', fontsize=14)
    axs[5, ll].set_xlabel(r'$x$', fontsize=14)

    
axs[0, 0].set_ylabel(r'$y$', fontsize=14)
axs[1, 0].set_ylabel(r'$y$', fontsize=14)
axs[2, 0].set_ylabel(r'$y$', fontsize=14)
axs[3, 0].set_ylabel(r'$y$', fontsize=14)
axs[4, 0].set_ylabel(r'$y$', fontsize=14)
axs[5, 0].set_ylabel(r'$y$', fontsize=14)


fig.text(0.09, 0.815, r'$z_{\rm s}$ sample', va='center', rotation='vertical', fontsize=16)
fig.text(0.09, 0.69, r'L15 sample', va='center', rotation='vertical', fontsize=16)
fig.text(0.09, 0.56, r'L15 sample + DA', va='center', rotation='vertical', fontsize=14)
fig.text(0.09, 0.44, r'$Euclid$ sample + DA', va='center', rotation='vertical', fontsize=14)
fig.text(0.09, 0.3, r'$Euclid$ sample + QC', va='center', rotation='vertical', fontsize=14)

fig.text(0.09, 0.17, r'$Euclid$ sample + DA + QC', va='center', rotation='vertical', fontsize=13)


plt.savefig('SOM_colourspace.pdf', format='pdf', bbox_inches='tight', dpi=300)

# -


