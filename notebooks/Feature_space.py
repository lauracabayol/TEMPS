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

# # DOMAIN ADAPTATION INTUITION

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import torch
from pathlib import Path
import seaborn as sns


#matplotlib settings
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from temps.archive import Archive 
from temps.utils import nmad
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule
from temps.plots import plot_nz

# ## LOAD DATA

#define here the directory containing the photometric catalogues
parent_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
modules_dir = Path('../data/models/')

# +
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(parent_dir/filename_valid)
cat = Table(hdu_list[1].data).to_pandas()
cat = cat[cat['FLAG_PHOT']==0]
cat = cat[cat['mu_class_L07']==1]

cat['SNR_VIS'] = cat.FLUX_VIS / cat.FLUXERR_VIS
#cat = cat[cat.SNR_VIS>10]
# -

ztarget = [cat['z_spec_S15'].values[ii] if cat['z_spec_S15'].values[ii]> 0 else cat['photo_z_L15'].values[ii] for ii in range(len(cat))]
specz_or_photo = [0 if cat['z_spec_S15'].values[ii]> 0 else 1 for ii in range(len(cat))]
ID = cat['ID']
VISmag = cat['MAG_VIS']
zsflag = cat['reliable_S15']
cat['ztarget']=ztarget
cat['specz_or_photo']=specz_or_photo

# ### EXTRACT PHOTOMETRY

photoz_archive = Archive(path = parent_dir,only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue= cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# ### MEASURE FEATURES

features_all = np.zeros((3,len(cat),10))
for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(modules_dir/f'modelF_{lab}.pt',map_location=torch.device('cpu')))
    
    features = nn_features(torch.Tensor(col))
    features = features.detach().cpu().numpy()  
    
    features_all[il]=features
    

# ### TRAIN AUTOENCODER TO REDUCE TO 2 DIMENSIONS

import torch
from torch import nn
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y,x


# +
from torch.utils.data import DataLoader, dataset, TensorDataset

ds =TensorDataset(torch.Tensor(features_all[0]))
train_loader = DataLoader(ds, batch_size=100, shuffle=True, drop_last=False)
        
# -

import torch.optim as optim
autoencoder = Autoencoder(input_dim=10,
                         latent_dim=2)
criterion = nn.L1Loss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

# + jupyter={"outputs_hidden": true}
# Define the number of epochs
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:  # Assuming 'train_loader' is your DataLoader
        # Forward pass
        outputs,f1  = autoencoder(data[0])
        
        loss_autoencoder = criterion(outputs, data[0])
        optimizer.zero_grad()
        
        # Backward pass
        loss_autoencoder.backward()
        
        # Update the weights
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss_autoencoder.item()
    
    # Print the average loss for the epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss / len(train_loader)))

print('Training finished')

# -

# #### EVALUTATE AUTOENCODER 

# cat.to_csv('features_cat.csv', header=True, sep=',')

indexes_specz = cat[(cat.specz_or_photo==0)&(cat.reliable_S15>0)].reset_index().index

features_all_reduced = np.zeros(shape=(3,len(cat),2))
for i in range(3):
    _, features = autoencoder(torch.Tensor(features_all[i]))
    features_all_reduced[i] = features.detach().cpu().numpy()

features_all.shape

# ### Plot the features

start = 0
end = len(cat)
all_values = set(range(start, end))
values_not_in_indexes_specz = all_values - set(indexes_specz)
indexes_nospecz = sorted(values_not_in_indexes_specz)

# +

# Create subplots with three panels
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Set style for all subplots
sns.set_style("white")

# First subplot
sns.kdeplot(x=features_all_reduced[0, indexes_nospecz,0],
            y=features_all_reduced[0, indexes_nospecz,1],
            clip=(-150, 150), 
            ax=axs[0],
            color='salmon')
sns.kdeplot(x=features_all_reduced[0, indexes_specz,0],
            y=features_all_reduced[0, indexes_specz,1],
            clip=(-150, 150), 
            ax=axs[0],
            color='lightskyblue')
            
axs[0].set_xlim(-150, 150)
axs[0].set_ylim(-150, 150)
axs[0].set_title(r'Trained on $z_{\rm s}$')

# Second subplot
sns.kdeplot(x=features_all_reduced[1, indexes_nospecz, 0],
            y=features_all_reduced[1, indexes_nospecz, 1],
            clip=(-50, 50), 
            ax=axs[1],
            color='salmon')
sns.kdeplot(x=features_all_reduced[1, indexes_specz, 0],
            y=features_all_reduced[1, indexes_specz,1],
            clip=(-50, 50), 
            ax=axs[1],
            color='lightskyblue')
axs[1].set_xlim(-50, 50)
axs[1].set_ylim(-50, 50)
axs[1].set_title('Trained on L15')

# Third subplot
features_all_reduced_nospecz = pd.DataFrame(features_all_reduced[2, indexes_nospecz, :]).drop_duplicates().values
sns.kdeplot(x=features_all_reduced[2, indexes_nospecz, 0],
            y=features_all_reduced[2, indexes_nospecz, 1],
            clip=(-1, 5), 
            ax=axs[2],
            color='salmon',
            label='Wide-field sample')
sns.kdeplot(x=features_all_reduced[2, indexes_specz, 0],
            y=features_all_reduced[2, indexes_specz,1],
            clip=(-1, 5), 
            ax=axs[2],
            color='lightskyblue',
            label=r'$z_{\rm s}$ sample')
axs[2].set_xlim(-2, 5)
axs[2].set_ylim(-2, 5)
axs[2].set_title('TEMPS')

axs[0].set_xlabel('Feature 1')
axs[1].set_xlabel('Feature 1')
axs[2].set_xlabel('Feature 1')
axs[0].set_ylabel('Feature 2')

# Create custom legend with desired colors
legend_labels = ['Wide-field sample', r'$z_{\rm s}$ sample']
legend_handles = [plt.Line2D([0], [0], color='salmon', lw=2),
                  plt.Line2D([0], [0], color='lightskyblue', lw=2)]
axs[2].legend(legend_handles, legend_labels, loc='upper right', fontsize=16)
# Adjust layout
plt.tight_layout()

#plt.savefig('Contourplot.pdf', bbox_inches='tight')
plt.show()

# -








