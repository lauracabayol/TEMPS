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

# # DOMAIN ADAPTATION INTUITION

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

# ## LOAD DATA

#define here the directory containing the photometric catalogues
parent_dir = '/data/astro/scratch2/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5'
modules_dir = '../data/models/'

# +
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(os.path.join(parent_dir,filename_valid))
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

photoz_archive = archive(path = parent_dir,only_zspec=False)
f, ferr = photoz_archive._extract_fluxes(catalogue= cat)
col, colerr = photoz_archive._to_colors(f, ferr)

# ### MEASURE FEATURES

features_all = np.zeros((3,len(cat),10))
for il, lab in enumerate(['z','L15','DA']):
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
    
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

# +
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

# + [markdown] jupyter={"source_hidden": true}
# cat.to_csv('features_cat.csv', header=True, sep=',')
# -

indexes_specz = cat[(cat.specz_or_photo==0)&(cat.reliable_S15>0)].reset_index().index

features_all_reduced = np.zeros(shape=(3,len(cat),2))
for i in range(3):
    _, features = autoencoder(torch.Tensor(features_all[i]))
    features_all_reduced[i] = features.detach().cpu().numpy()

# ### Plot the features

start = 0
end = len(cat)
all_values = set(range(start, end))
values_not_in_indexes_specz = all_values - set(indexes_specz)
indexes_nospecz = sorted(values_not_in_indexes_specz)

# +
import seaborn as sns

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
sns.kdeplot(x=features_all_reduced_nospecz[:, 0],
            y=features_all_reduced_nospecz[:, 1],
            clip=(-1, 5), 
            ax=axs[2],
            color='salmon',
            label='Wide-field sample')
sns.kdeplot(x=features_all_reduced_specz[:, 0],
            y=features_all_reduced_specz[:, 1],
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

plt.savefig('Contourplot.pdf', bbox_inches='tight')
plt.show()

# -







np.savetxt('features.txt',features_all_reduced.reshape(3*164816, 2))











# +
photoz_archive = archive(path = parent_dir,only_zspec=False)

fig, ax = plt.subplots(ncols = 3, figsize=(15,4), sharex=True, sharey=True)
colors = ['navy', 'goldenrod']
titles = [r'Training: $z_s$', r'Training: L15',r'Training: $z_s$ + DA']
x_min, x_max = -5,5
y_min, y_max = -5,5
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
density_grid = density_estimation(xy_grid).reshape(x_grid.shape)
for il, lab in enumerate(['z','L15','DA']):
    
    
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(torch.load(os.path.join(modules_dir,f'modelF_{lab}.pt')))
    
    for it, target_type in enumerate(['L15','zs']):
        if target_type=='zs':
            cat_sub = photoz_archive._select_only_zspec(cat)
            cat_sub = photoz_archive._clean_zspec_sample(cat_sub)
            
        elif target_type=='L15':
            cat_sub = photoz_archive._exclude_only_zspec(cat)
        else:
            assert False
            
        cat_sub = photoz_archive._clean_photometry(cat_sub)    
        print(cat_sub.shape)
        
        
            
        f, ferr = photoz_archive._extract_fluxes(cat_sub)
        col, colerr = photoz_archive._to_colors(f, ferr)

        features = nn_features(torch.Tensor(col))
        features = features.detach().cpu().numpy()
        

        #xy = np.vstack([features[:1000,0], features[:1000,1]])
        #zd = gaussian_kde(xy)(xy)
        #ax[il].scatter(features[:1000,0], features[:1000,1],c=zd, s=3)

        xy = np.vstack([features[:,0], features[:,1]])
        density_estimation = gaussian_kde(xy)

        # Define grid for plotting density lines

        xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
        density_grid = density_estimation(xy_grid).reshape(x_grid.shape)

        # Plot contour lines representing density
        ax[il].contour(x_grid, y_grid, density_grid, colors=colors[it], label = f'{target_type}')

        

    ax[il].set_title(titles[il])
    ax[il].set_xlim(-5,5)
    ax[il].set_ylim(-5,5)
    
    
ax[0].set_ylabel('Feature 1', fontsize=14)
#plt.ylabel('Feature 2', fontsize=14)    
            
    #assert False
            
            
        

    
    
# -

H

H

xedges

yedges

# +
import matplotlib.colors as colors
from matplotlib import path
import numpy as np
from matplotlib import pyplot as plt
try:
    from astropy.convolution import Gaussian2DKernel, convolve
    astro_smooth = True
except ImportError as IE:
    astro_smooth = False

np.random.seed(123)
#t = np.linspace(-5,1.2,1000)
x = features[:1000,0]
y = features[:1000,1]

H, xedges, yedges = np.histogram2d(x,y, bins=(10,10))
xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])

# Smooth the contours (if astropy is installed)
if astro_smooth:
    kernel = Gaussian2DKernel(x_stddev=1.)
    H=convolve(H,kernel)

fig,ax = plt.subplots(1, figsize=(7,6)) 
clevels = ax.contour(xmesh,ymesh,H.T,lw=.9,cmap='winter')#,zorder=90)
ax.scatter(x,y,s=3)
#ax.set_xlim(-20,5)
#ax.set_ylim(-20,5)

# Identify points within contours
#p = clevels.collections[0].get_paths()
#inside = np.full_like(x,False,dtype=bool)
#for level in p:
#    inside |= level.contains_points(zip(*(x,y)))

#ax.plot(x[~inside],y[~inside],'kx')
#plt.show(block=False)
# -

density_grid

features.shape, zd.shape

# + jupyter={"outputs_hidden": true}
xy = np.vstack([features[:,0], features[:,1]])
zd = gaussian_kde(xy)(xy)
plt.scatter(features[:,0], features[:,1],c=zd)


# +
# Make the base corner plot
figure = corner.corner(features[:,:2], quantiles=[0.16, 0.84], show_titles=False, color ='crimson')
corner.corner(samples2, fig=fig)
ndim=2
# Extract the axes
axes = np.array(figure.axes).reshape((ndim, ndim))

        
for a in axes[np.triu_indices(ndim)]:
    a.remove()

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Assuming 'features' is your data array with shape (n_samples, 2)

# Calculate the density estimate
xy = np.vstack([features[:,0], features[:,1]])
density_estimation = gaussian_kde(xy)

# Define grid for plotting density lines

xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
density_grid = density_estimation(xy_grid).reshape(x_grid.shape)

# Plot contour lines representing density
plt.contour(x_grid, y_grid, density_grid, colors='black')

# Optionally, you can add a scatter plot on top of the density lines for better visualization
#plt.scatter(features[:,0], features[:,1], color='blue', alpha=0.5)

# Set labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Density Lines Plot')

# Show plot
plt.show()

# -





corner_plot = corner.corner(Arinyo_preds, 
              labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$','$a_v$','$b_v$','$k_p$','$q_2$'], 
              truths=Arinyo_coeffs_central[test_snap],
              truth_color='crimson')

import corner
figure = corner.corner(features, quantiles=[0.16, 0.5, 0.84], show_titles=False)
axes = np.array(fig.axes).reshape((ndim, ndim))
for a in axes[np.triu_indices(ndim)]:
    a.remove()



# +
# My data
x = features[:,0]
y = features[:,1]

# Peform the kernel density estimate
k = stats.gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[-5:5,-5:5]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))



fig = plt.figure()
ax = fig.gca()


CS = ax.contour(xi, yi, zi.reshape(xi.shape), colors='crimson')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.show()
# -


