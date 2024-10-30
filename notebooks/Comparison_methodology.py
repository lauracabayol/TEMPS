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

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.table import Table

from temps.utils import nmad
from scipy import stats
from pathlib import Path
# -

#define here the directory containing the photometric catalogues
parent_dir = '/data/astro/scratch/lcabayol/EUCLID/DAz/DC2_results_to_share/'


# +
# List of FITS files to be processed
fits_files = [
    'GDE_RF_full.fits',
    'GDE_PHOSPHOROS_V2_full.fits',
    'OIL_LEPHARE_full.fits',
    'JDV_DNF_A_full.fits',
    'JSP_FRANKENZ_full.fits',
    'MBR_METAPHOR_full.fits',
    'GDE_ADABOOST_full.fits',
    'CSC_GPZ_best_full.fits',
    'SFO_CPZ_full.fits',
    'AAL_NNPZ_V3_full.fits'
]

# Corresponding redshift column names
redshift_columns = [
    'REDSHIFT_RF',
    'REDSHIFT_PHOSPHOROS',
    'REDSHIFT_LEPHARE',
    'REDSHIFT_DNF',
    'REDSHIFT_FRANKENZ',
    'REDSHIFT_METAPHOR',
    'REDSHIFT_ADABOOST',
    'REDSHIFT_GPZ',
    'REDSHIFT_CPZ',
    'REDSHIFT_NNPZ'
]

# Initialize an empty DataFrame for merging
merged_df = pd.DataFrame()

# Process each FITS file
for fits_file, redshift_col in zip(fits_files, redshift_columns):
    print(fits_file)
    # Open the FITS file
    hdu_list = fits.open(os.path.join(parent_dir,fits_file))
    df = Table(hdu_list[1].data).to_pandas()
    df = df[df.REDSHIFT!=0]
    df = df[['ID', 'VIS','SPECZ', 'REDSHIFT']].rename(columns={'REDSHIFT': redshift_col})
    # Merge with the main DataFrame
    if merged_df.empty:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on=['ID', 'VIS', 'SPECZ'], how='outer')


# -

# ## OPEN DATA

# +
modules_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(modules_dir/filename_valid)
cat_full = Table(hdu_list[1].data).to_pandas()

cat_full = cat_full[['ID','z_spec_S15','reliable_S15','mu_class_L07']]

merged_df['reliable_S15'] = cat_full.reliable_S15
merged_df['z_spec_S15'] = cat_full.z_spec_S15
merged_df['mu_class_L07'] = cat_full.mu_class_L07
merged_df['ID_catfull'] = cat_full.ID
# -

merged_df_specz  = merged_df[(merged_df.z_spec_S15>0)&(merged_df.SPECZ>0)&(merged_df.reliable_S15==1)&(merged_df.mu_class_L07==1)&(merged_df.VIS!=np.inf)]

# ## ONLY SPECZ SAMPLE

scatter, outliers =[],[]
for im, method in enumerate(redshift_columns):
    print(method)
    df_method = merged_df_specz.dropna(subset=method)
    zerr = (df_method.SPECZ - df_method[method] ) / (1 + df_method.SPECZ)
    print(len(zerr[np.abs(zerr)>0.15]) /len(zerr))
    scatter.append(nmad(zerr))
    outliers.append(len(zerr[np.abs(zerr)>0.15]) / len(df_method))
    

# +
labs = [
    'RF',
    'PHOSPHOROS',
    'LEPHARE',
    'DNF',
    'FRANKENZ',
    'METAPHOR',
    'ADABOOST',
    'GPZ',
    'CPZ',
    'NNPZ',
]

# Colors from colormap
cmap = plt.get_cmap('tab20')
colors = [cmap(i / len(labs)) for i in range(len(labs))]

# Plotting
plt.figure(figsize=(10, 6))
for i in range(len(labs)):
    plt.scatter(outliers[i]*100, scatter[i], color=colors[i], label=labs[i], marker = '^')

# Adding legend
plt.legend(fontsize=12)
plt.ylabel(r'NMAD $[\Delta z]$', fontsize=14)
plt.xlabel('Outlier fraction [%]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(5,35)
plt.ylim(0,0.14)

# Display plot
plt.show()
# -

# ### ADD TEMPS PREDICTIONS

import torch
from temps.archive import Archive 
from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule

# +
data_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(data_dir/filename_valid)
cat_phot = Table(hdu_list[1].data).to_pandas()
# -

cat_phot = cat_phot[cat_phot.ID.isin(merged_df_specz.ID_catfull)]

# +

#define here the directory containing the photometric catalogues
parent_dir_cats = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
filename_calib = 'euclid_cosmos_DC2_S1_v2.1_calib_clean.fits'
filename_valid = 'euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'
# -

photoz_archive = Archive(path_calib = parent_dir_cats/filename_calib, 
                         path_valid = parent_dir_cats/filename_valid,
                         only_zspec=False)
f = photoz_archive._extract_fluxes(catalogue= cat_phot)
col = photoz_archive._to_colors(f)
ID = cat_phot.ID

# +
modules_dir = Path('/nfs/pic.es/user/l/lcabayol/EUCLID/TEMPS/data/models')

nn_features = EncoderPhotometry()
nn_features.load_state_dict(torch.load(modules_dir / f'modelF_DA.pt',map_location=torch.device('cpu')))
nn_z = MeasureZ(num_gauss=6)
nn_z.load_state_dict(torch.load(modules_dir / f'modelZ_DA.pt', map_location=torch.device('cpu')))

temps_module = TempsModule(nn_features, nn_z)

z, pz, odds = temps_module.get_pz(input_data=torch.Tensor(col), 
                            return_pz=True)
df = pd.DataFrame(np.c_[ID, z], 
                  columns=['ID','TEMPS'])

df = df.dropna()
# -

merged_df_specz= merged_df_specz.merge(df, left_on='ID_catfull', right_on='ID')

# Corresponding redshift column names
redshift_columns = redshift_columns  + ['TEMPS']

scatter, outliers =[],[]
for im, method in enumerate(redshift_columns):
    print(method)
    df_method = merged_df_specz.dropna(subset=method)
    zerr = (df_method.SPECZ - df_method[method] ) / (1 + df_method.SPECZ)
    print(len(zerr[np.abs(zerr)>0.15]) /len(zerr))
    scatter.append(nmad(zerr))
    outliers.append(len(zerr[np.abs(zerr)>0.15]) / len(df_method))
    

# +
labs = [
    'RF',
    'PHOSPHOROS',
    'LEPHARE',
    'DNF',
    'FRANKENZ',
    'METAPHOR',
    'ADABOOST',
    'GPZ',
    'CPZ',
    'NNPZ',
    'TEMPS'
]

# Colors from colormap
cmap = plt.get_cmap('tab20')
colors = [cmap(i / len(labs)) for i in range(len(labs))]

# Plotting
plt.figure(figsize=(10, 6))
for i in range(len(labs)):
    plt.scatter(outliers[i]*100, scatter[i], color=colors[i], label=labs[i], marker = '^')

# Adding legend
plt.legend(fontsize=12)
plt.ylabel(r'NMAD $[\Delta z]$', fontsize=14)
plt.xlabel('Outlier fraction [%]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(5,35)
plt.ylim(0,0.14)

# Display plot
plt.show()
# -

# ## ANOTHER SELECTION

# +
# List of FITS files to be processed
fits_files = [
    'GDE_RF_full.fits',
    'GDE_PHOSPHOROS_V2_full.fits',
    'OIL_LEPHARE_full.fits',
    'JDV_DNF_A_full.fits',
    'JSP_FRANKENZ_full.fits',
    'MBR_METAPHOR_full.fits',
    'GDE_ADABOOST_full.fits',
    'CSC_GPZ_best_full.fits',
    'SFO_CPZ_full.fits',
    'AAL_NNPZ_V3_full.fits'
]

# Corresponding redshift column names
redshift_columns = [
    'REDSHIFT_RF',
    'REDSHIFT_PHOSPHOROS',
    'REDSHIFT_LEPHARE',
    'REDSHIFT_DNF',
    'REDSHIFT_FRANKENZ',
    'REDSHIFT_METAPHOR',
    'REDSHIFT_ADABOOST',
    'REDSHIFT_GPZ',
    'REDSHIFT_CPZ',
    'REDSHIFT_NNPZ'
]

use_columns = [
    'USE_RF',
    'USE_PHOSPHOROS',
    'USE_LEPHARE',
    'USE_DNF',
    'USE_FRANKENZ',
    'USE_METAPHOR',
    'USE_ADABOOST',
    'USE_GPZ',
    'USE_CPZ',
    'USE_NNPZ'
]

# Initialize an empty DataFrame for merging
merged_df = pd.DataFrame()

# Process each FITS file
for fits_file, redshift_col,use_col  in zip(fits_files, redshift_columns,use_columns):
    print(fits_file)
    # Open the FITS file
    hdu_list = fits.open(os.path.join(parent_dir,fits_file))
    df = Table(hdu_list[1].data).to_pandas()
    df = df[df.REDSHIFT!=0]
    df = df[['ID', 'VIS', 'SPECZ', 'REDSHIFT', 'L15PHZ', 'USE']].rename(columns={'REDSHIFT': redshift_col, 'USE': use_col})
    # Merge with the main DataFrame
    if merged_df.empty:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on=['ID', 'VIS', 'SPECZ','L15PHZ'], how='outer')


# -

merged_df['comp_z'] = np.where(merged_df['SPECZ'] > 0, merged_df['SPECZ'], merged_df['L15PHZ'])
#merged_df = merged_df[(merged_df.comp_z>0)&(merged_df.comp_z<4)&(merged_df.VIS>23.5)]
merged_df = merged_df[(merged_df.comp_z>0)&(merged_df.comp_z<4)&(merged_df.VIS<25)]

# +
modules_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(modules_dir/filename_valid)
cat_full = Table(hdu_list[1].data).to_pandas()

merged_df['ID_catfull'] = cat_full.ID

# +
data_dir = Path('/data/astro/scratch/lcabayol/insight/data/Euclid_EXT_MER_PHZ_DC2_v1.5')
filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'

hdu_list = fits.open(data_dir/filename_valid)
cat_phot = Table(hdu_list[1].data).to_pandas()
# -

cat_phot = cat_phot[cat_phot.ID.isin(merged_df.ID_catfull)]

photoz_archive = Archive(path_calib = parent_dir_cats/filename_calib, 
                         path_valid = parent_dir_cats/filename_valid,
                         only_zspec=False)
f = photoz_archive._extract_fluxes(catalogue= cat_phot)
col = photoz_archive._to_colors(f)
ID = cat_phot.ID


# +
modules_dir = Path('/nfs/pic.es/user/l/lcabayol/EUCLID/TEMPS/data/models')

nn_features = EncoderPhotometry()
nn_features.load_state_dict(torch.load(modules_dir/f'modelF_DA.pt',map_location=torch.device('cpu')))
nn_z = MeasureZ(num_gauss=6)
nn_z.load_state_dict(torch.load(modules_dir/f'modelZ_DA.pt',map_location=torch.device('cpu')))

temps_module = TempsModule(nn_features, nn_z)

z, pz, odds = temps_module.get_pz(input_data=torch.Tensor(col), 
                            return_pz=True)

nn_features = EncoderPhotometry()
nn_features.load_state_dict(torch.load(modules_dir/f'modelF_z.pt',map_location=torch.device('cpu')))
nn_z = MeasureZ(num_gauss=6)
nn_z.load_state_dict(torch.load(modules_dir/f'modelZ_z.pt',map_location=torch.device('cpu')))

temps_module = TempsModule(nn_features, nn_z)
znoda, pz, odds_noda = temps_module.get_pz(input_data=torch.Tensor(col), 
                            return_pz=True)

nn_features = EncoderPhotometry()
nn_features.load_state_dict(torch.load(modules_dir/f'modelF_L15.pt',map_location=torch.device('cpu')))
nn_z = MeasureZ(num_gauss=6)
nn_z.load_state_dict(torch.load(modules_dir/f'modelZ_L15.pt',map_location=torch.device('cpu')))

temps_module = TempsModule(nn_features, nn_z)
z_L15, pz, odds_L15 = temps_module.get_pz(input_data=torch.Tensor(col), 
                            return_pz=True)

df = pd.DataFrame(np.c_[ID, z, odds, znoda, odds_noda,z_L15, odds_L15], 
                  columns=['ID','TEMPS', 'flag_TEMPS', 'TEMPS_noda', 'flag_TEMPSnoda', 'TEMPS_L15', 'flag_L15'])

df = df.dropna()
# -

percent=0.2
df['USE_TEMPS'] = np.zeros(shape=len(df))
# Calculate the 50th percentile (median) value of 'Flag_temps'
threshold = df['flag_TEMPS'].quantile(percent)
# Set 'USE_TEMPS' to 1 if 'Flag_temps' is in the top 50% (greater than or equal to the threshold)
df['USE_TEMPS'] = np.where(df['flag_TEMPS'] >= threshold, 1, 0)

# +
percent=0.3
df['USE_TEMPS_noda'] = np.zeros(shape=len(df))
# Calculate the 50th percentile (median) value of 'Flag_temps'
threshold = df['flag_TEMPSnoda'].quantile(percent)

# Set 'USE_TEMPS' to 1 if 'Flag_temps' is in the top 50% (greater than or equal to the threshold)
df['USE_TEMPS_noda'] = np.where(df['flag_TEMPSnoda'] >= threshold, 1, 0)

# +
percent=0.3
df['USE_TEMPS_L15'] = np.zeros(shape=len(df))
# Calculate the 50th percentile (median) value of 'Flag_temps'
threshold = df['flag_L15'].quantile(percent)

# Set 'USE_TEMPS' to 1 if 'Flag_temps' is in the top 50% (greater than or equal to the threshold)
df['USE_TEMPS_L15'] = np.where(df['flag_L15'] >= threshold, 1, 0)
# -

merged_df_temps = merged_df.merge(df, left_on='ID_catfull', right_on='ID')

# Corresponding redshift column names
redshift_columns = [
    'REDSHIFT_RF',
    'REDSHIFT_PHOSPHOROS',
    'REDSHIFT_LEPHARE',
    'REDSHIFT_DNF',
    'REDSHIFT_FRANKENZ',
    'REDSHIFT_METAPHOR',
    'REDSHIFT_ADABOOST',
    'REDSHIFT_GPZ',
    'REDSHIFT_CPZ',
    'REDSHIFT_NNPZ'
]

redshift_columns = redshift_columns  + ['TEMPS', 'TEMPS_noda', 'TEMPS_L15']
use_columns = use_columns  + ['USE_TEMPS','USE_TEMPS_noda', 'USE_TEMPS_L15']

merged_df_temps = merged_df_temps[merged_df_temps.VIS <25]


scatter, outliers, size =[],[], []
for method, use in(zip(redshift_columns, use_columns)):
    print(method)
    #df_method = merged_df_temps.dropna(subset=method)
    df_method = merged_df_temps[(merged_df_temps.loc[:, method]>0.2)&(merged_df_temps.loc[:, method]<2.6)]
    df_method = df_method[df_method.VIS<24.5]
    norm_size = len(df_method)
    df_method = df_method[df_method.loc[:, use]==1]
    zerr = (df_method.comp_z - df_method[method] ) / (1 + df_method.comp_z)
    scatter.append(nmad(zerr))
    outliers.append(len(zerr[np.abs(zerr)>0.15]) / len(df_method))
    size.append(len(df_method)/norm_size)
    print(nmad(zerr),len(zerr[np.abs(zerr)>0.15]) / len(df_method), len(df_method) /norm_size )
    

scatter_faint, outliers_faint, size_faint =[],[], []
for method, use in(zip(redshift_columns, use_columns)):
    print(method)
    #df_method = merged_df_temps.dropna(subset=method)
    df_method = merged_df_temps[(merged_df_temps.loc[:,'VIS']>23.5)&(merged_df_temps.loc[:,'VIS']<25)]
    #df_method = df_method[df_method.loc[:, use]==1]
    #df_method = merged_df_temps[(merged_df_temps.loc[:,'VIS']>23.5)&(merged_df_temps.loc[:,'VIS']<24.5)]
    zerr = (df_method.comp_z - df_method[method] ) / (1 + df_method.comp_z)
    scatter_faint.append(nmad(zerr))
    outliers_faint.append(len(zerr[np.abs(zerr)>0.15]) / len(df_method))
    size_faint.append(len(df_method))
    print(nmad(zerr),len(zerr[np.abs(zerr)>0.15]) / len(df_method), len(df_method))
    

# +
import matplotlib.pyplot as plt
import numpy as np
from pastamarkers import markers

# Define labels for the models
labs = [
    'RF', 'PHOSPHOROS', 'LEPHARE', 'DNF', 'FRANKENZ', 'METAPHOR', 
    'ADABOOST', 'GPZ', 'CPZ', 'NNPZ', 'TEMPS', 'TEMPS - no DA', 'TEMPS - L15'
]

markers_pasta = [markers.penne, markers.conchiglie, markers.tortellini, markers.creste, markers.spaghetti,  markers.ravioli, markers.tagliatelle, markers.mezzelune,markers.puntine, markers.stelline , 's', 'o',  '^']

labs_faint = [f"{lab}_faint" for lab in labs]  # Labels for the faint data


# Colors from colormap
cmap = plt.get_cmap('tab20')
colors = [cmap(i / len(labs)) for i in range(len(labs))]

# Create subplots with 2 panels stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

# Plotting for the top panel
for i in range(len(labs)):
    if labs[i] == 'TEMPS - no DA' or  labs[i] == 'TEMPS - L15':
        ax1.scatter(np.nan, np.nan, color=colors[i], label=labs[i], marker=markers_pasta[i], s=300)
    elif labs[i]=='CPZ':
        ax1.scatter(outliers[i] * 100, scatter[i], color=colors[i], label=labs[i], marker=markers_pasta[i], s=300)
        ax1.text(outliers[i] * 100 -0.2, scatter[i] + 0.001, f'{int(np.around(size[i] * 100))}', fontsize=12, verticalalignment='bottom')
        
    elif labs[i]=='ADABOOST':
        ax1.scatter(outliers[i] * 100, scatter[i], color=colors[i], label=labs[i], marker=markers_pasta[i], s=300)
        ax1.text(outliers[i] * 100 - 0.5, scatter[i] - 0.004, f'{int(np.around(size[i] * 100))}', fontsize=12, verticalalignment='bottom')
                
    else:
        ax1.scatter(outliers[i] * 100, scatter[i], color=colors[i], label=labs[i], marker=markers_pasta[i], s=300)
        ax1.text(outliers[i] * 100 - 0.5, scatter[i] + 0.001, f'{int(np.around(size[i] * 100))}', fontsize=12, verticalalignment='bottom')

# Customizations for the top plot
ax1.set_ylabel(r'NMAD $[\Delta z]$', fontsize=24)
ax1.legend(fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=20)

# Plotting for the bottom panel (faint data)
for i in range(len(labs)):
    ax2.scatter(outliers_faint[i] * 100, scatter_faint[i], color=colors[i], label=labs[i], marker=markers_pasta[i], s=300)

# Customizations for the bottom plot
ax2.set_ylabel(r'NMAD $[\Delta z]$', fontsize=24)
ax2.set_xlabel('Outlier fraction [%]', fontsize=24)
ax2.tick_params(axis='both', which='major', labelsize=20)

# Display the plot
plt.tight_layout()
#plt.savefig('Comparison_paper.pdf', bbox_inches='tight')
plt.show()

