import numpy as np
import pandas as pd
from astropy.io import fits
import os
from astropy.table import Table
from scipy.spatial import KDTree

import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


class archive():
    def __init__(self, path, aperture=2, drop_stars=True, clean_photometry=True, convert_colors=True, extinction_corr=True, only_zspec=True, only_zspec_test=True, flags_kept=[3,3.1,3.4,3.5,4]):
        
        self.aperture = aperture
        self.flags_kept=flags_kept
        
        
        
        filename_calib='euclid_cosmos_DC2_S1_v2.1_calib_clean.fits'
        filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'
        
        hdu_list = fits.open(os.path.join(path,filename_calib))
        cat = Table(hdu_list[1].data).to_pandas()
        
        hdu_list = fits.open(os.path.join(path,filename_valid))
        cat_test = Table(hdu_list[1].data).to_pandas()
        
                
        if drop_stars==True:
            cat = cat[cat.mu_class_L07==1]
            cat_test = cat_test[cat_test.mu_class_L07==1]

        if clean_photometry==True:
            cat = self._clean_photometry(cat)
            cat_test = self._clean_photometry(cat_test)
            
        
        cat = self._set_combiend_target(cat)
            
                    
        self._set_training_data(cat, only_zspec=only_zspec, extinction_corr=extinction_corr, convert_colors=convert_colors)
        self._set_testing_data(cat_test, only_zspec=only_zspec_test, extinction_corr=extinction_corr, convert_colors=convert_colors)
        
            
    def _extract_fluxes(self,catalogue):
        columns_f = [f'FLUX_{x}_{self.aperture}' for x in ['G','R','I','Z','Y','J','H']]
        columns_ferr = [f'FLUXERR_{x}_{self.aperture}' for x in ['G','R','I','Z','Y','J','H']]

        f = catalogue[columns_f].values
        ferr = catalogue[columns_ferr].values
        return f, ferr
    
    def _to_colors(self, flux, fluxerr):
        """ Convert fluxes to colors"""
        color = flux[:,:-1] / flux[:,1:]
        color_err = fluxerr[:,:-1]**2 / flux[:,1:]**2 + flux[:,:-1]**2 / flux[:,1:]**4 * fluxerr[:,:-1]**2
        return color,color_err
    
    def _set_combiend_target(self, catalogue):
        catalogue['target_z'] = catalogue.apply(lambda row: row['z_spec_S15'] if row['z_spec_S15'] > 0 else row['photo_z_L15'], axis=1)
        
        return catalogue

    
    def _clean_photometry(self,catalogue):
        """ Drops all object with FLAG_PHOT!=0"""
        catalogue = catalogue[catalogue['FLAG_PHOT']==0]
        
        return catalogue
    
    def _correct_extinction(self,catalogue, f):
        """Corrects for extinction"""
        ext_correction_cols =  [f'EB_V_corr_FLUX_{x}' for x in ['G','R','I','Z','Y','J','H']]
        ext_correction = catalogue[ext_correction_cols].values
        
        f = f * ext_correction
        return f
    
    def _take_only_zspec(self,catalogue,cat_flag=None):
        """Selects only galaxies with spectroscopic redshift"""
        if cat_flag=='Calib':
            catalogue = catalogue[catalogue.z_spec_S15>0]
        elif cat_flag=='Valid':
            catalogue = catalogue[catalogue.z_spec_S15>0]
        return catalogue
    
    def _take_zspec_and_photoz(self,catalogue,cat_flag=None):
        """Selects only galaxies with spectroscopic redshift"""
        if cat_flag=='Calib':
            catalogue = catalogue[catalogue.target_z>0]
        elif cat_flag=='Valid':
            catalogue = catalogue[catalogue.z_spec_S15>0]
        return catalogue

    def _clean_zspec_sample(self,catalogue ,flags_kept=[3,3.1,3.4,3.5,4]):
        #[ 2.5,  3.5,  4. ,  1.5,  1.1, 13.5,  9. ,  3. ,  2.1,  9.5,  3.1,
        #1. ,  9.1,  2. ,  9.3,  1.4,  3.4, 11.5,  2.4, 13. , 14. , 12.1,
        #12.5, 13.1,  9.4, 11.1]
        
        catalogue = catalogue[catalogue.Q_f_S15.isin(flags_kept)]

        return catalogue
        


    def _match_gold_sample(self,catalogue_valid, catalogue_gold, max_distance_arcsec=2):
        max_distance_deg = max_distance_arcsec / 3600.0 

        gold_sample_radec = np.c_[catalogue_gold.RIGHT_ASCENSION,catalogue_gold.DECLINATION]
        valid_sample_radec = np.c_[catalogue_valid['RA'],catalogue_valid['DEC']]

        kdtree = KDTree(gold_sample_radec)
        distances, indices = kdtree.query(valid_sample_radec, k=1)

        specz_match_gold = catalogue_gold.FINAL_SPEC_Z.values[indices]

        zs = [specz_match_gold[i] if distance < max_distance_deg else -99 for i, distance in enumerate(distances)]

        catalogue_valid['z_spec_gold'] = zs

        return catalogue_valid

    
    def _set_training_data(self,catalogue, only_zspec=True, extinction_corr=True, convert_colors=True):
        
        
        
        if only_zspec:
            catalogue = self._take_only_zspec(catalogue, cat_flag='Calib')
            catalogue = self._clean_zspec_sample(catalogue, flags_kept=self.flags_kept)
        else:
            catalogue = self._take_zspec_and_photoz(catalogue, cat_flag='Calib')
            
            
        self.cat_train=catalogue
        f, ferr = self._extract_fluxes(catalogue)
        
        
        if extinction_corr==True:
            f = self._correct_extinction(catalogue,f)
                    
        if convert_colors==True:
            col, colerr = self._to_colors(f, ferr)
                        
            self.phot_train = col
            self.photerr_train = colerr
        else:
            self.phot_train = f
            self.photerr_train = ferr  
            
        if only_zspec==True:
            self.target_z_train = catalogue['z_spec_S15'].values
        else:
            self.target_z_train = catalogue['target_z'].values
            
        self.VIS_mag_train = catalogue['MAG_VIS'].values
        
    def _set_testing_data(self,catalogue, only_zspec=True, extinction_corr=True, convert_colors=True):
 
        if only_zspec:
            catalogue = self._take_only_zspec(catalogue, cat_flag='Valid')
            catalogue = self._clean_zspec_sample(catalogue)
            
        self.cat_test=catalogue
            
        f, ferr = self._extract_fluxes(catalogue)
                
        
        if extinction_corr==True:
            f = self._correct_extinction(catalogue,f)
            
        if convert_colors==True:
            col, colerr = self._to_colors(f, ferr)
            self.phot_test = col
            self.photerr_test = colerr
        else:
            self.phot_test = f
            self.photerr_test = ferr  
            
        self.target_z_test = catalogue['z_spec_S15'].values
        self.VIS_mag_test = catalogue['MAG_VIS'].values
            
        
    def get_training_data(self):
        return self.phot_train, self.photerr_train, self.target_z_train, self.VIS_mag_train

    def get_testing_data(self):
        return self.phot_test, self.photerr_test, self.target_z_test, self.VIS_mag_test

    def get_VIS_mag(self, catalogue):
        return catalogue[['MAG_VIS']].values
    
    def plot_zdistribution(self, plot_test=False, bins=50):
        _,_,specz = photoz_archive.get_training_data()
        plt.hist(specz, bins = bins, hisstype='step', color='navy', label=r'Training sample')

        if plot_test:
            _,_,specz_test = photoz_archive.get_training_data()
            plt.hist(specz, bins = bins, hisstype='step', color='goldenrod', label=r'Test sample',ls='--')

            
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.xlabel(r'Redshift', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        
        plt.show()