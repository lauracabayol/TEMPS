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
    def __init__(self, path, aperture=2, drop_stars=True, clean_photometry=True, convert_colors=True, extinction_corr=True, only_zspec=True, reliable_zspec=True):
        
        self.aperture = aperture
        
        self.weight_dict={(-99,0.99):0,
             (1,1.99):0.5,
             (2,2.99):0.75,
             (3,4):1,
             (9,9.99):0.25,
             (10,10.99):0,
             (11,11.99):0.5,
             (12,12.99):0.75,
             (13,14):1,
             (14.01,40):0
            }
        
        filename_calib='euclid_cosmos_DC2_S1_v2.1_calib_clean.fits'
        filename_valid='euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'
        filename_gold='Export_Gold_2023_07_03.csv'
        
        hdu_list = fits.open(os.path.join(path,filename_calib))
        cat = Table(hdu_list[1].data).to_pandas()
        
        hdu_list = fits.open(os.path.join(path,filename_valid))
        cat_test = Table(hdu_list[1].data).to_pandas()
        
        gold_sample = pd.read_csv(os.path.join(path,filename_gold))
        
        #cat_test = self._match_gold_sample(cat_test,gold_sample)        
        
        if drop_stars==True:
            cat = cat[cat.mu_class_L07==1]

        if clean_photometry==True:
            cat = self._clean_photometry(cat)
            cat_test = self._clean_photometry(cat_test)
            
        self._get_loss_weights(cat)
        
        cat = cat[cat.w_Q_f_S15>0]
            
        self._set_training_data(cat, only_zspec=only_zspec, reliable_zspec=reliable_zspec, extinction_corr=extinction_corr, convert_colors=convert_colors)
        
        
        self._set_testing_data(cat_test, only_zspec=only_zspec, reliable_zspec='Total', extinction_corr=extinction_corr, convert_colors=convert_colors)
        
        self._get_loss_weights(cat)
        
        #self.cat_test=cat_test
        #self.cat_train=cat
            
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

    def _clean_zspec_sample(self,catalogue ,kind=None):
        if kind==None:
            return catalogue
        elif kind=='Total':
            return catalogue[catalogue['reliable_S15']>0]
        elif kind=='Partial':
            return catalogue[(catalogue['w_Q_f_S15']>0.5)]
        
    def _map_weight(self,Qz):
        for key, value in self.weight_dict.items():
            if key[0] <= Qz <= key[1]:
                return value
    
    def _get_loss_weights(self,catalogue):
        catalogue['w_Q_f_S15'] = catalogue['Q_f_S15'].apply(self._map_weight)

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

    
    def _set_training_data(self,catalogue, only_zspec=True, reliable_zspec=True, extinction_corr=True, convert_colors=True):
        
        if only_zspec:
            catalogue = self._take_only_zspec(catalogue, cat_flag='Calib')
            catalogue = self._clean_zspec_sample(catalogue, kind=reliable_zspec)
            
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
            
        self.target_z_train = catalogue['z_spec_S15'].values
        self.target_qz_train = catalogue['w_Q_f_S15'].values
        
    def _set_testing_data(self,catalogue, only_zspec=True, reliable_zspec=True, extinction_corr=True, convert_colors=True):
 
        if only_zspec:
            catalogue = self._take_only_zspec(catalogue, cat_flag='Valid')
            catalogue = self._clean_zspec_sample(catalogue, kind=reliable_zspec)
            
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
            
        
    def get_training_data(self):
        return self.phot_train, self.photerr_train, self.target_z_train, self.target_qz_train

    def get_testing_data(self):
        return self.phot_test, self.photerr_test, self.target_z_test

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