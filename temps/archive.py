import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from matplotlib import rcParams
from pathlib import Path  
from loguru import logger


rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

class Archive:
    def __init__(self, 
                 path_calib, 
                 path_valid=None,
                 drop_stars=True, 
                 clean_photometry=True, 
                 convert_colors=True, 
                 extinction_corr=True, 
                 only_zspec=True, 
                 columns_photometry = ['FLUX_G_2','FLUX_R_2','FLUX_I_2','FLUX_Z_2','FLUX_Y_2','FLUX_J_2','FLUX_H_2'],
                 columns_ebv = ['EB_V_corr_FLUX_G','EB_V_corr_FLUX_R','EB_V_corr_FLUX_I','EB_V_corr_FLUX_Z','EB_V_corr_FLUX_Y','EB_V_corr_FLUX_J','EB_V_corr_FLUX_H'],
                 photoz_name="photo_z_L15",
                 specz_name="z_spec_S15",
                 target_test='specz', 
                 flags_kept=[3, 3.1, 3.4, 3.5, 4]):

        logger.info("Starting archive")
        self.flags_kept = flags_kept
        self.columns_photometry=columns_photometry
        self.columns_ebv=columns_ebv    

        
        if path_calib.suffix == ".fits":
            with fits.open(path_calib) as hdu_list:
                cat = Table(hdu_list[1].data).to_pandas()
            if path_valid != None:
                with fits.open(path_valid) as hdu_list:
                    cat_test = Table(hdu_list[1].data).to_pandas()    
                
        elif path_calib.suffix == ".csv":
            cat = pd.read_csv(path_calib)
            if path_valid != None:
                cat_test = pd.read_csv(path_valid)
        else:
            raise ValueError("Unsupported file format. Please provide a .fits or .csv file.")

        cat = cat.rename(columns ={f"{specz_name}":"specz",
                         f"{photoz_name}":"photo_z"})
        cat_test = cat_test.rename(columns ={f"{specz_name}":"specz",
                         f"{photoz_name}":"photo_z"})
        
        cat = cat[(cat['specz'] > 0) | (cat['photo_z'] > 0)]
        
        # Store the catalogs for later use
        self.cat = cat
        self.cat_test = cat_test
        
                
        if drop_stars==True:
            logger.info("dropping stars...")
            cat = cat[cat.mu_class_L07==1]
            cat_test = cat_test[cat_test.mu_class_L07==1]

        if clean_photometry==True:
            logger.info("cleaning stars...")
            cat = self._clean_photometry(cat)
            cat_test = self._clean_photometry(cat_test)
            
        
        cat = self._set_combiend_target(cat)
        cat_test = self._set_combiend_target(cat_test)
        
        
        
        cat = cat[cat.MAG_VIS<25]
        cat_test = cat_test[cat_test.MAG_VIS<25]
        
        cat = cat[cat.target_z<5]
        cat_test = cat_test[cat_test.target_z<5]
        
                    
                    
        self._set_training_data(cat, 
                                cat_test,
                                only_zspec=only_zspec, 
                                extinction_corr=extinction_corr, 
                                convert_colors=convert_colors)
        self._set_testing_data(cat_test, 
                               target=target_test, 
                               extinction_corr=extinction_corr, 
                               convert_colors=convert_colors)
        
            
    def _extract_fluxes(self,catalogue):
        f = catalogue[self.columns_photometry].values
        return f

    def _to_colors(self, flux):
        """ Convert fluxes to colors"""
        color = flux[:,:-1] / flux[:,1:]
        return color
    
    def _set_combiend_target(self, catalogue):
        catalogue['target_z'] = catalogue.apply(lambda row: row['specz'] 
                                                if row['specz'] > 0 
                                                else row['photo_z'], axis=1)
        
        return catalogue

    
    def _clean_photometry(self,catalogue):
        """ Drops all object with FLAG_PHOT!=0"""
        catalogue = catalogue[catalogue['FLAG_PHOT']==0]
        
        return catalogue
    
    def _correct_extinction(self,catalogue, f, return_ext_corr=False):
        """Corrects for extinction"""
        ext_correction = catalogue[self.columns_ebv].values
        f = f * ext_correction
        if return_ext_corr:
            return f, ext_correction
        else:
            return f
    
    def _select_only_zspec(self,catalogue,cat_flag=None):
        """Selects only galaxies with spectroscopic redshift"""
        if cat_flag=='Calib':
            catalogue = catalogue[catalogue.specz>0]
        elif cat_flag=='Valid':
            catalogue = catalogue[catalogue.specz>0]
        return catalogue
    
    def _exclude_only_zspec(self,catalogue):
        """Selects only galaxies without spectroscopic redshift"""
        catalogue = catalogue[(catalogue.specz<0)&(catalogue.photo_z>0)&(catalogue.photo_z<4)]
        return catalogue
    
    def _select_L15_sample(self,catalogue):
        """Selects only galaxies withoutidx spectroscopic redshift"""
        catalogue = catalogue[(catalogue.target_z>0)]
        catalogue = catalogue[(catalogue.target_z<4)]


        return catalogue
    
    def _take_zspec_and_photoz(self,catalogue,cat_flag=None):
        """Selects only galaxies with spectroscopic redshift"""
        if cat_flag=='Calib':
            catalogue = catalogue[catalogue.target_z>0]
        elif cat_flag=='Valid':
            catalogue = catalogue[catalogue.specz>0]
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

    
    def _set_training_data(self,catalogue, catalogue_da, only_zspec=True, extinction_corr=True, convert_colors=True):
        
        cat_da = self._exclude_only_zspec(catalogue_da)  
        target_z_train_DA = cat_da['photo_z'].values
  
        
        if only_zspec:
            logger.info("Selecting only galaxies with spectroscopic redshift")
            catalogue = self._select_only_zspec(catalogue, cat_flag='Calib')
            catalogue = self._clean_zspec_sample(catalogue, flags_kept=self.flags_kept)
        else:
            logger.info("Selecting galaxies with spectroscopic redshift and high-precision photo-z")
            catalogue = self._take_zspec_and_photoz(catalogue, cat_flag='Calib')
            
            
        self.cat_train=catalogue
        f = self._extract_fluxes(catalogue)
        f_DA = self._extract_fluxes(cat_da)
        idx = np.random.randint(0, len(f_DA), len(f))
        f_DA = f_DA[idx]
        target_z_train_DA = target_z_train_DA[idx]
        self.target_z_train_DA = target_z_train_DA
        
        
        if extinction_corr==True:
            logger.info("Correcting MW extinction")
            f = self._correct_extinction(catalogue,f)
                                
        if convert_colors==True:
            logger.info("Converting to colors")
            col = self._to_colors(f)
            col_DA = self._to_colors(f_DA)
            
            self.phot_train = col
            self.phot_train_DA = col_DA
        else:
            self.phot_train = f
            self.phot_train_DA = f_DA
            
        if only_zspec==True:
            self.target_z_train = catalogue['specz'].values
        else:
            self.target_z_train = catalogue['target_z'].values
            
        self.VIS_mag_train = catalogue['MAG_VIS'].values
        
    def _set_testing_data(self,catalogue, target='specz', extinction_corr=True, convert_colors=True):
        
        if target=='specz':
            catalogue = self._select_only_zspec(catalogue, cat_flag='Valid')
            catalogue = self._clean_zspec_sample(catalogue)
            self.target_z_test = catalogue['specz'].values
            
        elif target=='L15':
            catalogue = self._select_L15_sample(catalogue)
            self.target_z_test = catalogue['target_z'].values
                    
                        
        self.cat_test=catalogue
            
        f = self._extract_fluxes(catalogue)
        
        if extinction_corr==True:
            f = self._correct_extinction(catalogue,f)
            
        if convert_colors==True:
            col = self._to_colors(f)
            self.phot_test = col
        else:
            self.phot_test = f
            
        
        self.VIS_mag_test = catalogue['MAG_VIS'].values
            
        
    def get_training_data(self):
        return self.phot_train, self.target_z_train, self.VIS_mag_train, self.phot_train_DA, self.target_z_train_DA

    def get_testing_data(self):
        return self.phot_test, self.target_z_test, self.VIS_mag_test

    def get_VIS_mag(self, catalogue):
        return catalogue[['MAG_VIS']].values
