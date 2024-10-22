import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from matplotlib import rcParams
from pathlib import Path  
from loguru import logger


rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

class Archive:
    def __init__(self, path, 
                 aperture=2, 
                 drop_stars=True, 
                 clean_photometry=True, 
                 convert_colors=True, 
                 extinction_corr=True, 
                 only_zspec=True, 
                 all_apertures=False,
                 target_test='specz', flags_kept=[3, 3.1, 3.4, 3.5, 4]):

        
        logger.info("Starting archive")
        self.aperture = aperture
        self.all_apertures = all_apertures
        self.flags_kept = flags_kept
        
        filename_calib = 'euclid_cosmos_DC2_S1_v2.1_calib_clean.fits'
        filename_valid = 'euclid_cosmos_DC2_S1_v2.1_valid_matched.fits'
        
        # Use Path for file handling
        path_calib = Path(path) / filename_calib
        path_valid = Path(path) / filename_valid
        
        # Open the calibration FITS file
        with fits.open(path_calib) as hdu_list:
            cat = Table(hdu_list[1].data).to_pandas()
            cat = cat[(cat['z_spec_S15'] > 0) | (cat['photo_z_L15'] > 0)]
        
        # Open the validation FITS file
        with fits.open(path_valid) as hdu_list:
            cat_test = Table(hdu_list[1].data).to_pandas()
        
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
        if self.all_apertures:
            columns_f = [f'FLUX_{x}_{a}' for a in [1,2,3] for x in ['G','R','I','Z','Y','J','H']] 
            columns_ferr = [f'FLUXERR_{x}_{a}' for a in [1,2,3] for x in ['G','R','I','Z','Y','J','H'] ]
        else:
            columns_f = [f'FLUX_{x}_{self.aperture}' for x in ['G','R','I','Z','Y','J','H']]
            columns_ferr = [f'FLUXERR_{x}_{self.aperture}' for x in ['G','R','I','Z','Y','J','H']]

        f = catalogue[columns_f].values
        ferr = catalogue[columns_ferr].values
        return f, ferr
    
    def _extract_magnitudes(self,catalogue):
        if self.all_apertures:
            columns_m = [f'MAG_{x}_{a}' for a in [1,2,3] for x in ['G','R','I','Z','Y','J','H']] 
            columns_merr = [f'MAGERR_{x}_{a}' for a in [1,2,3] for x in ['G','R','I','Z','Y','J','H'] ]
        else:
            columns_m = [f'MAG_{x}_{self.aperture}' for x in ['G','R','I','Z','Y','J','H']]
            columns_merr = [f'MAGERR_{x}_{self.aperture}' for x in ['G','R','I','Z','Y','J','H']]

        m = catalogue[columns_m].values
        merr = catalogue[columns_merr].values
        return m, merr
    
    def _to_colors(self, flux, fluxerr):
        """ Convert fluxes to colors"""
        
        if self.all_apertures:

            for a in range(3):
                lim1 = 7*a
                lim2 = 7*(a+1)
                c = flux[:,lim1:(lim2-1)] / flux[:,(lim1+1):lim2]
                cerr = np.sqrt((fluxerr[:,lim1:(lim2-1)]/ flux[:,(lim1+1):lim2])**2 + (flux[:,lim1:(lim2-1)] / flux[:,(lim1+1):lim2]**2)**2 * fluxerr[:,(lim1+1):lim2]**2)
                
                if a==0:
                    color = c
                    color_err = cerr
                else:
                    color = np.concatenate((color,c),axis=1)
                    color_err = np.concatenate((color_err,cerr),axis=1)
            
        else:
            color = flux[:,:-1] / flux[:,1:]

            color_err = np.sqrt((fluxerr[:,:-1]/ flux[:,1:])**2 + (flux[:,:-1] / flux[:,1:]**2)**2 * fluxerr[:,1:]**2)
        return color,color_err
    
    def _set_combiend_target(self, catalogue):
        catalogue['target_z'] = catalogue.apply(lambda row: row['z_spec_S15'] 
                                                if row['z_spec_S15'] > 0 
                                                else row['photo_z_L15'], axis=1)
        
        return catalogue

    
    def _clean_photometry(self,catalogue):
        """ Drops all object with FLAG_PHOT!=0"""
        catalogue = catalogue[catalogue['FLAG_PHOT']==0]
        
        return catalogue
    
    def _correct_extinction(self,catalogue, f, return_ext_corr=False):
        """Corrects for extinction"""
        ext_correction_cols =  [f'EB_V_corr_FLUX_{x}' for x in ['G','R','I','Z','Y','J','H']]
        if self.all_apertures:
            ext_correction = catalogue[ext_correction_cols].values
            ext_correction = np.concatenate((ext_correction,ext_correction,ext_correction),axis=1)
        else:
            ext_correction = catalogue[ext_correction_cols].values
        
        f = f * ext_correction
        if return_ext_corr:
            return f, ext_correction
        else:
            return f
    
    def _select_only_zspec(self,catalogue,cat_flag=None):
        """Selects only galaxies with spectroscopic redshift"""
        if cat_flag=='Calib':
            catalogue = catalogue[catalogue.z_spec_S15>0]
        elif cat_flag=='Valid':
            catalogue = catalogue[catalogue.z_spec_S15>0]
        return catalogue
    
    def _exclude_only_zspec(self,catalogue):
        """Selects only galaxies without spectroscopic redshift"""
        catalogue = catalogue[(catalogue.z_spec_S15<0)&(catalogue.photo_z_L15>0)&(catalogue.photo_z_L15<4)]
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

    
    def _set_training_data(self,catalogue, catalogue_da, only_zspec=True, extinction_corr=True, convert_colors=True):
        
        cat_da = self._exclude_only_zspec(catalogue_da)  
        target_z_train_DA = cat_da['photo_z_L15'].values
  
        
        if only_zspec:
            logger.info("Selecting only galaxies with spectroscopic redshift")
            catalogue = self._select_only_zspec(catalogue, cat_flag='Calib')
            catalogue = self._clean_zspec_sample(catalogue, flags_kept=self.flags_kept)
        else:
            logger.info("Selecting galaxies with spectroscopic redshift and high-precision photo-z")
            catalogue = self._take_zspec_and_photoz(catalogue, cat_flag='Calib')
            
            
        self.cat_train=catalogue
        f, ferr = self._extract_fluxes(catalogue)
        
        f_DA, ferr_DA = self._extract_fluxes(cat_da)
        idx = np.random.randint(0, len(f_DA), len(f))
        f_DA, ferr_DA = f_DA[idx], ferr_DA[idx] 
        target_z_train_DA = target_z_train_DA[idx]
        self.target_z_train_DA = target_z_train_DA
        
        
        if extinction_corr==True:
            logger.info("Correcting MW extinction")
            f = self._correct_extinction(catalogue,f)
                                
        if convert_colors==True:
            logger.info("Converting to colors")
            col, colerr = self._to_colors(f, ferr)
            col_DA, colerr_DA = self._to_colors(f_DA, ferr_DA)
            
            self.phot_train = col
            self.photerr_train = colerr
            self.phot_train_DA = col_DA
            self.photerr_train_DA = colerr_DA
        else:
            self.phot_train = f
            self.photerr_train = ferr  
            self.phot_train_DA = f_DA
            self.photerr_train_DA = ferr_DA
            
        if only_zspec==True:
            self.target_z_train = catalogue['z_spec_S15'].values
        else:
            self.target_z_train = catalogue['target_z'].values
            
        self.VIS_mag_train = catalogue['MAG_VIS'].values
        
    def _set_testing_data(self,catalogue, target='specz', extinction_corr=True, convert_colors=True):
        
        if target=='specz':
            catalogue = self._select_only_zspec(catalogue, cat_flag='Valid')
            catalogue = self._clean_zspec_sample(catalogue)
            self.target_z_test = catalogue['z_spec_S15'].values
            
        elif target=='L15':
            catalogue = self._select_L15_sample(catalogue)
            self.target_z_test = catalogue['target_z'].values
                    
                        
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
            
        
        self.VIS_mag_test = catalogue['MAG_VIS'].values
            
        
    def get_training_data(self):
        return self.phot_train, self.photerr_train, self.target_z_train, self.VIS_mag_train, self.phot_train_DA, self.photerr_train_DA, self.target_z_train_DA

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