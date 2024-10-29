from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple, Union, List

# Set matplotlib configuration
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


@dataclass
class Archive:
    path_calib: Path
    path_valid: Optional[Path] = None
    drop_stars: bool = True
    clean_photometry: bool = True
    convert_colors: bool = True
    extinction_corr: bool = True
    only_zspec: bool = True
    columns_photometry: List[str] = field(
        default_factory=lambda: [
            "FLUX_G_2",
            "FLUX_R_2",
            "FLUX_I_2",
            "FLUX_Z_2",
            "FLUX_Y_2",
            "FLUX_J_2",
            "FLUX_H_2",
        ]
    )
    columns_ebv: List[str] = field(
        default_factory=lambda: [
            "EB_V_corr_FLUX_G",
            "EB_V_corr_FLUX_R",
            "EB_V_corr_FLUX_I",
            "EB_V_corr_FLUX_Z",
            "EB_V_corr_FLUX_Y",
            "EB_V_corr_FLUX_J",
            "EB_V_corr_FLUX_H",
        ]
    )
    photoz_name: str = "photo_z_L15"
    specz_name: str = "z_spec_S15"
    target_test: str = "specz"
    flags_kept: List[float] = field(default_factory=lambda: [3, 3.1, 3.4, 3.5, 4])

    def __post_init__(self):
        logger.info("Starting archive")

        # Load data based on the file format
        if self.path_calib.suffix == ".fits":
            with fits.open(self.path_calib) as hdu_list:
                self.cat = Table(hdu_list[1].data).to_pandas()
            if self.path_valid is not None:
                with fits.open(self.path_valid) as hdu_list:
                    self.cat_test = Table(hdu_list[1].data).to_pandas()

        elif self.path_calib.suffix == ".csv":
            self.cat = pd.read_csv(self.path_calib)
            if self.path_valid is not None:
                self.cat_test = pd.read_csv(self.path_valid)
        else:
            raise ValueError(
                "Unsupported file format. Please provide a .fits or .csv file."
            )

        self.cat = self.cat.rename(
            columns={f"{self.specz_name}": "specz", f"{self.photoz_name}": "photo_z"}
        )
        self.cat_test = self.cat_test.rename(
            columns={f"{self.specz_name}": "specz", f"{self.photoz_name}": "photo_z"}
        )

        self.cat = self.cat[(self.cat["specz"] > 0) | (self.cat["photo_z"] > 0)]

        # Apply operations based on the initialized parameters
        if self.drop_stars:
            logger.info("Dropping stars...")
            self.cat = self.cat[self.cat.mu_class_L07 == 1]
            self.cat_test = self.cat_test[self.cat_test.mu_class_L07 == 1]

        if self.clean_photometry:
            logger.info("Cleaning photometry...")
            self.cat = self._clean_photometry(catalogue=self.cat)
            self.cat_test = self._clean_photometry(catalogue=self.cat_test)

        self.cat = self._set_combined_target(self.cat)
        self.cat_test = self._set_combined_target(self.cat_test)

        # Apply magnitude and redshift cuts
        self.cat = self.cat[self.cat.MAG_VIS < 25]
        self.cat_test = self.cat_test[self.cat_test.MAG_VIS < 25]

        self.cat = self.cat[self.cat.target_z < 5]
        self.cat_test = self.cat_test[self.cat_test.target_z < 5]

        self._set_training_data(
            self.cat,
            self.cat_test,
            only_zspec=self.only_zspec,
            extinction_corr=self.extinction_corr,
            convert_colors=self.convert_colors,
        )
        self._set_testing_data(
            self.cat_test,
            target=self.target_test,
            extinction_corr=self.extinction_corr,
            convert_colors=self.convert_colors,
        )

    def _extract_fluxes(self, catalogue: pd.DataFrame) -> np.ndarray:
        """Extract fluxes from the given catalogue.

        Args:
            catalogue (pd.DataFrame): The input catalogue.

        Returns:
            np.ndarray: An array of fluxes.
        """
        f = catalogue[self.columns_photometry].values
        return f

    @staticmethod
    def _to_colors(flux: np.ndarray) -> np.ndarray:
        """Convert fluxes to colors.

        Args:
            flux (np.ndarray): The input fluxes.

        Returns:
            np.ndarray: An array of colors.
        """
        color = flux[:, :-1] / flux[:, 1:]
        return color

    @staticmethod
    def _set_combined_target(catalogue: pd.DataFrame) -> pd.DataFrame:
        """Set the combined target redshift based on available data.

        Args:
            catalogue (pd.DataFrame): The input catalogue.

        Returns:
            pd.DataFrame: Updated catalogue with the combined target redshift.
        """
        catalogue["target_z"] = catalogue.apply(
            lambda row: row["specz"] if row["specz"] > 0 else row["photo_z"], axis=1
        )
        return catalogue

    @staticmethod
    def _clean_photometry(catalogue: pd.DataFrame) -> pd.DataFrame:
        """Drops all objects with FLAG_PHOT != 0.

        Args:
            catalogue (pd.DataFrame): The input catalogue.

        Returns:
            pd.DataFrame: Cleaned catalogue.
        """
        catalogue = catalogue[catalogue["FLAG_PHOT"] == 0]
        return catalogue

    def _correct_extinction(
        self, catalogue: pd.DataFrame, f: np.ndarray, return_ext_corr: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Corrects for extinction based on the provided catalogue.

        Args:
            catalogue (pd.DataFrame): The input catalogue.
            f (np.ndarray): The flux values to correct.
            return_ext_corr (bool): Whether to return the extinction correction values.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Corrected fluxes, and optionally the extinction corrections.
        """
        ext_correction = catalogue[self.columns_ebv].values
        f = f * ext_correction
        if return_ext_corr:
            return f, ext_correction
        else:
            return f

    @staticmethod
    def _select_only_zspec(
        catalogue: pd.DataFrame, cat_flag: Optional[str] = None
    ) -> pd.DataFrame:
        """Selects only galaxies with spectroscopic redshift.

        Args:
            catalogue (pd.DataFrame): The input catalogue.
            cat_flag (Optional[str]): Indicates the catalogue type ('Calib' or 'Valid').

        Returns:
            pd.DataFrame: Filtered catalogue.
        """
        if cat_flag == "Calib":
            catalogue = catalogue[catalogue.specz > 0]
        elif cat_flag == "Valid":
            catalogue = catalogue[catalogue.specz > 0]
        return catalogue

    @staticmethod
    def take_zspec_and_photoz(
        catalogue: pd.DataFrame, cat_flag: Optional[str] = None
    ) -> pd.DataFrame:
        """Selects only galaxies with spectroscopic redshift"""
        if cat_flag == "Calib":
            catalogue = catalogue[catalogue.target_z > 0]
        elif cat_flag == "Valid":
            catalogue = catalogue[catalogue.specz > 0]
        return catalogue

    @staticmethod
    def exclude_only_zspec(catalogue: pd.DataFrame) -> pd.DataFrame:
        """Selects only galaxies without spectroscopic redshift.

        Args:
            catalogue (pd.DataFrame): The input catalogue.

        Returns:
            pd.DataFrame: Filtered catalogue.
        """
        catalogue = catalogue[
            (catalogue.specz < 0) & (catalogue.photo_z > 0) & (catalogue.photo_z < 4)
        ]
        return catalogue

    @staticmethod
    def _clean_zspec_sample(catalogue, flags_kept=[3, 3.1, 3.4, 3.5, 4]):
        catalogue = catalogue[catalogue.Q_f_S15.isin(flags_kept)]
        return catalogue

    @staticmethod
    def _select_L15_sample(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """Selects only galaxies within a specific redshift range.

        Args:
            catalogue (pd.DataFrame): The input catalogue.

        Returns:
            pd.DataFrame: Filtered catalogue.
        """
        catalogue = catalogue[(catalogue.target_z > 0) & (catalogue.target_z < 3)]
        return catalogue

    def _set_training_data(
        self,
        catalogue: pd.DataFrame,
        catalogue_da: pd.DataFrame,
        only_zspec: bool = True,
        extinction_corr: bool = True,
        convert_colors: bool = True,
    ) -> None:

        cat_da = Archive.exclude_only_zspec(catalogue_da)
        target_z_train_DA = cat_da["photo_z"].values

        if only_zspec:
            logger.info("Selecting only galaxies with spectroscopic redshift")
            catalogue = Archive._select_only_zspec(catalogue, cat_flag="Calib")
            catalogue = Archive._clean_zspec_sample(
                catalogue, flags_kept=self.flags_kept
            )
        else:
            logger.info(
                "Selecting galaxies with spectroscopic redshift and high-precision photo-z"
            )
            catalogue = Archive.take_zspec_and_photoz(catalogue, cat_flag="Calib")

        self.cat_train = catalogue
        f = self._extract_fluxes(catalogue)
        f_DA = self._extract_fluxes(cat_da)
        idx = np.random.randint(0, len(f_DA), len(f))
        f_DA = f_DA[idx]
        target_z_train_DA = target_z_train_DA[idx]
        self.target_z_train_DA = target_z_train_DA

        if extinction_corr == True:
            logger.info("Correcting MW extinction")
            f = self._correct_extinction(catalogue, f)

        if convert_colors == True:
            logger.info("Converting to colors")
            col = self._to_colors(f)
            col_DA = self._to_colors(f_DA)

            self.phot_train = col
            self.phot_train_DA = col_DA
        else:
            self.phot_train = f
            self.phot_train_DA = f_DA

        if only_zspec == True:
            self.target_z_train = catalogue["specz"].values
        else:
            self.target_z_train = catalogue["target_z"].values

        self.VIS_mag_train = catalogue["MAG_VIS"].values

    def _set_testing_data(
        self,
        cat_test: pd.DataFrame,
        target: str = "specz",
        extinction_corr: bool = True,
        convert_colors: bool = True,
    ) -> None:

        if target == "specz":
            cat_test = Archive._select_only_zspec(cat_test, cat_flag="Valid")
            cat_test = Archive._clean_zspec_sample(cat_test)
            self.target_z_test = cat_test["specz"].values

        elif target == "L15":
            cat_test = self._select_L15_sample(cat_test)
            self.target_z_test = cat_test["target_z"].values

        self.cat_test = cat_test

        f = self._extract_fluxes(cat_test)

        if extinction_corr == True:
            f = self._correct_extinction(cat_test, f)

        if convert_colors == True:
            col = self._to_colors(f)
            self.phot_test = col
        else:
            self.phot_test = f

        self.VIS_mag_test = cat_test["MAG_VIS"].values

    def get_training_data(self):
        return (
            self.phot_train,
            self.target_z_train,
            self.VIS_mag_train,
            self.phot_train_DA,
            self.target_z_train_DA,
        )

    def get_testing_data(self):
        return self.phot_test, self.target_z_test, self.VIS_mag_test

    def get_VIS_mag(self, catalogue):
        return catalogue[["MAG_VIS"]].values
