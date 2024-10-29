import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from loguru import logger
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, Tuple, List, Union

from temps.utils import maximum_mean_discrepancy


@dataclass
class TempsModule:
    """Attributes:
    model_f (nn.Module): The feature extraction model.
    model_z (nn.Module): The model for predicting z values.
    batch_size (int): Size of each batch for training. Default is 100.
    rejection_param (int): Parameter for rejection sampling. Default is 1.
    da (bool): Flag for enabling domain adaptation. Default is True.
    verbose (bool): Flag for verbose logging. Default is False.
    device (torch.device): Device to run the model on (CPU or GPU).
    ngaussians (int): Number of Gaussian components in the mixture model.
    """

    model_f: nn.Module
    model_z: nn.Module
    batch_size: int = 100
    rejection_param: int = 1
    da: bool = True
    verbose: bool = False
    device: torch.device = field(init=False)
    ngaussians: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization for setting up additional attributes."""
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.ngaussians: int = (
            self.model_z.ngaussians
        )  # Assuming ngaussians is an integer

    def _get_dataloaders(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        input_data_da: Optional[np.ndarray] = None,
        val_fraction: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders.

        Args:
            input_data (np.ndarray): The input features for training.
            target_data (np.ndarray): The target outputs for training.
            input_data_da (Optional[np.ndarray]): Input data for domain adaptation (if any).
            val_fraction (float): Fraction of data to use for validation. Default is 0.1.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders.
        """

        input_data = torch.Tensor(input_data)
        target_data = torch.Tensor(target_data)
        input_data_da = (
            torch.Tensor(input_data_da)
            if input_data_da is not None
            else input_data.clone()
        )

        dataset = TensorDataset(input_data, input_data_da, target_data)

        # Calculate sizes for training and validation sets
        total_size = len(dataset)
        val_size = int(total_size * val_fraction)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
        )

        loader_train = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        loader_val = DataLoader(val_dataset, batch_size=64, shuffle=True)

        return loader_train, loader_val

    def _loss_function(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        logmix: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss function for the model.

        Args:
            mean (torch.Tensor): Mean values predicted by the model.
            std (torch.Tensor): Standard deviation values predicted by the model.
            logmix (torch.Tensor): Logarithm of the mixture coefficients.
            true (torch.Tensor): True target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        log_prob = (
            logmix - 0.5 * (mean - true[:, None]).pow(2) / std.pow(2) - torch.log(std)
        )
        log_prob = torch.logsumexp(log_prob, dim=1)
        loss = -log_prob.mean()
        return loss

    def _loss_function_da(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Compute the KL divergence loss for domain adaptation.

        Args:
            f1 (torch.Tensor): Features from the primary domain.
            f2 (torch.Tensor): Features from the domain for adaptation.

        Returns:
            torch.Tensor: The KL divergence loss value.
        """

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss = kl_loss(f1, f2)
        return torch.log(loss)

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a NumPy array.

        Args:
            x (torch.Tensor): The input tensor to convert.

        Returns:
            np.ndarray: The converted NumPy array.
        """
        return x.detach().cpu().numpy()

    def train(
        self,
        input_data: np.ndarray,
        input_data_da: np.ndarray,
        target_data: np.ndarray,
        nepochs: int = 10,
        step_size: int = 100,
        val_fraction: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0,
    ) -> None:
        """Train the models using provided data.

        Args:
            input_data (np.ndarray): The input features for training.
            input_data_da (np.ndarray): Input data for domain adaptation.
            target_data (np.ndarray): The target outputs for training.
            nepochs (int): Number of training epochs. Default is 10.
            step_size (int): Step size for learning rate scheduling. Default is 100.
            val_fraction (float): Fraction of data to use for validation. Default is 0.1.
            lr (float): Learning rate for the optimizer. Default is 1e-3.
            weight_decay (float): Weight decay for regularization. Default is 0.
        """
        self.model_z.train()
        self.model_f.train()

        loader_train, loader_val = self._get_dataloaders(
            input_data, target_data, input_data_da, val_fraction
        )
        optimizer_z = optim.Adam(
            self.model_z.parameters(), lr=lr, weight_decay=weight_decay
        )
        optimizer_f = optim.Adam(
            self.model_f.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler_z = lr_scheduler.StepLR(optimizer_z, step_size=step_size, gamma=0.1)
        scheduler_f = lr_scheduler.StepLR(optimizer_f, step_size=step_size, gamma=0.1)

        self.model_z.to(self.device)
        self.model_f.to(self.device)

        loss_train, loss_validation = [], []

        for epoch in range(nepochs):
            _loss_train, _loss_validation = [], []
            logger.info(f"Epoch {epoch + 1}/{nepochs} starting...")
            for input_data, input_data_da, target_data in tqdm(
                loader_train, desc="Training", unit="batch"
            ):
                input_data, target_data = input_data.to(self.device), target_data.to(
                    self.device
                )
                if self.da:
                    input_data_da = input_data_da.to(self.device)

                optimizer_f.zero_grad()
                optimizer_z.zero_grad()

                features = self.model_f(input_data)
                features_da = self.model_f(input_data_da) if self.da else None

                mu, logsig, logmix_coeff = self.model_z(features)
                logsig = torch.clamp(logsig, -6, 2)
                sig = torch.exp(logsig)

                loss_z = self._loss_function(mu, sig, logmix_coeff, target_data)
                loss = loss_z + (
                    1e3
                    * maximum_mean_discrepancy(
                        features, features_da, kernel_type="rbf"
                    ).sum()
                    if self.da
                    else 0
                )

                _loss_train.append(loss_z.item())
                loss.backward()
                optimizer_f.step()
                optimizer_z.step()

            scheduler_f.step()
            scheduler_z.step()

            loss_train.append(np.mean(_loss_train))
            _loss_validation = self._validate(loader_val, target_data)

            logger.info(
                f"Epoch {epoch + 1}: Training Loss: {np.mean(_loss_train):.4f}, Validation Loss: {np.mean(_loss_validation):.4f}"
            )

    def _validate(
        self, loader_val: DataLoader, target_data: torch.Tensor
    ) -> List[float]:
        """Validate the model on the validation dataset."""

        self.model_z.eval()
        self.model_f.eval()
        _loss_validation = []

        with torch.no_grad():
            for input_data, _, target_data in tqdm(
                loader_val, desc="Validating", unit="batch"
            ):
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                features = self.model_f(input_data)
                mu, logsig, logmix_coeff = self.model_z(features)
                logsig = torch.clamp(logsig, -6, 2)
                sig = torch.exp(logsig)

                loss_val = self._loss_function(mu, sig, logmix_coeff, target_data)
                _loss_validation.append(loss_val.item())

        return _loss_validation

    def get_features(self, input_data: torch.Tensor) -> np.ndarray:
        """Extract features from the model for the given input data.

        Args:
            input_data (torch.Tensor): Input tensor containing the data for which features are to be extracted.

        Returns:
            np.ndarray: Numpy array of extracted features from the model.
        """

        self.model_f.eval()
        input_data = input_data.to(self.device)
        features = self.model_f(input_data)
        return self._to_numpy(features)

    def get_pz(
        self,
        input_data: torch.Tensor,
        return_pz: bool = True,
        return_flag: bool = True,
        return_odds: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],  # Return z and zerr
        Tuple[np.ndarray, np.ndarray],  # Return z, pz
        Tuple[np.ndarray, np.ndarray, np.ndarray],  # Return z, pz, odds
    ]:
        """Get the predicted redshift (z) values and their uncertainties from the model.

        This function predicts the photo-z for the input galaxies, computes the mean and standard
        deviation for the predicted redshifts, and optionally calculates the probability density function (PDF).

        Args:
            input_data (torch.Tensor): Input tensor containing galaxy data for which to predict redshifts.
            return_pz (bool, optional): Flag indicating whether to return the probability density function. Defaults to True.
            return_flag (bool, optional): Flag indicating whether to return additional information. Defaults to True.
            return_odds (bool, optional): Flag indicating whether to return the odds. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If return_pz is True, returns the PDF and possibly additional metrics.
                - If return_pz is False, returns a tuple containing the predicted redshifts and their uncertainties.
        """

        logger.info("Predicting photo-z for the input galaxies...")
        self.model_z.eval().to(self.device)
        self.model_f.eval().to(self.device)

        input_data = input_data.to(self.device)
        features = self.model_f(input_data)
        mu, logsig, logmix_coeff = self.model_z(features)
        logsig = torch.clamp(logsig, -6, 2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)
        z = (mix_coeff * mu).sum(dim=1)
        zerr = torch.sqrt(
            (mix_coeff * sig**2).sum(dim=1)
            + (mix_coeff * (mu - mu.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        )

        z = self._to_numpy(z)
        mu, mix_coeff, sig = map(self._to_numpy, (mu, mix_coeff, sig))

        if return_pz:
            logger.info("Returning p(z)")
            return self._calculate_pdf(z, mu, sig, mix_coeff, return_flag)
        else:
            return self._to_numpy(z), self._to_numpy(zerr)

    def _calculate_pdf(
        self,
        z: np.ndarray,
        mu: np.ndarray,
        sig: np.ndarray,
        mix_coeff: np.ndarray,
        return_flag: bool,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ]:
        """Calculate the probability density function (PDF) for the predicted redshifts.

        Args:
            z (np.ndarray): Predicted redshift values.
            mu (np.ndarray): Mean values for the Gaussian components.
            sig (np.ndarray): Standard deviations for the Gaussian components.
            mix_coeff (np.ndarray): Mixture coefficients for the Gaussian components.
            return_flag (bool): Flag indicating whether to calculate and return odds.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                - If return_flag is True, returns a tuple containing the redshift values, PDF, and the z-grid.
                - If return_flag is False, returns a tuple containing the redshift values and PDF.
        """

        zgrid = np.linspace(0, 5, 1000)
        pz = np.zeros((len(z), len(zgrid)))

        for ii in range(len(z)):
            for i in range(self.ngaussians):
                pz[ii] += mix_coeff[ii, i] * norm.pdf(zgrid, mu[ii, i], sig[ii, i])

        if return_flag:
            logger.info("Calculating and returning ODDS")
            pz /= pz.sum(axis=1, keepdims=True)
            return self._calculate_odds(z, pz, zgrid)
        return z, pz

    def _calculate_odds(
        self, z: np.ndarray, pz: np.ndarray, zgrid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the odds for the estimated redshifts based on the cumulative distribution.

        Args:
            z (np.ndarray): Predicted redshift values.
            pz (np.ndarray): Probability density function values.
            zgrid (np.ndarray): Grid of redshift values for evaluation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the predicted redshift values,
            PDF values, and calculated odds.
        """

        cumulative = np.cumsum(pz, axis=1)
        odds = np.array(
            [np.max(np.abs(cumulative[i] - 0.68)) for i in range(cumulative.shape[0])]
        )
        return z, pz, odds
