import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from loguru import logger
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

from temps.utils import maximum_mean_discrepancy


class TempsModule:
    """Class for managing temperature-related models and training."""

    def __init__(
        self,
        model_f,
        model_z,
        batch_size=100,
        rejection_param=1,
        da=True,
        verbose=False,
    ):
        self.model_z = model_z
        self.model_f = model_f
        self.da = da
        self.verbose = verbose
        self.ngaussians = model_z.ngaussians

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.rejection_parameter = rejection_param

    def _get_dataloaders(
        self, input_data, target_data, input_data_da=None, val_fraction=0.1
    ):
        """Create training and validation dataloaders."""
        input_data = torch.Tensor(input_data)
        target_data = torch.Tensor(target_data)
        input_data_da = (
            torch.Tensor(input_data_da)
            if input_data_da is not None
            else input_data.clone()
        )

        dataset = TensorDataset(input_data, input_data_da, target_data)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [int(len(dataset) * (1 - val_fraction)), int(len(dataset) * val_fraction)],
        )
        loader_train = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        loader_val = DataLoader(val_dataset, batch_size=64, shuffle=True)

        return loader_train, loader_val

    def _loss_function(self, mean, std, logmix, true):
        """Compute the loss function."""
        log_prob = (
            logmix - 0.5 * (mean - true[:, None]).pow(2) / std.pow(2) - torch.log(std)
        )
        log_prob = torch.logsumexp(log_prob, dim=1)
        loss = -log_prob.mean()
        return loss

    def _loss_function_da(self, f1, f2):
        """Compute the KL divergence loss for domain adaptation."""
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss = kl_loss(f1, f2)
        return torch.log(loss)

    def _to_numpy(self, x):
        """Convert a tensor to a NumPy array."""
        return x.detach().cpu().numpy()

    def train(
        self,
        input_data,
        input_data_da,
        target_data,
        nepochs=10,
        step_size=100,
        val_fraction=0.1,
        lr=1e-3,
        weight_decay=0,
    ):
        """Train the models using provided data."""
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

    def _validate(self, loader_val, target_data):
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

    def get_features(self, input_data):
        """Get features from the model."""
        self.model_f.eval()
        input_data = input_data.to(self.device)
        features = self.model_f(input_data)
        return self._to_numpy(features)

    def get_pz(self, input_data, return_pz=True, return_flag=True, return_odds=False):
        """Get the predicted z values and their uncertainties."""
        logger.info("Predicting photo-z for the input galaxies...")
        self.model_z.eval()
        self.model_f.eval()

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

        mu, mix_coeff, sig = map(self._to_numpy, (mu, mix_coeff, sig))

        if return_pz:
            logger.info("Returning p(z)")
            return self._calculate_pdf(z, mu, sig, mix_coeff, return_flag)
        else:
            return self._to_numpy(z), self._to_numpy(zerr)

    def _calculate_pdf(self, z, mu, sig, mix_coeff, return_flag):
        """Calculate the probability density function."""
        zgrid = np.linspace(0, 5, 1000)
        pz = np.zeros((len(z), len(zgrid)))

        for ii in range(len(z)):
            for i in range(self.ngaussians):
                pz[ii] += mix_coeff[ii, i] * norm.pdf(
                    zgrid, mu[ii, i], sig[ii, i]
                )

        if return_flag:
            logger.info("Calculating and returning ODDS")
            pz /= pz.sum(axis=1, keepdims=True)
            return self._calculate_odds(z, pz, zgrid)
        return self._to_numpy(z), pz

    def _calculate_odds(self, z, pz, zgrid):
        """Calculate odds based on the PDF."""
        logger.info('Calculating ODDS values')
        diff_matrix = np.abs(self._to_numpy(z)[:, None] - zgrid[None, :])
        idx_peak = np.argmax(pz, axis=1)
        zpeak = zgrid[idx_peak]
        idx_upper = np.argmin(np.abs((zpeak + 0.05)[:, None] - zgrid[None, :]), axis=1)
        idx_lower = np.argmin(np.abs((zpeak - 0.05)[:, None] - zgrid[None, :]), axis=1)

        odds = []
        for jj in range(len(pz)):
            odds.append(pz[jj,idx_lower[jj]:(idx_upper[jj]+1)].sum())
    
        odds = np.array(odds)
        return self._to_numpy(z), pz, odds

    def calculate_pit(self, input_data, target_data):
        logger.info('Calculating PIT values')
        
        pit_list = []

        self.model_f = self.model_f.eval()
        self.model_f = self.model_f.to(self.device)
        self.model_z = self.model_z.eval()
        self.model_z = self.model_z.to(self.device)

        input_data = input_data.to(self.device)
                

        features = self.model_f(input_data)
        mu, logsig, logmix_coeff = self.model_z(features)
        
        logsig = torch.clamp(logsig,-6,2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)
        
        mu,  mix_coeff, sig = mu.detach().cpu().numpy(),  mix_coeff.detach().cpu().numpy(), sig.detach().cpu().numpy() 
        
        for ii in range(len(input_data)):
            pit = (mix_coeff[ii] * norm.cdf(target_data[ii]*np.ones(mu[ii].shape),mu[ii], sig[ii])).sum()
            pit_list.append(pit)
        
        
        return pit_list
    
    def calculate_crps(self, input_data, target_data):
        logger.info('Calculating CRPS values')

        def measure_crps(cdf, t):
            zgrid = np.linspace(0,4,1000)
            Deltaz = zgrid[None,:] - t[:,None]
            DeltaZ_heaviside = np.where(Deltaz < 0,0,1)
            integral = (cdf-DeltaZ_heaviside)**2
            crps_value = integral.sum(1) / 1000

            return crps_value


        crps_list = []

        self.model_f = self.model_f.eval()
        self.model_f = self.model_f.to(self.device)
        self.model_z = self.model_z.eval()
        self.model_z = self.model_z.to(self.device)

        input_data = input_data.to(self.device)


        features = self.model_f(input_data)
        mu, logsig, logmix_coeff = self.model_z(features)
        logsig = torch.clamp(logsig,-6,2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)


        mu,  mix_coeff, sig = mu.detach().cpu().numpy(),  mix_coeff.detach().cpu().numpy(), sig.detach().cpu().numpy() 

        z = (mix_coeff * mu).sum(1)

        x = np.linspace(0, 4, 1000)
        pz = np.zeros(shape=(len(target_data), len(x)))
        for ii in range(len(input_data)):
            for i in range(6):
                pz[ii] += mix_coeff[ii,i] * norm.pdf(x, mu[ii,i], sig[ii,i])

        pz = pz / pz.sum(1)[:,None]


        cdf_z = np.cumsum(pz,1)

        crps_value = measure_crps(cdf_z, target_data)



        return crps_value

