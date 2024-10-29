import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from loguru import logger
from typing import Optional, Tuple, Union, List
from torch import nn
from scipy.stats import norm
import torch


def calculate_eta(df: pd.DataFrame) -> float:
    """Calculate the percentage of outliers in the DataFrame based on zwerr column."""
    return len(df[np.abs(df.zwerr) > 0.15]) / len(df) * 100


def nmad(data: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the normalized median absolute deviation (NMAD) of the data."""
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def sigma68(data: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the sigma68 metric, a robust measure of dispersion."""
    return 0.5 * (pd.Series(data).quantile(q=0.84) - pd.Series(data).quantile(q=0.16))


def maximum_mean_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_type: str = "rbf",
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
    - x: Tensor, samples from the source domain
    - y: Tensor, samples from the target domain
    - kernel_type: str, the type of kernel to be used ('linear', 'poly', 'rbf', 'sigmoid')
    - kernel_mul: float, multiplier for the kernel bandwidth
    - kernel_num: int, number of kernels for the MMD approximation

    Returns:
    - mmd_loss: Tensor, the MMD loss
    """
    x_kernel = compute_kernel(x, x, kernel_type, kernel_mul, kernel_num)
    y_kernel = compute_kernel(y, y, kernel_type, kernel_mul, kernel_num)
    xy_kernel = compute_kernel(x, y, kernel_type, kernel_mul, kernel_num)

    mmd_loss = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd_loss


def compute_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_type: str = "rbf",
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """
    Compute the kernel matrix based on the chosen kernel type.

    Args:
    - x: Tensor, samples
    - y: Tensor, samples
    - kernel_type: str, the type of kernel to be used ('linear', 'poly', 'rbf', 'sigmoid')
    - kernel_mul: float, multiplier for the kernel bandwidth
    - kernel_num: int, number of kernels for the MMD approximation

    Returns:
    - kernel_matrix: Tensor, the computed kernel matrix
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(x_size, y_size, dim)
    y = y.unsqueeze(0).expand(x_size, y_size, dim)

    kernel_input = (x - y).pow(2).mean(2)

    if kernel_type == "linear":
        kernel_matrix = kernel_input
    elif kernel_type == "poly":
        kernel_matrix = (1 + kernel_input / kernel_mul).pow(kernel_num)
    elif kernel_type == "rbf":
        kernel_matrix = torch.exp(-kernel_input / (2 * kernel_mul**2))
    elif kernel_type == "sigmoid":
        kernel_matrix = torch.tanh(kernel_mul * kernel_input)
    else:
        raise ValueError(
            "Invalid kernel type. Supported types are 'linear', 'poly', 'rbf', and 'sigmoid'."
        )

    return kernel_matrix


def select_cut(
    df: pd.DataFrame,
    completenss_lim: Optional[float] = None,
    nmad_lim: Optional[float] = None,
    outliers_lim: Optional[float] = None,
    return_df: bool = False,
) -> Union[Tuple[pd.DataFrame, float, pd.DataFrame], Tuple[float, pd.DataFrame]]:
    """
    Selects a cut based on one of the provided limits (completeness, NMAD, or outliers).

    Args:
    - df: DataFrame, containing the data
    - completenss_lim: float, optional limit on completeness
    - nmad_lim: float, optional limit on NMAD
    - outliers_lim: float, optional limit on outliers (eta)
    - return_df: bool, whether to return the filtered DataFrame

    Returns:
    - selected_cut: If return_df is False, returns the cut value and a DataFrame of cuts.
                    If return_df is True, returns the filtered DataFrame, cut value, and cuts DataFrame.
    """

    if (completenss_lim is None) and (nmad_lim is None) and (outliers_lim is None):
        raise ValueError("Select at least one cut")
    elif sum(c is not None for c in [completenss_lim, nmad_lim, outliers_lim]) > 1:
        raise ValueError("Select only one cut at a time")

    bin_edges = stats.mstats.mquantiles(df.odds, np.arange(0, 1.01, 0.1))
    scatter, eta, cmptnss, nobj = [], [], [], []

    for k in range(len(bin_edges) - 1):
        edge_min = bin_edges[k]
        edge_max = bin_edges[k + 1]

        df_bin = df[(df.odds > edge_min)]
        cmptnss.append(np.round(len(df_bin) / len(df), 2) * 100)
        scatter.append(nmad(df_bin.zwerr))
        eta.append(len(df_bin[np.abs(df_bin.zwerr) > 0.15]) / len(df_bin) * 100)
        nobj.append(len(df_bin))

    dfcuts = pd.DataFrame(
        data=np.c_[
            np.round(bin_edges[:-1], 5),
            np.round(nobj, 1),
            np.round(cmptnss, 1),
            np.round(scatter, 3),
            np.round(eta, 2),
        ],
        columns=["flagcut", "Nobj", "completeness", "nmad", "eta"],
    )

    if completenss_lim is not None:
        logger.info("Selecting cut based on completeness")
        selected_cut = dfcuts[dfcuts["completeness"] <= completenss_lim].iloc[0]

    elif nmad_lim is not None:
        logger.info("Selecting cut based on NMAD")
        selected_cut = dfcuts[dfcuts["nmad"] <= nmad_lim].iloc[0]

    elif outliers_lim is not None:
        logger.info("Selecting cut based on outliers")
        selected_cut = dfcuts[dfcuts["eta"] <= outliers_lim].iloc[0]

    logger.info(
        f"This cut provides completeness of {selected_cut['completeness']}, "
        f"nmad={selected_cut['nmad']} and eta={selected_cut['eta']}"
    )

    df_cut = df[(df.odds > selected_cut["flagcut"])]

    if return_df:
        return df_cut, selected_cut["flagcut"], dfcuts
    else:
        return selected_cut["flagcut"], dfcuts


def calculate_pit(
    self,
    model_f: nn.Module,
    model_z: nn.Module,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
) -> List[float]:

    logger.info("Calculating PIT values")

    pit_list = []

    model_f = model_f.eval()
    model_f = model_f.to(self.device)
    model_z = model_z.eval()
    model_z = model_z.to(self.device)

    input_data = input_data.to(self.device)

    features = model_f(input_data)
    mu, logsig, logmix_coeff = model_z(features)

    logsig = torch.clamp(logsig, -6, 2)
    sig = torch.exp(logsig)

    mix_coeff = torch.exp(logmix_coeff)

    mu, mix_coeff, sig = (
        mu.detach().cpu().numpy(),
        mix_coeff.detach().cpu().numpy(),
        sig.detach().cpu().numpy(),
    )

    for ii in range(len(input_data)):
        pit = (
            mix_coeff[ii]
            * norm.cdf(target_data[ii] * np.ones(mu[ii].shape), mu[ii], sig[ii])
        ).sum()
        pit_list.append(pit)

    return pit_list


def calculate_crps(
    self,
    model_f: nn.Module,
    model_z: nn.Module,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
) -> List[float]:
    logger.info("Calculating CRPS values")

    def measure_crps(cdf, t):
        zgrid = np.linspace(0, 4, 1000)
        Deltaz = zgrid[None, :] - t[:, None]
        DeltaZ_heaviside = np.where(Deltaz < 0, 0, 1)
        integral = (cdf - DeltaZ_heaviside) ** 2
        crps_value = integral.sum(1) / 1000

        return crps_value

    crps_list = []

    model_f = model_f.eval()
    model_f = model_f.to(self.device)
    model_z = model_z.eval()
    model_z = model_z.to(self.device)

    input_data = input_data.to(self.device)

    features = model_f(input_data)
    mu, logsig, logmix_coeff = model_z(features)
    logsig = torch.clamp(logsig, -6, 2)
    sig = torch.exp(logsig)

    mix_coeff = torch.exp(logmix_coeff)

    mu, mix_coeff, sig = (
        mu.detach().cpu().numpy(),
        mix_coeff.detach().cpu().numpy(),
        sig.detach().cpu().numpy(),
    )

    z = (mix_coeff * mu).sum(1)

    x = np.linspace(0, 4, 1000)
    pz = np.zeros(shape=(len(target_data), len(x)))
    for ii in range(len(input_data)):
        for i in range(6):
            pz[ii] += mix_coeff[ii, i] * norm.pdf(x, mu[ii, i], sig[ii, i])

    pz = pz / pz.sum(1)[:, None]

    cdf_z = np.cumsum(pz, 1)

    crps_value = measure_crps(cdf_z, target_data)

    return crps_value
