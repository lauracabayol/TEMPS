import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from loguru import logger


def caluclate_eta(df):
    return len(df[np.abs(df.zwerr)>0.15])/len(df) *100
    

def nmad(data):
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def sigma68(data):
    return 0.5 * (pd.Series(data).quantile(q=0.84) - pd.Series(data).quantile(q=0.16))


def maximum_mean_discrepancy(x, y, kernel_type="rbf", kernel_mul=2.0, kernel_num=5):
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


def compute_kernel(x, y, kernel_type="rbf", kernel_mul=2.0, kernel_num=5):
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
    df, completenss_lim=None, nmad_lim=None, outliers_lim=None, return_df=False
):

    if (completenss_lim is None) & (nmad_lim is None) & (outliers_lim is None):
        raise (ValueError("Select at least one cut"))
    elif sum(c is not None for c in [completenss_lim, nmad_lim, outliers_lim]) > 1:
        raise ValueError("Select only one cut at a time")

    else:
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
        logger.info("Selecting cut based on nmad")
        selected_cut = dfcuts[dfcuts["nmad"] <= nmad_lim].iloc[0]

    elif outliers_lim is not None:
        logger.info("Selecting cut based on outliers")
        selected_cut = dfcuts[dfcuts["eta"] <= outliers_lim].iloc[0]

    logger.info(
        f"This cut provides completeness of {selected_cut['completeness']}, nmad={selected_cut['nmad']} and eta={selected_cut['eta']}"
    )

    df_cut = df[(df.odds > selected_cut["flagcut"])]
    if return_df == True:
        return df_cut, selected_cut["flagcut"], dfcuts
    else:
        return selected_cut["flagcut"], dfcuts
