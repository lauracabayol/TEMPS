import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temps.utils import nmad, sigma68
from scipy import stats
from typing import List, Optional, Dict


def plot_photoz(
    df_list: List[pd.DataFrame],
    nbins: int,
    xvariable: str,
    metric: str,
    type_bin: str = "bin",
    label_list: Optional[List[str]] = None,
    samp: str = "zs",
    save: bool = False,
) -> None:
    """
    Plot photo-z metrics for multiple dataframes.

    Parameters:
    - df_list (List[pd.DataFrame]): List of dataframes containing data for plotting.
    - nbins (int): Number of bins for the histogram.
    - xvariable (str): Variable to plot on the x-axis.
    - metric (str): Metric to plot (e.g., 'sig68', 'bias', 'nmad', 'outliers').
    - type_bin (str, optional): Type of binning ('bin' or 'cum'). Default is 'bin'.
    - label_list (Optional[List[str]], optional): List of labels for each dataframe. Default is None.
    - samp (str, optional): Sample label for saving. Default is 'zs'.
    - save (bool, optional): If True, save the plot to a file. Default is False.

    Returns:
    None
    """
    # Plot properties
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    # Set x-axis label based on variable
    xvariable_lab = "VIS" if xvariable == "VISmag" else r"$z_{\rm s}$"

    # Calculate bin edges
    bin_edges = stats.mstats.mquantiles(
        df_list[0][xvariable].values, np.linspace(0.05, 1, nbins)
    )
    cmap = plt.get_cmap("Dark2")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    ydata_dict: Dict[str, List[float]] = {}

    # Loop through dataframes and calculate metrics
    for i, df in enumerate(df_list):
        ydata, xlab = [], []

        label = label_list[i]
        label_lab = {
            "zs": r"$z_{\rm s}$",
            "zs+L15": r"$z_{\rm s}$+L15",
            "TEMPS": "TEMPS",
        }.get(label, label)

        for k in range(len(bin_edges) - 1):
            edge_min = bin_edges[k]
            edge_max = bin_edges[k + 1]
            mean_mag = (edge_max + edge_min) / 2

            df_plot = (
                df[(df[xvariable] > edge_min) & (df[xvariable] < edge_max)]
                if type_bin == "bin"
                else df[(df[xvariable] < edge_max)]
            )

            xlab.append(mean_mag)
            if metric == "sig68":
                ydata.append(sigma68(df_plot.zwerr))
            elif metric == "bias":
                ydata.append(np.mean(df_plot.zwerr))
            elif metric == "nmad":
                ydata.append(nmad(df_plot.zwerr))
            elif metric == "outliers":
                ydata.append(
                    len(df_plot[np.abs(df_plot.zwerr) > 0.15]) / len(df_plot) * 100
                )

        ydata_dict[f"{i}"] = ydata
        color = cmap(i)
        ax1.plot(
            xlab,
            ydata,
            marker=".",
            lw=1,
            label=label_lab,
            color=color,
            ls=["--", ":", "-"][i],
        )

    ax1.set_ylabel(f"{metric} $[\Delta z]$", fontsize=18)
    ax1.grid(False)
    ax1.legend()

    # Plot ratios
    ax2.plot(
        xlab,
        np.array(ydata_dict["1"]) / np.array(ydata_dict["0"]),
        marker=".",
        color=cmap(1),
    )
    ax2.plot(
        xlab,
        np.array(ydata_dict["2"]) / np.array(ydata_dict["0"]),
        marker=".",
        color=cmap(2),
    )
    ax2.set_ylabel(r"Method $X$ / $z_{\rm z}$", fontsize=14)
    ax2.set_xlabel(f"{xvariable_lab}", fontsize=16)
    ax2.grid(True)

    if save:
        plt.savefig(f"{metric}_{xvariable}_{samp}.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_pz(m: int, pz: np.ndarray, specz: float) -> None:
    """
    Plot the Probability Density Function (PDF) for a given model and compare it with the spectroscopic redshift.

    Parameters:
    - m (int): Index for the model.
    - pz (np.ndarray): Probability density function values.
    - specz (float): Spectroscopic redshift value.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.linspace(0, 4, 1000), pz[m], label="PDF", color="navy")
    ax.axvline(specz[m], color="black", linestyle="--", label=r"$z_{\rm s}$")
    ax.set_xlabel(r"$z$", fontsize=18)
    ax.set_ylabel("Probability Density", fontsize=16)
    ax.legend(fontsize=18)
    plt.show()


def plot_zdistribution(archive, plot_test: bool = False, bins: int = 50) -> None:
    """
    Plot the distribution of redshifts for training and optionally test samples.

    Parameters:
    - archive: Data archive object containing the training data.
    - plot_test (bool, optional): If True, plot test sample distribution. Default is False.
    - bins (int, optional): Number of histogram bins. Default is 50.

    Returns:
    None
    """
    _, _, specz = archive.get_training_data()
    plt.hist(specz, bins=bins, histtype="step", color="navy", label=r"Training sample")

    if plot_test:
        _, _, specz_test = archive.get_training_data()
        plt.hist(
            specz_test,
            bins=bins,
            histtype="step",
            color="goldenrod",
            label=r"Test sample",
            linestyle="--",
        )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r"Redshift", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.legend()
    plt.show()


def plot_som_map(
    som_data: np.ndarray, plot_arg: str = "z", vmin: float = 0, vmax: float = 1
) -> None:
    """
    Plot the Self-Organizing Map (SOM) data.

    Parameters:
    - som_data (numpy.ndarray): The SOM data to be visualized.
    - plot_arg (str, optional): The column name to be plotted. Default is 'z'.
    - vmin (float, optional): Minimum value for color scaling. Default is 0.
    - vmax (float, optional): Maximum value for color scaling. Default is 1.

    Returns:
    None
    """
    plt.imshow(som_data, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(label=f"{plot_arg}")
    plt.xlabel(r"$x$ [pixel]", fontsize=14)
    plt.ylabel(r"$y$ [pixel]", fontsize=14)
    plt.show()


def plot_PIT(
    pit_list_1: List[float],
    pit_list_2: Optional[List[float]] = None,
    pit_list_3: Optional[List[float]] = None,
    sample: str = "specz",
    labels: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    """
    Plot Probability Integral Transform (PIT) values for given lists.

    Parameters:
    - pit_list_1 (List[float]): First list of PIT values.
    - pit_list_2 (Optional[List[float]], optional): Second list of PIT values. Default is None.
    - pit_list_3 (Optional[List[float]], optional): Third list of PIT values. Default is None.
    - sample (str, optional): Sample label for saving. Default is 'specz'.
    - labels (Optional[List[str]], optional): List of labels for each PIT list. Default is None.
    - save (bool, optional): If True, save the plot to a file. Default is True.

    Returns:
    None
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8, 6))
    kwargs = dict(bins=30, histtype="step", density=True, range=(0, 1))
    cmap = plt.get_cmap("Dark2")

    # Create a histogram
    ax.hist(pit_list_1, color=cmap(0), linestyle="--", **kwargs, label=labels[0])
    if pit_list_2 is not None:
        ax.hist(pit_list_2, color=cmap(1), linestyle="--", **kwargs, label=labels[1])
    if pit_list_3 is not None:
        ax.hist(pit_list_3, color=cmap(2), linestyle="--", **kwargs, label=labels[2])

    ax.set_xlabel("PIT values", fontsize=14)
    ax.set_ylabel("Normalized Counts", fontsize=14)
    ax.legend(fontsize=12)

    if save:
        plt.savefig(f"PIT_{sample}.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_outlier_ratio(
    outliers: np.ndarray, num_samp: int = 100, plot_mean: bool = True
) -> None:
    """
    Plot the outlier ratio as a function of the number of samples.

    Parameters:
    - outliers (np.ndarray): Outlier ratio data.
    - num_samp (int, optional): Number of samples for plotting. Default is 100.
    - plot_mean (bool, optional): If True, plot the mean of outliers. Default is True.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_samp + 1), outliers[:num_samp], label="Outlier Ratio")

    if plot_mean:
        plt.axhline(
            np.mean(outliers), color="red", linestyle="--", label="Mean Outlier Ratio"
        )

    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Outlier Ratio", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def plot_crps(
    crps_list_1: List[float],
    crps_list_2: Optional[List[float]] = None,
    crps_list_3: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    sample: str = "specz",
    save: bool = True,
) -> None:
    # Create a figure and axis
    # plot properties
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("Dark2")

    kwargs = dict(bins=50, histtype="step", density=True, range=(0, 1))

    # Create a histogram
    hist, bins, _ = ax.hist(
        crps_list_1, color=cmap(0), ls="--", **kwargs, label=labels[0]
    )
    if crps_list_2 is not None:
        hist, bins, _ = ax.hist(
            crps_list_2, color=cmap(1), ls=":", **kwargs, label=labels[1]
        )
    if crps_list_3 is not None:
        hist, bins, _ = ax.hist(
            crps_list_3, color=cmap(2), ls="-", **kwargs, label=labels[2]
        )

    # Add labels and a title
    ax.set_xlabel("CRPS Scores", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)

    # Add grid lines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Customize the x-axis
    ax.set_xlim(0, 0.5)

    # Make ticks larger
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Calculate the mean CRPS value
    mean_crps_1 = round(np.nanmean(crps_list_1), 2)
    mean_crps_2 = round(np.nanmean(crps_list_2), 2)
    mean_crps_3 = round(np.nanmean(crps_list_3), 2)

    # Add the mean CRPS value at the top-left corner
    ax.annotate(
        f"Mean CRPS {labels[0]}: {mean_crps_1}",
        xy=(0.57, 0.9),
        xycoords="axes fraction",
        fontsize=14,
        color=cmap(0),
    )
    ax.annotate(
        f"Mean CRPS {labels[1]}: {mean_crps_2}",
        xy=(0.57, 0.85),
        xycoords="axes fraction",
        fontsize=14,
        color=cmap(1),
    )
    ax.annotate(
        f"Mean CRPS {labels[2]}: {mean_crps_3}",
        xy=(0.57, 0.8),
        xycoords="axes fraction",
        fontsize=14,
        color=cmap(2),
    )

    if save == True:
        plt.savefig(f"{sample}_CRPS.pdf", bbox_inches="tight")

    # Show the plot
    plt.show()
