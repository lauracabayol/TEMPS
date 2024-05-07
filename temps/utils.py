import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from scipy.stats import gaussian_kde

def nmad(data):
    return 1.4826 * np.median(np.abs(data - np.median(data)))

def sigma68(data): return 0.5*(pd.Series(data).quantile(q = 0.84) - pd.Series(data).quantile(q = 0.16))

def plot_photoz(df_list, nbins, xvariable, metric, type_bin='bin',label_list=None, samp='zs', save=False):
    #plot properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    

    
    
    bin_edges = stats.mstats.mquantiles(df_list[0][xvariable].values, np.linspace(0.05, 1, nbins))
    print(bin_edges)
    cmap = plt.get_cmap('Dark2')  # Choose a colormap for coloring lines
    plt.figure(figsize=(6, 5))

    for i, df in enumerate(df_list):
        ydata, xlab = [], []

        for k in range(len(bin_edges)-1):
            edge_min = bin_edges[k]
            edge_max = bin_edges[k+1]

            mean_mag = (edge_max + edge_min) / 2

            if type_bin == 'bin':
                df_plot = df[(df[xvariable] > edge_min) & (df[xvariable] < edge_max)]
            elif type_bin == 'cum':
                df_plot = df[(df[xvariable] < edge_max)]
            else:
                raise ValueError("Only type_bin=='bin' for binned and 'cum' for cumulative are supported")

            xlab.append(mean_mag)
            if metric == 'sig68':
                ydata.append(sigma68(df_plot.zwerr))
            elif metric == 'bias':
                ydata.append(np.mean(df_plot.zwerr))
            elif metric == 'nmad':
                ydata.append(nmad(df_plot.zwerr))
            elif metric == 'outliers':
                ydata.append(len(df_plot[np.abs(df_plot.zwerr) > 0.15]) / len(df_plot)*100)
                
        print(ydata)
        color = cmap(i)  # Get a different color for each dataframe
        plt.plot(xlab, ydata, ls='-', marker='.', lw=1, label=f'{label_list[i]}', color=color)
        
    if xvariable == 'VISmag':
        xvariable_lab = 'VIS'
        


    plt.ylabel(f'{metric} $[\\Delta z]$', fontsize=18)
    plt.xlabel(f'{xvariable_lab}', fontsize=16)
    plt.grid(False)
    plt.legend()
    
    if save==True:
        plt.savefig(f'{metric}_{xvariable}_{samp}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_nz(df, bins=np.arange(0,5,0.2)):
    kwargs=dict( bins=bins,alpha=0.5)
    plt.hist(df.zs.values, color='grey', ls='-' ,**kwargs)
    counts, _, =np.histogram(df.z.values, bins=bins)
    
    plt.plot((bins[:-1]+bins[1:])*0.5,counts, color ='purple')
    
    #plt.legend(fontsize=14)
    plt.xlabel(r'Redshift', fontsize=14)
    plt.ylabel(r'Counts', fontsize=14)
    plt.yscale('log')
    
    plt.show()
    
    return


def plot_scatter(df, sample='specz', save=True):
    # Calculate the point density
    xy = np.vstack([df.zs.values,df.z.values])
    zd = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    plt.scatter(df.zs.values, df.z.values,c=zd, s=1)
    plt.xlim(0,5)
    plt.ylim(0,5)

    plt.xlabel(r'$z_{\rm s}$', fontsize = 14)
    plt.ylabel('$z$', fontsize = 14)

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    if save==True:
        plt.savefig(f'{sample}_scatter.pdf', dpi = 300, bbox_inches='tight')

    plt.show()  
    

    

def maximum_mean_discrepancy(x, y, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
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

def compute_kernel(x, y, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
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

    kernel_input = (x - y).pow(2).mean(2)  # Pairwise squared Euclidean distances

    if kernel_type == 'linear':
        kernel_matrix = kernel_input
    elif kernel_type == 'poly':
        kernel_matrix = (1 + kernel_input / kernel_mul).pow(kernel_num)
    elif kernel_type == 'rbf':
        kernel_matrix = torch.exp(-kernel_input / (2 * kernel_mul**2))
    elif kernel_type == 'sigmoid':
        kernel_matrix = torch.tanh(kernel_mul * kernel_input)
    else:
        raise ValueError("Invalid kernel type. Supported types are 'linear', 'poly', 'rbf', and 'sigmoid'.")

    return kernel_matrix


def select_cut(df, 
               completenss_lim=None, 
               nmad_lim = None, 
               outliers_lim=None, 
               return_df=False):
    
    
    if (completenss_lim is None)&(nmad_lim is None)&(outliers_lim is None):
        raise(ValueError("Select at least one cut"))
    elif sum(c is not None for c in [completenss_lim, nmad_lim, outliers_lim]) > 1:
        raise ValueError("Select only one cut at a time")
    
    else:
        bin_edges = stats.mstats.mquantiles(df.zflag, np.arange(0,1.01,0.1))
        scatter, eta, cmptnss, nobj = [],[],[], []

        for k in range(len(bin_edges)-1):
            edge_min = bin_edges[k]
            edge_max = bin_edges[k+1]

            df_bin = df[(df.zflag > edge_min)]    
    

            cmptnss.append(np.round(len(df_bin)/len(df),2)*100)
            scatter.append(nmad(df_bin.zwerr))
            eta.append(len(df_bin[np.abs(df_bin.zwerr)>0.15])/len(df_bin)*100)
            nobj.append(len(df_bin))
            
        dfcuts = pd.DataFrame(data=np.c_[np.round(bin_edges[:-1],5), np.round(nobj,1), np.round(cmptnss,1), np.round(scatter,3), np.round(eta,2)], columns=['flagcut', 'Nobj','completeness', 'nmad', 'eta'])
    
    if completenss_lim is not None:
        print('Selecting cut based on completeness')
        selected_cut = dfcuts[dfcuts['completeness'] <= completenss_lim].iloc[0]
        
    
    elif nmad_lim is not None:
        print('Selecting cut based on nmad')
        selected_cut = dfcuts[dfcuts['nmad'] <= nmad_lim].iloc[0]

        
    elif outliers_lim is not None:
        print('Selecting cut based on outliers')
        selected_cut = dfcuts[dfcuts['eta'] <= outliers_lim].iloc[0]


    print(f"This cut provides completeness of {selected_cut['completeness']}, nmad={selected_cut['nmad']} and eta={selected_cut['eta']}")

    df_cut = df[(df.zflag > selected_cut['flagcut'])]
    if return_df==True:
        return df_cut, selected_cut['flagcut'], dfcuts
    else:
        return selected_cut['flagcut'], dfcuts
        

