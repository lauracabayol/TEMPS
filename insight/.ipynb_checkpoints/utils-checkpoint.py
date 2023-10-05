import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def nmad(data):
    return 1.4826 * np.median(np.abs(data - np.median(data)))

def sigma68(data): return 0.5*(pd.Series(data).quantile(q = 0.84) - pd.Series(data).quantile(q = 0.16))


def plot_photoz_estimates(df, nbins,xvariable,metric, type_bin='bin'):
    bin_edges = stats.mstats.mquantiles(df[xvariable].values, np.linspace(0.1,1,nbins))
    ydata,xdata = [],[]
    
    
    for k in range(len(bin_edges)-1):
        edge_min = bin_edges[k]
        edge_max = bin_edges[k+1]

        mean_mag =  (edge_max + edge_min) / 2
        
        if type_bin=='bin':
            df_plot = df_test[(df_test[xvariable] > edge_min) & (df_test[xvariable] < edge_max)]
        elif type_bin=='cum':
            df_plot = df_test[(df_test[xvariable] < edge_max)]
        else:
            raise ValueError("Only type_bin=='bin' for binned and 'cum' for cumulative are supported")


        xdata.append(mean_mag)
        if metric=='sig68':
            ydata.append(sigma68(df_plot.zwerr))
            ylab=r'$\sigma_{\rm NMAD} [\Delta z]$'
        elif metric=='bias':
            ydata.append(np.median(df_plot.zwerr))
            ylab=r'Median $[\Delta z]$'
        elif metric=='nmad':
            ydata.append(nmad(df_plot.zwerr))
            ylab=r'$\sigma_{\rm NMAD} [\Delta z]$'
        elif metric=='outliers':
            ydata.append(len(df_plot[np.abs(df_plot.zwerr)>0.15])/len(df_plot) *100)
            ylab=r'$\eta$ [%]'
            
    if xvariable=='VISmag':
        xlab='VIS'
    elif xvariable=='zs':
        xlab=r'$z_{\rm spec}$'
    elif xvariable=='z':
        xlab=r'$z$'

    plt.plot(xdata,ydata, ls = '-', marker = '.', color = 'navy',lw = 1, label = '')
    plt.ylabel(f'{ylab}', fontsize = 18)
    plt.xlabel(f'{xlab}', fontsize = 16)

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.grid(False)
    
    plt.show()
   