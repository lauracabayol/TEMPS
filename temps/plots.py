import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temps.utils import nmad
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_photoz(df_list, nbins, xvariable, metric, type_bin='bin',label_list=None, samp='zs', save=False):
    #plot properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    if xvariable == 'VISmag':
        xvariable_lab = 'VIS'
    if xvariable == 'zs':
        xvariable_lab = r'$z_{\rm s}$'

    bin_edges = stats.mstats.mquantiles(df_list[0][xvariable].values, np.linspace(0.05, 1, nbins))
    cmap = plt.get_cmap('Dark2')  # Choose a colormap for coloring lines
    #plt.figure(figsize=(6, 5))
    ls = ['--',':','-']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ydata_dict = {}

    for i, df in enumerate(df_list):
        ydata, xlab = [], []
        
        label = label_list[i]
        
        if label == 'zs':
            label_lab = r'$z_{\rm s}$'
        if label == 'zs+L15':
            label_lab = r'$z_{\rm s}$+L15'    
        if label == 'TEMPS':
            label_lab = 'TEMPS'   

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
                
        ydata_dict[f'{i}'] = ydata
        color = cmap(i)  # Get a different color for each dataframe
        ax1.plot(xlab, ydata,marker='.', lw=1, label=label_lab, color=color, ls=ls[i])
        


    ax1.set_ylabel(f'{metric} $[\Delta z]$', fontsize=18)
    #ax1.set_xlabel(f'{xvariable_lab}', fontsize=16)
    ax1.grid(False)
    ax1.legend()
    
    # Plot ratios between lines in the upper panel
    
    ax2.plot(xlab, np.array(ydata_dict['1'])/np.array(ydata_dict['0']), marker='.', color = cmap(1))
    ax2.plot(xlab, np.array(ydata_dict['2'])/np.array(ydata_dict['0']), marker='.', color = cmap(2))
    ax2.set_ylabel(r'Method $X$ / $z_{\rm z}$', fontsize=14)
   

    ax2.set_xlabel(f'{xvariable_lab}', fontsize=16)
    ax2.grid(True)

    
    if save==True:
        plt.savefig(f'{metric}_{xvariable}_{samp}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pz(m, pz, specz):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the PDF with a label
    ax.plot(np.linspace(0, 4, 1000), pz[m], label='PDF', color='navy')

    # Add a vertical line for 'specz_test'
    ax.axvline(specz[m], color='black', linestyle='--', label=r'$z_{\rm s}$')

    # Add labels and a legend
    ax.set_xlabel(r'$z$', fontsize = 18)
    ax.set_ylabel('Probability Density', fontsize=16)
    ax.legend(fontsize = 18)

    # Display the plot
    plt.show()

    
def plot_zdistribution(archive, plot_test=False, bins=50):
    _,_,specz = archive.get_training_data()
    plt.hist(specz, bins = bins, hisstype='step', color='navy', label=r'Training sample')

    if plot_test:
        _,_,specz_test = archive.get_training_data()
        plt.hist(specz, bins = bins, hisstype='step', color='goldenrod', label=r'Test sample',ls='--')


    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel(r'Redshift', fontsize=14)
    plt.ylabel('Counts', fontsize=14)

    plt.show()
        
def plot_som_map(som_data, plot_arg = 'z', vmin=0, vmax=1):
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
    plt.imshow(som_data, vmin=vmin, vmax=vmax, cmap='viridis')  # Choose an appropriate colormap
    plt.colorbar(label=f'{plot_arg}')  # Add a colorbar with a label
    plt.xlabel(r'$x$ [pixel]', fontsize=14)  # Add an appropriate X-axis label
    plt.ylabel(r'$y$ [pixel]', fontsize=14)  # Add an appropriate Y-axis label
    plt.show()

    
def plot_PIT(pit_list_1, pit_list_2 = None, pit_list_3=None, sample='specz', labels=None, save =True):
    #plot properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(8, 6))
    kwargs=dict(bins=30, histtype='step', density=True, range=(0,1))
    cmap = plt.get_cmap('Dark2')
    

    # Create a histogram
    hist, bins, _ = ax.hist(pit_list_1,  color=cmap(0), ls='--', **kwargs, label=labels[0])
    if pit_list_2!= None:
        hist, bins, _ = ax.hist(pit_list_2,  color=cmap(1), ls=':', **kwargs, label=labels[1])
    if pit_list_3!= None:
        hist, bins, _ = ax.hist(pit_list_3,  color=cmap(2), ls='-', **kwargs, label=labels[2])

    
    # Add labels and a title
    ax.set_xlabel('PIT values', fontsize = 18)
    ax.set_ylabel('Frequency', fontsize = 18)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Customize the x-axis
    ax.set_xlim(0, 1)
    #ax.set_ylim(0,3)
    
    plt.legend(fontsize=12)

    # Make ticks larger
    ax.tick_params(axis='both', which='major', labelsize=14)
    if save==True:
        plt.savefig(f'{sample}_PIT.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()
    


    
def plot_nz(df_list, 
            zcuts = [0.1, 0.5, 1, 1.5, 2, 3, 4],
            save=False):
    # Plot properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16

    cmap = plt.get_cmap('Dark2')  # Choose a colormap for coloring lines
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(20, 8), sharex=True)

    for i, df in enumerate(df_list):
        dfplot = df_list[i].copy()  # Assuming df_list contains dataframes
        ax = axs[i]  # Selecting the appropriate subplot
        
        for iz in range(len(zcuts)-1):
            dfplot_z = dfplot[(dfplot['ztarget'] > zcuts[iz]) & (dfplot['ztarget'] < zcuts[iz + 1])]
            color = cmap(iz)  # Get a different color for each redshift
            
            zt_mean = np.median(dfplot_z.ztarget.values)
            zp_mean = np.median(dfplot_z.z.values)

            
            # Plot histogram on the selected subplot
            ax.hist(dfplot_z.z, bins=50, color=color, histtype='step', linestyle='-', density=True, range=(0, 4))
            ax.axvline(zt_mean, color=color, linestyle='-', lw=2)
            ax.axvline(zp_mean, color=color, linestyle='--', lw=2)
            
        ax.set_ylabel(f'Frequency', fontsize=14)
        ax.grid(False)
        ax.set_xlim(0, 3.5)
    
    axs[-1].set_xlabel(f'$z$', fontsize=18)
    
    if save:
        plt.savefig(f'nz_hist.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

    
    

def plot_crps(crps_list_1, crps_list_2 = None, crps_list_3=None, labels=None,  sample='specz', save =True):
    # Create a figure and axis
    #plot properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('Dark2')

    kwargs=dict(bins=50, histtype='step', density=True, range=(0,1))

    # Create a histogram
    hist, bins, _ = ax.hist(crps_list_1,  color=cmap(0), ls='--', **kwargs, label=labels[0])
    if crps_list_2 is not None:
        hist, bins, _ = ax.hist(crps_list_2,  color=cmap(1), ls=':', **kwargs, label=labels[1])
    if crps_list_3 is not None:
        hist, bins, _ = ax.hist(crps_list_3,  color=cmap(2), ls='-', **kwargs, label=labels[2])

    # Add labels and a title
    ax.set_xlabel('CRPS Scores', fontsize = 18)
    ax.set_ylabel('Frequency', fontsize = 18)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Customize the x-axis
    ax.set_xlim(0, 0.5)

    # Make ticks larger
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Calculate the mean CRPS value
    mean_crps_1 = round(np.nanmean(crps_list_1), 2)
    mean_crps_2 = round(np.nanmean(crps_list_2), 2)
    mean_crps_3 = round(np.nanmean(crps_list_3), 2)


    # Add the mean CRPS value at the top-left corner
    ax.annotate(f"Mean CRPS {labels[0]}: {mean_crps_1}", xy=(0.57, 0.9), xycoords='axes fraction', fontsize=14, color =cmap(0))
    ax.annotate(f"Mean CRPS {labels[1]}: {mean_crps_2}", xy=(0.57, 0.85), xycoords='axes fraction', fontsize=14, color =cmap(1))
    ax.annotate(f"Mean CRPS {labels[2]}: {mean_crps_3}", xy=(0.57, 0.8), xycoords='axes fraction', fontsize=14, color =cmap(2))

    
    if save==True:
        plt.savefig(f'{sample}_CRPS.pdf', bbox_inches='tight')

    # Show the plot
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
    
