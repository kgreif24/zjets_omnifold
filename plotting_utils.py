""" plotting_utils - This file contains some helper functions for generating plots,
usually histograms showing the quality of a given reweighting.

Author: Kevin Greif
Last updated 01/25/2024
python3
"""

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs

import numpy as np
import sys


############################ Default plotting settings for wandb logging ############################

# Default settings for histograms
common_hist_settings = {
    'histtype': 'bar',
    'density': True,
    'linear_scale': False,
    'bins': 150,
    'ylim': None,
    'rlim': [0.5, 1.5]
}

# Individual histogram settings
define_hists = [
    {
        'key': 'pT_l1',
        'xlabel': r'$p_{T,\mu1}$',
        'bins': np.linspace(0, 1e3, 150),
    },
    {
        'key': 'pT_l2',
        'xlabel': r'$p_{T,\mu2}$',
        'bins': np.linspace(0, 800, 150),
    },
    {
        'key': 'eta_l1',
        'xlabel': r'$\eta_{\mu1}$',
        'bins': np.linspace(-3, 3, 150)
    },
    {
        'key': 'eta_l2',
        'xlabel': r'$\eta_{\mu2}$',
        'bins': np.linspace(-3, 3, 150)
    },
    {
        'key': 'phi_l1',
        'xlabel': r'$\phi_{\mu1}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1]
    },
    {
        'key': 'phi_l2',
        'xlabel': r'$\phi_{\mu2}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1]
    },
    {
        'key': 'pT_ll',
        'xlabel': r'$p_{T, \mu\mu}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    {
        'key': 'y_ll',
        'xlabel': r'$y_{\mu\mu}$',
        'bins': np.linspace(-3, 3, 150)
    },
    {
        'key': 'pT_trackj1',
        'xlabel': r'$p_{T, j1}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    {
        'key': 'pT_trackj2',
        'xlabel': r'$p_{T, j2}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    {
        'key': 'y_trackj1',
        'xlabel': r'$y_{j1}$',
        'bins': np.linspace(-3, 3, 150)
    },
    {
        'key': 'y_trackj2',
        'xlabel': r'$y_{j2}$',
        'bins': np.linspace(-3, 3, 150)
    },
    {
        'key': 'phi_trackj1',
        'xlabel': r'$\phi_{j1}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1]
    },
    {
        'key': 'phi_trackj2',
        'xlabel': r'$\phi_{j2}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1]
    },
    {
        'key': 'm_trackj1',
        'xlabel': r'$m_{j1}$',
        'bins': np.linspace(0, 100, 150)
    },
    {
        'key': 'm_trackj2',
        'xlabel': r'$m_{j2}$',
        'bins': np.linspace(0, 100, 150)
    },
    {
        'key': 'tau1_trackj1',
        'xlabel': r'$\tau_{1,j1}$',
        'bins': np.linspace(0, 0.9, 150)
    },
    {
        'key': 'tau1_trackj2',
        'xlabel': r'$\tau_{1,j2}$',
        'bins': np.linspace(0, 0.9, 150)
    },
    {
        'key': 'tau2_trackj1',
        'xlabel': r'$\tau_{2,j1}$',
        'bins': np.linspace(0, 0.6, 150)
    },
    {
        'key': 'tau2_trackj2',
        'xlabel': r'$\tau_{2,j2}$',
        'bins': np.linspace(0, 0.6, 150)
    },
    {
        'key': 'tau3_trackj1',
        'xlabel': r'$\tau_{3,j1}$',
        'bins': np.linspace(0, 0.4, 150)
    },
    {
        'key': 'tau3_trackj2',
        'xlabel': r'$\tau_{3,j2}$',
        'bins': np.linspace(0, 0.4, 150)
    }
]

special_hists = []

# Overwrite default settings
default_settings = []
for hist_dict in define_hists:
    copy = common_hist_settings.copy()
    copy.update(hist_dict)
    default_settings.append(copy)

default_settings = default_settings + special_hists


###############################################################################################

def make_logged_plots(plot_data, labels, start_weights, outputs, definitions=default_settings, save_location='./plot_storage', display=False):
    """ make_logged_plots - This function will be called by the pytorch lightning module
    at the end of every validation epoch. It will generate histograms showing the quality of the 
    reweighting. The histograms will be logged to wandb.

    Histograms are defined in the define_hists list of dictionaries. This list is set as a 
    default argument to the function but can be overridden at any time.

    Plots are made from the "plot_datasets" contained in the pytorch lightning data module

    Arguments:
    plot_data - numpy array of the data to be plotted
    labels - numpy array of the labels for the data
    start_weights - numpy array of the weights used in network training
    outputs - numpy array of network outputs over the events contained in "plot_data"
    definitions - list of dictionaries describing each of the histograms to build
    save_location - location of directory for plot staging
    display - if true, display the plots to the screen

    Returns:
    dict - A dictionary of histograms with form {save_name: filepath}, where filepath
    is to a .png stored on local disk
    """

    # First calculate weights from outputs
    probs = 1 / (1 + np.exp(-outputs))
    derived_weights = probs / (1 - probs)

    # Make dictionary for return
    return_dict = {}

    # Get numpy array of the plotting data, labels, and starting weights
    # Indeces of relevant information in dataset are hardcoded here. Beware of changing them!
    end_weights = start_weights * derived_weights
    assert len(plot_data) == len(labels) == len(start_weights) == len(end_weights)

    # Loop over list of dictionaries
    for i, element in enumerate(definitions):

        # Pull plotting data for this particular histogram
        this_data = plot_data[:,i]

        # Modify if requested
        if 'simple_modifier' in element:
            this_data = element['simple_modifier'](this_data)
        
        # Make reweighting plot
        fig = plot_reweighting(
            this_data[labels==0],
            start_weights[labels==0],
            end_weights[labels==0],
            this_data[labels==1], 
            bins=element['bins'], 
            xlabel=element['xlabel'], 
            linear_scale=element['linear_scale'],
            ylim=element['ylim'],
            rlim=element['rlim']
        )

        # Save histogram
        hist_path = f'{save_location}/{element["key"]}.png'
        fig.savefig(hist_path, dpi=300)
        if display:
            plt.show()
        plt.close()
        return_dict[element['key']] = hist_path

    return return_dict


def add_ratios(fig):
    """ add_ratios - This function adds ratio pads to a given matplotlib figure.

    Arguments:
    fig - matplotlib axis to add ratio pads to

    Returns:
    ax - main matplotlib axis
    axr - ratio matplotlib axis
    """

    this_grid = gs.GridSpec(2, 1, figure=fig, height_ratios=(7, 2), hspace=0.0)

    axr = fig.add_subplot(this_grid[1,0])
    ax = fig.add_subplot(this_grid[0,0])

    return ax, axr


def add_stats_box(ax, data, low=0, high=1e20):
    """
    Add a ROOT-like stats box to a matplotlib histogram.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to add the stats box.
    data : array_like
        The data to be plotted in the histogram.
    """

    # Take subset of data within bounds
    data = data[(data >= low) & (data < high)]

    # Calculate the mean and standard deviation
    entries = len(data)
    mean = np.mean(data)
    std = np.std(data)

    # Add the stats box
    textstr = '\n'.join((
        r'$\mathrm{Entries}=%d$' % (entries, ),
        r'$\mathrm{Mean}=%.4f$' % (mean, ),
        r'$\mathrm{Std}=%.4f$' % (std, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)


def plot_reweighting(source_data, weight_start, weight_end, target_data, bins=150, xlabel="", linear_scale=False, ylim=None, rlim=None):
    """ plot_reweighting - This function will plot the quality of the reweighting
    along a given dimension. Inputs are the original data, the starting weights, the ending weights, and 
    the target data. Optional is the bins to be used in the plot, and the x-axis label.

    Arguments:
    source_data - numpy array of original data
    weight_start - numpy array of starting weights, set to vector of ones if there are none
    weights_end - numpy array of ending weights
    target_data - numpy array of target data
    bins - number of bins to use in the plot, if not set use mpl default w/ 150 bins
    xlabel - label to use for the x-axis
    linear_scale - if true, use linear scale for y-axis, otherwise use log scale
    ylim - if set, use this as the y-axis limits
    rlim - if set, use this as the ratio axis limits

    Returns:
    fig - matplotlib figure object
    """

    fig = plt.figure()
    ax, axr = add_ratios(fig)
    n_mc, bins, patches =  ax.hist(source_data, bins=bins, label='MC', density=True, alpha=0.5, weights=weight_start)
    n_rw, bins, patches = ax.hist(source_data, bins=bins, label='ReweightedMC', density=True, histtype='step', color='black', weights=weight_end)
    n_pd, bins, patches = ax.hist(target_data, bins=bins, label='PseudoData', density=True, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_ylabel('A.U.')
    if linear_scale:
        ax.set_yscale('linear')
    else:
        ax.set_yscale('log')
    ax.legend()

    axr.hlines(1, bins[0], bins[-1], color='k', linestyle='--', alpha=0.8)
    axr.plot(bins[:-1], n_mc / n_pd, color='#1f77b4', drawstyle='steps-post')
    axr.plot(bins[:-1], n_rw / n_pd, color='k', drawstyle='steps-post')
    axr.set_ylabel('MC / PseudoData')
    axr.set_xlim(ax.get_xlim())
    axr.set_xlabel(xlabel)
    if rlim is not None:
        axr.set_ylim(rlim)

    return fig
