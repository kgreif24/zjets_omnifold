""" plotting_utils - This file contains some helper functions for generating plots,
usually histograms showing the quality of a given reweighting.

Author: Kevin Greif
Last updated 01/25/2024
python3
"""

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs

from pytorch_lightning.utilities.rank_zero import *

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
    'rlim': [0.5, 1.5],
    'w1_eval': True,
    'truth_mc': False,
    'truth_data': False
}

# Individual histogram settings
event_hists = {
    'pT_l1': {
        'key': 'pT_l1',
        'xlabel': r'$p_{T,\mu1}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    'pT_l2': {
        'key': 'pT_l2',
        'xlabel': r'$p_{T,\mu2}$',
        'bins': np.linspace(0, 800, 150)
    },
    'eta_l1': {
        'key': 'eta_l1',
        'xlabel': r'$\eta_{\mu1}$',
        'bins': np.linspace(-3, 3, 150)
    },
    'eta_l2': {
        'key': 'eta_l2',
        'xlabel': r'$\eta_{\mu2}$',
        'bins': np.linspace(-3, 3, 150)
    },
    'phi_l1': {
        'key': 'phi_l1',
        'xlabel': r'$\phi_{\mu1}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1],
        'linear_scale': True,
        'ylim': [0, 0.2],
        'w1_eval': False
    },
    'phi_l2': {
        'key': 'phi_l2',
        'xlabel': r'$\phi_{\mu2}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1],
        'linear_scale': True,
        'ylim': [0, 0.2],
        'w1_eval': False
    },
    'pT_ll': {
        'key': 'pT_ll',
        'xlabel': r'$p_{T, \mu\mu}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    'y_ll': {
        'key': 'y_ll',
        'xlabel': r'$y_{\mu\mu}$',
        'bins': np.linspace(-3, 3, 150)
    },
    'pT_trackj1': {
        'key': 'pT_trackj1',
        'xlabel': r'$p_{T, j1}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    'pT_trackj2': {
        'key': 'pT_trackj2',
        'xlabel': r'$p_{T, j2}$',
        'bins': np.linspace(0, 1e3, 150)
    },
    'y_trackj1': {
        'key': 'y_trackj1',
        'xlabel': r'$y_{j1}$',
        'bins': np.linspace(-3, 3, 150)
    },
    'y_trackj2': {
        'key': 'y_trackj2',
        'xlabel': r'$y_{j2}$',
        'bins': np.linspace(-3, 3, 150)
    },
    'phi_trackj1': {
        'key': 'phi_trackj1',
        'xlabel': r'$\phi_{j1}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1],
        'linear_scale': True,
        'ylim': [0, 0.2],
        'w1_eval': False
    },
    'phi_trackj2': {
        'key': 'phi_trackj2',
        'xlabel': r'$\phi_{j2}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1],
        'linear_scale': True,
        'ylim': [0, 0.2],
        'w1_eval': False
    },
    'm_trackj1': {
        'key': 'm_trackj1',
        'xlabel': r'$m_{j1}$',
        'bins': np.linspace(0, 100, 150)
    },
    'm_trackj2': {
        'key': 'm_trackj2',
        'xlabel': r'$m_{j2}$',
        'bins': np.linspace(0, 100, 150)
    },
    'tau1_trackj1': {
        'key': 'tau1_trackj1',
        'xlabel': r'$\tau_{1,j1}$',
        'bins': np.linspace(0, 0.9, 150)
    },
    'tau1_trackj2': {
        'key': 'tau1_trackj2',
        'xlabel': r'$\tau_{1,j2}$',
        'bins': np.linspace(0, 0.9, 150)
    },
    'tau2_trackj1': {
        'key': 'tau2_trackj1',
        'xlabel': r'$\tau_{2,j1}$',
        'bins': np.linspace(0, 0.6, 150)
    },
    'tau2_trackj2': {
        'key': 'tau2_trackj2',
        'xlabel': r'$\tau_{2,j2}$',
        'bins': np.linspace(0, 0.6, 150)
    },
    'tau3_trackj1': {
        'key': 'tau3_trackj1',
        'xlabel': r'$\tau_{3,j1}$',
        'bins': np.linspace(0, 0.4, 150)
    },
    'tau3_trackj2': {
        'key': 'tau3_trackj2',
        'xlabel': r'$\tau_{3,j2}$',
        'bins': np.linspace(0, 0.4, 150)
    },
    'Ntracks_trackj1': {
        'key': 'Ntracks_trackj1',
        'xlabel': r'$N_{tracks,j1}$',
        'bins': np.arange(0, 80, 1)
    },
    'Ntracks_trackj2': {
        'key': 'Ntracks_trackj2',
        'xlabel': r'$N_{tracks,j2}$',
        'bins': np.arange(0, 80, 1)
    },
    'Ntracks': {
        'key': 'Ntracks',
        'xlabel': '# of tracks',
        'bins': np.arange(0, 256, 1)
    }
}

track_hists = {
    'alltrack_pt': {
        'key': 'alltrack_pt',
        'xlabel': r'$p_{T, track}$',
        'bins': np.linspace(0, 1e3, 150),
        'w1_eval': False
    },
    'alltrack_eta': {
        'key': 'alltrack_eta',
        'xlabel': r'$\eta_{track}$',
        'bins': np.linspace(-3, 3, 150),
        'w1_eval': False
    },
    'alltrack_phi': {
        'key': 'alltrack_phi',
        'xlabel': r'$\phi_{track}$',
        'bins': np.linspace(-3.2, 3.2, 150),
        'rlim': [0.9, 1.1],
        'linear_scale': True,
        'ylim': [0, 0.2],
        'w1_eval': False
    }
}

# Overwrite default settings function
def overwrite(defaults, new_dict):
    return_dict = {}
    for key, hist_dict in new_dict.items():
        copy = defaults.copy()
        copy.update(hist_dict)
        return_dict[key] = copy
    return return_dict

default_settings = overwrite(common_hist_settings, event_hists)
track_hists = overwrite(common_hist_settings, track_hists)

###############################################################################################

# Decorator to ensure that only rank zero process runs the function
@rank_zero_only
def make_logged_plots(
        plot_data, 
        labels, 
        start_weights, 
        end_weights=None, 
        definitions=default_settings, 
        save_location='./plot_storage', 
        display=False,
        **kwargs):
    """ make_logged_plots - This function will be called by the pytorch lightning module
    at the end of every validation epoch. It will generate histograms showing the quality of the 
    reweighting. The histograms will be logged to wandb.

    Histograms are defined in the define_hists list of dictionaries. This list is set as a 
    default argument to the function but can be overridden at any time.

    Plots are made from the "plot_datasets" contained in the pytorch lightning data module

    Arguments:
    plot_data - numpy array of the data to be plotted
    labels - numpy array of the labels for the data
    start_weights - numpy array of the weights used in network training, for all events
    end_weights - numpy array of new derived weights for all events.
        If left as None, don't plot reweighting
    definitions - list of dictionaries describing each of the histograms to build
    save_location - location of directory for plot staging
    display - if true, display the plots to the screen

    Returns:
    dict - A dictionary of histograms with form {save_name: filepath}, where filepath
    is to a .png stored on local disk
    """

    # Separate weights and calculate effeective number of events
    source_start_weights = start_weights[labels==0]
    target_weights = start_weights[labels==1]
    start_effective_events = np.sum(source_start_weights)**2 / np.sum(source_start_weights**2)
    if end_weights is not None:
        end_weights = end_weights[labels==0]
        end_effective_events = np.sum(end_weights)**2 / np.sum(end_weights**2)

    # Make dictionary for return
    return_dict = {}

    # Loop over dictionary of dictionaries
    for i, (key, element) in enumerate(definitions.items()):

        # Pull plotting data for this particular histogram
        this_data = plot_data[:,i]

        # Modify if requested
        if 'simple_modifier' in element:
            this_data = element['simple_modifier'](this_data)
        
        # Make reweighting plot
        fig = plot_reweighting(
            this_data[labels==0],
            this_data[labels==1], 
            source_start_weights,
            source_weight_end=end_weights,
            target_weight=target_weights,
            bins=element['bins'], 
            xlabel=element['xlabel'], 
            linear_scale=element['linear_scale'],
            ylim=element['ylim'],
            rlim=element['rlim'],
            **kwargs
        )

        # Save histogram
        hist_path = f'{save_location}/{key}.png'
        fig.savefig(hist_path, dpi=300)
        if display:
            plt.show()
        plt.close()
        return_dict[key] = hist_path

    # Also make histograms of the derived weights
    if end_weights is not None:
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(end_weights, bins=150, label='MC', density=True, histtype='step')
        ax.set_yscale('log')
        ax.set_xlabel('Derived Weights for MC')
        ax.set_ylabel('A.U.')
        # Add stats box
        add_stats_box(ax, end_weights)
        # Add the number of effective of events as a text box
        textstr = f'Start Effective Events: {start_effective_events:10.0f}' + \
                '\n' + f'End Effective Events: {end_effective_events:10.0f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.95, 0.75, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        plt.savefig(f'{save_location}/derived_weights.png', dpi=300)
        if display:
            plt.show()
        plt.close()
        return_dict['derived_weights'] = f'{save_location}/derived_weights.png'
    
    return return_dict

# Decorator to ensure that only rank zero process runs the function
@rank_zero_only
def make_inclusive_track_plots(
        track_data,
        labels,
        start_weights,
        end_weights=None,
        definitions=track_hists,
        save_location='./plot_storage',
        display=False,
        **kwargs):
    """ make_inclusive_track_plots - This function will make plots of the inclusive track kinematics.
    It will calculate the reweighting given a set of network inputs. Importantly the weights will be extended
    to the shape of the track input data, and then used to build histograms.

    Arguments:
    track_data - numpy array of the track kinematics to be plotted, in shape (n_events, 3, n_tracks)
    labels - numpy array of the labels for the data
    start_weights - numpy array of the weights used in network training for all events
    end_weights - numpy array of new derived weights for all events.
        If left as None, don't plot reweighting
    definitions - list of dictionaries describing each of the histograms to build, default setting is above
    save_location - location of directory in which to dump plots
    display - if true, display the plots to the screen
    """

    # Separate weights and extend to the track data shape 
    source_start_weights = start_weights[labels==0]
    source_start_weights = np.repeat(np.expand_dims(source_start_weights, axis=1), track_data.shape[2], axis=1)
    source_start_weights = np.ravel(source_start_weights)
    target_weights = start_weights[labels==1]
    target_weights = np.repeat(np.expand_dims(target_weights, axis=1), track_data.shape[2], axis=1)
    target_weights = np.ravel(target_weights)
    if end_weights is not None:
        end_weights = end_weights[labels==0]
        end_weights = np.repeat(np.expand_dims(end_weights, axis=1), track_data.shape[2], axis=1)
        end_weights = np.ravel(end_weights)

    # Make dictionary for return
    return_dict = {}

    # Loop over pT, eta, phi
    for i, (key, element) in enumerate(definitions.items()):

        # Pull plotting data for this particular histogram
        this_data = track_data[:,i,:]

        # Separate MC and pseudodata
        this_data_mc = np.ravel(this_data[labels==0,:])
        this_data_pd = np.ravel(this_data[labels==1,:])

        # Drop zero padded entries
        if i == 0:
            track_pt_mc = this_data_mc
            track_pt_pd = this_data_pd
            source_start_weights = source_start_weights[track_pt_mc != 0]
            target_weights = target_weights[track_pt_pd != 0]
            if end_weights is not None:
                end_weights = end_weights[track_pt_mc != 0]
        this_data_mc = this_data_mc[track_pt_mc != 0]
        this_data_pd = this_data_pd[track_pt_pd != 0]

        # Take exponential if this is pT
        if i == 0:
            this_data_mc = np.exp(this_data_mc)
            this_data_pd = np.exp(this_data_pd)
        
        # Make reweighting plot
        fig = plot_reweighting(
            this_data_mc,
            this_data_pd, 
            source_start_weights,
            source_weight_end=end_weights,
            target_weight=target_weights,
            bins=element['bins'], 
            xlabel=element['xlabel'], 
            linear_scale=element['linear_scale'],
            ylim=element['ylim'],
            rlim=element['rlim'],
            **kwargs
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


def plot_reweighting(
        source_data, 
        target_data, 
        source_weight_start, 
        source_weight_end=None,
        target_weight=None, 
        bins=150, 
        xlabel="", 
        linear_scale=False, 
        ylim=None, 
        rlim=None,
        names=None):
    """ plot_reweighting - This function will plot the quality of the reweighting
    along a given dimension. Inputs are the original data, the starting weights, the ending weights, and 
    the target data. Optional is the bins to be used in the plot, and the x-axis label.

    Arguments:
    source_data - numpy array of original data
    target_data - numpy array of target data
    source_weight_start - numpy array of starting weights for source, required
    source_weights_end - numpy array of ending weights for source. Optional if there are none
    target_weight - numpy array of weights for target, optional
    bins - number of bins to use in the plot, if not set use mpl default w/ 150 bins
    xlabel - label to use for the x-axis
    linear_scale - if true, use linear scale for y-axis, otherwise use log scale
    ylim - if set, use this as the y-axis limits
    rlim - if set, use this as the ratio axis limits
    names - an optional tuple of strings which sets the legends names for source and target data

    Returns:
    fig - matplotlib figure object
    """

    # Parse names
    if names is not None:
        name1, name2 = names
    else:
        name1 = 'MC'
        name2 = 'PseudoData'

    fig = plt.figure()
    ax, axr = add_ratios(fig)
    n_mc, bins, patches =  ax.hist(source_data, bins=bins, label=name1, density=True, alpha=0.5, weights=source_weight_start)
    if source_weight_end is not None:
        n_rw, bins, patches = ax.hist(source_data, bins=bins, label='Reweighted', density=True, histtype='step', color='black', weights=source_weight_end)
    if target_weight is None:
        target_weight = np.ones_like(target_data)
    n_pd, bins, patches = ax.hist(target_data, bins=bins, label=name2, density=True, alpha=0.5, weights=target_weight)
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
    if source_weight_end is not None:
        axr.plot(bins[:-1], n_rw / n_pd, color='k', drawstyle='steps-post')
    axr.set_ylabel(f'{name1}/{name2}')
    axr.set_xlim(ax.get_xlim())
    axr.set_xlabel(xlabel)
    if rlim is not None:
        axr.set_ylim(rlim)

    return fig
