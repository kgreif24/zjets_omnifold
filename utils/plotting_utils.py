""" plotting_utils - This file contains some helper functions for generating plots,
usually histograms showing the quality of a given reweighting.

Author: Kevin Greif
Last updated 01/25/2024
python3
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from pytorch_lightning.utilities.rank_zero import rank_zero_only

import numpy as np
from collections import OrderedDict


# ------- Default plotting settings for wandb logging ------- #

# IBU binning from first round analysis
use_ibu_bins = False
ibu_bins = OrderedDict()
ibu_bins["pT_l1"] = [25.0, 125.0, 200.0, 300.0, 400.0, 600.0, 800.0]
ibu_bins["pT_l2"] = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0]
ibu_bins["eta_l1"] = [
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
]
ibu_bins["eta_l2"] = [
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
]
ibu_bins["phi_l1"] = [
    -(np.pi + 1e-5),
    -2.8,
    -2.4,
    -2.0,
    -1.6,
    -1.2,
    -0.8,
    -0.4,
    0.0,
    0.4,
    0.8,
    1.2,
    1.6,
    2.0,
    2.4,
    2.8,
    (np.pi + 1e-5),
]
ibu_bins["phi_l2"] = [
    -(np.pi + 1e-5),
    -2.8,
    -2.4,
    -2.0,
    -1.6,
    -1.2,
    -0.8,
    -0.4,
    0.0,
    0.4,
    0.8,
    1.2,
    1.6,
    2.0,
    2.4,
    2.8,
    (np.pi + 1e-5),
]
ibu_bins["pT_trackj1"] = [5.0, 50.0, 100.0, 150.0, 200.0, 300.0, 1000.0]
ibu_bins["pT_trackj2"] = [5.0, 25.0, 50.0, 100.0, 500.0]
ibu_bins["y_trackj1"] = [
    -2.5,
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
]
ibu_bins["y_trackj2"] = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
ibu_bins["phi_trackj1"] = [
    -(np.pi + 1e-5),
    -2.8,
    -2.4,
    -2.0,
    -1.6,
    -1.2,
    -0.8,
    -0.4,
    0.0,
    0.4,
    0.8,
    1.2,
    1.6,
    2.0,
    2.4,
    2.8,
    (np.pi + 1e-5),
]
ibu_bins["phi_trackj2"] = [
    -(np.pi + 1e-5),
    -2.8,
    -2.4,
    -2.0,
    -1.6,
    -1.2,
    -0.8,
    -0.4,
    0.0,
    0.4,
    0.8,
    1.2,
    1.6,
    2.0,
    2.4,
    2.8,
    (np.pi + 1e-5),
]
ibu_bins["pT_ll"] = [200.0, 230.0, 300.0, 450.0, 600.0, 1000.0]
ibu_bins["y_ll"] = [
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
]
ibu_bins["Ntracks_trackj1"] = [0.5, 6.5, 10.5, 14.5, 19.5, 25.5, 39.5]
ibu_bins["Ntracks_trackj2"] = [0.5, 3.5, 7.5, 11.5, 15.5, 34.5]
ibu_bins["m_trackj1"] = [0.0, 8.0, 16.0, 24.0, 32.0, 42.0, 70.0]
ibu_bins["m_trackj2"] = [0.0, 5.0, 10.0, 20.0, 40.0]
ibu_bins["tau1_trackj1"] = [0.0, 0.05, 0.1, 0.17, 0.25, 0.35, 0.4, 0.9]
ibu_bins["tau1_trackj2"] = [0.0, 0.1, 0.2, 0.35, 0.5, 0.9]
ibu_bins["tau2_trackj1"] = [0.0, 0.025, 0.05, 0.08, 0.12, 0.17, 0.2, 0.5]
ibu_bins["tau2_trackj2"] = [0.0, 0.025, 0.1, 0.17, 0.25, 0.5]
ibu_bins["tau3_trackj1"] = [0.0, 0.025, 0.05, 0.1, 0.3]
ibu_bins["tau3_trackj2"] = [0.0, 0.025, 0.08, 0.14, 0.3]
ibu_bins["Ntracks"] = np.linspace(
    0, 256, 10
)  # Note not yet included in IBU so use Omnifold default binning
ibu_bins["sum_pT_tracks"] = np.linspace(0, 2000, 10)

# Default settings for histograms
common_hist_settings = {
    "histtype": "bar",
    "density": True,
    "linear_scale": False,
    "bins": 150,
    "ylim": None,
    "rlim": [0.5, 1.5],
    "w1_eval": True,
    "truth_mc": False,
    "truth_data": False,
}

# Individual histogram settings
event_hists = {
    "pT_l1": {
        "key": "pT_l1",
        "xlabel": r"$p_{T,\mu1}$",
        "bins": ibu_bins["pT_l1"] if use_ibu_bins else np.linspace(0, 1e3, 150),
    },
    "pT_l2": {
        "key": "pT_l2",
        "xlabel": r"$p_{T,\mu2}$",
        "bins": ibu_bins["pT_l2"] if use_ibu_bins else np.linspace(0, 800, 150),
    },
    "eta_l1": {
        "key": "eta_l1",
        "xlabel": r"$\eta_{\mu1}$",
        "bins": ibu_bins["eta_l1"] if use_ibu_bins else np.linspace(-3, 3, 150),
    },
    "eta_l2": {
        "key": "eta_l2",
        "xlabel": r"$\eta_{\mu2}$",
        "bins": ibu_bins["eta_l2"] if use_ibu_bins else np.linspace(-3, 3, 150),
    },
    "phi_l1": {
        "key": "phi_l1",
        "xlabel": r"$\phi_{\mu1}$",
        "bins": ibu_bins["phi_l1"] if use_ibu_bins else np.linspace(-3.2, 3.2, 150),
        "rlim": [0.9, 1.1],
        "linear_scale": True,
        "ylim": [0, 0.2],
    },
    "phi_l2": {
        "key": "phi_l2",
        "xlabel": r"$\phi_{\mu2}$",
        "bins": ibu_bins["phi_l2"] if use_ibu_bins else np.linspace(-3.2, 3.2, 150),
        "rlim": [0.9, 1.1],
        "linear_scale": True,
        "ylim": [0, 0.2],
    },
    "pT_ll": {
        "key": "pT_ll",
        "xlabel": r"$p_{T, \mu\mu}$",
        "bins": ibu_bins["pT_ll"] if use_ibu_bins else np.linspace(0, 1e3, 150),
    },
    "y_ll": {
        "key": "y_ll",
        "xlabel": r"$y_{\mu\mu}$",
        "bins": ibu_bins["y_ll"] if use_ibu_bins else np.linspace(-3, 3, 150),
    },
    "pT_trackj1": {
        "key": "pT_trackj1",
        "xlabel": r"$p_{T, j1}$",
        "bins": ibu_bins["pT_trackj1"] if use_ibu_bins else np.linspace(0, 1e3, 150),
    },
    "pT_trackj2": {
        "key": "pT_trackj2",
        "xlabel": r"$p_{T, j2}$",
        "bins": ibu_bins["pT_trackj2"] if use_ibu_bins else np.linspace(0, 1e3, 150),
    },
    "y_trackj1": {
        "key": "y_trackj1",
        "xlabel": r"$y_{j1}$",
        "bins": ibu_bins["y_trackj1"] if use_ibu_bins else np.linspace(-3, 3, 150),
    },
    "y_trackj2": {
        "key": "y_trackj2",
        "xlabel": r"$y_{j2}$",
        "bins": ibu_bins["y_trackj1"] if use_ibu_bins else np.linspace(-3, 3, 150),
    },
    "phi_trackj1": {
        "key": "phi_trackj1",
        "xlabel": r"$\phi_{j1}$",
        "bins": (
            ibu_bins["phi_trackj1"] if use_ibu_bins else np.linspace(-3.2, 3.2, 150)
        ),
        "rlim": [0.9, 1.1],
        "linear_scale": True,
        "ylim": [0, 0.2],
    },
    "phi_trackj2": {
        "key": "phi_trackj2",
        "xlabel": r"$\phi_{j2}$",
        "bins": (
            ibu_bins["phi_trackj2"] if use_ibu_bins else np.linspace(-3.2, 3.2, 150)
        ),
        "rlim": [0.9, 1.1],
        "linear_scale": True,
        "ylim": [0, 0.2],
    },
    "m_trackj1": {
        "key": "m_trackj1",
        "xlabel": r"$m_{j1}$",
        "bins": ibu_bins["m_trackj1"] if use_ibu_bins else np.linspace(0, 100, 150),
    },
    "m_trackj2": {
        "key": "m_trackj2",
        "xlabel": r"$m_{j2}$",
        "bins": ibu_bins["m_trackj2"] if use_ibu_bins else np.linspace(0, 100, 150),
    },
    "tau1_trackj1": {
        "key": "tau1_trackj1",
        "xlabel": r"$\tau_{1,j1}$",
        "bins": ibu_bins["tau1_trackj1"] if use_ibu_bins else np.linspace(0, 0.9, 150),
    },
    "tau1_trackj2": {
        "key": "tau1_trackj2",
        "xlabel": r"$\tau_{1,j2}$",
        "bins": ibu_bins["tau1_trackj2"] if use_ibu_bins else np.linspace(0, 0.9, 150),
    },
    "tau2_trackj1": {
        "key": "tau2_trackj1",
        "xlabel": r"$\tau_{2,j1}$",
        "bins": ibu_bins["tau2_trackj1"] if use_ibu_bins else np.linspace(0, 0.6, 150),
    },
    "tau2_trackj2": {
        "key": "tau2_trackj2",
        "xlabel": r"$\tau_{2,j2}$",
        "bins": ibu_bins["tau2_trackj2"] if use_ibu_bins else np.linspace(0, 0.6, 150),
    },
    "tau3_trackj1": {
        "key": "tau3_trackj1",
        "xlabel": r"$\tau_{3,j1}$",
        "bins": ibu_bins["tau3_trackj1"] if use_ibu_bins else np.linspace(0, 0.4, 150),
    },
    "tau3_trackj2": {
        "key": "tau3_trackj2",
        "xlabel": r"$\tau_{3,j2}$",
        "bins": ibu_bins["tau3_trackj2"] if use_ibu_bins else np.linspace(0, 0.4, 150),
    },
    "Ntracks_trackj1": {
        "key": "Ntracks_trackj1",
        "xlabel": r"$N_{tracks,j1}$",
        "bins": ibu_bins["Ntracks_trackj1"] if use_ibu_bins else np.arange(0, 80, 1),
    },
    "Ntracks_trackj2": {
        "key": "Ntracks_trackj2",
        "xlabel": r"$N_{tracks,j2}$",
        "bins": ibu_bins["Ntracks_trackj2"] if use_ibu_bins else np.arange(0, 80, 1),
    },
    "Ntracks": {
        "key": "Ntracks",
        "xlabel": "# of tracks",
        "bins": np.arange(0, 256, 1),
    },
    "sum_pT_tracks": {
        "key": "sum_pT_tracks",
        "xlabel": r"$H_{T, tracks}$",
        "bins": np.linspace(0, 2000, 150),
    },
}

track_hists = {
    "alltrack_pt": {
        "key": "alltrack_pt",
        "xlabel": r"$p_{T, track}$",
        "bins": np.linspace(0, 5, 150),
        "w1_eval": False,
    },
    "alltrack_eta": {
        "key": "alltrack_eta",
        "xlabel": r"$\eta_{track}$",
        "bins": np.linspace(-3, 3, 150),
        "w1_eval": False,
    },
    "alltrack_phi": {
        "key": "alltrack_phi",
        "xlabel": r"$\phi_{track}$",
        "bins": np.linspace(-np.pi, np.pi, 150),
        "rlim": [0.9, 1.1],
        "linear_scale": True,
        "w1_eval": False,
    },
    "alltrack_Ht": {
        "key": "alltrack_Ht",
        "xlabel": r"$H_{T, tracks}$",
        "bins": np.linspace(0, 2e3, 150),
        "w1_eval": False,
    },
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

# ----------------------------------------------------------------


# Decorator to ensure that only rank zero process runs the function
@rank_zero_only
def make_logged_plots(
    plot_data,
    labels,
    start_weights,
    end_weights,
    definitions=default_settings,
    save_location="./plot_storage",
    is_comp=False,
    region=0,
    display=False,
    **kwargs,
):
    """make_logged_plots - This function will be called by the pytorch lightning module
    at the end of every validation epoch. It will generate histograms showing the
    quality of the reweighting. The histograms will be logged to wandb.

    Histograms are defined in the define_hists list of dictionaries. This list is set
    as a default argument to the function but can be overridden at any time.

    Plots are made from the "plot_datasets" contained in the pytorch lightning data
    module.

    Arguments:
    plot_data - numpy array of the data to be plotted
    labels - numpy array of the labels for the data
    start_weights - numpy array of the weights used in network training, for all events
    end_weights - numpy array of new weights for all events.
        These are not multiplied by the start weights. This should be done
        independently of this function
    definitions - list of dictionaries describing each of the histograms to build
    save_location - location of directory for plot staging
    is_comp - set to true if we are plotting comparison between truth MC and truth PD.
        In this case the weight histogram is of the Omnifold weights in total, and
        should be labeled as such.
    display - if true, display the plots to the screen

    Returns:
    dict - A dictionary of histograms with form {save_name: filepath}, where filepath
    is to a .png stored on local disk
    """

    # Separate source and target weights
    source_start_weights = start_weights[labels == 0]
    source_end_weights = end_weights[labels == 0]
    target_start_weights = start_weights[labels == 1]
    # We don't care about end weights for target!

    # Calculate loss factor and effective number of events
    start_effective_events = np.sum(source_start_weights) ** 2 / np.sum(
        source_start_weights**2
    )
    end_effective_events = np.sum(source_end_weights) ** 2 / np.sum(
        source_end_weights**2
    )
    loss_factor = end_effective_events / start_effective_events

    # Make dictionary for return
    return_dict = {}

    # Calculate the interesting weights to plot. If we are plotting a reweighting for a
    # single step this is the network weights (end_weights).
    # If we are plotting a comparison between the reweighted truth MC and truth PD,
    # this is the omnifold weights (end_weights / start_weights)
    if is_comp:
        interesting_weights = source_end_weights / source_start_weights
        label = "Omnifold Weights"
    else:
        interesting_weights = source_end_weights
        label = "Network Weights"

    # Make histogram of the interesting weights, with a loss factor
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(interesting_weights, bins=150, label=label, density=True, histtype="step")
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("A.U.")
    add_stats_box(ax, interesting_weights)
    # Add the loss factor as a text box
    textstr = f"Loss Factor: {loss_factor:10.4f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax.text(
        0.95,
        0.75,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )
    save_path = f"{save_location}/delta_weights.png"
    plt.savefig(save_path, dpi=300)
    if display:
        plt.show()
    plt.close()
    return_dict["delta_weights"] = save_path

    # Loop over dictionary of dictionaries
    for i, (key, element) in enumerate(definitions.items()):

        # Pull plotting data for this particular histogram
        this_data = plot_data[:, i]

        # Modify if requested
        if "simple_modifier" in element:
            this_data = element["simple_modifier"](this_data)

        # Make reweighting plot
        fig = plot_reweighting(
            this_data[labels == 0],
            this_data[labels == 1],
            source_start_weights,
            source_weight_end=source_end_weights,
            target_weight=target_start_weights,
            bins=element["bins"],
            xlabel=element["xlabel"],
            linear_scale=element["linear_scale"],
            ylim=element["ylim"],
            rlim=element["rlim"],
            **kwargs,
        )

        # Save histogram
        hist_path = f"{save_location}/{key}.png"
        fig.savefig(hist_path, dpi=300)
        if display:
            plt.show()
        plt.close()
        return_dict[key] = hist_path

    return return_dict


# Decorator to ensure that only rank zero process runs the function
@rank_zero_only
def make_inclusive_track_plots(
    track_data,
    labels,
    start_weights,
    end_weights,
    definitions=track_hists,
    save_location="./plot_storage",
    display=False,
    **kwargs,
):
    """make_inclusive_track_plots - This function will make plots of the inclusive
    track kinematics. It will calculate the reweighting given a set of network
    inputs. Importantly the weights will be extended to the shape of the track
    input data, and then used to build histograms.

    Arguments:
    track_data - numpy array of the track kinematics to be plotted
        in shape (n_events, 3, n_tracks)
    labels - numpy array of the labels for the data
    start_weights - numpy array of the weights used in network training for all events
    end_weights - numpy array of new weights for all events.
        These are not multiplied by the start weights. This should be done
        independently of this function
    definitions - list of dictionaries describing each of the histograms to build,
        default setting is above
    save_location - location of directory in which to dump plots
    display - if true, display the plots to the screen
    """

    # Separate source and target weights
    source_start_weights = start_weights[labels == 0]
    source_end_weights = end_weights[labels == 0]
    target_start_weights = start_weights[labels == 1]
    # We don't care about end weights for target!

    # Extend to the track data shape
    rvl_source_start_weights = np.repeat(
        np.expand_dims(source_start_weights, axis=1), track_data.shape[2], axis=1
    )
    rvl_source_start_weights = np.ravel(rvl_source_start_weights)
    rvl_target_start_weights = np.repeat(
        np.expand_dims(target_start_weights, axis=1), track_data.shape[2], axis=1
    )
    rvl_target_start_weights = np.ravel(rvl_target_start_weights)
    rvl_source_end_weights = np.repeat(
        np.expand_dims(source_end_weights, axis=1), track_data.shape[2], axis=1
    )
    rvl_source_end_weights = np.ravel(rvl_source_end_weights)

    # Make dictionary for return
    return_dict = {}

    # Loop over pT, eta, phi
    for i, (key, element) in enumerate(definitions.items()):

        # Regular plotting for pt / eta / phi
        if key != "alltrack_Ht":

            # Pull plotting data for this particular histogram
            this_data = track_data[:, i, :]

            # Separate MC and pseudodata
            this_data_mc = np.ravel(this_data[labels == 0, :])
            this_data_pd = np.ravel(this_data[labels == 1, :])

            # Drop zero padded entries
            if i == 0:
                track_pt_mc = this_data_mc
                track_pt_pd = this_data_pd
                rvl_source_start_weights = rvl_source_start_weights[track_pt_mc != 0]
                rvl_target_start_weights = rvl_target_start_weights[track_pt_pd != 0]
                rvl_source_end_weights = rvl_source_end_weights[track_pt_mc != 0]
            this_data_mc = this_data_mc[track_pt_mc != 0]
            this_data_pd = this_data_pd[track_pt_pd != 0]

            # Take exponential if this is pT
            if i == 0:
                this_data_mc = np.exp(this_data_mc)
                this_data_pd = np.exp(this_data_pd)

            # Set weights
            this_plot_source_start_weights = rvl_source_start_weights
            this_plot_source_end_weights = rvl_source_end_weights
            this_plot_target_start_weights = rvl_target_start_weights

        # Otherwise, this is Ht
        else:

            # Pull pT for the tracks
            this_data = track_data[:, 0, :]

            # Take exponential and re-zero padded entries
            this_data = np.exp(this_data)
            this_data[this_data == 1] = 0

            # Calculate Ht for the tracks
            this_data_mc = np.sum(this_data[labels == 0, :], axis=1)
            this_data_pd = np.sum(this_data[labels == 1, :], axis=1)

            # Set weights
            this_plot_source_start_weights = source_start_weights
            this_plot_source_end_weights = source_end_weights
            this_plot_target_start_weights = target_start_weights

        # Make reweighting plot
        fig = plot_reweighting(
            this_data_mc,
            this_data_pd,
            this_plot_source_start_weights,
            source_weight_end=this_plot_source_end_weights,
            target_weight=this_plot_target_start_weights,
            bins=element["bins"],
            xlabel=element["xlabel"],
            linear_scale=element["linear_scale"],
            ylim=element["ylim"],
            rlim=element["rlim"],
            **kwargs,
        )

        # Save histogram
        hist_path = f'{save_location}/{element["key"]}.png'
        fig.savefig(hist_path, dpi=300)
        if display:
            plt.show()
        plt.close()
        return_dict[element["key"]] = hist_path

    return return_dict


def add_ratios(fig):
    """add_ratios - This function adds ratio pads to a given matplotlib figure.

    Arguments:
    fig - matplotlib axis to add ratio pads to

    Returns:
    ax - main matplotlib axis
    axr - ratio matplotlib axis
    """

    this_grid = gs.GridSpec(2, 1, figure=fig, height_ratios=(7, 2), hspace=0.0)

    axr = fig.add_subplot(this_grid[1, 0])
    ax = fig.add_subplot(this_grid[0, 0])

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
    textstr = "\n".join(
        (
            r"$\mathrm{Entries}=%d$" % (entries,),
            r"$\mathrm{Mean}=%.4f$" % (mean,),
            r"$\mathrm{Std}=%.4f$" % (std,),
        )
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )


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
    names=None,
):
    """plot_reweighting - This function will plot the quality of the reweighting
    along a given dimension. Inputs are the original data, the starting weights, the
    ending weights, and the target data.
    Optional is the bins to be used in the plot, and the x-axis label.

    Arguments:
    source_data - numpy array of original data
    target_data - numpy array of target data
    source_weight_start - numpy array of starting weights for source, required
    source_weights_end - numpy array of ending weights for source.
        Optional if there are none
    target_weight - numpy array of weights for target, optional
    bins - number of bins to use in the plot, if not set use mpl default w/ 150 bins
    xlabel - label to use for the x-axis
    linear_scale - if true, use linear scale for y-axis, otherwise use log scale
    ylim - if set, use this as the y-axis limits
    rlim - if set, use this as the ratio axis limits
    names - an optional tuple of strings which sets the legends names for
        source and target data

    Returns:
    fig - matplotlib figure object
    """

    # Parse names
    if names is not None:
        name1, name2 = names
    else:
        name1 = "MC"
        name2 = "PseudoData"

    fig = plt.figure()
    ax, axr = add_ratios(fig)
    n_mc, bins, patches = ax.hist(
        source_data,
        bins=bins,
        label=name1,
        density=True,
        alpha=0.5,
        weights=source_weight_start,
    )
    if source_weight_end is not None:
        n_rw, bins, patches = ax.hist(
            source_data,
            bins=bins,
            label="Reweighted",
            density=True,
            histtype="step",
            color="black",
            weights=source_weight_end,
        )
    if target_weight is None:
        target_weight = np.ones_like(target_data)
    n_pd, bins, patches = ax.hist(
        target_data,
        bins=bins,
        label=name2,
        density=True,
        alpha=0.5,
        weights=target_weight,
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_ylabel("A.U.")
    if linear_scale:
        ax.set_yscale("linear")
    else:
        ax.set_yscale("log")
    ax.legend()

    axr.hlines(1, bins[0], bins[-1], color="k", linestyle="--", alpha=0.8)
    axr.plot(bins[:-1], n_mc / n_pd, color="#1f77b4", drawstyle="steps-post")
    if source_weight_end is not None:
        axr.plot(bins[:-1], n_rw / n_pd, color="k", drawstyle="steps-post")
    axr.set_ylabel(f"{name1}/{name2}")
    axr.set_xlim(ax.get_xlim())
    axr.set_xlabel(xlabel)
    if rlim is not None:
        axr.set_ylim(rlim)

    return fig


def construct_hist_and_error(
    source, source_weight, target, target_weight, bins, bs_weights
):
    """construct_hist_and_error - This function will construct a histogram comparing
    source and target distributions. It will also quantify the raw statistical
    uncertainty in the comparison, by looking at the sum of the weights squared
    in each bin, for both the source and target distributions.
    Finally it will also calculate the error on the source
    distribution from some stochastic uncertainty using the "bs_weights" argument,
    which is a list of weights obtained via some ensembling or bootstrap.

    Arguments:
    source - numpy array of source data (N events,)
    source_weight - numpy array of source weights (N events,)
    target - numpy array of target data (N events,)
    target_weight - numpy array of target weights (N events,)
    bins - numpy array of bins to use in histogram
    bs_weights - list of numpy arrays of weights to use in error calculation

    Returns:
    nom - numpy array of source histogram
    target - numpy array of target histogram
    source_stat - numpy array of the variance of the source histogram due to
        statistical uncertainty
    target_stat - numpy array of the variance of the target histogram due to
        statistical uncertainty
    mbias - numpy array of the "method bias" uncertainty in the source histogram
    var - numpy array of the variance of the source histogram due to some
        stochastic uncertainty assessed through bootstrapping / ensembling
    """

    # Construct source and target histograms
    nom, _ = np.histogram(source, bins=bins, weights=source_weight, density=False)
    tar, _ = np.histogram(target, bins=bins, weights=target_weight, density=False)
    print("Nominal histogram: ", nom)

    # Normalize target histogram to source
    norm_factor = np.sum(nom) / np.sum(tar)
    tar = tar * norm_factor
    print("Target histogram: ", tar)

    # Find sum of squared weights in each bin
    source_sumw2 = np.histogram(source, bins=bins, weights=source_weight**2)[0]
    target_sumw2 = np.histogram(target, bins=bins, weights=target_weight**2)[0] * norm_factor

    # Calculate method bias
    mbias = (nom - tar) ** 2

    # Calculate stochastic uncertainty from bs_weights
    var_hists = []
    for i, bs in enumerate(bs_weights):
        varHist, _ = np.histogram(source, bins=bins, weights=bs, density=False)
        norm_factor = np.sum(nom) / np.sum(varHist)
        varHist = varHist * norm_factor
        print(f"Bootstrap {i} histogram: ", varHist)
        var_hists.append(varHist)
    var = np.var(var_hists, axis=0)

    return nom, tar, source_sumw2, target_sumw2, mbias, var


def unfold_performance_plot(
    source,
    source_weight,
    target,
    target_weight,
    bs_weights,
    plot_params={"color": "blue", "linear_scale": False, "xlabel": "Obs", "bins": None},
    err_multiple=1.0,
):
    """unfold_performance_plot - This function will generate a plot showing the
    performance of a unbinned unfolding (multifold or omnifold) in a given dimension.
    It accepts the source and target distributions, together with the weights applied
    to each, as well as the bootstrap weights used to estimate some uncertainties
    on the unfolding.

    Arguments:
    source - numpy array of source data (N events,)
    source_weight - numpy array of source weights (N events,)
    target - numpy array of target data (N events,)
    target_weight - numpy array of target weights (N events,)
    bins - numpy array of bins to use in histogram
    bs_weights - list of numpy arrays of weights to use in error calculation
    name - the name of the unfolding method to put in the legend
    plot_params - dictionary of parameters for the plot, containing:
        'color' (string) - the color of the markers for the unfolded data
        'name' (string) - the name of the unfolding method to put in the legend
        'linear_scale' (bool) - whether to use a linear scale for the y-axis,
        'xlabel' (string) - the label for the x-axis
        'bins' (array) - numpy array that gives the binning for the histograms.
    err_multiple - multiple to scale the error bars by

    Returns:
    fig - matplotlib figure object
    """

    # Construct histograms and errors
    bins = plot_params["bins"]
    nom, tar, source_stat_var, target_stat_var, mbias_var, nn_var = construct_hist_and_error(
        source, source_weight, target, target_weight, bins, bs_weights
    )

    # Calculate total variance
    total_var = source_stat_var + nn_var + mbias_var

    # Plot
    bin_centers = (bins[1:] + bins[:-1]) / 2
    fig, (ax, rax, vax) = plt.subplots(
        3, 1, figsize=(6, 6.8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    plt.subplots_adjust(hspace=0, top=0.95)

    # Densities
    plot_tar = np.append(tar, tar[-1])
    plot_target_stat = np.sqrt(np.append(target_stat_var, target_stat_var[-1]))
    ax.plot(bins, plot_tar, "--", label="Target", color="black", drawstyle="steps-post")
    ax.fill_between(
        bins,
        plot_tar - plot_target_stat,
        plot_tar + plot_target_stat,
        step="post",
        color="gray",
        alpha=0.3,
        label="Target stat. unc.",
    )
    ax.errorbar(
        bin_centers,
        nom,
        yerr=np.sqrt(total_var) * err_multiple,
        fmt=".",
        label=plot_params["name"],
        color=plot_params["color"],
    )
    if not plot_params["linear_scale"]:
        ax.set_yscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Normalized counts")
    ax.legend()

    # Ratios
    ratio = nom / tar
    ratio_err = ratio * np.sqrt(total_var / nom**2 + source_stat_var / tar**2)
    rax.axhline(1, color="black", linestyle="--")
    rax.errorbar(
        bin_centers, ratio, yerr=ratio_err, fmt=".", color=plot_params["color"]
    )
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.75, 1.0, 1.25])
    rax.set_ylabel("Ratio to target")
    rax.tick_params(axis="x", direction="in", bottom=True, top=False)

    # Uncertainties
    rel_source_stat_err = np.sqrt(source_stat_var) / nom
    rel_nn_err = np.sqrt(nn_var) / nom
    rel_mbias_err = np.sqrt(mbias_var) / tar
    rel_total_err = np.sqrt(rel_source_stat_err**2 + rel_nn_err**2 + rel_mbias_err**2)
    plot_stat_err = np.append(rel_source_stat_err, rel_source_stat_err[-1])
    plot_nn_err = np.append(rel_nn_err, rel_nn_err[-1])
    plot_mbias_err = np.append(rel_mbias_err, rel_mbias_err[-1])
    plot_total_err = np.append(rel_total_err, rel_total_err[-1])
    vax.plot(
        bins,
        plot_total_err,
        "--",
        color="black",
        label="Total unc.",
        drawstyle="steps-post",
    )
    vax.fill_between(
        bins, 0, plot_total_err, step="post", color="gray", alpha=0.3
    )
    vax.plot(bins, plot_nn_err, "-", color="blue", label="NN Init", drawstyle="steps-post")
    vax.plot(bins, plot_mbias_err, "-", color="red", label="Method bias", drawstyle="steps-post")
    vax.plot(bins, plot_stat_err, "-", color="green", label="MC stat", drawstyle="steps-post")
    vax.plot()
    vax.set_ylim(0, 0.2)
    vax.set_xlabel(plot_params["xlabel"])
    vax.set_ylabel("Uncertainty")
    vax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4)

    return fig


def ibu_performance_plot(obs_dict, target, target_weight, plot_params):
    """ibu_performance_plot - This function produces a plot of the performance of the
    IBU unfolding method in a given observable. It will show the target distribution,
    the unfolded distribution, and the bias of the unfolding.
    (Variance to be added, when I know what exactly that is...)

    Arguments:
    obs_dict (dictionary) - A dictionary containing all of the information about the
        IBU unfolding in this dimension
    target (numpy array) - The target distribution
    target_weight (numpy array) - The weights for the target distribution
    plot_params (dictionary) - A dictionary of parameters for the plot, containing:
        'color' (string) - the color of the markers for the unfolded data
        'name' (string) - the name of the unfolding method to put in the legend
        'linear_scale' (bool) - whether to use a linear scale for the y-axis,
        'xlabel' (string) - the label for the x-axis

    Note bins are taken from the IBU dictionary in this function

    Returns:
    fig - matplotlib figure object
    """

    bins = np.array(obs_dict["bins"])
    bin_centers = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    # Construct target histogram
    target_hist, _ = np.histogram(
        target, bins=bins, weights=target_weight, density=True
    )

    # Repeat last bin of target for plotting
    plot_tar = np.append(target_hist, target_hist[-1])

    # Normalize IBU histogram
    ibu_hist = np.array(obs_dict["ibu_bin_counts"])
    ibu_hist = ibu_hist / bin_widths / np.sum(ibu_hist)

    # Evaluate bias squared
    bias2 = (ibu_hist - target_hist) ** 2

    # Plot
    fig, (ax, rax, vax) = plt.subplots(
        3, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    plt.subplots_adjust(hspace=0)

    ax.plot(bins, plot_tar, "--", label="Target", color="black", drawstyle="steps-post")
    ax.plot(
        bin_centers,
        ibu_hist,
        ".",
        label=plot_params["name"],
        color=plot_params["color"],
    )
    if not plot_params["linear_scale"]:
        ax.set_yscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Normalized counts")
    ax.legend()

    rax.axhline(1, color="black", linestyle="--")
    rax.plot(bin_centers, ibu_hist / target_hist, ".", color=plot_params["color"])
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.75, 1.0, 1.25])
    rax.set_ylabel("Ratio to target")
    rax.tick_params(axis="x", direction="in", bottom=True, top=False)

    vax.plot(bin_centers, bias2 / target_hist**2, ".", color="blue", label="Bias^2")
    vax.set_yscale("log")
    vax.set_ylim(1e-6, 10)
    vax.set_xlabel(plot_params["xlabel"])
    vax.set_ylabel("Error / Target^2")
    vax.legend()

    return fig
