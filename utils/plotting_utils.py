""" plotting_utils - This file contains some helper functions for generating plots,
usually histograms showing the quality of a given reweighting.

Author: Kevin Greif
Last updated 01/25/2024
python3
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np


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

    # Calculate method bias
    mbias = (nom - tar) ** 2

    # Calculate stochastic uncertainty from bs_weights
    var_hists = []
    for i, bs in enumerate(bs_weights):
        varHist, _ = np.histogram(source, bins=bins, weights=bs, density=False)
        norm_factor = np.sum(nom) / np.sum(varHist)
        varHist = varHist * norm_factor
        var_hists.append(varHist)
    var = np.var(var_hists, axis=0)

    return nom, tar, source_sumw2, mbias, var


def unfold_performance_plot(
    source,
    source_weight,
    target,
    target_weight,
    bs_weights,
    plot_params={"color": "blue", "linear_yscale": False, "xlabel": "Obs", "bins": None},
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
        'linear_yscale' (bool) - whether to use a linear scale for the y-axis,
        'xlabel' (string) - the label for the x-axis
        'bins' (array) - numpy array that gives the binning for the histograms.

    Returns:
    fig - matplotlib figure object
    """

    # Construct histograms and errors
    bins = np.array(plot_params["ibubins"])
    nom, tar, source_stat_var, mbias_var, nn_var = construct_hist_and_error(
        source, source_weight, target, target_weight, bins, bs_weights
    )

    # Calculate total variance
    total_var = source_stat_var + nn_var

    # Plot
    bin_centers = (bins[1:] + bins[:-1]) / 2
    fig, (ax, rax, vax) = plt.subplots(
        3, 1, figsize=(6, 6.8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    plt.subplots_adjust(hspace=0, top=0.95)

    # Densities
    plot_tar = np.append(tar, tar[-1])
    ax.plot(bins, plot_tar, "--", label="Target", color="black", drawstyle="steps-post")
    ax.errorbar(
        bin_centers,
        nom,
        yerr=np.sqrt(total_var),
        fmt="o",
        label=plot_params["name"],
        color=plot_params["color"],
    )
    if not plot_params["linear_yscale"]:
        ax.set_yscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Normalized counts")
    ax.legend()

    # Ratios
    ratio = nom / tar
    ratio_err = ratio * np.sqrt(total_var / nom**2 + source_stat_var / tar**2)
    rax.axhline(1, color="black", linestyle="--")
    rax.errorbar(
        bin_centers, ratio, yerr=ratio_err, fmt="o", color=plot_params["color"]
    )
    rax.set_ylim(0.85, 1.15)
    rax.set_yticks([0.9, 1.0, 1.1])
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

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, top=0.95)

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
        'linear_yscale' (bool) - whether to use a linear scale for the y-axis,
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

    # Evaluate bias
    mbias_err = np.abs(ibu_hist - target_hist) / target_hist
    plot_mbias_err = np.append(mbias_err, mbias_err[-1])

    # Plot
    fig, (ax, rax, vax) = plt.subplots(
        3, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    plt.subplots_adjust(hspace=0, top=0.95)

    ax.plot(bins, plot_tar, "--", label="Target", color="black", drawstyle="steps-post")
    ax.plot(
        bin_centers,
        ibu_hist,
        ".",
        label=plot_params["name"],
        color=plot_params["color"],
    )
    if not plot_params["linear_yscale"]:
        ax.set_yscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Normalized counts")
    ax.legend()

    rax.axhline(1, color="black", linestyle="--")
    rax.plot(bin_centers, ibu_hist / target_hist, ".", color=plot_params["color"])
    rax.set_ylim(0.85, 1.15)
    rax.set_yticks([0.9, 1.0, 1.1])
    rax.set_ylabel("Ratio to target")
    rax.tick_params(axis="x", direction="in", bottom=True, top=False)

    vax.plot(bins, plot_mbias_err, "-", color="red", label="Method bias", drawstyle="steps-post")
    vax.set_ylim(0.0, 0.2)
    vax.set_xlabel(plot_params["xlabel"])
    vax.set_ylabel("Uncertainty")
    vax.legend()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, top=0.95)

    return fig
