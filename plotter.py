"""plotter.py - This module contains the Plotter class which produces all
basic reweighting plots. More advanced plots that contain uncertainty
information and comparisons are handled by classes that inherit from here.

Class also has a method that calculates the W1 distance between the source
and target distributions, using the WassersteinMetric class.

At the moment this class only uses pre-computed / track observables.
Hopefully I can also include code for calculating new observables with fastjet soon.

Author: Kevin Greif
Last updated 03/24/2025
python3
"""

import os
import subprocess
import pathlib
import yaml
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import wasserstein_metric
import utils.data_utils as du


class Plotter:
    """Plotter - This class produces all basic reweighting plots.
    It uses uproot to read root files into awkward arrays and matplotlib
    to build histograms and make plots. Can be used within the unfolding
    code to make reweighting plots within iterations, or in standalone
    plotting scripts. Can also set "verbosity" which controls how many
    pre-computed variable plots the class will produce.

    This class also computes the wasserstein distances
    that are used for early stopping and checkpointing in the unfolding.
    It does this for all plots for which w1_eval is set to True in the config,
    regardless of the verbosity level.
    """

    def __init__(
        self,
        source_path,
        target_path,
        store,
        use_truth=False,
        labels=None,
        verbosity=0,
        use_pdf=False,
        max_events=1e6,
        root_files=None,
    ):
        """Initializes the Plotter class with the source and target paths
        for the root files. The verbosity level controls how many plots are
        produced. The use_pdf flag controls whether to use pdf files for
        plotting or not.

        Args:
            source_path (str): Path to the source root file.
            target_path (str): Path to the target root file.
            store (str): Path to the directory for saving plots.
            use_truth (bool): Flag to use truth information. Default is False.
            labels (tuple of str): Tuple of labels for the source and target files.
                Default is None.
            verbosity (int): Verbosity level for plotting. Default is 0,
                which plots to sum_pT_tracks and Ntracks only. See plots_config.yml
                for details.
            use_pdf (bool): Flag to use pdf files for plotting. Default is False.
            max_events (int): Maximum number of events to include in plotting.
                Default is 5e6.
            root_files (list of str): List of root files to load histograms of
                fastjet observables from. In order [source_start, source_end, target].
                Defaults to None, in which case there should be no fastjet observables
                in the config.
        """

        # Store instance variables
        self.source_path = source_path
        self.target_path = target_path
        self.source_tree = uproot.open(source_path)["OmniTree"]
        self.target_tree = uproot.open(target_path)["OmniTree"]
        self.store = pathlib.Path(store)
        self.use_truth = use_truth
        self.labels = labels
        if labels is None:
            self.labels = ("Source", "Target")
        self.verbosity = verbosity
        self.use_pdf = use_pdf
        self.max_events = max_events
        self.root_files = root_files

        # Find number of events to use in plotting
        self.source_events = self.source_tree.num_entries
        if self.source_events > max_events:
            self.source_events = max_events
        self.target_events = self.target_tree.num_entries
        if self.target_events > max_events:
            self.target_events = max_events

        # Get appropriate pass190 flags, note we do not truncate
        pull_key = "truth_pass190" if use_truth else "pass190"
        self.source_pass190 = ak.to_numpy(
            self.source_tree[pull_key].array(entry_stop=self.source_events)
        )
        self.target_pass190 = ak.to_numpy(
            self.target_tree[pull_key].array(entry_stop=self.target_events)
        )

        # Get config from yaml
        with open("./utils/plots_config.yml", "r") as stream:
            config = yaml.safe_load(stream)

        # Loop through plots in config, keep only those that have verbosity less
        # than or equal to configured level
        self.plots = [
            config["plots"][plot]
            for plot in config["plots"]
            if config["plots"][plot]["verbosity_level"] <= verbosity
        ]

        # Detect whether we have any track level or fastjet observables
        self.track_level = False
        self.fastjet = False
        for plot in self.plots:
            if plot["type"] == "track":
                self.track_level = True
            if plot["type"] == "fastjet":
                self.fastjet = True

        # Get keys of plots that have w1_eval set to True for W1 computation
        self.w1_keys = [
            config["plots"][plot]["key"]
            for plot in config["plots"]
            if config["plots"][plot]["w1_eval"] is True
        ]

    def plot(self, source_start, source_end, target, recalculate=False, **kwargs):
        """plot - This function produces the reweighting plots given
        vectors of weights for the source and target data, and saves
        them to the proper directory. It returns a dictionary with the format
        {plot_name: path_to_file} for each plot produced.

        Args:
            source_start (np.array or str): Array of source starting weights
                or title of branch
            source_end (np.array or str): Array of source ending weights or
                title of branch
            target (np.array or str): Array of target weights, or title of branch
            recalculate (bool): If True, will recalculate fastjet observables
                even if the root files already exist.

        Returns:
            dict: Dictionary with the format {plot_name: path_to_file} for
                each plot produced
        """

        # Calculate fastjet observables if needed
        if self.fastjet:
            assert self.root_files is not None

            # Compile the fastjet package
            make_process = subprocess.run(
                ["make"], cwd="./fastjet/", capture_output=True
            )
            if make_process.returncode != 0:
                print("Error compiling fastjet package. Please check your setup.")
                print(make_process.stderr)

            # Run for each root file that does not exist
            weights = [source_start, source_end, target]
            for i, (use_weights, file) in enumerate(zip(weights, self.root_files)):
                if recalculate and pathlib.Path(file).exists():
                    os.remove(file)
                if not pathlib.Path(file).exists():
                    self._run_fastjet(i, use_weights, file)

        # Get event level weights
        gw_kwargs = {k: v for k, v in kwargs.items() if k == "use_train"}
        source_start = self._get_weights(source_start, **gw_kwargs)
        source_end = self._get_weights(source_end, **gw_kwargs)
        target = self._get_weights(target, is_target=True, **gw_kwargs)

        # Filter weights by pass190 flags
        source_start = source_start[self.source_pass190 == 1]
        source_end = source_end[self.source_pass190 == 1]
        target = target[self.target_pass190 == 1]

        # If we have track level observables, need to repeat the weights
        # for each track in the event
        if self.track_level:
            source_start_trk = self._get_track_weights(source_start)
            source_end_trk = self._get_track_weights(source_end)
            target_trk = self._get_track_weights(target, is_target=True)

        # Loop through plots and make histograms
        return_dict = {}
        for plot in self.plots:

            # Get histograms
            use_weights = [source_start, source_end, target]
            if plot["type"] == "track":
                # For track level observables, use the track weights
                use_weights = [source_start_trk, source_end_trk, target_trk]
            histograms, bins = self._get_histograms(plot, weights=use_weights)
            source_start_hist, source_end_hist, target_hist = histograms

            # Calculate ratios
            start_ratio = source_start_hist / target_hist
            end_ratio = source_end_hist / target_hist

            # Duplicate last bins for plotting
            source_start_hist = np.append(source_start_hist, source_start_hist[-1])
            source_end_hist = np.append(source_end_hist, source_end_hist[-1])
            target_hist = np.append(target_hist, target_hist[-1])
            start_ratio = np.append(start_ratio, start_ratio[-1])
            end_ratio = np.append(end_ratio, end_ratio[-1])

            # Make plot
            fig = plt.figure()
            ax, axr = self._add_ratios(fig)
            ax.plot(
                bins,
                source_start_hist,
                drawstyle="steps-post",
                label=self.labels[0],
                alpha=0.5,
            )
            ax.fill_between(
                bins,
                0,
                source_start_hist,
                step="post",
                alpha=0.5,
                color="#1f77b4",
            )
            ax.plot(
                bins,
                source_end_hist,
                drawstyle="steps-post",
                label="Reweighted",
                color="black",
            )
            ax.plot(
                bins,
                target_hist,
                drawstyle="steps-post",
                label=self.labels[1],
            )
            if plot["ylim"] is not None:
                ax.set_ylim(plot["ylim"])
            ax.set_ylabel(plot["ylabel"])
            if not plot["linear_yscale"]:
                ax.set_yscale("log")
            if plot["log_xscale"]:
                ax.set_xscale("log")
            ax.set_xticks([])
            ax.legend(loc="upper right", fontsize=8)
            axr.hlines(1, bins[0], bins[-1], color="black", linestyle="--", alpha=0.8)
            axr.plot(
                bins,
                start_ratio,
                drawstyle="steps-post",
                alpha=0.5,
            )
            axr.plot(
                bins,
                end_ratio,
                drawstyle="steps-post",
                color="black",
            )
            axr.set_xlabel(plot["xlabel"])
            if plot["log_xscale"]:
                axr.set_xscale("log")
            axr.set_ylabel("Ratio to target")
            axr.set_ylim(plot["rlim"])

            fig.tight_layout()

            # Save plot and add to return dictionary
            extension = ".pdf" if self.use_pdf else ".png"
            store_name = self.store / (plot["key"] + extension)
            fig.savefig(store_name, dpi=300)
            plt.close(fig)
            return_dict[plot["key"]] = store_name

        return return_dict

    def wasserstein_distance(self, source_start, source_end, target, **kwargs):
        """wasserstein_distance - This function computes the wasserstein
        distances for the given source and target distributions. It uses
        the WassersteinMetric class to compute the distances.

        Args:
            source_start (np.array or str): Array of source starting weights
                or title of branch
            source_end (np.array or str): Array of source ending weights or
                title of branch
            target (np.array or str): Array of target weights, or title of branch

        Returns:
            tuple (float, float): Tuple of wasserstein distances to target for
                source_start and source_end distributions
        """

        # Get event level weights
        gw_kwargs = {k: v for k, v in kwargs.items() if k == "use_train"}
        source_start = self._get_weights(source_start, **gw_kwargs)
        source_end = self._get_weights(source_end, **gw_kwargs)
        target = self._get_weights(target, is_target=True, **gw_kwargs)

        # Filter weights by pass190 flags
        source_start = source_start[self.source_pass190 == 1]
        source_end = source_end[self.source_pass190 == 1]
        target = target[self.target_pass190 == 1]

        # Get data and labels for W1 calculation
        w1_keys = du.get_w1_obs()
        source_w1_obs = du.get_observables(
            self.source_tree,
            w1_keys,
            get_truth=self.use_truth,
            stop=self.source_events,
        )
        target_w1_obs = du.get_observables(
            self.target_tree,
            w1_keys,
            get_truth=self.use_truth,
            stop=self.target_events,
        )
        w1_data = np.concatenate((source_w1_obs, target_w1_obs), axis=0)
        labels = np.concatenate(
            (np.zeros(len(source_w1_obs)), np.ones(len(target_w1_obs))), axis=0
        )

        # Concatenate weights
        start_weights = np.concatenate((source_start, target), axis=0)
        end_weights = np.concatenate((source_end, target), axis=0)

        # Calculate distances
        w1 = wasserstein_metric.WassersteinOne()
        w1.update(w1_data, start_weights, labels)
        w1_start_value = w1.compute(from_torch=False)
        w1.reset()
        w1.update(w1_data, end_weights, labels)
        w1_end_value = w1.compute(from_torch=False)
        w1.reset()

        return w1_start_value, w1_end_value

    def _get_histograms(self, plot_dict, weights=None):
        """_get_histograms - This function computes histograms for a given
        plot dictionary (observable) and vector of weights.
        It uses the key from the plot dictionary to access the data from the
        trees if the observable is precomputed or a track variable.
        If the observable is computed using fastjet it directly loads the
        histograms from the root files whose path is provided in the
        root_files argument, in order [start, end, target]. In this case the
        weights vector is not used.

        Arguments:
            plot_dict (dict): Dictionary containing the plot configuration
                including the key for the observable.
            weights (list of np.array): Weights to use for histogram
                computation, in order [source_start, source_end, target].
                If the observable is computed using fastjet, this argument
                is ignored. Default is None.
            root_files (list of str): List of root files to load histograms from,
                in order [source_start, source_end, target]. Default is None.
                Required if the observable is not present in the trees and
                was computed using fastjet.

        Returns:
            tuple: A tuple containing the histograms for the source start,
                source end, and target distributions. The histograms are
                numpy arrays representing the counts in each bin.
            np.array: The bin edges for the histograms.
        """

        # If the observable is computed using fastjet, we need to load
        # the histograms using uproot
        if plot_dict["type"] == "fastjet":

            key = plot_dict["key"]
            source_start_hist, bins = uproot.open(self.root_files[0])[key].to_numpy()
            source_end_hist, bins2 = uproot.open(self.root_files[1])[key].to_numpy()
            target_hist, bins3 = uproot.open(self.root_files[2])[key].to_numpy()
            assert np.array_equal(bins, bins2)
            assert np.array_equal(bins, bins3)

        # Else the data is loaded and binned from the trees directly
        else:

            assert len(weights) == 3

            # Get filtered data
            pull_key = (
                "truth_" + plot_dict["key"] if self.use_truth else plot_dict["key"]
            )
            source_data, target_data = self._get_data(pull_key)

            # Get bins and make histograms, selecting correct weights
            # depending on whether we are using track level or event level
            # observables
            bins = np.linspace(
                plot_dict["binlow"], plot_dict["binhigh"], plot_dict["nbins"]
            )
            source_start_hist, _ = np.histogram(
                source_data, bins=bins, weights=weights[0], density=False
            )
            source_end_hist, _ = np.histogram(
                source_data, bins=bins, weights=weights[1], density=False
            )
            target_hist, _ = np.histogram(
                target_data, bins=bins, weights=weights[2], density=False
            )

        # Normalize histograms
        source_start_hist = self._normalize_to(source_start_hist, val=1.0)
        source_end_hist = self._normalize_to(source_end_hist, val=1.0)
        target_hist = self._normalize_to(target_hist, val=1.0)

        return (source_start_hist, source_end_hist, target_hist), bins

    def _normalize_to(self, hist, val=1.0):
        """_normalize_to - This function normalizes a histogram to a given value.

        Arguments:
            hist (np.array): Histogram to normalize.
            val (float): Value to normalize the histogram to. Default is 1.0.

        Returns:
            np.array: Normalized histogram.
        """
        if np.sum(hist) == 0:
            return hist
        return hist / np.sum(hist) * val

    def _get_weights(self, weights, is_target=False, use_train=False):
        """_get_weights - This function gets a set of weights for use in
        plotting. The weights argument can be a string, either pointing to
        weights stored as a .npz file, or a branch name in the tree. It
        can also be a numpy array, in which case the function simply
        handles the truncation to the number of events used in plotting.

        Arguments:
            weights (str or np.array): Path to the weights file,
                branch name, or numpy array
            is_target (bool): If weights is a branch name, set to
                true to pull from target tree
            use_train (bool): Flag to use training weights if
                weights is a .npz file.

        Returns:
            weights (np.array): Array of weights
        """

        max_events = self.target_events if is_target else self.source_events

        # Numpy array case
        if type(weights) is np.ndarray:
            if len(weights) > max_events:
                weights = weights[:max_events]

        # Path to .npz case
        elif weights.endswith(".npz"):

            # Load weights from file
            weights = np.load(weights)
            if use_train:
                weights = weights["train"]
            else:
                weights = weights["test"]

            # Truncate if needed
            max_events = self.target_events if is_target else self.source_events
            if len(weights) > max_events:
                weights = weights[:max_events]

        # Branch name case
        else:
            if is_target:
                weights = ak.to_numpy(
                    self.target_tree[weights].array(entry_stop=self.target_events)
                )
            else:
                weights = ak.to_numpy(
                    self.source_tree[weights].array(entry_stop=self.source_events)
                )

        return weights

    def _get_track_weights(self, weights, is_target=False):
        """_get_track_weights - This function broadcasts a set of weights
        such that its shape matches track level observables, where an event
        weight is repeated for each track in a given event.
        Whether to use the source or target data as a broadcasting
        template is controlled by the "is_target" flag.

        Args:
            weights (np.array): Array of weights to broadcast
            is_target (bool): Flag to use target data as broadcasting template

        Returns:
            weights (np.array): Array of weights broadcasted to track level
        """

        # Truncate pass190 flags to number of events used in plotting
        source_pass190 = self.source_pass190[: self.source_events]
        target_pass190 = self.target_pass190[: self.target_events]

        # Get track pTs to serve as template
        pull_key = "truth_pT_tracks" if self.use_truth else "pT_tracks"
        if is_target:
            track_data = self.target_tree[pull_key].array(entry_stop=self.target_events)
            track_data = track_data[target_pass190 == 1]
        else:
            track_data = self.source_tree[pull_key].array(entry_stop=self.source_events)
            track_data = track_data[source_pass190 == 1]

        # Broadcast weights
        weights, track_data = ak.broadcast_arrays(ak.from_numpy(weights), track_data)

        return ak.to_numpy(ak.flatten(weights, axis=None))

    def _get_data(self, key):
        """_get_data - This function gets the data for a given key
        from the source and target trees. It returns the data as
        numpy arrays. The data is filtered by the pass190 flags.

        If the data is track level, it will be flattened to a 1D array
        before returning.

        Args:
            key (str): Key to get data for

        Returns:
            source_data (ak.Array): Source data
            target_data (ak.Array): Target data
        """

        source_data = self.source_tree[key].array(entry_stop=self.source_events)
        target_data = self.target_tree[key].array(entry_stop=self.target_events)

        # Filter data by pass190 flags
        source_data = source_data[self.source_pass190 == 1]
        target_data = target_data[self.target_pass190 == 1]

        # Flatten data and send to numpy
        source_data = ak.to_numpy(ak.flatten(source_data, axis=None))
        target_data = ak.to_numpy(ak.flatten(target_data, axis=None))

        return source_data, target_data

    def _add_ratios(self, fig):
        """_add_ratios - This function adds ratio pads to a given matplotlib figure.

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

    def _run_fastjet(self, index, weights, save_file):
        """_run_fastjet - This function computes fastjet observables for a given
        data set and weights, and saves them to a root file.
        Computation is done by launching a subprocess which compiles and executes
        the C++ code in the fastjet subdirectory.
        The distribution is controlled by the index argument, which is an integer
        that accepts the following settings:
        0 - Source start (source_start)
        1 - Source end (source_end)
        2 - Target (target)

        Arguments:
        index - int - The index of the distribution to compute fastjet observables for.
            This determines which ROOT file is used as input.
        weights - str - The path to the weights file or branch name to use for
            fastjet computation. This will be passed to the C++ code as an argument.
        save_file - str - The path to the root file where the fastjet observables

        No returns
        """

        # Check if the index is valid
        if index not in [0, 1, 2]:
            raise ValueError("Index must be 0, 1, or 2 for fastjet computation.")

        # Determine the commands to run based on the index
        inpath = self.source_path if index in [0, 1] else self.target_path
        inpath = "../" + inpath
        weightpath = "../" + weights if weights.endswith(".npz") else weights
        outpath = "../" + save_file
        command = [
            "./doHisto.out",
            "--file", inpath,
            "--weights", weightpath,
            "--outFile", outpath,
            "--maxEvents", str(self.max_events),
        ]
        if self.use_truth:
            command += "--truth"

        # Run fastjet compuatation
        try:
            print(f"Calculating fastjet observables with command: {command}")
            subprocess.run(command, cwd="./fastjet/", capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running fastjet computation: {e.stderr}")
            print(f"Return code: {e.returncode}")
