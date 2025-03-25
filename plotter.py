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

import yaml
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import wasserstein_metric
import utils.data_utils as du


class Plotter:
    """ Plotter - This class produces all basic reweighting plots.
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
        """

        # Store instance variables
        self.source_tree = uproot.open(source_path)["OmniTree"]
        self.target_tree = uproot.open(target_path)["OmniTree"]
        self.store = store
        self.use_truth = use_truth
        self.labels = labels
        if labels is None:
            self.labels = ("Source", "Target")
        self.verbosity = verbosity
        self.use_pdf = use_pdf
        self.max_events = max_events

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

        # Detect whether we have any track level observables
        self.track_level = False
        for plot in self.plots:
            if plot["type"] == "track":
                self.track_level = True
                break

        # Get keys of plots that have w1_eval set to True for W1 computation
        self.w1_keys = [
            config["plots"][plot]["key"]
            for plot in config["plots"]
            if config["plots"][plot]["w1_eval"] is True
        ]

    def plot(self, source_start, source_end, target, **kwargs):
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

        Returns:
            dict: Dictionary with the format {plot_name: path_to_file} for
                each plot produced
        """

        # Get event level weights
        source_start = self._get_weights(source_start, **kwargs)
        source_end = self._get_weights(source_end, **kwargs)
        target = self._get_weights(target, is_target=True, **kwargs)

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

            # Get filtered data
            pull_key = "truth_" + plot["key"] if self.use_truth else plot["key"]
            source_data, target_data = self._get_data(pull_key)

            # Get bins and make histograms, selecting correct weights
            # depending on whether we are using track level or event level
            # observables
            bins = np.linspace(plot["binlow"], plot["binhigh"], plot["nbins"])
            source_start_use = (
                source_start_trk if plot["type"] == "track" else source_start
            )
            source_start_hist, _ = np.histogram(
                source_data, bins=bins, weights=source_start_use, density=True
            )
            source_end_use = source_end_trk if plot["type"] == "track" else source_end
            source_end_hist, _ = np.histogram(
                source_data, bins=bins, weights=source_end_use, density=True
            )
            target_use = target_trk if plot["type"] == "track" else target
            target_hist, _ = np.histogram(
                target_data, bins=bins, weights=target_use, density=True
            )

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
            ax.set_xticks([])
            ax.set_ylabel(plot["ylabel"])
            if not plot["linear_scale"]:
                ax.set_yscale("log")
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
        source_start = self._get_weights(source_start, **kwargs)
        source_end = self._get_weights(source_end, **kwargs)
        target = self._get_weights(target, is_target=True, **kwargs)

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

    def _get_weights(self, weights, is_target=False, use_train=False):
        """ _get_weights - This function gets a set of weights for use in
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
        source_pass190 = self.source_pass190[:self.source_events]
        target_pass190 = self.target_pass190[:self.target_events]

        # Get track pTs to serve as template
        pull_key = "truth_pT_tracks" if self.use_truth else "pT_tracks"
        if is_target:
            track_data = self.target_tree[pull_key].array(
                entry_stop=self.target_events
            )
            track_data = track_data[target_pass190 == 1]
        else:
            track_data = self.source_tree[pull_key].array(
                entry_stop=self.source_events
            )
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
