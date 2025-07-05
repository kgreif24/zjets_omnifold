"""plotter.py - This module contains the Plotter class which produces all
basic reweighting plots. More advanced plots that contain uncertainty
information and comparisons are handled by classes that inherit from here.

Class also has a method that calculates the W1 distance between the source
and target distributions, using the WassersteinMetric class.

Author: Kevin Greif
Last updated 06/04/2025
python3
"""

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
        ibu_bins=False,
        kinematic_region=0,
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
            ibu_bins (bool): If true use coarse IBU bins instead of fine bins
            kinematic_region (int): If set to one of the following values,
                restricts plotting to the following kinematic regions:
                0: No cuts, all events are used.
                1. High pT_Z: pT_j2 > 50 GeV, pT_ll > 350 GeV
                2. Electroweak enhanced: m_jj > 200 GeV, |dy_jj| > 2
                3. Diboson enhanced: pT_j1 > 32 GeV
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
        self.ibu_bins = ibu_bins
        self.kinematic_region = kinematic_region

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

        # Apply kinematic cuts
        if kinematic_region != 0:
            print("Applying kinematic cuts for region:", kinematic_region)
            if self.verbosity >= 3:
                print(
                    "Verbosity is greater than 3, please ensure fastjet"
                    " observables are calculated in the limited phase space!"
                )
            self.apply_kinematic_cuts(kinematic_region)

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
                assert self.root_files is not None

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
            source_start_hist, bins = self._get_histogram(
                plot,
                weights=source_start_trk if plot["type"] == "track" else source_start,
                density=True,
                root_index=0,
            )
            source_end_hist, _ = self._get_histogram(
                plot,
                weights=source_end_trk if plot["type"] == "track" else source_end,
                density=True,
                root_index=1,
            )
            target_hist, _ = self._get_histogram(
                plot,
                weights=target_trk if plot["type"] == "track" else target,
                density=True,
                is_target=True,
                root_index=2,
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
                target_hist,
                drawstyle="steps-post",
                label=self.labels[1],
            )
            ax.plot(
                bins,
                source_end_hist,
                drawstyle="steps-post",
                label="Reweighted",
                color="black",
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
        source_w1_obs = np.stack([self._get_data(key) for key in w1_keys], axis=-1)
        target_w1_obs = np.stack(
            [self._get_data(key, is_target=True) for key in w1_keys],
            axis=-1,
        )
        w1_data = np.concatenate((source_w1_obs, target_w1_obs), axis=0)
        labels = np.concatenate(
            (np.zeros(len(source_w1_obs)), np.ones(len(target_w1_obs))), axis=0
        )

        # Concatenate weights
        start_weights = np.concatenate((source_start, target), axis=0)
        end_weights = np.concatenate((source_end, target), axis=0)

        # Calculate distances
        w1 = wasserstein_metric.WassersteinOne(sync_on_compute=False)
        w1.update(w1_data, start_weights, labels)
        w1_start_value = w1.compute(from_torch=False)
        w1.reset()
        w1.update(w1_data, end_weights, labels)
        w1_end_value = w1.compute(from_torch=False)
        w1.reset()

        return w1_start_value, w1_end_value

    def _get_histogram(
        self,
        plot_dict,
        weights=None,
        density=False,
        root_index=0,
        return_variance=False,
        **kwargs,
    ):
        """_get_histogram - This function computes a histogram for a given
        plot dictionary (observable) and vector of weights.
        It uses the key from the plot dictionary to access the data from the
        trees if the observable is precomputed or a track variable.
        If the observable is computed using fastjet it directly loads the
        histograms from the root files whose path is provided in the
        root_files argument. In this case the weights vector is not used.

        Some fastjet observables (Lund planes) are 2D histograms. In the case
        that these histograms are requested, the function returns a tuple
        of np.arrays (binsx, binsy) in place of a single np.array for the
        bins.

        Arguments:
            plot_dict (dict): Dictionary containing the plot configuration
                including the key for the observable.
            weights (np.array): Weights to use for histogram in building
                the histogram
            density (bool): If True, will normalize the histogram to
                form a probability density function (PDF). Default is False.
            return_variance (bool): If True, will return the variance of the
                histogram instead of the histogram itself.
                Default is False, note this only works for
                fastjet observables.
            root_index (int): Index of the root file to use for fastjet
                observables. Default is 0, or the first root file provided.

        Returns:
            tuple: A tuple containing the histograms for the source start,
                source end, and target distributions. The histograms are
                numpy arrays representing the counts in each bin.
            np.array: The bin edges for the histograms.
        """

        # If the observable is computed using fastjet, we need to load
        # the histograms using uproot
        if plot_dict["type"] == "fastjet":

            assert root_index < len(self.root_files)

            tobject = uproot.open(self.root_files[root_index])[plot_dict["key"]]
            if "TH2" in tobject.classname:
                hist, binsx, binsy = tobject.to_numpy()
                bins = (binsx, binsy)
            else:
                hist, bins = tobject.to_numpy()

            if return_variance:
                hist = tobject.variances()

        # Else the data is loaded and binned from the trees directly
        else:
            assert not return_variance

            # Get filtered data
            data = self._get_data(plot_dict["key"], **kwargs)

            # Make histogram
            if self.ibu_bins:
                bins = np.array(plot_dict["ibubins"])
                if self.kinematic_region != 0 and "region_bins" in plot_dict:
                    if str(self.kinematic_region) in plot_dict["region_bins"]:
                        bins = np.array(
                            plot_dict["region_bins"][str(self.kinematic_region)]
                        )
            else:
                bins = np.linspace(
                    plot_dict["binlow"], plot_dict["binhigh"], plot_dict["nbins"]
                )
            hist, bins = np.histogram(data, bins=bins, weights=weights, density=False)

        # Normalize histogram if desired
        if density:
            assert not return_variance
            hist = self._normalize_to(hist, val=1.0)

        return hist, bins

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

        Note if the weights argument points to a .npz file, the weights
        are multiplied by the correct weights in the ROOT tree. If
        use_truth is True, this is the "weight_mc" branch, otherwise
        it is the "weight" branch.

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

            # Load ROOT weights
            key = "weight_mc" if self.use_truth else "weight"
            if is_target:
                root_weights = ak.to_numpy(
                    self.target_tree[key].array(entry_stop=self.target_events)
                )
            else:
                root_weights = ak.to_numpy(
                    self.source_tree[key].array(entry_stop=self.source_events)
                )

            # Load weights from file
            weights = np.load(weights)
            if "nominal-ensemble-central" in weights.files:
                assert not use_train
                weights = weights["nominal-ensemble-central"]
            elif use_train:
                weights = weights["train"]
            else:
                weights = weights["test"]

            # Truncate if needed
            max_events = self.target_events if is_target else self.source_events
            if len(weights) > max_events:
                weights = weights[:max_events]

            # Multiply by ROOT weights
            weights *= root_weights

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

    def _get_data(self, key, is_target=False):
        """_get_data - This function gets the data for a given key
        from the source or target tree. It returns the data as
        numpy arrays. The data is filtered by the pass190 flags.

        If the data is track level, it will be flattened to a 1D array
        before returning.

        Args:
            key (str): Key to get data for
            is_target (bool): If true, pull from the target tree instead of source

        Returns:
            (ak.Array): The data as an awkward array
        """

        if self.use_truth:
            key = "truth_" + key

        if is_target:
            data = self.target_tree[key].array(entry_stop=self.target_events)
            data = data[self.target_pass190 == 1]
        else:
            data = self.source_tree[key].array(entry_stop=self.source_events)
            data = data[self.source_pass190 == 1]

        return ak.to_numpy(ak.flatten(data, axis=None))

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

    def apply_kinematic_cuts(self, region):
        """apply_kinematic_cuts - Applies kinematic cuts to the source_pass190 and
        target_pass190 vectors. Can use this to restrict the plotting to a
        specific phase space.

        Args:
            region (int): The kinematic region to apply cuts for.
                0: No cuts, all events are used.
                1: High pT_Z: pT_j2 > 50 GeV, pT_ll > 350 GeV
                2: Electroweak enhanced: m_jj > 200 GeV, |dy_jj| > 2
                3: Diboson enhanced: pT_j1 > 32 GeV

        Returns:
            None
        """

        # Get kinematic region masks
        source_mask = self.get_kinematic_region(
            self.source_tree, region, evts=self.source_events, use_truth=self.use_truth
        )
        target_mask = self.get_kinematic_region(
            self.target_tree, region, evts=self.target_events, use_truth=self.use_truth
        )

        # Apply masks to pass190 flags
        self.source_pass190 = np.logical_and(self.source_pass190, source_mask)
        self.target_pass190 = np.logical_and(self.target_pass190, target_mask)

    def get_kinematic_region(self, tree, region, evts=99999999, use_truth=True):
        """get_kinematic_region - This function returns a boolean mask for the
        given tree that selects events in the given kinematic region.

        Arguments:
            tree (uproot.TTree): The tree to which the cuts should be applied.
            region (int): The kinematic region to apply cuts for.
                0: No cuts, all events are used.
                1: High pT_Z: pT_j2 > 50 GeV, pT_ll > 350 GeV
                2: Electroweak enhanced: m_jj > 200 GeV, |dy_jj| > 2
                3: Diboson enhanced: pT_j1 > 32 GeV
            evts (int): The maximum number of events to pull from the tree.
            use_truth (bool): If true, use truth information to apply cuts.

        Returns:
            np.array: Boolean mask for the events in the given kinematic region.
        """

        prekey = "truth_" if use_truth else ""
        N = tree.num_entries
        if N > evts:
            N = evts

        if region == 0:
            # No cuts, all events are used
            return np.ones(N, dtype=bool)
        elif region == 1:
            pT_j2 = ak.to_numpy(tree[prekey + "pT_trackj2"].array(entry_stop=N))
            pT_ll = ak.to_numpy(tree[prekey + "pT_ll"].array(entry_stop=N))
            return np.logical_and(pT_j2 > 50, pT_ll > 350)
        elif region == 2:
            m_jj, dy_jj = du.get_jj_info(tree, use_truth=use_truth, stop=N)
            return np.logical_and(m_jj > 200, np.abs(dy_jj) > 2)
        elif region == 3:
            m_j1 = ak.to_numpy(tree[prekey + "pT_trackj1"].array(entry_stop=N))
            return m_j1 > 32
        else:
            raise ValueError(
                f"Invalid kinematic region {region}. Must be one of 0, 1, 2, or 3."
            )
