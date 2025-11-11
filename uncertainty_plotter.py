"""uncertainty_plotter.py - This module provides a subclass of the Plotter
class that builds uncertainty plots. It will only make plots comparing
truth pseudodata to re-weighted truth level MC, as uncertainties are only
defined in this context.

Author: Kevin Greif
Last updated 03.28.2025
python3
"""

import tqdm
import numpy as np
import uproot
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import plotter


class UncertaintyPlotter(plotter.Plotter):
    """
    UncertaintyPlotter is a subclass of Plotter that is specialized for
    building uncertainty plots. It will only make plots comparing truth
    pseudodata to re-weighted truth level MC, as uncertainties are only
    defined in this context.
    """

    def __init__(
        self,
        source_path,
        target_path,
        hv_path,
        data_path,
        store,
        root_files=None,
        target2_path=None,
        data_comparison_mode=False,
        **kwargs,
    ):
        """
        Initialize the UncertaintyPlotter class by calling the parent class's
        constructor.

        Fixes the use_truth argument to true.

        Arguments:
            source_path (str): Path to the source file containing the Omnifold
                weights.
            target_path (str): Path to the target file containing the target
                weights.
            hv_path (str): Path to the file containing the sherpa sample reweighted
                to look like MG that is used for the hidden variable uncertainty.
            data_path (str): Path to the data file containing the data.
            store (str): Path to the directory where the plots will be stored.
            root_files (list of str): List of root files to load histograms of
                fastjet observables from. In order [source_end, target, hv, data].
                Defaults to None, in which case there should be no fastjet observables
                in the config.
            target2_path (str, optional): Path to the second target file for dual
                truth-level generator comparison. If provided, enables dual target mode.
            data_comparison_mode (bool): If True, compares data measurement to truth
                generators instead of pseudodata to target. Removes method bias and
                flips ratio calculations. Default is False.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(
            source_path,
            target_path,
            store,
            use_truth=True,
            root_files=root_files,
            **kwargs,
        )

        # Store new functionality flags
        self.target2_path = target2_path
        self.data_comparison_mode = data_comparison_mode
        self.dual_target_mode = target2_path is not None

        # Systematic uncertainty settings
        self.active_systs = {
            "nn-init": {
                "name": "NN Init",
                "color": "aqua",
                "plot_ratio": False,
                "stochastic": True,
            },
            "track-eff": {
                "name": "Track eff.",
                "color": "purple",
                "plot_ratio": False,
                "stochastic": False,
            },
            "jet-track-eff": {
                "name": "Jet track eff.",
                "color": "pink",
                "plot_ratio": False,
                "stochastic": False,
            },
            "track-fake": {
                "name": "Track fake",
                "color": "brown",
                "plot_ratio": False,
                "stochastic": False,
            },
            "track-scale": {
                "name": "Track scale",
                "color": "gray",
                "plot_ratio": False,
                "stochastic": False,
            },
            "muon-id": {
                "name": "Muon ID",
                "color": "lightgreen",
                "plot_ratio": False,
                "stochastic": False,
            },
            "muon-ms": {
                "name": "Muon MS",
                "color": "lightblue",
                "plot_ratio": False,
                "stochastic": False,
            },
            "muon-resbias": {
                "name": "Muon resolution bias",
                "color": "deepskyblue",
                "plot_ratio": False,
                "stochastic": False,
            },
            "muon-scale": {
                "name": "Muon scale",
                "color": "teal",
                "plot_ratio": False,
                "stochastic": False,
            },
            "hv": {
                "name": "Hidden variable",
                "color": "orange",
                "plot_ratio": False,
                "stochastic": False,
            },
            "dd": {
                "name": "Data driven",
                "color": "red",
                "plot_ratio": False,
                "stochastic": False,
            },
            "data-stat": {
                "name": "Data stat",
                "color": "blue",
                "plot_ratio": False,
                "stochastic": True,
            },
            "mc-stat": {
                "name": "MC stat",
                "color": "green",
                "plot_ratio": False,
                "stochastic": True,
            },
        }

        # Get the sherpa tree
        if "hv" in self.active_systs.keys():
            self.sherpa_path = hv_path
            self.sherpa_tree = uproot.open(hv_path)["OmniTree"]
            self.sherpa_events = self.sherpa_tree.num_entries
            if self.sherpa_events > self.max_events:
                self.sherpa_events = self.max_events

        # Get the data tree
        self.data_path = data_path
        self.data_tree = uproot.open(data_path)["OmniTree"]
        self.data_events = self.data_tree.num_entries
        if self.data_events > self.max_events:
            self.data_events = self.max_events

        # Verify that we have the correct number of root files
        # if there are fastjet observables
        if self.fastjet:
            expected_files = 5 if self.dual_target_mode else 4
            assert (
                len(self.root_files) == expected_files
            ), f"Must have {expected_files} root files for fastjet observables"
            assert self.root_files[0] is not None, "Must have a source_end root file"
            assert self.root_files[1] is not None, "Must have a target root file"
            assert self.root_files[2] is not None, "Must have a hv root file"
            assert self.root_files[3] is not None, "Must have a data root file"
            if self.dual_target_mode:
                assert self.root_files[4] is not None, "Must have a target2 root file"

        # Initialize caches for additional trees - flags will be loaded on demand
        # For data, since no pass190, we can cache a dummy
        self._pass190_cache["data"] = np.ones(self.data_events, dtype=bool)

        # Initialize second target if provided
        if self.dual_target_mode:
            self.target2_tree = uproot.open(target2_path)["OmniTree"]
            self.target2_events = self.target2_tree.num_entries
            if self.target2_events > self.max_events:
                self.target2_events = self.max_events

        # Hardcode symlog x ticks
        self.symlog_raw_xticks = np.array(
            [
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                0.5,
                1 - 1e-1,
                1 - 1e-2,
            ]
        )
        self.symlog_xticklabels = np.array(
            [
                r"$10^{-5}$",
                r"$10^{-4}$",
                r"$10^{-3}$",
                r"$10^{-2}$",
                r"$10^{-1}$",
                r"$0.5$",
                r"$1-10^{-1}$",
                r"$1-10^{-2}$",
            ]
        )

        # Note: kinematic cuts will be applied in plot() method when needed

        # Define uncertainty merging groups
        self.uncertainty_groups = {
            "Tracking": ["track-eff", "jet-track-eff", "track-fake", "track-scale"],
            "Unfolding": ["dd", "hv"],
            "Muon": ["muon-id", "muon-ms", "muon-resbias", "muon-scale"],
        }

        # Control whether to hide individual uncertainties when merging
        self.hide_individual_uncertainties = True

    def plot(self, of_weights, color="blue", **kwargs):
        """plot - Override of the base class plot method. Here we will build
        uncertainty plots comparing truth pseudodata to re-weighted truth
        level MC. The weights argument controls which weights will be used .

        Arguments:
        of_weights (str) - The weights produced by the Omnifold algorithm.
            Should be a string which can contain wildcards. These
            wildcards will be globbed over for the purpose of calculating the NN
            initialization uncertainty.
        color (str): Color to use for the histogram and ratio plots.

        Returns:
            dict: Dictionary with the format {plot_name: path_to_file} for
                each plot produced
        """

        # Ensure kinematic cuts are applied if needed
        self._ensure_kinematic_cuts_applied()

        # Load weights from numpy file
        weights_file = np.load(of_weights)

        # Efficiently load and process all weights
        processed_weights = self._load_and_process_weights_efficiently(weights_file)

        # Apply source weights vectorized
        final_weights = self._apply_source_weights_vectorized(processed_weights)

        # Get target weights efficiently
        target = self._get_target_weights_efficiently()
        if "dd" in self.active_systs:
            target_dd = self._get_target_dd_weights_efficiently()
            final_weights["target_dd"] = target_dd

        # Package into use weights dict
        weights_dict = {
            "target": target,
            **final_weights,
        }

        # Get second target weights if in dual target mode
        if self.dual_target_mode:
            target2 = self._get_weights_target2("weight_mc")
            target2_pass190 = self._get_cached_pass190_flags("target2")
            target2 = target2[target2_pass190 == 1]
            weights_dict["target2"] = target2

        # If we have track level observables, need to repeat the weights
        # for each track in the event
        if self.track_level:
            print("Calculating track level weights!")

            # Process all weights in batch for efficiency
            # A bit of a hack to remove the hv weights from the systematic dictionary
            # because they are handled separately
            source_weights_dict = final_weights.copy()
            source_weights_dict.pop("ensemble")
            if "hv" in source_weights_dict:
                source_weights_dict.pop("hv")

            # Process all source weights at once
            weights_dict_trk = self._get_track_weights_batch(source_weights_dict)

            # Process target and hv weights separately
            target_trk = self._get_track_weights(
                target,
                tree_type="target",
            )
            weights_dict_trk["target"] = target_trk
            if "hv" in self.active_systs:
                hv_trk = self._get_track_weights(
                    final_weights["hv"],
                    tree_type="sherpa",
                )
                weights_dict_trk["hv"] = hv_trk

            # Process ensemble weights efficiently
            if final_weights["ensemble"].size > 0:
                ensemble_weights_trk = np.zeros(
                    (
                        len(weights_dict_trk["central"]),
                        final_weights["ensemble"].shape[1],
                    )
                )
                for i in tqdm.tqdm(range(final_weights["ensemble"].shape[1])):
                    ensemble_weights_trk[:, i] = self._get_track_weights(
                        final_weights["ensemble"][:, i],
                        tree_type="source",
                    )
                weights_dict_trk["ensemble"] = ensemble_weights_trk
            else:
                weights_dict_trk["ensemble"] = np.zeros(
                    (len(weights_dict_trk["central"]), 0)
                )

            # Process target2 weights if in dual target mode
            if self.dual_target_mode:
                target2_trk = self._get_track_weights(
                    target2,
                    tree_type="target2",
                )
                weights_dict_trk["target2"] = target2_trk

        # Loop through plots and make histograms
        return_dict = {}
        for plot in self.plots:

            # Pick weight dict
            use_weight_dict = (
                weights_dict_trk if plot["type"] == "track" else weights_dict
            )

            # Get histograms
            nominal_plot = plot.copy()
            if nominal_plot["type"] == "fastjet":
                nominal_plot["key"] = "nominal-" + nominal_plot["key"]
            source_hist, bins = self._get_histogram(
                nominal_plot,
                weights=(use_weight_dict["central"]),
            )
            target_hist, _ = self._get_histogram(
                nominal_plot,
                weights=use_weight_dict["target"],
                is_target=True,
                root_index=1,  # This only effects histogram for fastjet observables
            )

            # Get second target histogram if in dual target mode
            if self.dual_target_mode:
                target2_hist, _ = self._get_histogram_target2(
                    nominal_plot,
                    weights=use_weight_dict["target2"],
                    root_index=4,  # This only effects histogram for fastjet observables
                )

            # MC stat uncertainty
            if "mc-stat" in self.active_systs:
                source_stat_var, _ = self._get_histogram(
                    nominal_plot,
                    weights=(use_weight_dict["central"] ** 2),
                    return_variance=True if plot["type"] == "fastjet" else False,
                )
                self.active_systs["mc-stat"].update({"var": source_stat_var})

            # Data stat uncertainty, note this is copied from reco level
            if "data-stat" in self.active_systs:
                data_hist, _ = self._get_data_histogram(nominal_plot)
                data_var = source_hist**2 / data_hist
                self.active_systs["data-stat"].update({"var": data_var})

            # NN initialization uncertainty
            if "nn-init" in self.active_systs:
                var_hists = []
                for i in range(use_weight_dict["ensemble"].shape[1]):
                    member_weights = use_weight_dict["ensemble"][:, i]
                    member_plot = plot.copy()
                    if member_plot["type"] == "fastjet":
                        member_plot["key"] = "nominal-" + str(i) + "-" + plot["key"]
                    member_hist, _ = self._get_histogram(
                        member_plot,
                        weights=member_weights,
                    )
                    # Remember to normalize to the source histogram!
                    norm_factor = np.sum(source_hist) / np.sum(member_hist)
                    member_hist *= norm_factor
                    var_hists.append(member_hist)
                nn_init_var = np.var(var_hists, axis=0) / (len(var_hists) - 1)
                self.active_systs["nn-init"].update({"var": nn_init_var})

            # Systematic uncertainties
            for key in self.active_systs:
                # Skip stochastic systematic uncertainties
                if self.active_systs[key]["stochastic"]:
                    continue
                wgts = use_weight_dict[key]
                syst_plot = plot.copy()
                if syst_plot["type"] == "fastjet":
                    syst_plot["key"] = key + "-" + plot["key"]
                if key == "hv":
                    hv_plot = syst_plot.copy()
                    syst_hist, _ = self._get_sherpa_histogram(
                        hv_plot,
                        weights=wgts,
                    )
                    estimate_hist = source_hist
                elif key == "dd":
                    syst_hist, _ = self._get_histogram(
                        syst_plot,
                        weights=wgts,
                    )
                    target_plot = plot.copy()  # Make sure you don't copy "syst_plot"
                    if target_plot["type"] == "fastjet":
                        target_plot["key"] = "target_dd-" + plot["key"]
                    estimate_hist, _ = self._get_histogram(
                        target_plot,
                        weights=use_weight_dict["target_dd"],
                        # Note the data driven weights are actually for the source data!
                        is_target=False,
                    )
                else:
                    syst_hist, _ = self._get_histogram(
                        syst_plot,
                        weights=wgts,
                    )
                    estimate_hist = source_hist
                norm_factor = np.sum(estimate_hist) / np.sum(syst_hist)
                syst_hist *= norm_factor
                # If we are plotting the ratio, add this systematic histogram
                # to the dictionary to pass into the build plot function
                if self.active_systs[key]["plot_ratio"]:
                    self.active_systs[key]["ratio"] = syst_hist / estimate_hist
                syst_var = np.abs(syst_hist - estimate_hist) ** 2
                self.active_systs[key].update({"var": syst_var})

            # Apply uncertainty merging after all individual uncertainties
            # are calculated
            self._apply_uncertainty_merging()

            # Make and save plot
            if type(bins) is tuple:
                fig = self._build_2d_uncert_plots(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    color=color,
                    target2_hist=target2_hist if self.dual_target_mode else None,
                )
            else:
                fig = self._build_uncert_plot(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    color=color,
                    target2_hist=target2_hist if self.dual_target_mode else None,
                )
            extension = ".pdf" if self.use_pdf else ".png"
            store_name = self.store / (plot["key"] + extension)
            fig.savefig(store_name, dpi=300)
            plt.close(fig)
            return_dict[plot["key"]] = store_name

            # Create and save uncertainty budget plot
            if not type(bins) is tuple:  # Only create for 1D plots
                budget_fig = self._plot_uncertainty_budget(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    target2_hist=target2_hist if self.dual_target_mode else None,
                )
                budget_name = plot["key"] + "_uncert_budget" + extension
                budget_store_name = self.store / budget_name
                budget_fig.savefig(budget_store_name, dpi=300)
                plt.close(budget_fig)
                return_dict[plot["key"] + "_uncert_budget"] = budget_store_name

            # Clean up merged uncertainties after each plot
            self._remove_merged_uncertainties()

        return return_dict

    def _get_sherpa_histogram(self, plot_dict, weights):
        """_get_sherpa_histogram - Get the sherpa histogram for a given observable.
        Need to have a separate routine because the base class functions assume data
        will be drawn from the source or target tree. For the hidden variable
        uncertainty, we need to draw data from a different tree.

        Arguments:
            plot_dict (dict): Dictionary containing the plotting style information
            weights (np.array): Array of weights to use for the histogram.

        Returns:
            hist (np.array): Array of histogram values.
            bins (np.array): Array of bin edges.
        """

        assert "hv" in self.active_systs.keys()

        # If the observable is computed using fastjet, we need to load the data
        # histogram from the correct ROOT file
        if plot_dict["type"] == "fastjet":
            sherpa_hist, bins = self._get_histogram(
                plot_dict,
                root_index=2,
            )

        # Else we can load the data and weights from the sherpa tree
        else:
            # Get sherpa data using shared method
            sherpa_data = self._get_filtered_data(
                self.sherpa_tree,
                plot_dict["key"],
                self._get_cached_pass190_flags("sherpa"),
                use_truth=True,
                max_events=self.sherpa_events,
            )

            # Create histogram using shared method
            sherpa_hist, bins = self._create_histogram_from_data(
                sherpa_data, plot_dict, weights=weights, density=False
            )

        return sherpa_hist, bins

    def _get_weights_target2(self, weights):
        """_get_weights_target2 - Get weights from the second target tree.

        Arguments:
            weights (str): Branch name to get weights from

        Returns:
            weights (np.array): Array of weights from target2 tree
        """
        return ak.to_numpy(
            self.target2_tree[weights].array(entry_stop=self.target2_events)
        )

    def _get_histogram_target2(
        self,
        plot_dict,
        weights=None,
        density=False,
        root_index=4,
        return_variance=False,
    ):
        """_get_histogram_target2 - Get histogram from the second target tree.

        Arguments:
            plot_dict (dict): Dictionary containing the plot configuration
            weights (np.array): Weights to use for histogram
            density (bool): If True, normalize to PDF
            root_index (int): Index of root file for fastjet observables
            return_variance (bool): If True, return variance instead of histogram

        Returns:
            tuple: (histogram, bins)
        """
        # If the observable is computed using fastjet, we need to load
        # the histograms using uproot
        if plot_dict["type"] == "fastjet":
            tobject = self._get_cached_root_object(root_index, plot_dict["key"])
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

            # Get filtered data from target2 tree
            data = self._get_data_target2(plot_dict["key"])

            # Create histogram using shared method
            hist, bins = self._create_histogram_from_data(
                data, plot_dict, weights=weights, density=False
            )

        # Normalize histogram if desired
        if density:
            assert not return_variance
            hist = self._normalize_to(hist, val=1.0)

        return hist, bins

    def _get_data_target2(self, key):
        """_get_data_target2 - Get data from the second target tree.

        Arguments:
            key (str): Key to get data for

        Returns:
            np.array: The data as a numpy array
        """
        return self._get_filtered_data(
            self.target2_tree,
            key,
            self._get_cached_pass190_flags("target2"),
            use_truth=True,
            max_events=self.target2_events,
        )

    def _get_data_histogram(self, plot_dict):
        """_get_data_histogram - Get the data histogram for a given observable.
        We'll use a simple sqrt(N) error at reco level as a stand-in for the
        full data stat uncertainty.

        Arguments:
            plot_dict (dict): Dictionary containing the plotting style information

        Returns:
            hist (np.array): Array of histogram values.
            bins (np.array): Array of bin edges.
        """

        # If the observable is computed using fastjet, we need to load the data
        # histogram from the correct ROOT file
        if plot_dict["type"] == "fastjet":
            data_hist, bins = self._get_histogram(
                plot_dict,
                root_index=3,
            )

        # Else we can load the data histogram from the data tree
        else:
            # Create a dummy pass190 array (all True) for data
            data = self._get_filtered_data(
                self.data_tree,
                plot_dict["key"],
                self._get_cached_pass190_flags("data"),
                use_truth=False,
                max_events=self.data_events,
            )

            # Create histogram using shared method
            data_hist, bins = self._create_histogram_from_data(
                data, plot_dict, weights=None, density=False
            )

        return data_hist, bins

    def _build_uncert_plot(
        self,
        plot,
        bins,
        source_hist,
        target_hist,
        color="blue",
        target2_hist=None,
    ):
        """_build_uncert_plot - Produce an uncertainty plot for a given observable.
        This plot will compare the source histogram to the target histogram, and
        additionally draw all of the uncertainties from the variances contained in
        the optional variances argument detailed below.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            bins (np.array): Array of bin edges for the histogram.
            source_hist (np.array): Array of source histogram values.
            target_hist (np.array): Array of target histogram values.
            color (str): Color to use for the histogram and ratio plots.
            target2_hist (np.array, optional): Array of second target histogram values.

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        # Handle different comparison modes
        if self.data_comparison_mode:
            # In data comparison mode, we compare data to truth generators
            # No method bias calculation, and ratio is target/data
            norm_factor = np.sum(source_hist) / np.sum(target_hist)
            norm_target_hist = norm_factor * target_hist
            ratio = norm_target_hist / source_hist  # Flipped ratio
            mbias = None
            rel_mbias = None
        else:
            # Standard pseudodata vs target comparison
            norm_factor = np.sum(source_hist) / np.sum(target_hist)
            norm_target_hist = norm_factor * target_hist
            ratio = source_hist / norm_target_hist

            # Find method bias
            mbias = (source_hist - norm_target_hist) ** 2
            rel_mbias = np.sqrt(mbias) / norm_target_hist

        # Calculate total variance and uncertainty
        total_var = np.sum(
            [self.active_systs[key]["var"] for key in self.active_systs], axis=0
        )
        total_uncert = np.sqrt(total_var)
        rel_total_uncert = total_uncert / source_hist

        # Duplicate last bins for all step plots
        plot_target_hist = np.append(norm_target_hist, norm_target_hist[-1])
        if rel_mbias is not None:
            rel_mbias = np.append(rel_mbias, rel_mbias[-1])

        # Plot
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_errors = (bins[1:] - bins[:-1]) / 2
        fig, (ax, rax) = plt.subplots(
            2,
            1,
            figsize=(6, 4.8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        plt.subplots_adjust(hspace=0, top=0.95)

        # If the x-axis is a symmetric log scale, transform the bins
        if plot["symlog_xscale"]:
            bins = self._transform_to_symlog(bins)
            bin_centers = self._transform_to_symlog(bin_centers)

        # Densities
        if self.data_comparison_mode:
            # In data comparison mode: target as purple points, data as black points
            ax.errorbar(
                bin_centers,
                plot_target_hist[:-1],  # Remove the duplicated last bin for errorbar
                fmt="o",
                label="Sherpa",
                color="purple",
            )

            # Add second target if in dual target mode
            if self.dual_target_mode and target2_hist is not None:
                norm_factor2 = np.sum(source_hist) / np.sum(target2_hist)
                norm_target2_hist = norm_factor2 * target2_hist
                ratio2 = norm_target2_hist / source_hist
                ax.errorbar(
                    bin_centers,
                    norm_target2_hist,
                    fmt="o",
                    label="MadGraph",
                    color="orange",
                )

            ax.errorbar(
                bin_centers,
                source_hist,
                xerr=bin_errors,
                yerr=total_uncert,
                fmt="+",
                label="Data",
                color="black",
            )
        else:
            # Standard mode: target as dashed line, unfolded as colored points
            ax.plot(
                bins,
                plot_target_hist,
                "--",
                label="Target",
                color="black",
                drawstyle="steps-post",
            )

            ax.errorbar(
                bin_centers,
                source_hist,
                yerr=total_uncert,
                fmt="o",
                label="Unfolded",
                color=color,
            )

        # Set tick parameters
        if plot["symlog_xscale"]:
            xticks = self._transform_to_symlog(self.symlog_raw_xticks)
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", direction="in", top=True)
        else:
            ax.tick_params(axis="x", direction="in", top=True)

        if not plot["linear_yscale"]:
            ax.set_yscale("log")
        if plot["log_xscale"]:
            ax.set_xscale("log")

        # Set y-axis label
        if plot["ylabel"] is not None:
            ax.set_ylabel(plot["ylabel"])
        else:
            ax.set_ylabel("Counts")
        ax.legend()

        # Ratios
        rax.axhline(1, color="black", linestyle="--")
        rax.errorbar(
            bin_centers,
            ratio,
            xerr=bin_errors,
            yerr=rel_total_uncert,
            fmt="o",
            color="purple" if self.data_comparison_mode else color,
        )
        if self.data_comparison_mode and self.dual_target_mode:
            rax.errorbar(
                bin_centers,
                ratio2,
                xerr=bin_errors,
                yerr=rel_total_uncert,
                fmt="o",
                color="orange",
            )
        for key, syst_ratio in self.active_systs.items():
            if syst_ratio["plot_ratio"]:
                rax.plot(
                    bin_centers,
                    syst_ratio["ratio"],
                    ".",
                    color=syst_ratio["color"],
                )
        rax.set_ylim(0.5, 1.5)
        rax.set_yticks([0.5, 1.0, 1.5])
        if self.data_comparison_mode:
            rax.set_ylabel("Ratio to data")
        else:
            rax.set_ylabel("Ratio to target")
        rax.set_xlabel(plot["xlabel"])

        # Set tick parameters
        if plot["symlog_xscale"]:
            xticks = self._transform_to_symlog(self.symlog_raw_xticks)
            rax.set_xticks(xticks)
            rax.set_xticklabels(self.symlog_xticklabels, rotation=45)
        else:
            rax.tick_params(axis="x", direction="in", bottom=True, top=False)

        # Finalize layout and return
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, top=0.95)

        return fig

    def _build_2d_uncert_plots(
        self,
        plot,
        bins,
        source_hist,
        target_hist,
        color="blue",
        target2_hist=None,
    ):
        """_build_2d_uncert_plots - Produce a 2D uncertainty plot for a given
        observable. This plot will compare the source histogram to the target
        histogram, and additionally draw all of the uncertainties from the
        variances contained in self.active_systs.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            bins (tuple): Tuple of two arrays of bin edges for the histogram.
            source_hist (np.array): Array of source histogram values.
            target_hist (np.array): Array of target histogram values.
            color (str): Color to use for the histogram and ratio plots.

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        # Handle different comparison modes
        if self.data_comparison_mode:
            # In data comparison mode, we compare data to truth generators
            # No method bias calculation, and ratio is target/data
            norm_factor = np.sum(source_hist) / np.sum(target_hist)
            norm_target_hist = norm_factor * target_hist
            ratio = norm_target_hist / source_hist  # Flipped ratio
            mbias = None
            rel_mbias = None
        else:
            # Standard pseudodata vs target comparison
            norm_factor = np.sum(source_hist) / np.sum(target_hist)
            norm_target_hist = norm_factor * target_hist
            ratio = source_hist / norm_target_hist

            # Find method bias
            mbias = (source_hist - norm_target_hist) ** 2
            rel_mbias = np.sqrt(mbias) / norm_target_hist

        # Calculate total variance and uncertainty
        total_var = np.sum(
            [self.active_systs[key]["var"] for key in self.active_systs], axis=0
        )
        total_uncert = np.sqrt(total_var)
        rel_total_uncert = total_uncert / source_hist

        # If the relative MC stat error is larger than 10%, mask the bin
        bin_mask = np.zeros_like(source_hist, dtype=bool)
        rel_mc_stat = np.sqrt(self.active_systs["mc-stat"]["var"]) / source_hist
        bin_mask[rel_mc_stat > 0.1] = True
        ratio = np.ma.masked_where(bin_mask, ratio)
        rel_total_uncert = np.ma.masked_where(bin_mask, rel_total_uncert)
        if rel_mbias is not None:
            rel_mbias = np.ma.masked_where(bin_mask, rel_mbias)

        # Define custom color maps
        ratio_cmap = matplotlib.cm.get_cmap("coolwarm").copy()
        ratio_cmap.set_bad(color="white")
        uncert_cmap = matplotlib.cm.get_cmap("summer").copy()
        uncert_cmap.set_bad(color="white")

        # Plot
        fig, (ax, uax) = plt.subplots(1, 2, figsize=(12, 6))

        # Ratio plot
        cax = ax.pcolormesh(
            bins[0],
            bins[1],
            ratio.T,
            cmap=ratio_cmap,
            vmin=0.9,
            vmax=1.1,
            shading="auto",
        )
        if self.data_comparison_mode:
            fig.colorbar(cax, ax=ax, label="Ratio to data")
        else:
            fig.colorbar(cax, ax=ax, label="Ratio to target")
        ax.set_xlabel(plot["xlabel"])
        ax.set_ylabel(plot["ylabel"])
        ax.set_title(plot["title"])

        # Uncertainty plot
        cuax = uax.pcolormesh(
            bins[0],
            bins[1],
            rel_total_uncert.T,
            cmap=uncert_cmap,
            vmin=0,
            vmax=0.2,
            shading="auto",
        )
        fig.colorbar(cuax, ax=uax, label="Total uncertainty")
        uax.set_xlabel(plot["xlabel"])
        uax.set_ylabel(plot["ylabel"])
        uax.set_title(plot["title"])

        # Add method bias as contours (only if method bias is available)
        if rel_mbias is not None:
            uax.contour(
                bins[0][:-1],
                bins[1][:-1],
                rel_mbias.T,
                levels=[0.1],
                colors="red",
                linewidths=2,
            )
            # Add legend for contour
            uax.plot([], [], color="red", linewidth=2, label="10% method bias")
            uax.legend(loc="upper right")

        fig.tight_layout()
        return fig

    def _plot_uncertainty_budget(
        self,
        plot,
        bins,
        source_hist,
        target_hist,
        target2_hist=None,
    ):
        """plot_uncertainty_budget - Create a standalone plot showing just the
        uncertainty budget. This plot will show the total uncertainty and
        individual contributions from each source.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            bins (np.array): Array of bin edges for the histogram.
            source_hist (np.array): Array of source histogram values.
            target_hist (np.array): Array of target histogram values.

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        # Handle different comparison modes
        if self.data_comparison_mode:
            # In data comparison mode, no method bias calculation
            norm_factor = np.sum(source_hist) / np.sum(target_hist)
            norm_target_hist = norm_factor * target_hist
            mbias = None
            rel_mbias = None
        else:
            # Standard pseudodata vs target comparison
            norm_factor = np.sum(source_hist) / np.sum(target_hist)
            norm_target_hist = norm_factor * target_hist

            # Find method bias
            mbias = (source_hist - norm_target_hist) ** 2
            rel_mbias = np.sqrt(mbias) / norm_target_hist

        # Calculate total variance and uncertainty
        total_var = np.sum(
            [self.active_systs[key]["var"] for key in self.active_systs], axis=0
        )
        total_uncert = np.sqrt(total_var)
        rel_total_uncert = total_uncert / source_hist

        # Duplicate last bins for all step plots
        plot_source_hist = np.append(source_hist, source_hist[-1])
        if rel_mbias is not None:
            plot_mbias = np.append(rel_mbias, rel_mbias[-1])
        plot_total_uncert = np.append(rel_total_uncert, rel_total_uncert[-1])
        plot_systs = {
            key: np.append(
                self.active_systs[key]["var"], self.active_systs[key]["var"][-1]
            )
            for key in self.active_systs
        }

        # Create figure
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        # If the x-axis is a symlog scale, transform the bins
        if plot["symlog_xscale"]:
            bins = self._transform_to_symlog(bins)

        # Plot total uncertainty
        ax.plot(
            bins,
            plot_total_uncert,
            "--",
            color="black",
            label="Total unc.",
            drawstyle="steps-post",
            linewidth=2,
        )

        # Plot individual uncertainties
        for key in plot_systs:
            ax.plot(
                bins,
                np.sqrt(plot_systs[key]) / plot_source_hist,
                "-",
                color=self.active_systs[key]["color"],
                label=self.active_systs[key]["name"],
                drawstyle="steps-post",
            )

        # Plot method bias only if available (not in data comparison mode)
        if rel_mbias is not None:
            ax.fill_between(
                bins,
                0,
                plot_mbias,
                step="post",
                color="gray",
                alpha=0.3,
                label="Method bias",
            )

        # Set tick parameters
        if plot["symlog_xscale"]:
            xticks = self._transform_to_symlog(self.symlog_raw_xticks)
            ax.set_xticks(xticks)
            ax.set_xticklabels(self.symlog_xticklabels, rotation=45)
        else:
            ax.tick_params(axis="x", direction="in", top=True)

        # Set other plot properties
        if rel_mbias is not None:
            top_uncert = np.max(np.concatenate([rel_total_uncert, rel_mbias]))
        else:
            top_uncert = np.max(rel_total_uncert)
        if top_uncert > 0.2 or np.isnan(top_uncert):
            ax.set_ylim(bottom=0.0, top=0.2)
        else:
            ax.set_ylim(bottom=0.0, top=top_uncert * 1.1)
        if plot["log_xscale"]:
            ax.set_xscale("log")
        ax.set_xlabel(plot["xlabel"])
        ax.set_ylabel("Uncertainty budget")
        if plot["symlog_xscale"]:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8
            )
        else:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8
            )

        # Finalize layout
        fig.tight_layout()
        if plot["symlog_xscale"]:
            fig.subplots_adjust(bottom=0.3)
        else:
            fig.subplots_adjust(bottom=0.2)

        return fig

    def _merge_uncertainties(self, group_name, individual_uncertainties):
        """Merge multiple uncertainties in quadrature.

        Arguments:
            group_name (str): Name for the merged uncertainty group
            individual_uncertainties (list): List of uncertainty names to merge

        Returns:
            dict: Dictionary containing the merged uncertainty information
        """
        if not individual_uncertainties:
            return None

        # Check that all uncertainties exist and have variance data
        available_uncertainties = []
        for uncert_name in individual_uncertainties:
            if (
                uncert_name in self.active_systs
                and "var" in self.active_systs[uncert_name]
            ):
                available_uncertainties.append(uncert_name)

        if not available_uncertainties:
            return None

        # Merge variances in quadrature
        merged_var = np.sum(
            [self.active_systs[uncert]["var"] for uncert in available_uncertainties],
            axis=0,
        )

        # Create merged uncertainty entry
        merged_uncertainty = {
            "name": group_name.title(),
            # Use color of first uncertainty
            "color": self.active_systs[available_uncertainties[0]]["color"],
            "plot_ratio": False,  # Default to not plotting ratio
            "stochastic": False,  # Merged uncertainties are typically not stochastic
            "var": merged_var,
            # Keep track of what was merged
            "merged_from": available_uncertainties,
        }

        return merged_uncertainty

    def _apply_uncertainty_merging(self):
        """Apply uncertainty merging based on self.uncertainty_groups.
        This method creates merged uncertainties and temporarily adds them to
        active_systs. Individual uncertainties that are grouped are hidden.
        """
        self._merged_uncertainties = {}
        self._hidden_uncertainties = {}  # Store hidden uncertainties for restoration

        for group_name, individual_uncertainties in self.uncertainty_groups.items():
            merged_uncert = self._merge_uncertainties(
                group_name, individual_uncertainties
            )
            if merged_uncert is not None:
                self._merged_uncertainties[group_name] = merged_uncert
                # Temporarily add to active_systs for plotting
                self.active_systs[group_name] = merged_uncert

                # Hide individual uncertainties that are being grouped (if enabled)
                if self.hide_individual_uncertainties:
                    for uncert_name in individual_uncertainties:
                        if uncert_name in self.active_systs:
                            # Store the original uncertainty for restoration
                            self._hidden_uncertainties[uncert_name] = (
                                self.active_systs[uncert_name]
                            )
                            # Remove from active_systs to hide it
                            del self.active_systs[uncert_name]

    def _remove_merged_uncertainties(self):
        """Remove merged uncertainties from active_systs after plotting and restore
        hidden individual uncertainties.
        """
        # Remove merged uncertainties
        for group_name in self._merged_uncertainties.keys():
            if group_name in self.active_systs:
                del self.active_systs[group_name]

        # Restore hidden individual uncertainties (if any were hidden)
        if hasattr(self, '_hidden_uncertainties'):
            for uncert_name, uncert_data in self._hidden_uncertainties.items():
                self.active_systs[uncert_name] = uncert_data

    def _transform_to_symlog(self, x):
        """_transform_to_symlog - Transform a linear scale to a symmetric log scale.
        This is used to transform the x-axis of some EEC plots to a symmetric log scale.
        """
        return np.where(x < 0.5, np.log10(x + 1e-10), -1 * np.log10(1 - x + 1e-10))

    def _inverse_transform_from_symlog(self, y):
        """_inverse_transform_from_symlog - Inverse of the symmetric log
        transformation.
        """
        return np.where(y < 0, 10**y - 1e-10, 1 - 10 ** (-y) + 1e-10)

    def _load_and_process_weights_efficiently(self, weights_file):
        """Efficiently load and process all weights from the weights file.

        This method optimizes the weight loading process by:
        1. Loading all weights in vectorized operations
        2. Applying filters in batch
        3. Pre-allocating arrays to avoid repeated allocation
        4. Processing ensemble weights more efficiently

        Arguments:
            weights_file (np.lib.npyio.NpzFile): The loaded weights file

        Returns:
            dict: Dictionary containing all processed weights
        """
        # Get basic info
        central_weights = weights_file["nominal-central"]
        nraw_events = len(central_weights)
        max_events_nominal = min(nraw_events, self.max_events)
        if "hv" in self.active_systs:
            max_events_sherpa = min(self.sherpa_events, self.max_events)

        # Pre-allocate result dictionary
        result = {}

        # Process central weights
        central_weights = central_weights[:max_events_nominal]
        source_pass190 = self._get_cached_pass190_flags("source")
        central_weights = central_weights[source_pass190 == 1]
        result["central"] = central_weights

        # Process ensemble weights more efficiently
        ens_names = [
            f for f in weights_file.files if ("nominal" in f) and ("central" not in f)
        ]

        if ens_names:
            # Load all ensemble weights at once
            ensemble_weights = np.array(
                [weights_file[name][:max_events_nominal] for name in ens_names]
            ).T  # Shape: (n_events, n_ensemble)

            # Apply filter to all ensemble weights at once
            ensemble_weights = ensemble_weights[source_pass190 == 1, :]
            result["ensemble"] = ensemble_weights
        else:
            result["ensemble"] = np.zeros((len(central_weights), 0))

        # Process systematic weights efficiently
        for syst in self.active_systs:
            if not self.active_systs[syst]["stochastic"]:
                syst_key = syst + "-central"
                if syst_key in weights_file.files:
                    if syst == "hv":
                        sherpa_pass190 = self._get_cached_pass190_flags("sherpa")
                        syst_weights = weights_file[syst_key][:max_events_sherpa][
                            sherpa_pass190 == 1
                        ]
                    else:
                        syst_weights = weights_file[syst_key][:max_events_nominal][
                            source_pass190 == 1
                        ]
                    result[syst] = syst_weights

        return result

    def _get_source_weights_efficiently(self):
        """Efficiently get and cache source weights to avoid repeated loading."""
        if not hasattr(self, "_cached_source_weights"):
            self._cached_source_weights = self._get_weights(
                "weight_mc", is_target=False
            )
            source_pass190 = self._get_cached_pass190_flags("source")
            self._cached_source_weights = self._cached_source_weights[
                source_pass190 == 1
            ]

        return self._cached_source_weights

    def _get_sherpa_weights_efficiently(self):
        """Efficiently get and cache sherpa weights to avoid repeated loading."""
        if not hasattr(self, "_cached_sherpa_weights"):
            self._cached_sherpa_weights = ak.to_numpy(
                self.sherpa_tree["weight_mc"].array(entry_stop=self.sherpa_events)
            )
            sherpa_pass190 = self._get_cached_pass190_flags("sherpa")
            self._cached_sherpa_weights = self._cached_sherpa_weights[
                sherpa_pass190 == 1
            ]

        return self._cached_sherpa_weights

    def _get_target_weights_efficiently(self):
        """Efficiently get and cache target weights to avoid repeated loading."""
        if not hasattr(self, "_cached_target_weights"):
            self._cached_target_weights = self._get_weights("weight_mc", is_target=True)
            target_pass190 = self._get_cached_pass190_flags("target")
            self._cached_target_weights = self._cached_target_weights[
                target_pass190 == 1
            ]

        return self._cached_target_weights

    def _get_target_dd_weights_efficiently(self):
        """Efficiently get and cache target data-driven weights."""
        if not hasattr(self, "_cached_target_dd_weights"):
            self._cached_target_dd_weights = self._get_weights(
                "target_dd", is_target=False
            )
            source_pass190 = self._get_cached_pass190_flags("source")
            self._cached_target_dd_weights = self._cached_target_dd_weights[
                source_pass190 == 1
            ]

        return self._cached_target_dd_weights

    def _apply_source_weights_vectorized(self, weights_dict):
        """Apply source weights to all weight arrays in a vectorized manner.

        Arguments:
            weights_dict (dict): Dictionary containing weight arrays

        Returns:
            dict: Dictionary with source weights applied
        """
        source_weights = self._get_source_weights_efficiently()
        if "hv" in self.active_systs:
            sherpa_weights = self._get_sherpa_weights_efficiently()

        result = {}

        # Apply source weights to central and ensemble weights
        if "central" in weights_dict:
            result["central"] = weights_dict["central"] * source_weights

        if "ensemble" in weights_dict and weights_dict["ensemble"].size > 0:
            result["ensemble"] = weights_dict["ensemble"] * np.expand_dims(
                source_weights, axis=1
            )

        # Apply appropriate weights to systematic weights
        for syst in self.active_systs:
            if self.active_systs[syst]["stochastic"]:
                continue
            weights = weights_dict[syst]
            if syst == "hv":
                result[syst] = weights * sherpa_weights
            else:
                result[syst] = weights * source_weights

        return result

    def _get_track_weights_batch(self, weights_dict, tree_type="source"):
        """Process multiple weight arrays to track level in a single operation.

        This method optimizes track weight processing by:
        1. Loading track data only once
        2. Processing all weights in batch
        3. Avoiding repeated track data access

        Arguments:
            weights_dict (dict): Dictionary containing weight arrays to process
            tree_type (str): Type of tree to use for track data

        Returns:
            dict: Dictionary with track-level weights
        """
        # Get track data once
        track_data = self._get_cached_track_data(tree_type=tree_type)

        # Process all weights at once
        result = {}
        for name, weights in weights_dict.items():
            if weights.size > 0:  # Only process non-empty arrays
                weights_ak, _ = ak.broadcast_arrays(ak.from_numpy(weights), track_data)
                result[name] = ak.to_numpy(ak.flatten(weights_ak, axis=None))
            else:
                result[name] = np.array([])

        return result

    def _get_cached_pass190_flags(self, tree_type="source"):
        """Override to handle additional tree types: sherpa, target2, data."""
        if tree_type in self._pass190_cache:
            return self._pass190_cache[tree_type]

        if tree_type in ["source", "target"]:
            return super()._get_cached_pass190_flags(tree_type)
        elif tree_type == "sherpa":
            tree = self.sherpa_tree
            max_events = self.sherpa_events
        elif tree_type == "target2":
            tree = self.target2_tree
            max_events = self.target2_events
        elif tree_type == "data":
            # Data has no pass190, all events pass
            flags = np.ones(self.data_events, dtype=bool)
            self._pass190_cache[tree_type] = flags
            return flags
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}")

        pull_key = "truth_pass190" if self.use_truth else "pass190"
        flags = ak.to_numpy(tree[pull_key].array(entry_stop=max_events))

        self._pass190_cache[tree_type] = flags
        return flags

    def _ensure_kinematic_cuts_applied(self):
        """Override to apply kinematic cuts to all trees including additional ones
        in UncertaintyPlotter (sherpa, data, and optionally target2).
        This method is idempotent.
        """
        if self._kinematic_region != 0 and not hasattr(self, "_kinematic_cuts_applied"):
            print("Applying kinematic cuts for region:", self._kinematic_region)
            if self.verbosity >= 3:
                print(
                    "Verbosity is greater than 3, please ensure fastjet"
                    " observables are calculated in the limited phase space!"
                )

            # First call the parent method to handle source and target trees
            super().apply_kinematic_cuts(self._kinematic_region)

            # Apply kinematic cuts to sherpa tree
            if "hv" in self.active_systs:
                sherpa_mask = self.get_kinematic_region(
                    self.sherpa_tree,
                    self._kinematic_region,
                    evts=self.sherpa_events,
                    use_truth=self.use_truth,
                )
                sherpa_pass190 = self._get_cached_pass190_flags("sherpa")
                self._pass190_cache["sherpa"] = np.logical_and(
                    sherpa_pass190, sherpa_mask
                )

            # Apply kinematic cuts to data tree
            # Note: data doesn't have pass190, so we just apply the kinematic mask
            # directly
            data_mask = self.get_kinematic_region(
                self.data_tree,
                self._kinematic_region,
                evts=self.data_events,
                use_truth=False,  # Data has no truth
            )
            # For data, we store the combined mask (no pass190 to combine with)
            self._pass190_cache["data"] = data_mask

            # Apply kinematic cuts to target2 tree if in dual target mode
            if self.dual_target_mode:
                target2_mask = self.get_kinematic_region(
                    self.target2_tree,
                    self._kinematic_region,
                    evts=self.target2_events,
                    use_truth=self.use_truth,
                )
                target2_pass190 = self._get_cached_pass190_flags("target2")
                self._pass190_cache["target2"] = np.logical_and(
                    target2_pass190, target2_mask
                )

            # Mark as applied to make this method idempotent
            self._kinematic_cuts_applied = True

    def _get_cached_track_data(self, tree_type="source"):
        """Override to handle additional tree types."""
        cache_key = tree_type
        if cache_key in self._track_data_cache:
            return self._track_data_cache[cache_key]

        pull_key = "truth_pT_tracks" if self.use_truth else "pT_tracks"

        if tree_type in ["source", "target"]:
            return super()._get_cached_track_data(tree_type)
        elif tree_type == "sherpa":
            tree = self.sherpa_tree
            max_events = self.sherpa_events
            pass190 = self._get_cached_pass190_flags("sherpa")
        elif tree_type == "target2":
            tree = self.target2_tree
            max_events = self.target2_events
            pass190 = self._get_cached_pass190_flags("target2")
        elif tree_type == "data":
            tree = self.data_tree
            max_events = self.data_events
            pass190 = self._get_cached_pass190_flags("data")
            pull_key = "pT_tracks"  # Data has no truth
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}")

        track_data = tree[pull_key].array(entry_stop=max_events)
        track_data = track_data[pass190 == 1]

        # Cache the original structure (not flattened) for proper broadcasting
        self._track_data_cache[cache_key] = track_data
        return track_data
