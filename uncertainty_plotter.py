"""uncertainty_plotter.py - This module provides a subclass of the Plotter
class that builds uncertainty plots. It can be run in two modes:
    1. Compare truth pseudodata to re-weighted truth level MC
    2. Compare data to truth generators
The data comparison mode removes method bias and flips the ratio calculation.
It also supports dual target mode for comparing against two truth-level generators.

Author: Kevin Greif
Last updated 11.22.2025
python3
"""

import tqdm
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.stats as stats
import plotter
from analyze import uncertainties as uncert_module


class UncertaintyPlotter(plotter.Plotter):
    """
    UncertaintyPlotter is a subclass of Plotter that is specialized for
    building uncertainty plots. It can be run in two modes:
    1. Compare truth pseudodata to re-weighted truth level MC
    2. Compare data to truth generators
    The data comparison mode removes method bias and flips the ratio calculation.
    It also supports dual target mode for comparing against two truth-level generators.
    """

    def __init__(
        self,
        source_path,
        target_path,
        hv_path,
        store,
        root_files=None,
        target2_path=None,
        data_comparison_mode=False,
        do_chi2_test=False,
        uncertainty_definitions=None,
        uncertainty_groups=None,
        plot_signed_uncerts=False,
        **kwargs,
    ):
        """
        Initialize the UncertaintyPlotter class by calling the parent class's
        constructor.

        Fixes the use_truth argument to true.

        Arguments:
            source_path (str): Path to the source file containing the Omnifold
                weights.
            target_path (str): Path to the target file containing the target weights.
                Must be a single ROOT file.
            hv_path (str): Path to the file containing the sherpa sample reweighted
                to look like MG that is used for the hidden variable uncertainty.
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
            do_chi2_test (bool): If True, performs a chi^2 test and prints the results.
                Default is False.
            uncertainty_definitions (dict): Dictionary mapping uncertainty keys to their
                definitions (name, color, etc.). If None, uses default definitions.
            uncertainty_groups (dict): Dictionary mapping group names to lists of
                uncertainty keys. If None, uses default groups.
            plot_signed_uncerts (bool): If True, generates an additional plot showing
                signed uncertainties for non-stochastic systematics grouped into five
                panels. Default is False.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Ensure target_path is a single string (not a list)
        if isinstance(target_path, list):
            raise ValueError(
                "target_path must be a single ROOT file path (string). "
                "Multiple target files are not supported. "
                "Use target2_path for multiple files if needed."
            )

        # Initialize parent class with single target path
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
        self.do_chi2_test = do_chi2_test
        self.plot_signed_uncerts = plot_signed_uncerts

        # Create UncertaintyCalculator instance
        self.uncertainty_calculator = uncert_module.UncertaintyCalculator(
            uncertainty_definitions=uncertainty_definitions,
            uncertainty_groups=uncertainty_groups,
        )

        # Hardcode luminosity
        self.luminosity = 140.1  # fb^-1

        # Get the sherpa tree
        if "hv" in self.uncertainty_calculator.uncertainty_definitions:
            self.sherpa_path = hv_path
            self.sherpa_tree = uproot.open(hv_path)["OmniTree"]
            self.sherpa_events = self.sherpa_tree.num_entries
            if self.max_events > 0 and self.sherpa_events > self.max_events:
                self.sherpa_events = self.max_events

        # Verify that we have the correct number of root files
        # if there are fastjet observables
        if self.fastjet:
            expected_files = 4 if self.dual_target_mode else 3
            assert (
                len(self.root_files) == expected_files
            ), f"Must have {expected_files} root files for fastjet observables"
            assert self.root_files[0] is not None, "Must have a source_end root file"
            assert self.root_files[1] is not None, "Must have a target root file"
            assert self.root_files[2] is not None, "Must have a hv root file"
            if self.dual_target_mode:
                assert self.root_files[3] is not None, "Must have a target2 root file"

        # Initialize second target if provided
        if self.dual_target_mode:
            self.target2_tree = uproot.open(target2_path)["OmniTree"]
            self.target2_events = self.target2_tree.num_entries
            if self.max_events > 0 and self.target2_events > self.max_events:
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
        weights_dict = self._load_and_process_weights_efficiently(weights_file)

        # Get target weights efficiently
        # (get MG weights if in data comparison mode)
        mg_weight_names = self.uncertainty_calculator.madgraph_uncertainties
        target_weights = self._get_target_weights_efficiently(
            get_mg_weights=mg_weight_names if self.data_comparison_mode else None
        )

        # Get second target weights if in dual target mode
        target2_weights = {}
        if self.dual_target_mode:
            # Get sherpa uncertainty weights if in data comparison mode
            sherpa_weight_names = self.uncertainty_calculator.sherpa_uncertainties
            get_sherpa_weights = (
                sherpa_weight_names if self.data_comparison_mode else None
            )
            target2_weights = self._get_target2_weights_efficiently(
                get_sherpa_weights=get_sherpa_weights
            )

        # If we have track level observables, need to repeat the weights
        # for each track in the event
        if self.track_level:
            print("Calculating track level weights!")

            # Process all weights in batch for efficiency
            # Remove hv weights from the systematic dictionary because they are
            # handled separately (they come from sherpa tree, not source tree)
            source_weights_dict = weights_dict.copy()
            if "weights_hv" in source_weights_dict:
                source_weights_dict.pop("weights_hv")

            # Process all source weights at once (including ensemble and bootstrap_data)
            source_weights_trk = self._get_track_weights_batch(source_weights_dict)

            # Process hv weights
            if "hv" in self.uncertainty_calculator.uncertainty_definitions:
                source_weights_trk["weights_hv"] = self._get_track_weights(
                    weights_dict["weights_hv"],
                    tree_type="sherpa",
                )

            # Process target weights
            target_weights_trk = {}
            target_weights_trk["target"] = self._get_track_weights(
                target_weights["target"],
                tree_type="target",
            )

            # Process target theory weights if they exist
            mg_weight_names = self.uncertainty_calculator.madgraph_uncertainties
            if self.data_comparison_mode:
                for weight_name in mg_weight_names:
                    if weight_name in target_weights:
                        target_weights_trk[weight_name] = self._get_track_weights(
                            target_weights[weight_name],
                            tree_type="target",
                        )

            # Process target2 weights if in dual target mode
            if self.dual_target_mode:
                target2_weights_trk = {}
                target2_weights_trk["target2"] = self._get_track_weights(
                    target2_weights["target2"],
                    tree_type="target2",
                )

                # Process target2 theory weights if they exist
                sherpa_weight_names = self.uncertainty_calculator.sherpa_uncertainties
                if self.data_comparison_mode:
                    for weight_name in sherpa_weight_names:
                        if weight_name in weights_dict:
                            target2_weights_trk[weight_name] = self._get_track_weights(
                                weights_dict[weight_name],
                                tree_type="target2",
                            )

        # Loop through plots and make histograms
        histogram_data = {}  # Store histogram data for .npz export
        for plot in self.plots:
            print(f"Plotting {plot['key']}")

            # Pick weight dict
            use_weight_source = (
                source_weights_trk if plot["type"] == "track" else weights_dict
            )
            use_weight_target = (
                target_weights_trk if plot["type"] == "track" else target_weights
            )
            if self.dual_target_mode:
                use_weight_target2 = (
                    target2_weights_trk if plot["type"] == "track" else target2_weights
                )
            else:
                use_weight_target2 = None

            # Build measured histogram and begin compiling uncertainty dict
            all_hists = {}
            nominal_plot = plot.copy()
            if nominal_plot["type"] == "fastjet":
                nominal_plot["key"] = "nominal-" + nominal_plot["key"]
            source_hist, source_hist_var, bins = self._get_histogram(
                nominal_plot,
                weights=(use_weight_source["central"]),
            )
            all_hists["nominal"] = (source_hist, source_hist_var, bins)

            # Build target histograms
            target_hists = {}
            target_nominal_hist = None
            for wgt_name, wgts in use_weight_target.items():
                target_plot = plot.copy()
                if target_plot["type"] == "fastjet":
                    target_plot["key"] = (
                        self._theory_weight_name_to_hist_name(wgt_name)
                        + "-"
                        + plot["key"]
                    )
                target_hist_tuple = self._get_histogram(
                    target_plot,
                    weights=wgts,
                    is_target=True,
                    root_index=1,  # Only effects histogram for fastjet observables
                )
                if wgt_name == "target":
                    if not plot["cross_section"]:
                        norm_factor = np.sum(source_hist) / np.sum(target_hist_tuple[0])
                        target_hist_tuple = (
                            target_hist_tuple[0] * norm_factor,
                            target_hist_tuple[1] * norm_factor**2,
                            target_hist_tuple[2],
                        )
                    target_nominal_hist = target_hist_tuple[0]
                else:
                    norm_factor = np.sum(target_nominal_hist) / np.sum(
                        target_hist_tuple[0]
                    )
                    target_hist_tuple = (
                        target_hist_tuple[0] * norm_factor,
                        target_hist_tuple[1] * norm_factor**2,
                        target_hist_tuple[2],
                    )
                target_hists[wgt_name] = target_hist_tuple

            if self.dual_target_mode:
                target2_hists = {}
                target2_nominal_hist = None
                for wgt_name, wgts in use_weight_target2.items():
                    target2_plot = plot.copy()
                    if target2_plot["type"] == "fastjet":
                        target2_plot["key"] = (
                            self._theory_weight_name_to_hist_name(wgt_name)
                            + "-"
                            + plot["key"]
                        )
                    target2_hist_tuple = self._get_histogram_target2(
                        target2_plot,
                        weights=wgts,
                        root_index=3,  # Only effects histogram for fastjet observables
                    )
                    if wgt_name == "target2":
                        if not plot["cross_section"]:
                            norm_factor = np.sum(source_hist) / np.sum(
                                target2_hist_tuple[0]
                            )
                            target2_hist_tuple = (
                                target2_hist_tuple[0] * norm_factor,
                                target2_hist_tuple[1] * norm_factor**2,
                                target2_hist_tuple[2],
                            )
                        target2_nominal_hist = target2_hist_tuple[0]
                    else:
                        norm_factor = np.sum(target2_nominal_hist) / np.sum(
                            target2_hist_tuple[0]
                        )
                        target2_hist_tuple = (
                            target2_hist_tuple[0] * norm_factor,
                            target2_hist_tuple[1] * norm_factor**2,
                            target2_hist_tuple[2],
                        )
                    target2_hists[wgt_name] = target2_hist_tuple

            # Build ensemble histograms for NN stability uncertainty
            if "nn-stability" in self.uncertainty_calculator.uncertainty_definitions:
                if use_weight_source["ensemble"].size > 0:
                    for i in range(use_weight_source["ensemble"].shape[1]):
                        member_weights = use_weight_source["ensemble"][:, i]
                        member_plot = plot.copy()
                        # Only modify key for fastjet observables
                        # (to load from correct ROOT file)
                        if member_plot["type"] == "fastjet":
                            member_plot["key"] = (
                                "weights_ensemble_" + str(i) + "-" + plot["key"]
                            )
                        member_hist, member_hist_var, _ = self._get_histogram(
                            member_plot,
                            weights=member_weights,
                        )
                        # Use key format expected by UncertaintyCalculator
                        all_hists[f"ensemble_{i}"] = (
                            member_hist,
                            member_hist_var,
                            bins,
                        )

            # Build ensemble histograms for data stat uncertainty
            if "data-stat" in self.uncertainty_calculator.uncertainty_definitions:
                if (
                    "bootstrap_data" in use_weight_source
                    and use_weight_source["bootstrap_data"].size > 0
                ):
                    for i in range(use_weight_source["bootstrap_data"].shape[1]):
                        member_weights = use_weight_source["bootstrap_data"][:, i]
                        member_plot = plot.copy()
                        # Only modify key for fastjet observables
                        # (to load from correct ROOT file)
                        if member_plot["type"] == "fastjet":
                            member_plot["key"] = (
                                "weights_bootstrap_data_" + str(i) + "-" + plot["key"]
                            )
                        member_hist, member_hist_var, _ = self._get_histogram(
                            member_plot,
                            weights=member_weights,
                        )
                        # Use key format expected by UncertaintyCalculator
                        all_hists[f"bootstrap_data_{i}"] = (
                            member_hist,
                            member_hist_var,
                            bins,
                        )

            # Build ensemble histograms for mc stat bootstrap uncertainty
            if "mc-stat-bs" in self.uncertainty_calculator.uncertainty_definitions:
                if (
                    "bootstrap_mc" in use_weight_source
                    and use_weight_source["bootstrap_mc"].size > 0
                ):
                    for i in range(use_weight_source["bootstrap_mc"].shape[1]):
                        member_weights = use_weight_source["bootstrap_mc"][:, i]
                        member_plot = plot.copy()
                        # Only modify key for fastjet observables
                        # (to load from correct ROOT file)
                        if member_plot["type"] == "fastjet":
                            member_plot["key"] = (
                                "weights_bootstrap_mc_" + str(i) + "-" + plot["key"]
                            )
                        member_hist, member_hist_var, _ = self._get_histogram(
                            member_plot,
                            weights=member_weights,
                        )
                        # Use key format expected by UncertaintyCalculator
                        all_hists[f"bootstrap_mc_{i}"] = (
                            member_hist,
                            member_hist_var,
                            bins,
                        )

            # Build systematic uncertainty histograms
            # Iterate over uncertainty definitions and build histograms
            for syst_key in self.uncertainty_calculator.uncertainty_definitions:
                syst_def = self.uncertainty_calculator.uncertainty_definitions[syst_key]
                # Skip stochastic uncertainties (handled separately by calculator)
                if syst_def.get("stochastic", False):
                    continue

                # Get weights from weight dict using "weights_" prefix
                weight_key = f"weights_{syst_key}"
                if weight_key not in use_weight_source:
                    continue
                wgts = use_weight_source[weight_key]
                syst_plot = plot.copy()
                if syst_plot["type"] == "fastjet":
                    syst_plot["key"] = syst_key + "-" + plot["key"]

                if syst_key == "hv":
                    hv_plot = syst_plot.copy()
                    all_hists[syst_key] = self._get_sherpa_histogram(
                        hv_plot,
                        weights=wgts,
                    )
                elif syst_key == "dd":
                    all_hists[syst_key] = self._get_histogram(
                        syst_plot,
                        weights=wgts,
                    )
                    # Also need target_dd histogram
                    target_plot = plot.copy()
                    if target_plot["type"] == "fastjet":
                        target_plot["key"] = "target_dd-" + plot["key"]
                    all_hists["target_dd"] = self._get_histogram(
                        target_plot,
                        weights=use_weight_source["target_dd"],
                        is_target=False,
                    )
                else:
                    all_hists[syst_key] = self._get_histogram(
                        syst_plot,
                        weights=wgts,
                    )

            # Calculate uncertainties using UncertaintyCalculator
            syst_uncerts, syst_covs, syst_info = (
                self.uncertainty_calculator.calculate_uncertainties(
                    all_hists, measured_key="nominal"
                )
            )
            total_var = np.sum(np.array(list(syst_uncerts.values())) ** 2, axis=0)
            total_uncert = np.sqrt(total_var)
            total_cov = np.sum(list(syst_covs.values()), axis=0)
            chi2_cov = np.sum(
                [
                    syst_covs[key]
                    for key in syst_covs.keys()
                    if key not in ["Muon", "Tracking", "lumi", "pileup"]
                ],
                axis=0,
            )

            # Store histogram data for .npz export
            histogram_data[plot["key"] + "_hist"] = source_hist
            histogram_data[plot["key"] + "_uncert"] = total_uncert
            histogram_data[plot["key"] + "_cov"] = total_cov
            histogram_data[plot["key"] + "_bins"] = bins

            # Calculate theory uncertainties for targets if in data comparison mode
            target_uncert = None
            target2_uncert = None
            if self.data_comparison_mode:
                # Calculate MadGraph theory uncertainty for target
                target_uncert = (
                    self.uncertainty_calculator.get_total_theory_uncertainty(
                        target_hists, measured_key="target", is_madgraph=True
                    )
                )

                # Calculate Sherpa theory uncertainty for target2 if in dual target mode
                if self.dual_target_mode:
                    target2_uncert = (
                        self.uncertainty_calculator.get_total_theory_uncertainty(
                            target2_hists,
                            measured_key="target2",
                            is_madgraph=False,
                        )
                    )

            # Make and save plot
            fig = self._build_uncert_plot(
                plot,
                bins,
                source_hist,
                total_uncert,
                target_hists["target"][0],
                color=color,
                target2_hist=(
                    target2_hists["target2"][0] if self.dual_target_mode else None
                ),
                target_uncert=target_uncert,
                target2_uncert=target2_uncert,
            )
            extension = ".pdf" if self.use_pdf else ".png"
            store_name = self.store / (plot["key"] + extension)
            fig.savefig(store_name, dpi=300)
            plt.close(fig)

            # Create and save uncertainty budget plot
            budget_fig = self._plot_uncertainty_budget(
                plot,
                bins,
                source_hist,
                target_hists["target"][0],
                syst_uncerts,
                syst_info,
            )
            budget_name = plot["key"] + "_uncert_budget" + extension
            budget_store_name = self.store / budget_name
            budget_fig.savefig(budget_store_name, dpi=300)
            plt.close(budget_fig)

            # Create and save correlation matrix plot
            corr_fig = self._plot_correlation_matrix(plot, total_cov, bins)
            corr_name = plot["key"] + "_corr_matrix" + extension
            corr_store_name = self.store / corr_name
            corr_fig.savefig(corr_store_name, dpi=300)
            plt.close(corr_fig)
            histogram_data[plot["key"] + "_corr_matrix"] = corr_store_name

            # Create and save signed uncertainties plot
            if self.plot_signed_uncerts:
                signed_fig = self._plot_signed_uncertainties(
                    plot, bins, all_hists
                )
                signed_name = plot["key"] + "_signed_uncerts" + extension
                signed_store_name = self.store / signed_name
                signed_fig.savefig(signed_store_name, dpi=300)
                plt.close(signed_fig)

            # If not in data comparison mode, calculate chi^2 test and p-value
            if not self.data_comparison_mode and self.do_chi2_test:
                dof = len(bins) - 1
                D = source_hist - target_hists["target"][0]
                chi2 = D.dot(np.linalg.inv(chi2_cov)).dot(D.T)
                p_value = 1 - stats.chi2.cdf(chi2, dof)
                obs = str(plot["key"])
                print(
                    f"{obs:<20} dof: {dof:<7} χ2: {chi2:.5f} \t p value: {p_value:.4f}"
                )
                histogram_data[plot["key"] + "_dof"] = dof
                histogram_data[plot["key"] + "_chi2"] = chi2
                histogram_data[plot["key"] + "_p_value"] = p_value

        # Save histogram data to .npz file
        npz_path = self.store / "omnifold_histograms.npz"
        np.savez(npz_path, **histogram_data)
        print(f"Saved histogram data to: {npz_path}")

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
            hist_var (np.array): Array of histogram variance values.
            bins (np.array): Array of bin edges.
        """

        assert "hv" in self.uncertainty_calculator.uncertainty_definitions

        # If the observable is computed using fastjet, we need to load the data
        # histogram from the correct ROOT file
        if plot_dict["type"] == "fastjet":
            sherpa_hist, sherpa_hist_var, bins = self._get_histogram(
                plot_dict,
                root_index=2,
            )

        # Else we can load the data and weights from the sherpa tree
        else:
            # Check cache first
            # use_truth is always True for sherpa
            cache_key = (plot_dict["key"], "sherpa", True)
            if cache_key in self._filtered_data_cache:
                sherpa_data = self._filtered_data_cache[cache_key]
            else:
                # Get sherpa data using shared method
                sherpa_data = self._get_filtered_data(
                    self.sherpa_tree,
                    plot_dict["key"],
                    self._get_cached_pass190_flags("sherpa"),
                    use_truth=True,
                    max_events=self.sherpa_events,
                )
                # Cache the result
                self._filtered_data_cache[cache_key] = sherpa_data

            # Create histogram using shared method
            sherpa_hist, bins = self._create_histogram_from_data(
                sherpa_data, plot_dict, weights=weights, density=False
            )

            # Calculate variance
            bins_edges = self._get_bins_for_plot(plot_dict)
            if weights is not None:
                sherpa_hist_var, _ = np.histogram(
                    sherpa_data, bins=bins_edges, weights=weights**2, density=False
                )
            else:
                sherpa_hist_var = sherpa_hist.copy()

        return sherpa_hist, sherpa_hist_var, bins

    def _get_weights_target2(self, weights):
        """_get_weights_target2 - Get weights from the second target tree.

        Arguments:
            weights (str): Branch name to get weights from

        Returns:
            weights (np.array): Array of weights from target2 tree.
        """
        return ak.to_numpy(
            self.target2_tree[weights].array(entry_stop=self.target2_events)
        )

    def _get_histogram_target2(
        self,
        plot_dict,
        weights=None,
        density=False,
        root_index=3,
    ):
        """_get_histogram_target2 - Get histogram from the second target tree(s).
        Always returns the variance of the histogram. Handles multiple trees by
        concatenating data before histogramming.

        Arguments:
            plot_dict (dict): Dictionary containing the plot configuration
            weights (np.array): Weights to use for histogram (already concatenated
                if multiple trees)
            density (bool): If True, normalize to PDF
            root_index (int): Index of root file for fastjet observables

        Returns:
            tuple: (histogram, variance, bins) where:
                - histogram (np.array): Histogram values
                - variance (np.array): Variance of the histogram
                - bins (np.array or tuple): Bin edges
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

            # Get variance from ROOT object
            variance = tobject.variances()

        # Else the data is loaded and binned from the trees directly
        else:
            # Get filtered data from target2 tree(s) - already concatenated
            data = self._get_data_target2(plot_dict["key"])

            # Create histogram using shared method
            hist, bins = self._create_histogram_from_data(
                data, plot_dict, weights=weights, density=False
            )

            # Calculate variance
            if weights is not None:
                # For weighted histograms, variance is sum of squared weights per bin
                bins_edges = self._get_bins_for_plot(plot_dict)
                variance, _ = np.histogram(
                    data, bins=bins_edges, weights=weights**2, density=False
                )
            else:
                # For unweighted histograms, use Poisson statistics (variance = counts)
                variance = hist.copy()

        # Normalize histogram if desired
        if density:
            # Calculate normalization factor
            norm_factor = np.sum(hist)
            if norm_factor > 0:
                hist = hist / norm_factor
                # Variance scales as the square of the normalization factor
                variance = variance / (norm_factor**2)

        return hist, variance, bins

    def _get_data_target2(self, key):
        """_get_data_target2 - Get data from the second target tree.

        This method caches the filtered/flattened data to avoid repeated
        disk reads for the same observable.

        Arguments:
            key (str): Key to get data for

        Returns:
            np.array: The data as a numpy array
        """
        # Check cache first
        # use_truth is always True for target2
        cache_key = (key, "target2", True)
        if cache_key in self._filtered_data_cache:
            return self._filtered_data_cache[cache_key]

        data = self._get_filtered_data(
            self.target2_tree,
            key,
            self._get_cached_pass190_flags("target2"),
            use_truth=True,
            max_events=self.target2_events,
        )

        # Cache the result
        self._filtered_data_cache[cache_key] = data
        return data

    def _build_uncert_plot(
        self,
        plot,
        bins,
        source_hist,
        source_total_uncert,
        target_hist,
        color="blue",
        target2_hist=None,
        target_uncert=None,
        target2_uncert=None,
    ):
        """_build_uncert_plot - Produce an uncertainty plot for a given observable.
        This plot will compare the source histogram to the target histogram, and
        additionally draw all of the uncertainties from the variances contained in
        syst_vars.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            bins (np.array): Array of bin edges for the histogram.
            source_hist (np.array): Array of source histogram values.
            source_total_uncert (np.array): Array of source total uncertainty values.
                Should be percent uncertainties that are multiplied by measured bin
                counts to get the error bars.
            target_hist (np.array): Array of target histogram values.
            color (str): Color to use for the histogram and ratio plots.
            target2_hist (np.array, optional): Array of second target histogram values.
            target_uncert (np.array, optional): Total theory uncertainty for target
                histogram. Only used in data_comparison_mode.
            target2_uncert (np.array, optional): Total theory uncertainty for target2
                histogram. Only used in data_comparison_mode with dual_target_mode.

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        # Scale histograms by bin width for plotting
        if plot["cross_section"]:
            source_hist, _ = self._scale_histogram_by_bin_width(
                source_hist, None, bins
            )
            target_hist, _ = self._scale_histogram_by_bin_width(
                target_hist, None, bins
            )
            if target2_hist is not None:
                target2_hist, _ = self._scale_histogram_by_bin_width(
                    target2_hist, None, bins
                )

            print(f"Divided by bin width for key {plot['key']}")
            print(f"Source hist: {source_hist}")
            print(f"Target hist: {target_hist}")

        # Handle different comparison modes
        if self.data_comparison_mode:
            # In data comparison mode, we compare data to truth generators
            # No method bias calculation, and ratio is target/data
            ratio = target_hist / source_hist  # Flipped ratio
            mbias = None
            rel_mbias = None
        else:
            # Standard pseudodata vs target comparison
            ratio = source_hist / target_hist
            mbias = (source_hist - target_hist) ** 2
            rel_mbias = np.sqrt(mbias) / target_hist

        # Calculate relative uncertainties for ratio plots in data_comparison_mode
        rel_ratio_uncert = None
        rel_ratio2_uncert = None
        if self.data_comparison_mode:
            # For ratio = norm_target_hist / source_hist, uncertainty includes:
            # - Data uncertainty: source_total_uncert
            # - Target theory uncertainty: target_var / norm_target_hist
            if target_uncert is not None:
                rel_ratio_uncert = np.sqrt(source_total_uncert**2 + target_uncert**2)
            else:
                rel_ratio_uncert = source_total_uncert

        # Duplicate last bins for all step plots
        plot_target_hist = np.append(target_hist, target_hist[-1])
        if rel_mbias is not None:
            rel_mbias = np.append(rel_mbias, rel_mbias[-1])

        # Plot
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_errors = (bins[1:] - bins[:-1]) / 2
        error_bars = source_total_uncert * source_hist
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
            # Calculate bin widths for uncertainty boxes
            bin_widths = 2 * bin_errors

            # In data comparison mode: target as purple points, data as black points
            # Plot target as purple points first to get y-values
            target_y_values = plot_target_hist[:-1]  # Remove the duplicated last bin
            ax.errorbar(
                bin_centers,
                target_y_values,
                fmt="o",
                label="MadGraph",
                color="purple",
                zorder=2,  # In front of boxes
            )

            # Plot theory uncertainty boxes for target (behind the points)
            if target_uncert is not None:
                box_height = 2 * target_uncert * target_y_values
                for x, y, h, w in zip(
                    bin_centers, target_y_values, box_height, bin_widths
                ):
                    # Center the box on the point: bottom = y - h/2
                    box = Rectangle(
                        (x - w / 2, y - h / 2),
                        w,
                        h,
                        alpha=0.3,
                        facecolor="purple",
                        edgecolor=None,
                        zorder=1,  # Behind the points
                    )
                    ax.add_patch(box)

            # Add second target if in dual target mode
            if self.dual_target_mode and target2_hist is not None:
                ratio2 = target2_hist / source_hist

                # Calculate relative uncertainty for target2 ratio
                # Includes data uncertainty and target2 theory uncertainty
                if target2_uncert is not None:
                    rel_ratio2_uncert = np.sqrt(
                        source_total_uncert**2 + target2_uncert**2
                    )
                else:
                    rel_ratio2_uncert = source_total_uncert

                # Plot target2 as orange points first to get y-values
                ax.errorbar(
                    bin_centers,
                    target2_hist,
                    fmt="o",
                    label="Sherpa",
                    color="orange",
                    zorder=2,  # In front of boxes
                )

                # Plot theory uncertainty boxes for target2 (behind the points)
                if target2_uncert is not None:
                    box_height = 2 * target2_uncert * target2_hist
                    for x, y, h, w in zip(
                        bin_centers, target2_hist, box_height, bin_widths
                    ):
                        # Center the box on the point: bottom = y - h/2
                        box = Rectangle(
                            (x - w / 2, y - h / 2),
                            w,
                            h,
                            alpha=0.3,
                            facecolor="orange",
                            edgecolor=None,
                            zorder=1,  # Behind the points
                        )
                        ax.add_patch(box)

            ax.errorbar(
                bin_centers,
                source_hist,
                xerr=bin_errors,
                yerr=error_bars,
                fmt="+",
                label="Data",
                color="black",
                zorder=3,  # In front of everything
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
                yerr=error_bars,
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
        # Use combined uncertainty (data + theory) in data_comparison_mode
        ratio_uncert = (
            rel_ratio_uncert if rel_ratio_uncert is not None else source_total_uncert
        )
        rax.errorbar(
            bin_centers,
            ratio,
            xerr=bin_errors,
            yerr=ratio_uncert,
            fmt="+",
            color="purple" if self.data_comparison_mode else color,
        )
        if self.data_comparison_mode and self.dual_target_mode:
            # Use combined uncertainty (data + target2 theory) for target2
            rax.errorbar(
                bin_centers,
                ratio2,
                xerr=bin_errors,
                yerr=rel_ratio2_uncert,
                fmt="+",
                color="orange",
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

    def _plot_uncertainty_budget(
        self,
        plot,
        bins,
        source_hist,
        target_hist,
        syst_uncerts,
        syst_info,
    ):
        """plot_uncertainty_budget - Create a standalone plot showing just the
        uncertainty budget. This plot will show the total uncertainty and
        individual contributions from each source.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            bins (np.array): Array of bin edges for the histogram.
            source_hist (np.array): Array of source histogram values.
            target_hist (np.array): Array of target histogram values.
            syst_uncerts (dict): Dictionary mapping uncertainty keys to
                fractional uncertainty arrays.
            syst_info (dict): Dictionary mapping uncertainty keys to their metadata.

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        # Handle different comparison modes
        if self.data_comparison_mode:
            # In data comparison mode, no method bias calculation
            mbias = None
            rel_mbias = None
        else:
            # Find method bias
            if not plot["cross_section"]:
                norm_factor = np.sum(source_hist) / np.sum(target_hist)
                norm_target_hist = norm_factor * target_hist
            else:
                norm_target_hist = target_hist
            mbias = (source_hist - norm_target_hist) ** 2
            rel_mbias = np.sqrt(mbias) / norm_target_hist

        # Calculate total uncertainty
        total_vars = np.sum(np.array(list(syst_uncerts.values())) ** 2, axis=0)
        total_uncert = np.sqrt(total_vars)

        # Duplicate last bins for all step plots
        if rel_mbias is not None:
            plot_mbias = np.append(rel_mbias, rel_mbias[-1])
        plot_total_uncert = np.append(total_uncert, total_uncert[-1])

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
        for key, uncert in syst_uncerts.items():
            info = syst_info[key]
            plot_uncert = np.append(uncert, uncert[-1])
            ax.plot(
                bins,
                plot_uncert,
                "-",
                color=info.get("color", "black"),
                label=info.get("name", key),
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
            top_uncert = np.max(np.concatenate([total_uncert, rel_mbias]))
        else:
            top_uncert = np.max(total_uncert)
        if plot["ulim"] is not None:
            ax.set_ylim(top=plot["ulim"])
        elif top_uncert > 0.2 or np.isnan(top_uncert):
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

    def _plot_correlation_matrix(self, plot, total_cov, bins):
        """_plot_correlation_matrix - Create a correlation matrix plot from the
        covariance matrix.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            total_cov (np.array): Total covariance matrix (n_bins x n_bins)
            bins (np.array): Array of bin edges for labeling

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """
        # Calculate correlation matrix from covariance matrix
        # correlation[i,j] = covariance[i,j] / sqrt(covariance[i,i] * covariance[j,j])
        n_bins = total_cov.shape[0]
        std_devs = np.sqrt(np.diag(total_cov))

        # Handle zero standard deviations to avoid division by zero
        std_devs = np.where(std_devs == 0, 1, std_devs)

        # Calculate correlation matrix
        corr_matrix = total_cov / np.outer(std_devs, std_devs)

        # Clip values to [-1, 1] to handle numerical precision issues
        corr_matrix = np.clip(corr_matrix, -1, 1)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Create the heatmap with origin='lower' to match the attached style
        # (smallest bins at bottom-left)
        im = ax.imshow(
            corr_matrix,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            aspect="equal",
            origin="lower",
        )

        # Add colorbar
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Create bin labels from bin edges (round to hundredths for decimal edges)
        bin_labels = [f"({bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(len(bins) - 1)]

        # Set ticks and labels
        ax.set_xticks(np.arange(n_bins))
        ax.set_yticks(np.arange(n_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_yticklabels(bin_labels)

        # Add correlation values as text annotations
        for i in range(n_bins):
            for j in range(n_bins):
                # Choose text color based on background for readability
                corr_val = corr_matrix[i, j]
                text_color = "white" if abs(corr_val) < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{corr_val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        # Set title using the xlabel from plot config (contains variable name)
        ax.set_title(f"Correlation Matrix: {plot['xlabel']}", fontsize=12)

        # Finalize layout
        fig.tight_layout()

        return fig

    def _plot_signed_uncertainties(
        self,
        plot,
        bins,
        all_hists,
        measured_key="nominal",
    ):
        """Create a plot showing signed (non-absolute-value) ratios of systematic
        variations to the nominal histogram, grouped into five panels.

        Each panel corresponds to one uncertainty group:
        1. Unfolding (dd, hv, hvhad)
        2. Muon calibration and efficiency
        3. Tracking uncertainties
        4. Theory uncertainties
        5. Remaining non-stochastic uncertainties

        Arguments:
            plot (dict): Dictionary containing the plotting style information.
            bins (np.array): Array of bin edges for the histogram.
            all_hists (dict): Dictionary mapping histogram names to tuples of
                (hist, hist_var, bins).
            measured_key (str): Key in all_hists for the nominal distribution.

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        nominal_hist = all_hists[measured_key][0]

        # Hardcode the five signed-uncertainty groups
        unfolding_keys = ["dd", "hv", "hvhad"]
        muon_keys = [
            "muCalID", "muCalMS", "muCalResBias", "muCalScale",
            "muEffReco", "muEffIso", "muEffTrack", "muEffTrig",
        ]
        tracking_keys = ["trackEffMain", "trackEffJet", "trackFake", "trackPtScale"]
        theory_keys = [
            "theoryQCD", "theoryPDF", "theoryAlphaS", "theoryPSsoft",
            "theoryPSjet", "theoryMPI", "theoryPSscale",
        ]
        predefined_keys = set(
            unfolding_keys + muon_keys + tracking_keys + theory_keys
        )

        # Calculate signed ratios for every non-stochastic uncertainty present
        signed_ratios = {}
        for syst_key, syst_def in (
            self.uncertainty_calculator.uncertainty_definitions.items()
        ):
            if syst_def.get("stochastic", False):
                continue
            if syst_key not in all_hists:
                continue

            syst_hist = all_hists[syst_key][0]
            if syst_key == "dd":
                target_dd_hist = all_hists["target_dd"][0]
                signed_ratios[syst_key] = syst_hist / target_dd_hist
            else:
                signed_ratios[syst_key] = syst_hist / nominal_hist

        # Build ordered list of groups, filtering to keys actually present
        remaining_keys = [
            k for k in signed_ratios if k not in predefined_keys
        ]
        groups = [
            ("Unfolding", [k for k in unfolding_keys if k in signed_ratios]),
            ("Muon", [k for k in muon_keys if k in signed_ratios]),
            ("Tracking", [k for k in tracking_keys if k in signed_ratios]),
            ("Theory", [k for k in theory_keys if k in signed_ratios]),
            ("Remaining", remaining_keys),
        ]

        # Create figure with 5 stacked panels
        fig, axes = plt.subplots(
            5, 1, figsize=(8, 12), sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1]},
        )
        plt.subplots_adjust(hspace=0)

        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_errors = (bins[1:] - bins[:-1]) / 2

        # Handle symlog x-axis transformation
        if plot["symlog_xscale"]:
            bins_plot = self._transform_to_symlog(bins)
            bin_centers = self._transform_to_symlog(bin_centers)
            bin_errors = (bins_plot[1:] - bins_plot[:-1]) / 2

        for i, (group_name, group_keys) in enumerate(groups):
            ax = axes[i]

            for syst_key in group_keys:
                ratio = signed_ratios[syst_key]
                syst_def = (
                    self.uncertainty_calculator.uncertainty_definitions[syst_key]
                )
                label = syst_def.get("name", syst_key)
                color = syst_def.get("color", "black")
                ax.errorbar(
                    bin_centers,
                    ratio,
                    xerr=bin_errors,
                    yerr=0,
                    fmt="o",
                    color=color,
                    label=label,
                    markersize=3,
                )

            ax.axhline(1, color="black", linestyle="--", linewidth=0.5)
            ax.set_ylabel(group_name, fontsize=10)
            if group_keys:
                ax.legend(fontsize=6, loc="best", ncol=2)

            ax.tick_params(axis="both", direction="in")

            if plot["log_xscale"]:
                ax.set_xscale("log")

            # Hide x tick labels for all panels except the bottom one
            if i < 4:
                ax.tick_params(labelbottom=False)

        # Label the x-axis only on the bottom panel
        axes[-1].set_xlabel(plot["xlabel"])

        if plot["symlog_xscale"]:
            xticks = self._transform_to_symlog(self.symlog_raw_xticks)
            axes[-1].set_xticks(xticks)
            axes[-1].set_xticklabels(self.symlog_xticklabels, rotation=45)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, top=0.95)

        return fig

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
        central_weights = weights_file["weights_nominal"]

        # Pre-allocate result dictionary
        result = {}

        # Process central weights
        central_weights = central_weights[: self.source_events]
        source_pass190 = self._get_cached_pass190_flags("source")
        central_weights = central_weights[source_pass190 == 1]
        result["central"] = central_weights

        # Process ensemble weights more efficiently
        ens_names = [
            f
            for f in weights_file.files
            if ("weights_ensemble" in f) and ("weights_nominal" not in f)
        ]

        if ens_names:
            # Load all ensemble weights at once
            ensemble_weights = np.array(
                [weights_file[name][: self.source_events] for name in ens_names]
            ).T  # Shape: (n_events, n_ensemble)

            # Apply filter to all ensemble weights at once
            ensemble_weights = ensemble_weights[source_pass190 == 1, :]
            result["ensemble"] = ensemble_weights

        # Process data bootstrap weights efficiently
        dbootstrap_names = [
            f for f in weights_file.files if "weights_bootstrap_data" in f
        ]
        if dbootstrap_names:
            dbootstrap_weights = np.array(
                [weights_file[name][: self.source_events] for name in dbootstrap_names]
            ).T
            # Apply filter to all bootstrap weights at once
            dbootstrap_weights = dbootstrap_weights[source_pass190 == 1, :]
            result["bootstrap_data"] = dbootstrap_weights

        # Process mc stat bootstrap weights efficiently
        mc_stat_bs_names = [
            f for f in weights_file.files if "weights_bootstrap_mc" in f
        ]
        if mc_stat_bs_names:
            mc_stat_bs_weights = np.array(
                [weights_file[name][: self.source_events] for name in mc_stat_bs_names]
            ).T
            mc_stat_bs_weights = mc_stat_bs_weights[source_pass190 == 1, :]
            result["bootstrap_mc"] = mc_stat_bs_weights

        # Process systematic weights efficiently
        # Keys in .npz file have "weights_" prefix, strip it to get
        # UncertaintyCalculator keys
        for syst_key in self.uncertainty_calculator.uncertainty_definitions:
            syst_def = self.uncertainty_calculator.uncertainty_definitions[syst_key]
            if syst_def.get("stochastic", False):
                continue
            # Look for weight key with "weights_" prefix
            weight_key = f"weights_{syst_key}"
            if weight_key in weights_file.files:
                if syst_key == "hv":
                    sherpa_pass190 = self._get_cached_pass190_flags("sherpa")
                    syst_weights = weights_file[weight_key][: self.sherpa_events][
                        sherpa_pass190 == 1
                    ]
                else:
                    syst_weights = weights_file[weight_key][: self.source_events][
                        source_pass190 == 1
                    ]
                result[weight_key] = syst_weights
            # If the weight key is "dd", we need to get the target_dd weights
            if weight_key == "weights_dd":
                target_dd = weights_file["target_dd"][: self.source_events][
                    source_pass190 == 1
                ]
                result["target_dd"] = target_dd

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

    def _get_target_weights_efficiently(self, get_mg_weights=None):
        """Efficiently get and cache target weights to avoid repeated loading."""
        if not hasattr(self, "_cached_target_weights"):
            self._cached_target_weights = self._get_weights("weight_mc", is_target=True)
            target_pass190 = self._get_cached_pass190_flags("target")
            self._cached_target_weights = self._cached_target_weights[
                target_pass190 == 1
            ]
            # Divide by luminosity to get cross section
            self._cached_target_weights /= self.luminosity
        result = {"target": self._cached_target_weights}
        if get_mg_weights is not None:
            if not hasattr(self, "_cached_mg_weights"):
                self._cached_mg_weights = {}
                for weight_name in get_mg_weights:
                    wgts = self._get_weights(weight_name, is_target=True)
                    target_pass190 = self._get_cached_pass190_flags("target")
                    wgts = wgts[target_pass190 == 1]
                    # Note MG theory weights are multiplied by the weight_mc branch!
                    wgts *= self._cached_target_weights
                    self._cached_mg_weights[weight_name] = wgts
                result.update(**self._cached_mg_weights)
        return result

    def _get_target2_weights_efficiently(self, get_sherpa_weights=None):
        """Efficiently get and cache target2 weights to avoid repeated loading.

        Arguments:
            get_sherpa_weights (list, optional): List of weight names to load from
                sherpa_uncertainties. If provided, loads these weights and adds
                them to the result.

        Returns:
            dict: Dictionary containing "target2" key and optionally theory
                uncertainty weight keys.
        """
        if not hasattr(self, "_cached_target2_weights"):
            target2_weights = self._get_weights_target2("weight_mc")
            target2_pass190 = self._get_cached_pass190_flags("target2")
            target2_weights = target2_weights[target2_pass190 == 1]
            # Divide by luminosity to get cross section
            target2_weights /= self.luminosity
            self._cached_target2_weights = target2_weights

        result = {"target2": self._cached_target2_weights}

        if get_sherpa_weights is not None:
            if not hasattr(self, "_cached_sherpa_weights"):
                self._cached_sherpa_weights = {}
                for weight_name in get_sherpa_weights:
                    wgts = self._get_weights_target2(weight_name)
                    target2_pass190 = self._get_cached_pass190_flags("target2")
                    wgts = wgts[target2_pass190 == 1]
                    # Note Sherpa theory weights ARE NOT multiplied by weight_mc!
                    # Divide by luminosity to get cross section
                    wgts /= self.luminosity
                    self._cached_sherpa_weights[weight_name] = wgts
                result.update(**self._cached_sherpa_weights)

        return result

    def _get_track_weights_batch(self, weights_dict, tree_type="source"):
        """Process multiple weight arrays to track level in a single operation.

        This method optimizes track weight processing by:
        1. Loading track data only once
        2. Processing all weights in batch
        3. Avoiding repeated track data access
        4. Handling both 1D and 2D weight arrays (e.g., ensemble, bootstrap_data)

        Arguments:
            weights_dict (dict): Dictionary containing weight arrays to process.
                Can contain 1D arrays (n_events,) or 2D arrays (n_events, n_members).
            tree_type (str): Type of tree to use for track data

        Returns:
            dict: Dictionary with track-level weights. 1D arrays become 1D track arrays,
                2D arrays become 2D track arrays (n_tracks, n_members).
        """
        # Get track data once
        track_data = self._get_cached_track_data(tree_type=tree_type)

        # Process all weights at once
        result = {}
        n_tracks = None  # Will be determined from first processed weight

        # First pass: process all non-empty weights
        for name, weights in weights_dict.items():
            if weights.size > 0:  # Only process non-empty arrays
                # Check if this is a 2D array (e.g., ensemble, bootstrap_data)
                if weights.ndim == 2 and weights.shape[1] > 0:
                    # Process each column and stack them back into a 2D array
                    n_members = weights.shape[1]
                    track_weights_list = []
                    for i in tqdm.tqdm(range(n_members), desc=f"Processing {name}"):
                        weights_1d = weights[:, i]
                        weights_ak, _ = ak.broadcast_arrays(
                            ak.from_numpy(weights_1d), track_data
                        )
                        track_weights_list.append(
                            ak.to_numpy(ak.flatten(weights_ak, axis=None))
                        )
                    # Stack columns back into 2D array: (n_tracks, n_members)
                    result[name] = np.column_stack(track_weights_list)
                    # Store track count from first processed weight
                    if n_tracks is None:
                        n_tracks = len(result[name])
                else:
                    # Process 1D array as before
                    weights_ak, _ = ak.broadcast_arrays(
                        ak.from_numpy(weights), track_data
                    )
                    result[name] = ak.to_numpy(ak.flatten(weights_ak, axis=None))
                    # Store track count from first processed weight
                    if n_tracks is None:
                        n_tracks = len(result[name])

        # Second pass: handle empty arrays using track count from first pass
        # If no weights were processed, get track count from track_data
        if n_tracks is None:
            n_tracks = len(track_data) if hasattr(track_data, "__len__") else 0

        for name, weights in weights_dict.items():
            if weights.size == 0:  # Handle empty arrays
                # Preserve 2D structure if original was 2D
                if weights.ndim == 2:
                    n_cols = weights.shape[1] if len(weights.shape) > 1 else 0
                    result[name] = np.zeros((n_tracks, n_cols))
                else:
                    result[name] = np.array([])

        return result

    def _get_cached_pass190_flags(self, tree_type="source"):
        """Override to handle additional tree types: sherpa, target2."""
        if tree_type in self._pass190_cache:
            return self._pass190_cache[tree_type]

        if tree_type in ["source", "target"]:
            return super()._get_cached_pass190_flags(tree_type)
        elif tree_type == "sherpa":
            tree = self.sherpa_tree
            max_events = self.sherpa_events
            pull_key = "truth_pass190" if self.use_truth else "pass190"
            flags = ak.to_numpy(tree[pull_key].array(entry_stop=max_events))
            self._pass190_cache[tree_type] = flags
            return flags
        elif tree_type == "target2":
            tree = self.target2_tree
            max_events = self.target2_events
            pull_key = "truth_pass190" if self.use_truth else "pass190"
            flags = ak.to_numpy(tree[pull_key].array(entry_stop=max_events))
            self._pass190_cache[tree_type] = flags
            return flags
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}")

    def apply_kinematic_cuts(self, region):
        """Override to handle target2 trees when applying kinematic cuts."""
        # Call parent method to handle source and target trees
        super().apply_kinematic_cuts(region)

    def _ensure_kinematic_cuts_applied(self):
        """Override to apply kinematic cuts to all trees including additional ones
        in UncertaintyPlotter (sherpa, data, and optionally target2).
        This method is idempotent.
        """
        if self._kinematic_region != -1 and not hasattr(
            self, "_kinematic_cuts_applied"
        ):
            print("Applying kinematic cuts for region:", self._kinematic_region)
            if self.verbosity >= 3:
                print(
                    "Verbosity is greater than 2, please ensure fastjet"
                    " observables are calculated in the limited phase space!"
                )

            # Apply kinematic cuts (handles multiple target trees)
            self.apply_kinematic_cuts(self._kinematic_region)

            # Apply kinematic cuts to sherpa tree
            if "hv" in self.uncertainty_calculator.uncertainty_definitions:
                sherpa_mask = self.get_kinematic_region(
                    self.sherpa_tree,
                    self._kinematic_region,
                    evts=self.sherpa_events,
                    use_truth=True,
                )
                sherpa_pass190 = self._get_cached_pass190_flags("sherpa")
                self._pass190_cache["sherpa"] = np.logical_and(
                    sherpa_pass190, sherpa_mask
                )

            # Apply kinematic cuts to target2 tree if in dual target mode
            if self.dual_target_mode:
                target2_mask = self.get_kinematic_region(
                    self.target2_tree,
                    self._kinematic_region,
                    evts=self.target2_events,
                    use_truth=True,
                )
                target2_pass190 = self._get_cached_pass190_flags("target2")
                self._pass190_cache["target2"] = np.logical_and(
                    target2_pass190, target2_mask
                )

            # Mark as applied to make this method idempotent
            self._kinematic_cuts_applied = True

    def _get_cached_track_data(self, tree_type="source"):
        """Override to handle additional tree types: sherpa, target2."""
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
            track_data = tree[pull_key].array(entry_stop=max_events)
            track_data = track_data[pass190 == 1]
            self._track_data_cache[cache_key] = track_data
            return track_data
        elif tree_type == "target2":
            tree = self.target2_tree
            max_events = self.target2_events
            pass190 = self._get_cached_pass190_flags("target2")
            track_data = tree[pull_key].array(entry_stop=max_events)
            track_data = track_data[pass190 == 1]
            self._track_data_cache[cache_key] = track_data
            return track_data
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}")

    def _theory_weight_name_to_hist_name(self, weight_name):
        """_theory_weight_name_to_hist_name - Convert a theory weight name to a
        histogram name.

        Arguments:
            weight_name (str): The name of the theory weight.

        Returns:
            str: The name of the histogram.
        """
        if weight_name == "target":
            return "nominal"
        elif weight_name == "target2":
            return "nominal"
        elif weight_name == "w_QCD_dd":
            return "theoryQCD"
        elif weight_name == "w_PDF_CT18nnlo":
            return "theoryPDF"
        elif weight_name == "w_Alpha_s1":
            return "theoryAlphaS"
        elif weight_name == "w_Var1Down":
            return "theoryPSsoft"
        elif weight_name == "w_Var2Down":
            return "theoryPSjet"
        elif weight_name == "w_MPIDown":
            return "theoryMPI"
        elif weight_name == "w_RenDown":
            return "theoryPSscale"
        elif weight_name == "PS_ME_QCD_dd":
            return "theoryQCD"
        elif weight_name == "PS_ME_PDF_CT18nnlo":
            return "theoryPDF"
        elif weight_name == "PS_ME_Alpha_s1":
            return "theoryAlphaS"
        else:
            raise ValueError(f"Weight name {weight_name} not recognized!")
