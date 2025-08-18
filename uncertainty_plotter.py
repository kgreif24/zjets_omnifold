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
        self, source_path, target_path, hv_path, data_path, store, root_files, **kwargs
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

        # Get the sherpa tree
        self.sherpa_path = hv_path
        self.sherpa_tree = uproot.open(hv_path)["OmniTree"]
        self.sherpa_events = self.sherpa_tree.num_entries
        if self.sherpa_events > self.max_events:
            self.sherpa_events = self.max_events
        self.sherpa_pass190 = ak.to_numpy(
            self.sherpa_tree["truth_pass190"].array(entry_stop=self.sherpa_events)
        )

        # Get the data tree
        self.data_path = data_path
        self.data_tree = uproot.open(data_path)["OmniTree"]
        self.data_events = self.data_tree.num_entries
        if self.data_events > self.max_events:
            self.data_events = self.max_events

        # Systematic uncertainty settings
        self.active_systs = {
            "nn-init": {
                "name": "NN Init",
                "color": "blue",
                "plot_ratio": False,
                "stochastic": True,
            },
            "track-eff": {
                "name": "Track eff.",
                "color": "purple",
                "plot_ratio": False,
                "stochastic": False,
            },
            "mc-stat": {
                "name": "MC stat",
                "color": "green",
                "plot_ratio": False,
                "stochastic": True,
            },
            "hidden-variable": {
                "name": "Hidden variable",
                "color": "orange",
                "plot_ratio": False,
                "stochastic": False,
                "tree": self.sherpa_tree,
            },
            "data-stat": {
                "name": "Data stat",
                "color": "aqua",
                "plot_ratio": False,
                "stochastic": True,
                "tree": self.data_tree,
            },
        }

        # Verify that we have the correct number of root files
        # if there are fastjet observables
        if self.fastjet:
            assert (
                len(self.root_files) == 4
            ), "Must have 4 root files for fastjet observables"
            assert self.root_files[0] is not None, "Must have a source_end root file"
            assert self.root_files[1] is not None, "Must have a target root file"
            assert self.root_files[2] is not None, "Must have a hv root file"
            assert self.root_files[3] is not None, "Must have a data root file"

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

        # Load weights from numpy file
        weights_file = np.load(of_weights)

        # Get central weights, nominal ensemble weights, and systematic weights
        central_weights = weights_file["nominal-ensemble-central"]
        nraw_events = len(central_weights)
        central_weights = central_weights[: self.max_events]
        central_weights = central_weights[self.source_pass190 == 1]

        ens_names = [
            f for f in weights_file.files if ("nominal" in f) and ("central" not in f)
        ]
        ensemble_weights = np.zeros((nraw_events, len(ens_names)))
        for i, name in enumerate(ens_names):
            ensemble_weights[:, i] = weights_file[name]
        ensemble_weights = ensemble_weights[: self.max_events, :]
        ensemble_weights = ensemble_weights[self.source_pass190 == 1, :]

        systematic_weights = {}
        for syst in self.active_systs:
            if self.active_systs[syst]["stochastic"]:
                continue
            systematic_weights[syst] = weights_file[syst + "-central"]
            systematic_weights[syst] = systematic_weights[syst][: self.max_events]
            if syst == "hidden-variable":
                systematic_weights[syst] = systematic_weights[syst][
                    self.sherpa_pass190 == 1
                ]
            else:
                systematic_weights[syst] = systematic_weights[syst][
                    self.source_pass190 == 1
                ]

        # Multiply by source weights
        source = self._get_weights("weight_mc", is_target=False)
        source = source[self.source_pass190 == 1]
        source_sherpa = ak.to_numpy(
            self.sherpa_tree["weight_mc"].array(entry_stop=self.sherpa_events)
        )
        source_sherpa = source_sherpa[self.sherpa_pass190 == 1]
        central_weights *= source
        ensemble_weights *= np.expand_dims(source, axis=1)
        for syst in systematic_weights:
            if syst == "hidden-variable":
                systematic_weights[syst] *= source_sherpa
            else:
                systematic_weights[syst] *= source

        # Get target weights
        target = self._get_weights("weight_mc", is_target=True)
        target = target[self.target_pass190 == 1]

        # If we have track level observables, need to repeat the weights
        # for each track in the event
        if self.track_level:
            print("Calculating track level weights!")
            central_weights_trk = self._get_track_weights(central_weights)
            target_trk = self._get_track_weights(target, is_target=True)
            ensemble_weights_trk = np.zeros(
                (len(central_weights_trk), ensemble_weights.shape[1])
            )
            for i in tqdm.tqdm(range(ensemble_weights.shape[1])):
                ensemble_weights_trk[:, i] = self._get_track_weights(
                    ensemble_weights[:, i]
                )
            systematic_weights_trk = {
                syst: self._get_track_weights(syst_wgt)
                for syst, syst_wgt in tqdm.tqdm(systematic_weights.items())
            }

        # Loop through plots and make histograms
        return_dict = {}
        for plot in self.plots:

            # Get histograms
            nominal_plot = plot.copy()
            if nominal_plot["type"] == "fastjet":
                nominal_plot["key"] = "nominal-ensemble-" + nominal_plot["key"]
            source_hist, bins = self._get_histogram(
                nominal_plot,
                weights=(
                    central_weights_trk if plot["type"] == "track" else central_weights
                ),
            )
            target_hist, _ = self._get_histogram(
                nominal_plot,
                weights=target_trk if plot["type"] == "track" else target,
                is_target=True,
                root_index=1,  # This only effects histogram for fastjet observables
            )

            # MC stat uncertainty
            if "mc-stat" in self.active_systs:
                source_stat_var, _ = self._get_histogram(
                    nominal_plot,
                    weights=(
                        central_weights_trk**2
                        if plot["type"] == "track"
                        else central_weights**2
                    ),
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
                use_nominal_weights = (
                    ensemble_weights_trk
                    if plot["type"] == "track"
                    else ensemble_weights
                )
                for i in range(use_nominal_weights.shape[1]):
                    member_weights = use_nominal_weights[:, i]
                    member_plot = plot.copy()
                    if member_plot["type"] == "fastjet":
                        member_plot["key"] = (
                            "nominal-ensemble-" + str(i) + "-" + plot["key"]
                        )
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
            use_syst_weights = (
                systematic_weights_trk
                if plot["type"] == "track"
                else systematic_weights
            )
            for key, wgts in use_syst_weights.items():
                syst_plot = plot.copy()
                if syst_plot["type"] == "fastjet":
                    syst_plot["key"] = key + "-" + plot["key"]
                if key == "hidden-variable":
                    syst_hist, _ = self._get_sherpa_histogram(
                        syst_plot,
                        weights=wgts,
                    )
                else:
                    syst_hist, _ = self._get_histogram(
                        syst_plot,
                        weights=wgts,
                    )
                norm_factor = np.sum(source_hist) / np.sum(syst_hist)
                syst_hist *= norm_factor
                # If we are plotting the ratio, add this systematic histogram
                # to the dictionary to pass into the build plot function
                if self.active_systs[key]["plot_ratio"]:
                    self.active_systs[key]["hist"] = syst_hist
                syst_var = np.abs(syst_hist - source_hist) ** 2
                self.active_systs[key].update({"var": syst_var})

            # Make and save plot
            if type(bins) is tuple:
                fig = self._build_2d_uncert_plots(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    color=color,
                )
            else:
                fig = self._build_uncert_plot(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    color=color,
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
                )
                budget_name = plot["key"] + "_uncert_budget" + extension
                budget_store_name = self.store / budget_name
                budget_fig.savefig(budget_store_name, dpi=300)
                plt.close(budget_fig)
                return_dict[plot["key"] + "_uncert_budget"] = budget_store_name

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

        # If the observable is computed using fastjet, we need to load the data
        # histogram from the correct ROOT file
        if plot_dict["type"] == "fastjet":
            sherpa_hist, bins = self._get_histogram(
                plot_dict,
                root_index=2,
            )

        # Else we can load the data and weights from the sherpa tree
        else:

            get_key = "truth_" + plot_dict["key"]
            sherpa_data = self.sherpa_tree[get_key].array(entry_stop=self.sherpa_events)
            sherpa_p190 = self.sherpa_tree["truth_pass190"].array(
                entry_stop=self.sherpa_events
            )
            if self.kinematic_region != 0:
                cuts = self.get_kinematic_region(
                    self.sherpa_tree, self.kinematic_region, evts=self.sherpa_events
                )
                # Unfortunate hack needed because the weights are already filtered
                weights = weights[cuts[sherpa_p190 == 1] == 1]
                sherpa_p190 = np.logical_and(sherpa_p190, cuts)
            sherpa_data = sherpa_data[sherpa_p190 == 1]
            sherpa_data = ak.to_numpy(ak.flatten(sherpa_data, axis=None))

            # Get the bins
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

            # Get the histogram
            sherpa_hist, _ = np.histogram(sherpa_data, bins=bins, weights=weights)

        return sherpa_hist, bins

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

            data = self.data_tree[plot_dict["key"]].array(entry_stop=self.data_events)
            if self.kinematic_region != 0:
                cuts = self.get_kinematic_region(
                    self.data_tree,
                    self.kinematic_region,
                    evts=self.data_events,
                    use_truth=False,
                )
                data = data[cuts == 1]
            data = ak.to_numpy(ak.flatten(data, axis=None))

            # Get the bins
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

            # Get the histogram
            data_hist, _ = np.histogram(data, bins=bins)

        return data_hist, bins

    def _build_uncert_plot(
        self,
        plot,
        bins,
        source_hist,
        target_hist,
        color="blue",
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

        Returns:
            fig (matplotlib.figure.Figure): Figure object for the plot.
        """

        # Normalize target histogram to the source, and take ratio
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
        rel_mbias = np.append(rel_mbias, rel_mbias[-1])

        # Plot
        bin_centers = (bins[1:] + bins[:-1]) / 2
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
            label="Omnifold",
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
            yerr=rel_total_uncert,
            fmt="o",
            color=color,
        )
        for key, syst_ratio in self.active_systs.items():
            if syst_ratio["plot_ratio"]:
                rax.plot(
                    bin_centers,
                    syst_ratio["hist"] / norm_target_hist,
                    ".",
                    color=syst_ratio["color"],
                )
        rax.set_ylim(0.5, 1.5)
        rax.set_yticks([0.5, 1.0, 1.5])
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

        # Normalize target histogram to the source, and take ratio
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

        # Add method bias as contours
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

        # Normalize target histogram to the source, and take ratio
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

        # Always want to plot the method bias
        ax.plot(
            bins,
            plot_mbias,
            "-",
            color="red",
            label="Method bias",
            drawstyle="steps-post",
        )
        ax.fill_between(bins, 0, plot_mbias, step="post", color="gray", alpha=0.3)

        # Set tick parameters
        if plot["symlog_xscale"]:
            xticks = self._transform_to_symlog(self.symlog_raw_xticks)
            ax.set_xticks(xticks)
            ax.set_xticklabels(self.symlog_xticklabels, rotation=45)
        else:
            ax.tick_params(axis="x", direction="in", top=True)

        # Set other plot properties
        top_uncert = np.max(np.concatenate([rel_total_uncert, rel_mbias]))
        if top_uncert > 0.2 or np.isnan(top_uncert):
            ax.set_ylim(bottom=0.0, top=0.2)
        else:
            ax.set_ylim(bottom=0.0, top=top_uncert * 1.1)
        if plot["log_xscale"]:
            ax.set_xscale("log")
        ax.set_xlabel(plot["xlabel"])
        ax.set_ylabel("Uncertainty")
        ax.set_title("Uncertainty Budget")
        if plot["symlog_xscale"]:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
        else:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

        # Finalize layout
        fig.tight_layout()
        if plot["symlog_xscale"]:
            fig.subplots_adjust(bottom=0.3)
        else:
            fig.subplots_adjust(bottom=0.2)

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
