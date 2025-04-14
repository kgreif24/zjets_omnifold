"""uncertainty_plotter.py - This module provides a subclass of the Plotter
class that builds uncertainty plots. It will only make plots comparing
truth pseudodata to re-weighted truth level MC, as uncertainties are only
defined in this context.

Author: Kevin Greif
Last updated 03.28.2025
python3
"""

import os
import glob
import pathlib
import subprocess
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import plotter


class UncertaintyPlotter(plotter.Plotter):
    """
    UncertaintyPlotter is a subclass of Plotter that is specialized for
    building uncertainty plots. It will only make plots comparing truth
    pseudodata to re-weighted truth level MC, as uncertainties are only
    defined in this context.
    """

    def __init__(self, source_path, target_path, store, **kwargs):
        """
        Initialize the UncertaintyPlotter class by calling the parent class's
        constructor.

        Fixes the use_truth argument to true.

        Arguments:
            source_path (str): Path to the source file containing the Omnifold
                weights.
            target_path (str): Path to the target file containing the target
                weights.
            store (str): Path to the directory where the plots will be stored.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(
            source_path,
            target_path,
            store,
            use_truth=True,
            **kwargs,
        )

    def plot(self, of_weights, color="blue", recalculate=False, **kwargs):
        """ plot - Override of the base class plot method. Here we will build
        uncertainty plots comparing truth pseudodata to re-weighted truth
        level MC. The weights argument controls which weights will be used .

        Arguments:
        of_weights (str) - The weights produced by the Omnifold algorithm.
            Should be a string which can contain wildcards. These
            wildcards will be globbed over for the purpose of calculating the NN
            initialization uncertainty.
        color (str): Color to use for the histogram and ratio plots.
        recalculate (bool): If True, will recalculate fastjet observables
            even if the root files already exist.

        Returns:
            dict: Dictionary with the format {plot_name: path_to_file} for
                each plot produced
        """

        # Form glob from pattern for weights
        weights_list = sorted(glob.glob(of_weights))
        if not weights_list:
            raise ValueError(f"No files found for the pattern: {of_weights}")

        # Load, truncate, and filter weights
        raw_ensemble_weights = [np.load(f)["test"] for f in weights_list]
        t_ensemble_weights = [
            ens[: self.max_events] for ens in raw_ensemble_weights
        ]
        ensemble_weights = [
            ens[self.source_pass190 == 1] for ens in t_ensemble_weights
        ]

        # Calculate central weights from ensemble, then truncate and filter
        raw_central_weights = self._calculate_central(raw_ensemble_weights)
        t_central_weights = raw_central_weights[: self.max_events]
        central_weights = t_central_weights[self.source_pass190 == 1]

        # Get target weights
        target = self._get_weights("weight_mc", is_target=True)
        target = target[self.target_pass190 == 1]

        # If we have track level observables, need to repeat the weights
        # for each track in the event
        if self.track_level:
            central_weights_trk = self._get_track_weights(central_weights)
            target_trk = self._get_track_weights(target, is_target=True)
            ensemble_weights_trk = []
            for ens in tqdm.tqdm(ensemble_weights):
                ensemble_weights_trk.append(self._get_track_weights(ens))

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

            # Save raw central weights to .npz file for input to fastjet
            central_weights_file = pathlib.Path(self.store) / "central_weights.npz"
            np.savez(central_weights_file, test=raw_central_weights)

            # Run for each root file that does not exist
            # Note we need to raise all paths by one directory
            weights = [pathlib.Path("..") / central_weights_file, "weight_mc"]
            for i, (use_weights, file) in enumerate(zip(weights, self.root_files)):
                if recalculate and pathlib.Path(file).exists():
                    os.remove(file)
                if not pathlib.Path(file).exists():
                    # Need to raise paths by one directory
                    up_file = pathlib.Path("..") / file
                    up_weights_list = [pathlib.Path("..") / w for w in weights_list]
                    self._run_fastjet(
                        use_weights,
                        up_file,
                        # Don't run ensemble for target, only for source
                        ens_weights=up_weights_list if i == 0 else None,
                        is_target=(i == 1),
                    )

        # Loop through plots and make histograms
        return_dict = {}
        for plot in self.plots:

            # Get histograms
            source_hist, bins = self._get_histogram(
                plot,
                weights=(
                    central_weights_trk if plot["type"] == "track" else central_weights
                ),
            )
            target_hist, _ = self._get_histogram(
                plot,
                weights=target_trk if plot["type"] == "track" else target,
                is_target=True,
                root_index=1,  # This only effects histogram for fastjet observables
            )

            # Calculate uncertainties
            variance_dict = {}

            # MC stat uncertainty
            source_stat_var, _ = self._get_histogram(
                plot,
                weights=(
                    central_weights_trk**2
                    if plot["type"] == "track"
                    else central_weights**2
                ),
            )
            variance_dict["mc_stat"] = {
                "name": "MC stat",
                "color": "green",
                "values": source_stat_var,
            }

            # NN initialization uncertainty
            var_hists = []
            use_ensemble_weights = (
                ensemble_weights_trk if plot["type"] == "track" else ensemble_weights
            )
            for i, member_weights in enumerate(use_ensemble_weights):
                member_hist, _ = self._get_histogram(
                    plot, weights=member_weights, ens_index=i+1
                )
                # Remember to normalize to the source histogram!
                norm_factor = np.sum(source_hist) / np.sum(member_hist)
                member_hist *= norm_factor
                var_hists.append(member_hist)
            nn_init_var = np.var(var_hists, axis=0) / len(var_hists)
            variance_dict["nn_init"] = {
                "name": "NN Init",
                "color": "blue",
                "values": nn_init_var,
            }

            # Make and save plot
            if type(bins) is tuple:
                fig = self._build_2d_uncert_plots(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    variances=variance_dict,
                    color=color,
                )
            else:
                fig = self._build_uncert_plot(
                    plot,
                    bins,
                    source_hist,
                    target_hist,
                    variances=variance_dict,
                    color=color,
                )
            extension = ".pdf" if self.use_pdf else ".png"
            store_name = self.store / (plot["key"] + extension)
            fig.savefig(store_name, dpi=300)
            plt.close(fig)
            return_dict[plot["key"]] = store_name

        return return_dict

    def _calculate_central(self, ensemble_weights):
        """_calculate_central - Calculate the central weight from the ensemble
        weights, for now just with a simple mean.
        """
        print(
            f"Calculating central weight from {len(ensemble_weights)} ensemble weights"
        )
        return np.mean(np.array(ensemble_weights).clip(max=100), axis=0)

    def _build_uncert_plot(
        self, plot, bins, source_hist, target_hist, variances=None, color="blue"
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
            variances (dict): Dictionary of dictionaries containined the 
                following information for each uncertainty:
                    - name (str): Name of the uncertainty.
                    - color (str): Color for plotting the uncertainty.
                    - values (np.array): Array of uncertainty values in each bin
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

        # If we have uncertainties, calculate total variance and uncertainty
        if variances is not None:
            total_var = []
            for var in variances.values():
                total_var.append(var["values"])
                # Convert from variances to relative uncertatinties here
                var["values"] = np.sqrt(var["values"]) / source_hist
            total_var = np.sum(total_var, axis=0)
            total_uncert = np.sqrt(total_var)
            ratio_uncert = total_uncert / norm_target_hist
            rel_total_uncert = total_uncert / source_hist

        # Duplicate last bins for all step plots
        norm_target_hist = np.append(norm_target_hist, norm_target_hist[-1])
        rel_mbias = np.append(rel_mbias, rel_mbias[-1])
        rel_total_uncert = np.append(rel_total_uncert, rel_total_uncert[-1])
        if variances is not None:
            for var in variances.values():
                var["values"] = np.append(var["values"], var["values"][-1])

        # Plot
        bin_centers = (bins[1:] + bins[:-1]) / 2
        fig, (ax, rax, vax) = plt.subplots(
            3,
            1,
            figsize=(6, 6.8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        plt.subplots_adjust(hspace=0, top=0.95)

        # Densities
        ax.plot(
            bins,
            norm_target_hist,
            "--",
            label="Target",
            color="black",
            drawstyle="steps-post",
        )
        ax.errorbar(
            bin_centers,
            source_hist,
            yerr=total_uncert if variances is not None else None,
            fmt="o",
            label="Omnifold",
            color=color,
        )
        if not plot["linear_yscale"]:
            ax.set_yscale("log")
        if plot["log_xscale"]:
            ax.set_xscale("log")
        ax.tick_params(axis="x", direction="in", top=True)
        ax.set_ylabel("Counts")
        ax.legend()

        # Ratios
        rax.axhline(1, color="black", linestyle="--")
        rax.errorbar(
            bin_centers,
            ratio,
            yerr=ratio_uncert if variances is not None else None,
            fmt="o",
            color=color,
        )
        rax.set_ylim(0.85, 1.15)
        rax.set_yticks([0.9, 1.0, 1.1])
        rax.set_ylabel("Ratio to target")
        rax.tick_params(axis="x", direction="in", bottom=True, top=False)

        # Uncertainties
        if variances is not None:
            vax.plot(
                bins,
                rel_total_uncert,
                "--",
                color="black",
                label="Total unc.",
                drawstyle="steps-post",
            )
            vax.fill_between(
                bins, 0, rel_total_uncert, step="post", color="gray", alpha=0.3
            )
            for var in variances.values():
                vax.plot(
                    bins,
                    var["values"],
                    "-",
                    color=var["color"],
                    label=var["name"],
                    drawstyle="steps-post",
                )

        # Always want to plot the method bias in the bottom panel
        vax.plot(
            bins,
            rel_mbias,
            "-",
            color="red",
            label="Method bias",
            drawstyle="steps-post",
        )
        vax.set_ylim(0, 0.2)
        vax.set_xlabel(plot["xlabel"])
        vax.set_ylabel("Uncertainty")
        vax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4)

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
        variances=None,
        color="blue",
    ):
        """_build_2d_uncert_plots - Produce a 2D uncertainty plot for a given
        observable. This plot will compare the source histogram to the target
        histogram, and additionally draw all of the uncertainties from the
        variances contained in the optional variances argument detailed below.

        Arguments:
            plot (dict): Dictionary containing the plotting style information
            bins (tuple): Tuple of two arrays of bin edges for the histogram.
            source_hist (np.array): Array of source histogram values.
            target_hist (np.array): Array of target histogram values.
            variances (dict): Dictionary of dictionaries containined the 
                following information for each uncertainty:
                    - name (str): Name of the uncertainty.
                    - color (str): Color for plotting the uncertainty.
                    - values (np.array): Array of uncertainty values in each bin
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

        # If we have uncertainties, calculate total variance and uncertainty
        if variances is not None:
            total_var = []
            for var in variances.values():
                total_var.append(var["values"])
                # Convert from variances to relative uncertatinties here
                var["values"] = np.sqrt(var["values"]) / source_hist
            total_var = np.sum(total_var, axis=0)
            total_uncert = np.sqrt(total_var)
            ratio_uncert = total_uncert / norm_target_hist
            rel_total_uncert = total_uncert / source_hist

        # Drop the bottom row and zero the upper triangle
        xbins, ybins = bins
        ybins = ybins[1:]
        ratio = ratio[:,1:]

        # Plot
        fig = plt.figure()
        ax = plt.gca()
        cax = ax.pcolormesh(
            xbins,
            ybins,
            ratio.T,
            cmap="coolwarm",
            vmin=0.9,
            vmax=1.1,
            shading="auto",
        )
        fig.colorbar(cax, ax=ax)
        ax.set_xlabel(plot["xlabel"])
        ax.set_ylabel(plot["ylabel"])

        return fig

