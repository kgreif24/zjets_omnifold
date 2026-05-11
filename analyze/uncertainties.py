"""
Uncertainty calculation and management for jet analysis.

This module provides a class for calculating and managing systematic uncertainties
from correlation dimension histograms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats


class UncertaintyCalculator:
    """Calculate and manage systematic uncertainties from histograms.

    Attributes:
    -----------
    uncertainty_definitions : dict
        Dictionary mapping uncertainty keys to their definitions (name, color, etc.)
    uncertainty_groups : dict
        Dictionary mapping group names to lists of uncertainty keys
    hide_individual_uncertainties : bool
        If True, hide (don't return) individual uncertainties
        when they are part of a group
    """

    def __init__(
        self,
        uncertainty_definitions: Optional[Dict[str, Dict]] = None,
        uncertainty_groups: Optional[Dict[str, List[str]]] = None,
        hide_individual_uncertainties: bool = True,
        multifold_nn_init: bool = False,
    ):
        """Initialize the UncertaintyCalculator.

        Arguments:
        ----------
        uncertainty_definitions : dict, optional
            Dictionary mapping uncertainty keys to definitions. Each definition
            should contain at a minimum:
            - "name": Display name for the uncertainty
            - "color": Color for plotting
            - "stochastic": bool, whether this is a stochastic uncertainty
            - "prefix": str or None, prefix for finding ensemble members
            If None, uses default definitions.
        uncertainty_groups : dict, optional
            Dictionary mapping group names to lists of uncertainty keys.
            If None, uses default groups.
        hide_individual_uncertainties : bool, optional
            If True, hide individual uncertainties when they are part of a group
            (default: True).
        multifold_nn_init : bool, optional
            If True, use the multifold nn-stability uncertainty, which only differs
            from the Omnifold one by an additional numeric factor
        """
        if uncertainty_definitions is None:
            uncertainty_definitions = self._get_default_definitions()

        if uncertainty_groups is None:
            uncertainty_groups = self._get_default_groups()

        self.uncertainty_definitions = uncertainty_definitions
        self.uncertainty_groups = uncertainty_groups
        self.hide_individual_uncertainties = hide_individual_uncertainties
        self.multifold_nn_init = multifold_nn_init

        # Hardcode the theory uncertainties, since we will only ever care about
        # the total theory uncertainty and don't need to visualize the budget
        self.madgraph_uncertainties = [
            "weights_theoryQCD",
            "weights_theoryPDF",
            "weights_theoryAlphaS",
            "weights_theoryPSjet",
            "weights_theoryPSsoft",
            "weights_theoryMPI",
            "weights_theoryPSscale",
            "weights_ns_theory_diboson_up",
            "weights_ns_theory_ew_zjj_up",
        ]
        self.sherpa_uncertainties = [
            "weights_theoryQCD",
            "weights_theoryPDF",
            "weights_theoryAlphaS",
        ]

    @staticmethod
    def _get_default_definitions() -> Dict[str, Dict]:
        """Get default uncertainty definitions.

        Returns:
        --------
        dict : Dictionary of default uncertainty definitions.
        """
        return {
            "nn-stability": {
                "name": "NN stability",
                "color": "aqua",
                "stochastic": True,
                "prefix": "ensemble_",
            },
            "trackEffMain": {
                "name": "Track eff.",
                "color": "purple",
                "stochastic": False,
                "prefix": None,
            },
            "trackEffJet": {
                "name": "Jet track eff.",
                "color": "pink",
                "stochastic": False,
                "prefix": None,
            },
            "trackFake": {
                "name": "Track fake",
                "color": "brown",
                "stochastic": False,
                "prefix": None,
            },
            "trackPtScale": {
                "name": "Track scale",
                "color": "gray",
                "stochastic": False,
                "prefix": None,
            },
            "muCalID": {
                "name": "Muon ID",
                "color": "lightgreen",
                "stochastic": False,
                "prefix": None,
            },
            "muCalMS": {
                "name": "Muon MS",
                "color": "red",
                "stochastic": False,
                "prefix": None,
            },
            "muCalResBias": {
                "name": "Muon resolution bias",
                "color": "deepskyblue",
                "stochastic": False,
                "prefix": None,
            },
            "muCalScale": {
                "name": "Muon scale",
                "color": "purple",
                "stochastic": False,
                "prefix": None,
            },
            "muEffReco": {
                "name": "Muon eff. reco.",
                "color": "orange",
                "stochastic": False,
                "prefix": None,
            },
            "muEffIso": {
                "name": "Muon eff. iso.",
                "color": "yellow",
                "stochastic": False,
                "prefix": None,
            },
            "muEffTrack": {
                "name": "Muon eff. track.",
                "color": "gold",
                "stochastic": False,
                "prefix": None,
            },
            "muEffTrig": {
                "name": "Muon eff. trig.",
                "color": "pink",
                "stochastic": False,
                "prefix": None,
            },
            "pileup": {
                "name": "Pileup",
                "color": "brown",
                "stochastic": False,
                "prefix": None,
            },
            "dd": {
                "name": "Data driven",
                "color": "red",
                "stochastic": False,
                "prefix": None,
            },
            "hv": {
                "name": "Hidden variable",
                "color": "blue",
                "stochastic": False,
                "prefix": None,
            },
            "hvhad": {
                "name": "Hadron composition",
                "color": "orange",
                "stochastic": False,
                "prefix": None,
            },
            "mc-stat-test": {
                "name": "MC stat (test)",
                "color": "green",
                "stochastic": True,
                "prefix": "bootstrap_mc_test_",
            },
            "mc-stat-train": {
                "name": "MC stat (train)",
                "color": "magenta",
                "stochastic": True,
                "prefix": "bootstrap_mc_",
            },
            "lumi": {
                "name": "Luminosity",
                "color": "pink",
                "stochastic": False,
                "prefix": None,
            },
            "data-stat": {
                "name": "Data stat",
                "color": "mediumslateblue",
                "stochastic": True,
                "prefix": "bootstrap_data_",
            },
            "theoryQCD": {
                "name": "Theory QCD",
                "color": "gold",
                "stochastic": False,
                "prefix": None,
            },
            "theoryPDF": {
                "name": "Theory PDF",
                "color": "goldenrod",
                "stochastic": False,
                "prefix": None,
            },
            "theoryAlphaS": {
                "name": "Theory AlphaS",
                "color": "mediumslateblue",
                "stochastic": False,
                "prefix": None,
            },
            "theoryPSsoft": {
                "name": "Theory PS soft",
                "color": "indigo",
                "stochastic": False,
                "prefix": None,
            },
            "theoryPSjet": {
                "name": "Theory PS jet",
                "color": "crimson",
                "stochastic": False,
                "prefix": None,
            },
            "theoryMPI": {
                "name": "Theory MPI",
                "color": "dimgray",
                "stochastic": False,
                "prefix": None,
            },
            "theoryPSscale": {
                "name": "Theory PS scale",
                "color": "saddlebrown",
                "stochastic": False,
                "prefix": None,
            },
            "topBackground": {
                "name": "Top background",
                "color": "navajowhite",
                "stochastic": False,
                "prefix": None,
            },
            "nonstrongDiboson": {
                "name": "Non-strong diboson",
                "color": "rebeccapurple",
                "stochastic": False,
                "prefix": None,
            },
            "nonstrongEW": {
                "name": "Non-strong EW",
                "color": "sandybrown",
                "stochastic": False,
                "prefix": None,
            },
        }

    @staticmethod
    def _get_default_groups() -> Dict[str, List[str]]:
        """Get default uncertainty groups.

        Returns:
        --------
        dict : Dictionary of default uncertainty groups.
        """
        return {
            "Tracking": ["trackEffMain", "trackEffJet", "trackFake", "trackPtScale"],
            "Unfolding": ["dd", "hv", "hvhad"],
            "Muon": [
                "muCalID",
                "muCalMS",
                "muCalResBias",
                "muCalScale",
                "muEffReco",
                "muEffIso",
                "muEffTrack",
                "muEffTrig",
            ],
            "MC Stat": ["mc-stat-test", "mc-stat-train"],
            "Data Stat": ["data-stat"],
            "Theory": [
                "theoryQCD",
                "theoryPDF",
                "theoryAlphaS",
                "theoryPSsoft",
                "theoryPSjet",
                "theoryMPI",
                "theoryPSscale",
            ],
            "Non-strong": ["nonstrongDiboson", "nonstrongEW"],
        }

    def add_uncertainty(
        self,
        key: str,
        name: str,
        color: str,
        stochastic: bool = False,
        prefix: Optional[str] = None,
    ):
        """Add or update an uncertainty definition.

        Arguments:
        ----------
        key : str
            Key for the uncertainty (used in histogram dictionary).
        name : str
            Display name for the uncertainty.
        color : str
            Color for plotting.
        stochastic : bool, optional
            Whether this is a stochastic uncertainty (default: False).
        prefix : str or None, optional
            Prefix for finding ensemble members (default: None).
        """
        self.uncertainty_definitions[key] = {
            "name": name,
            "color": color,
            "stochastic": stochastic,
            "prefix": prefix,
        }

    def add_uncertainty_group(
        self, group_name: str, uncertainty_keys: List[str], replace: bool = False
    ):
        """Add or update an uncertainty group.

        Arguments:
        ----------
        group_name : str
            Name of the group.
        uncertainty_keys : list of str
            List of uncertainty keys to include in the group.
        replace : bool, optional
            If True, replace existing group. If False, merge with existing
            (default: False).
        """
        if replace or group_name not in self.uncertainty_groups:
            self.uncertainty_groups[group_name] = uncertainty_keys
        else:
            # Merge with existing, avoiding duplicates
            existing = set(self.uncertainty_groups[group_name])
            new = set(uncertainty_keys)
            self.uncertainty_groups[group_name] = list(existing | new)

    def remove_uncertainty(self, key: str):
        """Remove an uncertainty definition.

        Arguments:
        ----------
        key : str
            Key of the uncertainty to remove.
        """
        if key in self.uncertainty_definitions:
            del self.uncertainty_definitions[key]

    def remove_uncertainty_group(self, group_name: str):
        """Remove an uncertainty group.

        Arguments:
        ----------
        group_name : str
            Name of the group to remove.
        """
        if group_name in self.uncertainty_groups:
            del self.uncertainty_groups[group_name]

    def calculate_uncertainties(
        self,
        all_hists: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        measured_key: str = "nominal",
        smooth_hv: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict]]:
        """Calculate all uncertainties from histogram dictionary.
        Will only calculate uncertainties that are defined in the
        uncertainty_definitions dictionary, the rest will be ignored.

        Will also fill the covariance matrices for all uncertainties and
        return a dictionary of covariance matrices for all of the uncertainties
        which can be used for plotting.

        Arguments:
        ----------
        all_hists : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
            Dictionary mapping histogram names to tuples of (hist, hist_var, bins)
            where:
            - hist: The histogram values
            - hist_var: The variance of the histogram values (sum of squared weights)
            - bins: The bin edges
        measured_key : str, optional
            Key in all_hists for the measured/unfolded distribution (default: "nominal")
        smooth_hv : bool, optional
            If True, smooth the hidden variable uncertainties only (default: True)

        Returns:
        --------
        syst_uncerts : dict[str, np.ndarray]
            Dictionary mapping uncertainty names to signed fractional uncertainty
            arrays. Non-stochastic uncertainties retain their sign; stochastic
            uncertainties (mc-stat, nn-stability, etc.) are inherently non-negative
            since they are derived from std/sqrt(var). Uncertainties are not grouped.
            Call process_signed_uncertainties to obtain absolute-valued, grouped
            uncertainties suitable for budget plots and total uncertainty computation.
        syst_covs : dict[str, np.ndarray]
            Dictionary mapping uncertainty names to covariance matrices.
            Covariance matrices are computed using signed uncertainties.
        syst_info : dict[str, dict]
            Dictionary mapping uncertainty names to metadata (name, color, etc.).
            Useful for plotting only uncertainties that are active (in the dictionary)
        """

        # Extract measured histogram
        if measured_key not in all_hists:
            available = list(all_hists.keys())
            raise KeyError(
                f"Key '{measured_key}' not found in all_hists."
                f" Available keys: {available}"
            )
        measured_hist, measured_hist_var, bins = all_hists[measured_key]

        # Calculate systematic uncertainties
        syst_uncerts = {}
        syst_covs = {}
        syst_info = {}

        # MC test statistical uncertainty
        mc_stat_def = self.uncertainty_definitions.get("mc-stat-test")
        if mc_stat_def is not None:

            # There are three possible ways to compute MC test uncertainty:
            # 1. Bootstrap the MC test sample
            # 2. Use the weighted Poisson counting error in the measured histogram
            # 3. For distributions where bins are correlated, a covariance matrix is built and the sqrt of the diagonal of the covariance matrix gives the uncertainty
            # Default to 2 unless the bootstrap hists are provided, in which case use 1
            prefix = mc_stat_def.get("prefix", "bootstrap_mc_test_")
            mc_stat_test_bs_keys = [
                key for key in all_hists.keys() if key.startswith(prefix)
            ]
            if measured_hist_var.ndim == 2:
                var = np.diag(measured_hist_var)
                cov = measured_hist_var # Is a 2x2 non-diagonal cov matrix
                syst_uncerts["mc-stat-test"] = np.sqrt(var) / measured_hist
                syst_covs["mc-stat-test"]    = cov
                
            # Note with bootstraps fill off diagonal elements of the covariance matrix
            elif len(mc_stat_test_bs_keys) > 0:
                mc_stat_test_bs_hists = np.array(
                    [all_hists[key][0] for key in mc_stat_test_bs_keys]
                )
                mc_stat_test_bs_uncert_unnorm = np.std(mc_stat_test_bs_hists, axis=0)
                syst_uncerts["mc-stat-test"] = (
                    mc_stat_test_bs_uncert_unnorm / measured_hist
                )
                syst_covs["mc-stat-test"] = self._fill_covariance_matrix(
                    mc_stat_test_bs_hists,
                    means=np.mean(mc_stat_test_bs_hists, axis=0),
                )
            # With only Poisson bin counts the covariance matrix is diagonal
            else:
                mc_stat_uncert_unnorm = np.sqrt(measured_hist_var)
                syst_uncerts["mc-stat-test"] = mc_stat_uncert_unnorm / measured_hist
                syst_covs["mc-stat-test"] = np.diag(measured_hist_var)
            syst_info["mc-stat-test"] = mc_stat_def.copy()

        
        # MC statistical uncertainty (from bootstrap MC stat uncertainty)
        mc_stat_bs_def = self.uncertainty_definitions.get("mc-stat-train")
        if mc_stat_bs_def is not None:
            prefix = mc_stat_bs_def.get("prefix", "bootstrap_mc_")
            mc_stat_bs_keys = [
                key for key in all_hists.keys() if key.startswith(prefix)
            ]
            mc_stat_bs_hists = np.array([all_hists[key][0] for key in mc_stat_bs_keys])
            mc_stat_bs_uncert_unnorm = np.std(mc_stat_bs_hists, axis=0)
            syst_uncerts["mc-stat-train"] = mc_stat_bs_uncert_unnorm / measured_hist
            syst_covs["mc-stat-train"] = self._fill_covariance_matrix(
                mc_stat_bs_hists,
                means=np.mean(mc_stat_bs_hists, axis=0),
            )
            syst_info["mc-stat-train"] = mc_stat_bs_def.copy()

        # Data statistical uncertainty
        data_stat_def = self.uncertainty_definitions.get("data-stat")
        if data_stat_def is not None:
            prefix = data_stat_def.get("prefix", "bootstrap_data_")
            data_stat_keys = [key for key in all_hists.keys() if key.startswith(prefix)]
            data_stat_hists = np.array([all_hists[key][0] for key in data_stat_keys])
            data_stat_uncert_unnorm = np.std(data_stat_hists, axis=0)
            syst_uncerts["data-stat"] = data_stat_uncert_unnorm / measured_hist
            syst_covs["data-stat"] = self._fill_covariance_matrix(
                data_stat_hists,
                means=np.mean(data_stat_hists, axis=0),
            )
            syst_info["data-stat"] = data_stat_def.copy()

        # NN initialization uncertainty
        nn_init_def = self.uncertainty_definitions.get("nn-stability")
        if nn_init_def is not None:
            prefix = nn_init_def.get("prefix", "ensemble_")
            nn_init_keys = [key for key in all_hists.keys() if key.startswith(prefix)]
            ensemble_hists = np.array([all_hists[key][0] for key in nn_init_keys])
            nens = len(ensemble_hists)
            # Calculate variance across ensemble members
            nn_init_uncert_unnorm = np.std(ensemble_hists, axis=0) / np.sqrt(nens)
            if self.multifold_nn_init:
                nn_init_uncert_unnorm *= (
                    1.253  # Additional factor for multifold nn-stability
                )
            syst_uncerts["nn-stability"] = nn_init_uncert_unnorm / measured_hist
            syst_covs["nn-stability"] = (
                self._fill_covariance_matrix(
                    ensemble_hists,
                    means=np.mean(ensemble_hists, axis=0),
                )
                / nens
            )
            syst_info["nn-stability"] = nn_init_def.copy()

        # Data driven uncertainty (from difference between "dd" and "dd-target")
        dd_def = self.uncertainty_definitions.get("dd")
        if dd_def is not None:
            dd_hist, _, _ = all_hists["dd"]
            dd_target_hist, _, _ = all_hists["target_dd"]
            dd_uncert_unnorm = dd_hist - dd_target_hist
            syst_uncerts["dd"] = dd_uncert_unnorm / measured_hist
            # Note we re-normalize the dd uncertainty to match the measured histogram!
            syst_covs["dd"] = self._fill_covariance_matrix(
                [dd_uncert_unnorm * (measured_hist / dd_target_hist)],
            )
            syst_info["dd"] = dd_def.copy()

        # Other systematic uncertainties (from differences with nominal)
        for syst_key, syst_def in self.uncertainty_definitions.items():
            if syst_key in ["nn-stability", "mc-stat", "data-stat", "dd"]:
                continue  # Already handled

            # Check if this systematic exists in all_hists
            if syst_key in all_hists:
                syst_hist, _, _ = all_hists[syst_key]
                uncert_unnorm = syst_hist - measured_hist
                if syst_key in ["hv", "hvhad"] and smooth_hv:
                    bin_centers = (bins[1:] + bins[:-1]) / 2
                    uncert_unnorm = self._smooth_uncertainty(uncert_unnorm, bin_centers)
                syst_uncerts[syst_key] = uncert_unnorm / measured_hist
                syst_covs[syst_key] = self._fill_covariance_matrix([uncert_unnorm])
                syst_info[syst_key] = syst_def.copy()

        return syst_uncerts, syst_covs, syst_info

    def process_signed_uncertainties(
        self,
        signed_uncerts: Dict[str, np.ndarray],
        syst_covs: Dict[str, np.ndarray],
        syst_info: Dict[str, Dict],
        ungrouped: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict]]:
        """Process signed fractional uncertainties into unsigned grouped uncertainties.

        Takes the output of calculate_uncertainties (signed, ungrouped) and returns
        absolute-valued, grouped uncertainties suitable for uncertainty budget plots
        and total uncertainty computation.

        Covariance matrices are filled with signed uncertainties and passed through
        unchanged (only grouped by summing) to preserve correct statistical properties.

        Arguments:
        ----------
        signed_uncerts : dict[str, np.ndarray]
            Signed fractional uncertainties from calculate_uncertainties.
        syst_covs : dict[str, np.ndarray]
            Covariance matrices from calculate_uncertainties.
        syst_info : dict[str, dict]
            Metadata from calculate_uncertainties.
        ungrouped : bool
            If True, do not group uncertainties and only return the absolute values

        Returns:
        --------
        syst_uncerts : dict[str, np.ndarray]
            Unsigned (absolute value applied) fractional uncertainties, grouped
            in quadrature according to uncertainty_groups.
        syst_covs : dict[str, np.ndarray]
            Covariance matrices grouped by summing.
        syst_info : dict[str, dict]
            Metadata for each uncertainty (grouped entries reflect group names).
        """
        # Apply absolute value to non-stochastic uncertainties.
        # Stochastic uncertainties (nn-stability, mc-stat, data-stat) are already
        # non-negative since they are computed via std / sqrt(var).
        abs_uncerts = {}
        for key, uncert in signed_uncerts.items():
            syst_def = self.uncertainty_definitions.get(key, {})
            if syst_def.get("stochastic", False):
                abs_uncerts[key] = uncert
            else:
                abs_uncerts[key] = np.abs(uncert)

        # Shallow-copy covs and info so _apply_grouping does not mutate the
        # caller's dictionaries when it deletes individual keys.
        covs_copy = dict(syst_covs)
        info_copy = dict(syst_info)

        # Apply grouping (adds in quadrature for uncerts, sums for covs).
        if self.uncertainty_groups and not ungrouped:
            abs_uncerts, covs_copy, info_copy = self._apply_grouping(
                abs_uncerts, covs_copy, info_copy
            )

        return abs_uncerts, covs_copy, info_copy

    def _apply_grouping(
        self,
        syst_uncerts: Dict[str, np.ndarray],
        syst_covs: Dict[str, np.ndarray],
        syst_info: Dict[str, Dict],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict]]:
        """Apply uncertainty grouping by merging in quadrature.

        Arguments:
        ----------
        syst_uncerts : dict[str, np.ndarray]
            Dictionary of individual uncertainty fractional uncertainties.
        syst_covs : dict[str, np.ndarray]
            Dictionary of individual uncertainty covariance matrices.
        syst_info : dict[str, dict]
            Dictionary of individual uncertainty metadata.

        Returns:
        --------
        syst_uncerts : dict[str, np.ndarray]
            Dictionary with grouped uncertainties added.
        syst_covs : dict[str, np.ndarray]
            Dictionary with grouped uncertainty covariance matrices added.
        syst_info : dict[str, dict]
            Dictionary with grouped uncertainty metadata added.
        """
        merged_uncertainties = {}
        hidden_uncertainties = {}

        for group_name, individual_uncertainties in self.uncertainty_groups.items():
            # Check which uncertainties are available
            available_uncertainties = []
            for uncert_name in individual_uncertainties:
                if uncert_name in syst_uncerts:
                    available_uncertainties.append(uncert_name)

            if not available_uncertainties:
                continue

            # Merge variances in quadrature
            merged_uncert = np.sqrt(
                np.sum(
                    [syst_uncerts[uncert] ** 2 for uncert in available_uncertainties],
                    axis=0,
                )
            )

            # Merge covariance matrices by adding
            merged_cov = np.sum(
                [syst_covs[uncert] for uncert in available_uncertainties],
                axis=0,
            )

            # Get color from first available uncertainty
            first_color = syst_info[available_uncertainties[0]]["color"]

            # Create merged uncertainty entry
            merged_uncertainties[group_name] = {
                "uncert": merged_uncert,
                "cov": merged_cov,
                "merged_from": available_uncertainties,
                "info": {
                    "name": group_name.title(),
                    "color": first_color,
                    "stochastic": False,
                    "prefix": None,
                },
            }

            # Hide individual uncertainties if enabled
            if self.hide_individual_uncertainties:
                for uncert_name in available_uncertainties:
                    hidden_uncertainties[uncert_name] = {
                        "uncert": syst_uncerts[uncert_name],
                        "cov": syst_covs[uncert_name],
                        "info": syst_info[uncert_name],
                    }
                    del syst_uncerts[uncert_name]
                    del syst_covs[uncert_name]
                    del syst_info[uncert_name]

        # Add merged uncertainties to syst_vars and syst_info
        for group_name, merged_uncert in merged_uncertainties.items():
            syst_uncerts[group_name] = merged_uncert["uncert"]
            syst_covs[group_name] = merged_uncert["cov"]
            syst_info[group_name] = merged_uncert["info"]

        return syst_uncerts, syst_covs, syst_info

    def get_total_theory_uncertainty(
        self,
        all_hists: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        measured_key: str = "target",
        is_madgraph: bool = True,
    ) -> np.ndarray:
        """Calculate total MadGraph uncertainty (square root of sum of variances).

        Arguments:
        ----------
        all_hists : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
            Dictionary mapping histogram names to tuples of (hist, hist_var, bins).
        measured_key : str, optional
            Key in all_hists for the measured/unfolded distribution
            (default: "target").
        is_madgraph : bool, optional
            Whether the theory uncertainties are MadGraph uncertainties.
            If set to False, the theory uncertainties are Sherpa uncertainties.
            (default: True).

        Returns:
        --------
        np.ndarray : Total MadGraph uncertainty (standard deviation) array.
        """

        central_hist, central_hist_var, _ = all_hists[measured_key]
        if central_hist_var.ndim == 2:
            var = np.diag(central_hist_var)
            cov = central_hist_var # Is a 2x2 non-diagonal cov matrix
            syst_uncerts = [np.sqrt(var) / central_hist]
        else:
            syst_uncerts = [np.sqrt(central_hist_var) / central_hist]

        if is_madgraph:
            weight_names = self.madgraph_uncertainties
        else:
            weight_names = self.sherpa_uncertainties
        for weight_name in weight_names:
            if weight_name in all_hists:
                syst_hist, _, _ = all_hists[weight_name]
                syst_uncert = np.abs(syst_hist - central_hist) / central_hist
                syst_uncerts.append(syst_uncert)
        total_var = np.sum(np.array(syst_uncerts) ** 2, axis=0)
        return np.sqrt(total_var)

    # Method that calculates a covariance matrix based on one of two types of inputs:
    # 1. a list of Hessian uncertainty variations (aka 'nuisance parameters')
    #    this is activated when there is no third argument
    #    as usual, such uncertainty components are uncorrelated wrt each other but
    #    fully correlated across bins.
    # 2. Bootstrap variations (aka resamplings), activated when uncerts_mean is defined
    #    These uncertainties have a magnitude corresponding to the sample covariance
    # Note method includes an option to decorrelate the uncertainties between bins,
    # as is usually done for unfolding uncertainties
    def _fill_covariance_matrix(
        self,
        syst_hists: List[np.ndarray],
        means: np.ndarray = None,
    ):
        """Fill the covariance matrix for a list of uncertainty histograms.

        Note uncertainties are not normalized by the measured histogram, so the
        covariance matrix is not normalized by the measured histogram.

        Arguments:
        ----------
        syst_hists : List[np.ndarray]
            List of histograms representing the uncertainty variations.
        means : np.ndarray, optional
            Mean values of the uncertainty variations.
            Including this argument activates the bootstrap covariance calculation
            as opposed to the Hessian one.

        Returns:
        --------
        np.ndarray : Covariance matrix with shape (n_bins, n_bins).
        """
        # Convert to 2D array: shape (n_hists, n_bins)
        H = np.asarray(syst_hists)

        if means is None:
            # Hessian: v[i,j] = sum_k(H[k,i] * H[k,j]) = H.T @ H
            cov = H.T @ H
        else:
            # Bootstrap: sample covariance = (H - means).T @ (H - means) / (n - 1)
            centered = H - means  # Broadcasting: (n_hists, n_bins) - (n_bins,)
            cov = centered.T @ centered / (len(syst_hists) - 1)
        return cov

    # Gaussian Kernel smoothing
    # Does not consider the 'uncertainty on the uncertainty'
    def _smooth_uncertainty(
        self, uncert: np.ndarray, bin_centers: np.ndarray
    ) -> np.ndarray:
        # Parameter for smoothing
        # The Gaussian Kernel width will be 'full-range'/Nsig
        Nsig = 10
        xrange = bin_centers[-1] - bin_centers[0]
        logScale = bin_centers[0] > 0
        if logScale:
            xrange = np.log(bin_centers[-1]) - np.log(bin_centers[0])
        kernel_width = xrange / Nsig

        # will hold the uncertainty
        smooth = uncert.copy()
        for bin_i in range(0, len(bin_centers)):
            x_i = np.log(bin_centers[bin_i]) if logScale else bin_centers[bin_i]
            sumw = sumwy = 0
            for bin_j in range(0, len(bin_centers)):
                x_j = np.log(bin_centers[bin_j]) if logScale else bin_centers[bin_j]
                # Kernel weight
                w = stats.norm.pdf((x_j - x_i) / kernel_width)
                sumw += w
                sumwy += w * uncert[bin_j]

            smooth[bin_i] = sumwy / sumw
        return smooth
