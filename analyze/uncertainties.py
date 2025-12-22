"""
Uncertainty calculation and management for jet analysis.

This module provides a class for calculating and managing systematic uncertainties
from correlation dimension histograms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


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
            "w_QCD_dd",
            "w_PDF_CT18nnlo",
            "w_Alpha_s1",
            "w_Var2Down",
            "w_Var1Down",
            "w_MPIDown",
            "w_RenDown",
        ]
        self.sherpa_uncertainties = [
            "PS_ME_QCD_dd",
            "PS_ME_PDF_CT18nnlo",
            "PS_ME_Alpha_s1",
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
                "color": "lightblue",
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
                "color": "teal",
                "stochastic": False,
                "prefix": None,
            },
            "muEffReco": {
                "name": "Muon eff. reco.",
                "color": "lightseagreen",
                "stochastic": False,
                "prefix": None,
            },
            "muEffIso": {
                "name": "Muon eff. iso.",
                "color": "seagreen",
                "stochastic": False,
                "prefix": None,
            },
            "muEffTrack": {
                "name": "Muon eff. track.",
                "color": "skyblue",
                "stochastic": False,
                "prefix": None,
            },
            "muEffTrig": {
                "name": "Muon eff. trig.",
                "color": "cadetblue",
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
            "mc-stat": {
                "name": "MC stat",
                "color": "green",
                "stochastic": True,
                "prefix": None,
            },
            "lumi": {
                "name": "Luminosity",
                "color": "pink",
                "stochastic": False,
                "prefix": None,
            },
            "data-stat": {
                "name": "Data stat",
                "color": "blue",
                "stochastic": True,
                "prefix": "bootstrap_data_",
            },
            # "theoryQCD": {
            #     "name": "Theory QCD",
            #     "color": "chartreuse",
            #     "stochastic": False,
            #     "prefix": None,
            # },
            # "theoryPDF": {
            #     "name": "Theory PDF",
            #     "color": "lawngreen",
            #     "stochastic": False,
            #     "prefix": None,
            # },
            # "theoryAlphaS": {
            #     "name": "Theory AlphaS",
            #     "color": "olive",
            #     "stochastic": False,
            #     "prefix": None,
            # },
            # "theoryPSsoft": {
            #     "name": "Theory PS soft",
            #     "color": "palegreen",
            #     "stochastic": False,
            #     "prefix": None,
            # },
            # "theoryPSjet": {
            #     "name": "Theory PS jet",
            #     "color": "lightgreen",
            #     "stochastic": False,
            #     "prefix": None,
            # },
            # "theoryMPI": {
            #     "name": "Theory MPI",
            #     "color": "aquamarine",
            #     "stochastic": False,
            #     "prefix": None,
            # },
            # "theoryPSscale": {
            #     "name": "Theory PS scale",
            #     "color": "lime",
            #     "stochastic": False,
            #     "prefix": None,
            # },
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
            "Unfolding": ["dd", "hv"],
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
            # "Theory": [
            #     # "theoryQCD",
            #     # "theoryPDF",
            #     "theoryAlphaS",
            #     # "theoryPSsoft",
            #     # "theoryPSjet",
            #     # "theoryMPI",
            #     # "theoryPSscale",
            # ],
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
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Calculate all uncertainties from histogram dictionary.
        Will only calculate uncertainties that are defined in the
        uncertainty_definitions dictionary, the rest will be ignored.

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

        Returns:
        --------
        syst_uncerts : dict[str, np.ndarray]
            Dictionary mapping uncertainty names to fractional uncertainty arrays.
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
        measured_hist, measured_hist_var, _ = all_hists[measured_key]

        # Calculate systematic uncertainties
        syst_uncerts = {}
        syst_info = {}

        # MC statistical uncertainty (from variance in measured distribution)
        mc_stat_def = self.uncertainty_definitions.get("mc-stat")
        if mc_stat_def is not None:
            syst_uncerts["mc-stat"] = np.sqrt(measured_hist_var) / measured_hist
            syst_info["mc-stat"] = mc_stat_def.copy()

        # Data statistical uncertainty
        data_stat_def = self.uncertainty_definitions.get("data-stat")
        if data_stat_def is not None:
            prefix = data_stat_def.get("prefix", "bootstrap_data_")
            data_stat_keys = [key for key in all_hists.keys() if key.startswith(prefix)]
            data_stat_hists = np.array([all_hists[key][0] for key in data_stat_keys])
            syst_uncerts["data-stat"] = np.std(data_stat_hists, axis=0) / measured_hist
            syst_info["data-stat"] = data_stat_def.copy()

        # NN initialization uncertainty
        nn_init_def = self.uncertainty_definitions.get("nn-stability")
        if nn_init_def is not None:
            prefix = nn_init_def.get("prefix", "ensemble_")
            nn_init_keys = [key for key in all_hists.keys() if key.startswith(prefix)]
            ensemble_hists = np.array([all_hists[key][0] for key in nn_init_keys])
            nens = len(ensemble_hists)
            # Calculate variance across ensemble members
            nn_init_uncert = np.std(ensemble_hists, axis=0) / np.sqrt(nens)
            if self.multifold_nn_init:
                nn_init_uncert *= 1.253  # Additional factor for multifold nn-stability
            syst_uncerts["nn-stability"] = nn_init_uncert / measured_hist
            syst_info["nn-stability"] = nn_init_def.copy()

        # Data driven uncertainty (from difference between "dd" and "dd-target")
        dd_def = self.uncertainty_definitions.get("dd")
        if dd_def is not None:
            dd_hist, _, _ = all_hists["dd"]
            dd_target_hist, _, _ = all_hists["target_dd"]
            syst_uncerts["dd"] = np.abs(dd_hist - dd_target_hist) / dd_target_hist
            syst_info["dd"] = dd_def.copy()

        # Other systematic uncertainties (from differences with nominal)
        for syst_key, syst_def in self.uncertainty_definitions.items():
            if syst_key in ["nn-stability", "mc-stat", "data-stat", "dd"]:
                continue  # Already handled

            # Check if this systematic exists in all_hists
            if syst_key in all_hists:
                syst_hist, _, _ = all_hists[syst_key]

                # Calculate variance as squared difference
                uncert = np.abs(syst_hist - measured_hist) / measured_hist
                syst_uncerts[syst_key] = uncert
                syst_info[syst_key] = syst_def.copy()

        # Apply uncertainty grouping
        if self.uncertainty_groups:
            syst_uncerts, syst_info = self._apply_grouping(syst_uncerts, syst_info)

        return syst_uncerts, syst_info

    def _apply_grouping(
        self,
        syst_uncerts: Dict[str, np.ndarray],
        syst_info: Dict[str, Dict],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Apply uncertainty grouping by merging in quadrature.

        Arguments:
        ----------
        syst_uncerts : dict[str, np.ndarray]
            Dictionary of individual uncertainty fractional uncertainties.
        syst_info : dict[str, dict]
            Dictionary of individual uncertainty metadata.

        Returns:
        --------
        syst_uncerts : dict[str, np.ndarray]
            Dictionary with grouped uncertainties added.
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
            merged_uncert = np.sqrt(np.sum(
                [syst_uncerts[uncert]**2 for uncert in available_uncertainties],
                axis=0,
            ))

            # Get color from first available uncertainty
            first_color = syst_info[available_uncertainties[0]]["color"]

            # Create merged uncertainty entry
            merged_uncertainties[group_name] = {
                "name": group_name.title(),
                "color": first_color,
                "uncert": merged_uncert,
                "merged_from": available_uncertainties,
            }

            # Hide individual uncertainties if enabled
            if self.hide_individual_uncertainties:
                for uncert_name in available_uncertainties:
                    hidden_uncertainties[uncert_name] = {
                        "uncert": syst_uncerts[uncert_name],
                        "info": syst_info[uncert_name],
                    }
                    del syst_uncerts[uncert_name]
                    del syst_info[uncert_name]

        # Add merged uncertainties to syst_vars and syst_info
        for group_name, merged_uncert in merged_uncertainties.items():
            syst_uncerts[group_name] = merged_uncert["uncert"]
            syst_info[group_name] = {
                "name": merged_uncert["name"],
                "color": merged_uncert["color"],
                "stochastic": False,
            }

        return syst_uncerts, syst_info

    def get_total_uncertainty(
        self,
        all_hists: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        measured_key: str = "nominal",
    ) -> np.ndarray:
        """Calculate total uncertainty (square root of sum of variances).

        Arguments:
        ----------
        all_hists : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
            Dictionary mapping histogram names to tuples of (hist, hist_var, bins).
        measured_key : str, optional
            Key in all_hists for the measured/unfolded distribution
            (default: "nominal").

        Returns:
        --------
        np.ndarray : Total uncertainty (standard deviation) array.
        """
        syst_uncerts, _ = self.calculate_uncertainties(all_hists, measured_key)
        total_var = np.sum(np.array(list(syst_uncerts.values()))**2, axis=0)
        return np.sqrt(total_var)

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
        total_var = np.sum(np.array(syst_uncerts)**2, axis=0)
        return np.sqrt(total_var)
