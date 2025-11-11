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
        """
        if uncertainty_definitions is None:
            uncertainty_definitions = self._get_default_definitions()

        if uncertainty_groups is None:
            uncertainty_groups = self._get_default_groups()

        self.uncertainty_definitions = uncertainty_definitions
        self.uncertainty_groups = uncertainty_groups
        self.hide_individual_uncertainties = hide_individual_uncertainties

    @staticmethod
    def _get_default_definitions() -> Dict[str, Dict]:
        """Get default uncertainty definitions.

        Returns:
        --------
        dict : Dictionary of default uncertainty definitions.
        """
        return {
            "nn-init": {
                "name": "NN Init",
                "color": "aqua",
                "stochastic": True,
                "prefix": "nn-init-",
            },
            "track-eff": {
                "name": "Track eff.",
                "color": "purple",
                "stochastic": False,
                "prefix": None,
            },
            "jet-track-eff": {
                "name": "Jet track eff.",
                "color": "pink",
                "stochastic": False,
                "prefix": None,
            },
            "track-fake": {
                "name": "Track fake",
                "color": "brown",
                "stochastic": False,
                "prefix": None,
            },
            "track-scale": {
                "name": "Track scale",
                "color": "gray",
                "stochastic": False,
                "prefix": None,
            },
            "muon-id": {
                "name": "Muon ID",
                "color": "lightgreen",
                "stochastic": False,
                "prefix": None,
            },
            "muon-ms": {
                "name": "Muon MS",
                "color": "lightblue",
                "stochastic": False,
                "prefix": None,
            },
            "muon-resbias": {
                "name": "Muon resolution bias",
                "color": "deepskyblue",
                "stochastic": False,
                "prefix": None,
            },
            "muon-scale": {
                "name": "Muon scale",
                "color": "teal",
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
                "color": "orange",
                "stochastic": False,
                "prefix": None,
            },
            "mc-stat": {
                "name": "MC stat",
                "color": "green",
                "stochastic": True,
                "prefix": None,
            },
            "data-stat": {
                "name": "Data stat",
                "color": "blue",
                "stochastic": True,
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
            "Tracking": ["track-eff", "jet-track-eff", "track-fake", "track-scale"],
            "Unfolding": ["dd", "hv"],
            "Muon": [
                "muon-id",
                "muon-ms",
                "muon-resbias",
                "muon-scale",
            ],
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
        syst_vars : dict[str, np.ndarray]
            Dictionary mapping uncertainty names to variance arrays.
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
        syst_vars = {}
        syst_info = {}

        # MC statistical uncertainty (from variance in measured distribution)
        mc_stat_def = self.uncertainty_definitions.get("mc-stat")
        if mc_stat_def is not None:
            syst_vars["mc-stat"] = measured_hist_var
            syst_info["mc-stat"] = mc_stat_def.copy()

        # Data statistical uncertainty (from variance in reco level data histogram)
        data_stat_def = self.uncertainty_definitions.get("data-stat")
        if data_stat_def is not None:
            _, data_hist_var, _ = all_hists["data-stat"]
            syst_vars["data-stat"] = data_hist_var
            syst_info["data-stat"] = data_stat_def.copy()

        # NN initialization uncertainty (from variance across ensemble members)
        nn_init_def = self.uncertainty_definitions.get("nn-init")
        if nn_init_def is not None:
            prefix = nn_init_def.get("prefix", "nn-init-")
            nn_init_keys = [
                key
                for key in all_hists.keys()
                if key.startswith(prefix) and key != "nn-init"
            ]
            if nn_init_keys:
                # Extract all ensemble member histograms
                ensemble_hists = []
                for key in sorted(nn_init_keys):
                    hist, _, _ = all_hists[key]
                    ensemble_hists.append(hist)

                if ensemble_hists:
                    ensemble_hists = np.array(ensemble_hists)
                    # Calculate variance across ensemble members
                    # Normalize each member to match the nominal
                    for i in range(len(ensemble_hists)):
                        norm_factor = np.sum(measured_hist) / np.sum(ensemble_hists[i])
                        ensemble_hists[i] *= norm_factor
                    nn_init_var = np.var(ensemble_hists, axis=0) / (
                        len(ensemble_hists) - 1
                    )
                    syst_vars["nn-init"] = nn_init_var
                    syst_info["nn-init"] = nn_init_def.copy()

        # Data driven uncertainty (from difference between "dd" and "dd-target")
        dd_def = self.uncertainty_definitions.get("dd")
        dd_target_def = self.uncertainty_definitions.get("dd-target")
        if dd_def is not None and dd_target_def is not None:
            dd_hist, _, _ = all_hists["dd"]
            dd_target_hist, _, _ = all_hists["dd-target"]
            dd_var = np.abs(dd_hist - dd_target_hist) ** 2
            syst_vars["dd"] = dd_var
            syst_info["dd"] = dd_def.copy()

        # Other systematic uncertainties (from differences with nominal)
        for syst_key, syst_def in self.uncertainty_definitions.items():
            if syst_key in ["nn-init", "mc-stat", "data-stat", "dd"]:
                continue  # Already handled

            # Check if this systematic exists in all_hists
            if syst_key in all_hists:
                syst_hist, _, _ = all_hists[syst_key]

                # Normalize to match measured distribution
                norm_factor = np.sum(measured_hist) / np.sum(syst_hist)
                syst_hist *= norm_factor

                # Calculate variance as squared difference
                syst_var = np.abs(syst_hist - measured_hist) ** 2
                syst_vars[syst_key] = syst_var
                syst_info[syst_key] = syst_def.copy()

        # Apply uncertainty grouping
        if self.uncertainty_groups:
            syst_vars, syst_info = self._apply_grouping(syst_vars, syst_info)

        return syst_vars, syst_info

    def _apply_grouping(
        self,
        syst_vars: Dict[str, np.ndarray],
        syst_info: Dict[str, Dict],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Apply uncertainty grouping by merging in quadrature.

        Arguments:
        ----------
        syst_vars : dict[str, np.ndarray]
            Dictionary of individual uncertainty variances.
        syst_info : dict[str, dict]
            Dictionary of individual uncertainty metadata.

        Returns:
        --------
        syst_vars : dict[str, np.ndarray]
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
                if uncert_name in syst_vars:
                    available_uncertainties.append(uncert_name)

            if not available_uncertainties:
                continue

            # Merge variances in quadrature
            merged_var = np.sum(
                [syst_vars[uncert] for uncert in available_uncertainties],
                axis=0,
            )

            # Get color from first available uncertainty
            first_color = syst_info[available_uncertainties[0]]["color"]

            # Create merged uncertainty entry
            merged_uncertainties[group_name] = {
                "name": group_name.title(),
                "color": first_color,
                "var": merged_var,
                "merged_from": available_uncertainties,
            }

            # Hide individual uncertainties if enabled
            if self.hide_individual_uncertainties:
                for uncert_name in available_uncertainties:
                    hidden_uncertainties[uncert_name] = {
                        "var": syst_vars[uncert_name],
                        "info": syst_info[uncert_name],
                    }
                    del syst_vars[uncert_name]
                    del syst_info[uncert_name]

        # Add merged uncertainties to syst_vars and syst_info
        for group_name, merged_uncert in merged_uncertainties.items():
            syst_vars[group_name] = merged_uncert["var"]
            syst_info[group_name] = {
                "name": merged_uncert["name"],
                "color": merged_uncert["color"],
                "stochastic": False,
            }

        return syst_vars, syst_info

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
        syst_vars, _ = self.calculate_uncertainties(all_hists, measured_key)
        total_var = np.sum(list(syst_vars.values()), axis=0)
        return np.sqrt(total_var)
