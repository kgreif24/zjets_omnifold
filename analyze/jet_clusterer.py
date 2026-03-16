"""
Jet clustering class for parallel processing of events.
"""

import numpy as np
import awkward as ak
import fastjet as fj
import vector
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import contextlib
import os


# Module-level variable to store worker instance (set by initializer)
_worker_instance = None


def _init_worker(clusterer_instance):
    """
    Initialize worker process with a JetClusterer instance.
    This is called once per worker process, avoiding repeated pickling.
    """
    global _worker_instance
    _worker_instance = clusterer_instance


class JetClusterer:
    """
    Class for clustering jets from particle kinematics with parallel processing.
    """

    def __init__(self, pt, eta, phi, masses=None):
        """
        Initialize the JetClusterer with particle kinematics.

        Parameters:
        -----------
        pt : awkward.Array
            Transverse momenta of particles, shape (n_events, n_particles)
        eta : awkward.Array
            Pseudorapidities of particles, shape (n_events, n_particles)
        phi : awkward.Array
            Azimuthal angles of particles, shape (n_events, n_particles)
        masses : awkward.Array, optional
            Masses of particles, shape (n_events, n_particles).
            If None, assumes massless particles.
        """
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.masses = masses

    @staticmethod
    def _process_single_event_cluster(args):
        """
        Worker function to process a single event for parallel execution.
        Uses the module-level worker instance set by the initializer.

        This is a static method so it can be easily pickled for multiprocessing.
        The worker instance is set via the module-level _worker_instance variable
        by the Pool initializer.

        Parameters:
        -----------
        args : tuple
            Tuple containing (event_idx, algorithm, R, ptmin) where event_idx is the
            index to extract from the worker instance arrays.

        Returns:
        --------
        jet_constituents : list[np.ndarray]
            List of numpy arrays, one per jet, each with shape (n_constituents, 4)
            where columns are (E, px, py, pz). Returns empty list if processing fails
            or if no jets above ptmin.
        """
        event_idx, algorithm, R, ptmin = args

        try:
            # Extract event data from awkward arrays (conversion happens in parallel)
            # Use instance arrays set by initializer
            event_pt = ak.to_numpy(_worker_instance.pt[event_idx])
            event_eta = ak.to_numpy(_worker_instance.eta[event_idx])
            event_phi = ak.to_numpy(_worker_instance.phi[event_idx])
            event_masses = (
                ak.to_numpy(_worker_instance.masses[event_idx])
                if _worker_instance.masses is not None
                else None
            )

            # Suppress fastjet splash screen output by redirecting stdout/stderr
            # This is especially important in parallel processing where it
            # prints multiple times
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                    devnull
                ):

                    # Convert particles to PseudoJets using class method
                    particles = _worker_instance._particles_to_pseudojets(
                        event_pt, event_eta, event_phi, event_masses
                    )

                    if len(particles) == 0:
                        return []

                    # Create jet definition (recreate here to avoid pickling issues)
                    jet_def = fj.JetDefinition(algorithm, R)

                    # Cluster jets
                    cluster_seq = fj.ClusterSequence(particles, jet_def)

            # Extract jets above ptmin threshold
            jets = cluster_seq.inclusive_jets(ptmin)

            # Extract constituents for each jet using class method
            # Keep cluster_seq in scope while extracting constituents
            jet_constituents = []
            for jet in jets:
                constituents = _worker_instance._find_constituents(jet)
                if constituents is not None and len(constituents) > 0:
                    jet_constituents.append(constituents)

            return jet_constituents

        except Exception:
            # Return empty list if event processing fails
            raise Exception(f"Error processing event {event_idx}")

    def _particles_to_pseudojets(self, pt, eta, phi, masses=None):
        """
        Convert particle kinematics for a single event into fastjet PseudoJet objects.

        Parameters:
        -----------
        pt : array-like
            Transverse momenta of particles (in GeV)
        eta : array-like
            Pseudorapidities of particles
        phi : array-like
            Azimuthal angles of particles (in radians)
        masses : array-like, optional
            Masses of particles (in GeV). If None, assumes massless particles.

        Returns:
        --------
        pseudojets : list
            List of fastjet.PseudoJet objects for all particles

        Raises:
        -------
        ValueError
            If any NaN values are found in pt, eta, or phi
            If any negative pT values are found
            If masses is provided and any negative mass values are found
        """
        # Convert to numpy arrays if not already
        pt = np.asarray(pt)
        eta = np.asarray(eta)
        phi = np.asarray(phi)

        # Check for NaN values
        if np.any(np.isnan(pt)) or np.any(np.isnan(eta)) or np.any(np.isnan(phi)):
            raise ValueError("NaN values found in pt, eta, or phi")

        # Check for negative pT values
        if np.any(pt < 0):
            raise ValueError("Negative pT values found")

        # Handle masses
        if masses is None:
            masses = np.zeros_like(pt)
        else:
            masses = np.asarray(masses)
            # Check for NaN in masses
            if np.any(np.isnan(masses)):
                raise ValueError("NaN values found in masses")
            # Check for negative masses
            if np.any(masses < 0):
                raise ValueError("Negative mass values found")

        # Use vector package to compute four-vectors efficiently
        vectors = vector.array({"pt": pt, "eta": eta, "phi": phi, "mass": masses})

        # Extract Cartesian components and energy
        px = vectors.px
        py = vectors.py
        pz = vectors.pz
        energy = vectors.energy

        # Create fastjet PseudoJet objects
        pseudojets = []
        for i in range(len(pt)):
            pj = fj.PseudoJet(
                float(px[i]), float(py[i]), float(pz[i]), float(energy[i])
            )
            pj.set_user_index(i)  # Store original index for track-to-jet assignment
            pseudojets.append(pj)

        return pseudojets

    def _find_constituents(self, jet: fj.PseudoJet) -> np.ndarray | None:
        """
        Extract the constituents of a fastjet PseudoJet object and return a numpy array.

        Parameters:
        -----------
        jet : fastjet.PseudoJet
            The fastjet PseudoJet object

        Returns:
        --------
        constituents : np.ndarray | None
            A numpy array of the constituents in the shape (n_constituents, 4)
            where columns are (E, px, py, pz). Returns None if no constituents.

        Notes:
        ------
        - The cluster sequence that built the pseudojet must be in scope.
        """
        # Get constituents
        constituents = jet.constituents()

        if len(constituents) == 0:
            return None

        # Loop over constituents and store in numpy array
        constituents_array = np.zeros((len(constituents), 4))
        for i, constituent in enumerate(constituents):
            constituents_array[i] = [
                constituent.E(),
                constituent.px(),
                constituent.py(),
                constituent.pz(),
            ]

        return constituents_array

    @staticmethod
    def _process_single_event_with_track_assignment(args):
        """
        Cluster jets for a single event and return a track-to-jet assignment map.

        Each track is assigned the index of the jet it belongs to, where jets are
        ordered by descending transverse momentum (pT).

        Parameters
        ----------
        args : tuple
            Tuple containing: (event_idx, algorithm, R, ptmin)
                event_idx : int
                    Index of the event to process.
                algorithm : fastjet.JetAlgorithm
                    FastJet clustering algorithm to use.
                R : float
                    Jet radius parameter.
                ptmin : float
                    Minimum jet transverse momentum threshold.

        Returns
        -------
        jets_list : list[list[float]]
            Each jet is a list: [pt, eta, phi, mass], ordered by descending pT.
            Empty events return an empty list: [].

        track_to_jet_list : list[int]
            Track assignment indices:
                0  -> highest-pT jet
                1  -> second highest-pT jet
                ...
              -1 -> track does not belong to any jet passing ptmin
            Empty events return an empty list: [].
        """
        event_idx, algorithm, R, ptmin = args

        try:
            # Convert event tracks to NumPy arrays for FastJet input
            event_pt = ak.to_numpy(_worker_instance.pt[event_idx])
            event_eta = ak.to_numpy(_worker_instance.eta[event_idx])
            event_phi = ak.to_numpy(_worker_instance.phi[event_idx])
            event_masses = (
                ak.to_numpy(_worker_instance.masses[event_idx])
                if _worker_instance.masses is not None
                else None
            )

            n_tracks = len(event_pt)

            # Initialize track assignment as pure Python list
            track_to_jet_list = [-1] * n_tracks

            # Suppress FastJet stdout/stderr
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                    devnull
                ):

                    particles = _worker_instance._particles_to_pseudojets(
                        event_pt, event_eta, event_phi, event_masses
                    )

                    # Handle events with no tracks
                    if len(particles) == 0:
                        return [], track_to_jet_list

                    jet_def = fj.JetDefinition(algorithm, R)
                    cluster_seq = fj.ClusterSequence(particles, jet_def)

            # Jets passing pt threshold
            jets = cluster_seq.inclusive_jets(ptmin)

            if len(jets) == 0:
                return [], track_to_jet_list

            # Sort jets by descending pT
            jets = sorted(jets, key=lambda j: j.pt(), reverse=True)

            # Convert jets to simple Python lists
            jets_list = [[j.pt(), j.rap(), j.phi(), j.m()] for j in jets]

            # Assign tracks to jets
            for jet_rank, jet in enumerate(jets):
                for c in jet.constituents():
                    idx = c.user_index()
                    if idx >= 0:
                        track_to_jet_list[idx] = jet_rank

            return jets_list, track_to_jet_list

        except Exception:
            raise Exception(f"Error processing event {event_idx}")

    def cluster_events(
        self,
        algorithm=fj.antikt_algorithm,
        R=1.0,
        ptmin: float = 500.0,
        n_jobs: int | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Cluster jets for multiple events from particle kinematics and extract
        constituents. Returns pickleable numpy arrays, enabling parallel processing.

        Parameters:
        -----------
        algorithm : fastjet.JetAlgorithm, optional
            Jet clustering algorithm (default: fj.antikt_algorithm)
            Options: fj.antikt_algorithm, fj.kt_algorithm, fj.cambridge_algorithm
        R : float, optional
            Jet clustering radius parameter (default: 1.0)
        ptmin : float, optional
            Minimum pT threshold for clustered jets in GeV (default: 500.0)
            Only jets with pT >= ptmin are returned.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available CPUs.
            If 1, runs sequentially. (default: None)

        Returns:
        --------
        event_jet_constituents : list[list[np.ndarray]]
            List of lists, one per event. Each inner list contains numpy arrays,
            one per jet. Each numpy array has shape (n_constituents, 4) where
            columns are (E, px, py, pz).
            Events with no jets above ptmin will have empty lists.
            Events with failed processing will have empty lists.

        Notes:
        ------
        - All returned objects are pickleable numpy arrays, enabling parallel
          processing
        - Each event returns a list of jet constituent arrays
        - To flatten across events:
          flat_jets = [jet for event in result for jet in event]
        """
        n_events = len(self.pt)

        # Determine number of jobs
        if n_jobs is None:
            n_jobs = cpu_count()
        elif n_jobs < 1:
            n_jobs = 1

        # Prepare arguments for each event
        # Use Pool initializer to pass arrays once per worker process (not per task)
        # This avoids repeated pickling while still parallelizing the conversion
        print(f"Preparing {n_events} events for clustering")
        event_indices = list(range(n_events))
        # Only pass event indices and algorithm parameters
        # (arrays passed via initializer)
        event_args = [(i, algorithm, R, ptmin) for i in event_indices]

        # Process events
        # Returns list of lists of numpy arrays, all pickleable for parallel processing
        print(f"Processing {n_events} events with {n_jobs} jobs")
        if n_jobs == 1:
            # Sequential processing - use instance directly
            event_jet_constituents = []
            for args in tqdm(event_args, desc="Clustering events", unit="event"):
                # Temporarily set worker instance for sequential processing
                global _worker_instance
                _worker_instance = self
                event_jet_constituents.append(self._process_single_event_cluster(args))
        else:
            # Parallel processing
            # Use imap to get results as they complete, allowing progress bar updates
            # Use initializer to pass instance once per worker (avoid repeated pickling)
            with Pool(
                processes=n_jobs, initializer=_init_worker, initargs=(self,)
            ) as pool:
                # Use imap which returns an iterator, allowing tqdm to update progress
                # The progress bar will only print in the main process
                imap_result = pool.imap(self._process_single_event_cluster, event_args)
                event_jet_constituents = list(
                    tqdm(
                        imap_result,
                        total=n_events,
                        desc="Clustering events",
                        unit="event",
                    )
                )

        return event_jet_constituents

    def cluster_tracks_to_jets(
        self,
        algorithm=fj.antikt_algorithm,
        R=1.0,
        ptmin: float = 500.0,
        n_jobs: int | None = None,
    ) -> list[np.ndarray]:
        """
        Cluster jets for multiple events and return track-to-jet assignment arrays.

        For each event, tracks are assigned the index of the jet they belong to,
        where jets are ordered by descending transverse momentum (pT). Tracks that
        do not belong to any jet above the ptmin threshold receive a value of -1.

        Parameters:
        -----------
        algorithm : fastjet.JetAlgorithm, optional
            Jet clustering algorithm (default: fj.antikt_algorithm).
            Options include: fj.antikt_algorithm, fj.kt_algorithm, fj.cambridge_algorithm.
        R : float, optional
            Jet clustering radius parameter (default: 1.0).
        ptmin : float, optional
            Minimum jet pT threshold in GeV (default: 500.0).
            Only jets with pT >= ptmin are considered.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available CPUs.
            If 1, runs sequentially. (default: None)

        Returns:
        --------
        jets : list[np.ndarray[4]]
            List of arrays, one per event. Each array has shape (n_jets, 4) where columns are (pT, rapidity, phi, mass) for each jet that passes the ptmin threshold.
        event_track_assignment : list[np.ndarray]
            List of arrays, one per event. Each array has shape (n_tracks)
            and contains integer jet labels for each track:

            0  -> track belongs to the highest-pT jet
            1  -> track belongs to the second highest-pT jet
            2  -> track belongs to the third highest-pT jet
            -1 -> track does not belong to any jet passing the ptmin threshold

        Notes:
        ------
        - All returned objects are pickleable numpy arrays, enabling parallel processing.
        - Each event returns a single array mapping tracks to jet indices.
        """

        n_events = len(self.pt)

        # Determine number of jobs
        if n_jobs is None:
            n_jobs = cpu_count()
        elif n_jobs < 1:
            n_jobs = 1

        print(f"Preparing {n_events} events for clustering")

        event_indices = list(range(n_events))
        event_args = [(i, algorithm, R, ptmin) for i in event_indices]

        print(f"Processing {n_events} events with {n_jobs} jobs")

        if n_jobs == 1:
            event_track_assignment = []
            jets = []
            track_assignment_i = []

            for args in tqdm(event_args, desc="Clustering events", unit="event"):
                global _worker_instance
                _worker_instance = self

                jets_i, track_assignment_i = (
                    self._process_single_event_with_track_assignment(args)
                )

                jets.append(jets_i)
                event_track_assignment.append(track_assignment_i)

        else:
            with Pool(
                processes=n_jobs, initializer=_init_worker, initargs=(self,)
            ) as pool:
                imap_result = pool.imap(
                    self._process_single_event_with_track_assignment, event_args
                )

                results = list(
                    tqdm(
                        imap_result,
                        total=n_events,
                        desc="Clustering events",
                        unit="event",
                    )
                )

            jets, event_track_assignment = map(list, zip(*results))

        return jets, event_track_assignment
