import numpy as np
import awkward as ak
import numba as nb
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class _ProfileWorker:
    pass


def jet_radial_profile_parallel(
    jets,
    event_jet_indices,
    track_pt,
    track_eta,
    track_phi,
    track_mass,
    annulus_edges,
    weights_dict,
    only_associated=True,
    jet_pt_range=None,
    jet_y_max=None,
    use_rho_old=False,
    compute_psi=False,
    only_leading_jet=False,
    max_events=-1,
    n_jobs=None,
):
    """
    Parallel version of jet_radial_profile.
    """
    print("Parallel execution enabled with n_jobs =", n_jobs)

    # ---------------- Setup ----------------
    n_events = len(event_jet_indices)

    if max_events > 0:
        n_events = min(n_events, max_events)

    annulus_edges = np.asarray(annulus_edges)
    n_bins = len(annulus_edges) - 1

    if n_jobs is None:
        n_jobs = cpu_count()

    # ---------------- Worker container ----------------
    worker = _ProfileWorker()
    worker.track_pt = track_pt
    worker.track_eta = track_eta
    worker.track_phi = track_phi
    worker.track_mass = track_mass
    worker.jets = jets
    worker.event_jet_indices = event_jet_indices
    worker._profile_event = _profile_event

    # ---------------- Weight categories ----------------
    ensemble = [k for k in weights_dict if k.startswith("ensemble_")]
    bootstrap_mc = [k for k in weights_dict if k.startswith("bootstrap_mc_")]
    bootstrap_data = [k for k in weights_dict if k.startswith("bootstrap_data_")]

    special = set(["nominal"] + ensemble + bootstrap_mc + bootstrap_data)
    syst_weights = [k for k in weights_dict if k not in special]

    weights_np = {k: np.asarray(v) for k, v in weights_dict.items()}
    nominal_w = weights_np["nominal"]
    # ---------------- Accumulators ----------------
    total_profile_nom = np.zeros(n_bins)
    total_profile_nom_var = np.zeros(n_bins)
    n_jets_counter = 0.0

    total_profile_var = {name: np.zeros(n_bins) for name in syst_weights}
    total_profile_var_var = {name: np.zeros(n_bins) for name in syst_weights}
    n_jets_var = {name: 0.0 for name in syst_weights}

    replica_profiles = {
        k: np.zeros(n_bins) for k in ensemble + bootstrap_mc + bootstrap_data
    }
    replica_profile_var = {
        k: np.zeros(n_bins) for k in ensemble + bootstrap_mc + bootstrap_data
    }
    n_jets_replica = {k: 0.0 for k in ensemble + bootstrap_mc + bootstrap_data}

    # ---------------- Event sampling ----------------
    if max_events > 0:
        sample_indices = np.random.choice(
            len(event_jet_indices), size=n_events, replace=False
        )
    else:
        sample_indices = np.arange(n_events)

    event_args = [
        (
            int(ievt),
            annulus_edges,
            compute_psi,
            only_leading_jet,
            only_associated,
            jet_pt_range,
            jet_y_max,
            use_rho_old,
        )
        for ievt in sample_indices
    ]

    # ---------------- Parallel execution ----------------
    if n_jobs == 1:
        _init_profile_worker(worker)
        results = [
            _process_single_event_profile(args)
            for args in tqdm(
                event_args,
                desc="Computing radial profile",
                unit="event",
            )
        ]
    else:
        with Pool(
            processes=n_jobs,
            initializer=_init_profile_worker,
            initargs=(worker,),
        ) as pool:

            imap_result = pool.imap(_process_single_event_profile, event_args)

            results = list(
                tqdm(
                    imap_result,
                    total=len(event_args),
                    desc="Computing radial profile",
                    unit="event",
                )
            )

    # ---------------- Vectorized Reduction ----------------
    profile_dict = {}

    evt_list = [r for r in results if r is not None]

    if len(evt_list) == 0:
        return {}

    # Build arrays
    profiles = np.stack([r[0] for r in evt_list])  # (n_evt, n_bins)
    n_jets_arr = np.array([r[1] for r in evt_list])  # (n_evt,)
    ievt_arr = np.array([r[2] for r in evt_list])  # (n_evt,)

    # ---------------- Nominal ----------------
    w = nominal_w[ievt_arr]
    print("Here")
    total_profile_nom = np.sum(profiles * w[:, None], axis=0)
    total_profile_nom_var = np.sum(profiles * (w**2)[:, None], axis=0)
    n_jets_counter = np.sum(n_jets_arr * w)

    if n_jets_counter > 0:
        total_profile_nom /= n_jets_counter
        total_profile_nom_var /= n_jets_counter**2

    profile_dict["nominal"] = (
        total_profile_nom.copy(),
        total_profile_nom_var.copy(),
        annulus_edges,
    )

    # ---------------- Systematics ----------------
    if syst_weights:
        names = syst_weights

        # (n_weights, n_evt)
        W = np.stack([weights_np[name][ievt_arr] for name in names])

        star_mask = np.array([name.startswith("*") for name in names])
        if np.any(star_mask):
            W[star_mask] *= w

        total = W @ profiles
        total_var = (W**2) @ profiles
        n_jets_w = W @ n_jets_arr

        for i, name in enumerate(names):
            if n_jets_w[i] > 0:
                mean = total[i] / n_jets_w[i]
                var = total_var[i] / n_jets_w[i] ** 2
            else:
                mean = np.zeros(n_bins)
                var = np.zeros(n_bins)

            key = name[1:] if name.startswith("*") else name
            profile_dict[key] = (mean.copy(), var.copy(), annulus_edges)

    # ---------------- Replicas ----------------
    if replica_profiles:
        keys = list(replica_profiles.keys())

        W = np.stack([weights_np[k][ievt_arr] for k in keys])

        total = W @ profiles
        total_var = (W**2) @ profiles
        n_jets_w = W @ n_jets_arr

        for i, key in enumerate(keys):
            if n_jets_w[i] > 0:
                mean = total[i] / n_jets_w[i]
                var = total_var[i] / n_jets_w[i] ** 2
            else:
                mean = np.zeros(n_bins)
                var = np.zeros(n_bins)

            profile_dict[key] = (mean.copy(), var.copy(), annulus_edges)

    return profile_dict


@nb.njit
def _profile_event(
    tracks_pt,
    tracks_eta,
    tracks_phi,
    tracks_mass,
    event_jet_indices,
    jets,
    annulus_edges,
    compute_psi,
    only_leading_jet,
    only_associated,
    jet_pt_range,
    jet_y_max,
    use_rho_old,
):
    """Compute per-event jet radial profile.

    Accumulates either rho (radial momentum density) or psi (cumulative profile)
    for jets in a single event, using track information and annular binning.

    Arguments:
    ----------
    tracks_pt, tracks_eta, tracks_phi, tracks_mass : array-like
        Track kinematics for the event.
    event_jet_indices : array-like
        Mapping of tracks to associated jet indices.
    jets : array-like
        Jet kinematics as (pt, y, phi, mass).
    annulus_edges : array-like
        Radial bin edges (ΔR).
    compute_psi : bool
        If True, compute cumulative psi profile; otherwise rho.
    only_leading_jet : bool
        If True, use only the leading jet.
    only_associated : bool
        If True, use only tracks associated to each jet.
    jet_pt_range : tuple or None
        Optional (min, max) jet pT selection.
    jet_y_max : float or None
        Maximum jet rapidity.
    use_rho_old : bool
        If True, use legacy rho normalization.

    Returns:
    --------
    profile : np.ndarray
        Accumulated rho or psi profile for the event.
    n_selected_jets : int
        Number of jets passing selection.
    """

    pi = np.pi
    n_bins = len(annulus_edges) - 1
    rho_sum = np.zeros(n_bins)
    psi_sum = np.zeros(n_bins)
    n_selected_jets = 0

    # precompute annulus normalization
    if use_rho_old:
        annulus_norm = annulus_edges[1:] - annulus_edges[:-1]
    else:
        annulus_norm = np.pi * (annulus_edges[1:] ** 2 - annulus_edges[:-1] ** 2)

    n_tracks = tracks_pt.shape[0]

    for jidx in range(jets.shape[0]):
        jet_pt, jet_y, jet_phi, jet_mass = jets[jidx]

        # Only leading jet if requested
        if only_leading_jet and jidx != 0:
            continue

        # Jet kinematic cuts
        if jet_pt_range is not None and (
            jet_pt < jet_pt_range[0] or jet_pt > jet_pt_range[1]
        ):
            continue
        if jet_y_max is not None and abs(jet_y) > jet_y_max:
            continue

        n_selected_jets += 1

        # Track selection depending on only_associated
        if only_associated:
            mask = np.zeros(n_tracks, dtype=np.bool_)
            for t in range(n_tracks):
                if event_jet_indices[t] == jidx:
                    mask[t] = True
        else:
            mask = np.ones(n_tracks, dtype=np.bool_)

        n_selected = np.sum(mask)
        if n_selected == 0:
            continue

        # select tracks
        pt = tracks_pt[mask]
        eta = tracks_eta[mask]
        phi = tracks_phi[mask]
        mass = tracks_mass[mask]

        # compute 4-vectors
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
        y = 0.5 * np.log((E + pz) / (E - pz))

        # radial distance
        dphi = (phi - jet_phi + pi) % (2 * pi) - pi
        dy = y - jet_y
        dR = np.sqrt(dy**2 + dphi**2)

        # bin indices
        bin_idx = np.digitize(dR, annulus_edges, right=False) - 1
        # select valid bins
        valid_mask = (bin_idx >= 0) & (bin_idx < n_bins)
        if np.sum(valid_mask) == 0:
            continue

        bin_idx = bin_idx[valid_mask]
        px_sel = px[valid_mask]
        py_sel = py[valid_mask]

        # vector sum per bin
        px_sum = np.zeros(n_bins)
        py_sum = np.zeros(n_bins)
        # pt_sum = np.zeros(n_bins)
        for i in range(bin_idx.shape[0]):
            px_sum[bin_idx[i]] += px_sel[i]
            py_sum[bin_idx[i]] += py_sel[i]
            # pt_sum[bin_idx[i]] += pt[valid_mask][i] # linear pt sum

        pt_vec = np.sqrt(px_sum**2 + py_sum**2)
        # pt_vec = pt_sum

        # compute either rho or psi
        if compute_psi:
            px_cumulative = np.zeros(n_bins)
            py_cumulative = np.zeros(n_bins)
            pt_cumulative = np.zeros(n_bins)
            for i in range(n_bins):
                if i == 0:
                    px_cumulative[i] = px_sum[i]
                    py_cumulative[i] = py_sum[i]
                else:
                    px_cumulative[i] = px_cumulative[i - 1] + px_sum[i]
                    py_cumulative[i] = py_cumulative[i - 1] + py_sum[i]
                pt_cumulative[i] = np.sqrt(
                    px_cumulative[i] ** 2 + py_cumulative[i] ** 2
                )
            psi_sum += pt_cumulative / jet_pt
        else:
            if use_rho_old:
                rho_sum += pt_vec / (annulus_norm * jet_pt)
            else:
                rho_sum += pt_vec / annulus_norm
    return (psi_sum if compute_psi else rho_sum), n_selected_jets


# Reuse global worker pattern
_profile_worker_instance = None


def _init_profile_worker(instance):
    global _profile_worker_instance

    _profile_worker_instance = instance


def _process_single_event_profile(args):
    """
    Worker: compute per-event profile + jet count.
    """
    (
        ievt,
        annulus_edges,
        compute_psi,
        only_leading_jet,
        only_associated,
        jet_pt_range,
        jet_y_max,
        use_rho_old,
    ) = args

    inst = _profile_worker_instance

    try:
        # Convert jagged → numpy (done in parallel)
        t_pt = np.asarray(ak.to_numpy(inst.track_pt[ievt]))
        t_eta = np.asarray(ak.to_numpy(inst.track_eta[ievt]))
        t_phi = np.asarray(ak.to_numpy(inst.track_phi[ievt]))
        t_mass = np.asarray(ak.to_numpy(inst.track_mass[ievt]))
        evt_jets = np.asarray(ak.to_numpy(inst.jets[ievt]))
        indices = np.asarray(ak.to_numpy(inst.event_jet_indices[ievt]), dtype=np.int64)

        if len(evt_jets) == 0:
            return None  # skip

        profile_evt, n_jets = inst._profile_event(
            t_pt,
            t_eta,
            t_phi,
            t_mass,
            indices,
            evt_jets,
            annulus_edges,
            compute_psi,
            only_leading_jet,
            only_associated,
            jet_pt_range,
            jet_y_max,
            use_rho_old,
        )

        return profile_evt, n_jets, ievt

    except Exception as e:
        raise RuntimeError(f"Error processing event {ievt}") from e
