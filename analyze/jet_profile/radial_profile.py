import numpy as np
import awkward as ak
import numba as nb
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Reuse global worker pattern
_profile_worker_instance = None

class _ProfileWorker:
    pass

def _init_profile_worker(worker):
    global _profile_worker_instance
    _profile_worker_instance = worker
    worker.jets_list    = ak.to_list(worker.jets)
    worker.indices_list = ak.to_list(worker.event_jet_indices)

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

def _process_single_event_profile(args):
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
        # -------- FAST path --------
        
        t_pt   = inst.track_pt[ievt].to_numpy()
        t_eta  = inst.track_eta[ievt].to_numpy()
        t_phi  = inst.track_phi[ievt].to_numpy()
        t_mass = inst.track_mass[ievt].to_numpy()

        # Jets + indices: already converted once
        evt_jets = np.array(inst.jets_list[ievt], dtype=np.float64)
        indices  = np.asarray(inst.indices_list[ievt], dtype=np.int64)

        if len(evt_jets) == 0:
            return None

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
    returns the covariance matrix of the MEAN, and taking sqrt(diag(cov)) gives the error on the MEAN. correlatinon matrix is unnafected by this.
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
    worker.jets       = jets
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
                desc="Computing radial profile parallel",
                unit="event",
            )
        ]
    else:
        with Pool(
            processes=n_jobs,
            initializer=_init_profile_worker,
            initargs=(worker,),
        ) as pool:

            imap_result = pool.imap(_process_single_event_profile, event_args, chunksize = 100)

            results = list(
                tqdm(
                    imap_result,
                    total=len(event_args),
                    desc="Computing radial profile parallel",
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
    total_profile_nom = np.sum(profiles * w[:, None], axis=0) # the means!
    n_jets_counter = np.sum(n_jets_arr * w)
    if n_jets_counter > 0:
        total_profile_nom /= n_jets_counter

     # ---------------- MC test stat uncertainty ----------------
    diff  = profiles - n_jets_arr[:, None] * total_profile_nom
    a = (w / n_jets_counter) ** 2  
    total_profile_cov = (diff * a[:, None]).T @ diff

    # ---------------- also MC Test uncertainty, found via bootstraping ----------------
  #  n_bootstrap = 100
  #  n_evt = profiles.shape[0]
  #  R = np.random.poisson(1, size=(n_evt, n_bootstrap))
  #  R_w = R * w[:, None]
  #  n_jets_boot = (R_w.T @ n_jets_arr)
  #  profile_boot_mean = (R_w.T @ profiles) / n_jets_boot[:, None]
  #  diff = profile_boot_mean - total_profile_nom[None, :]
  #  total_profile_cov = (diff.T @ diff) / n_bootstrap
    
    profile_dict["nominal"] = (total_profile_nom.copy(), total_profile_cov.copy(), annulus_edges)


    # ---------------- Systematics ----------------
    if syst_weights:
        names = syst_weights

        # (n_weights, n_evt)
        W = np.stack([weights_np[name][ievt_arr] for name in names])

        star_mask = np.array([name.startswith("*") for name in names])
        if np.any(star_mask):
            W[star_mask] *= w

        total = W @ profiles
        n_jets_w = W @ n_jets_arr

        for i, name in enumerate(names):
            if n_jets_w[i] > 0:
                mean = total[i] / n_jets_w[i]
                diff = profiles - n_jets_arr[:, None] * mean
                wi   = W[i] / n_jets_w[i]  # event-level normalized weights
                cov = ((diff * (wi * wi)[:, None]).T @ diff)
            else:
                mean = np.zeros(n_bins)
                cov = np.zeros((n_bins, n_bins), dtype=np.float64)

            key = name[1:] if name.startswith("*") else name
            profile_dict[key] = (mean.copy(), cov.copy(), annulus_edges)

    # ---------------- Replicas ----------------
    if replica_profiles:
        keys = list(replica_profiles.keys())
        

        W = np.stack([weights_np[k][ievt_arr] for k in keys])

        total = W @ profiles
        n_jets_w = W @ n_jets_arr

        for i, key in enumerate(keys):
            if n_jets_w[i] > 0:
                mean = total[i] / n_jets_w[i]
                diff = profiles - n_jets_arr[:, None] * mean
                wi   = W[i] / n_jets_w[i]  # event-level normalized weights
                cov  = ((diff * (wi * wi)[:, None]).T @ diff)
            else:
                mean = np.zeros(n_bins)
                cov  = np.zeros((n_bins, n_bins), dtype=np.float64)

            profile_dict[key] = (mean.copy(), cov.copy(), annulus_edges)

    return profile_dict



# ---------------- Parallel over events ----------------
@nb.njit(parallel=True)
def _profile_events_nb(
    track_pt_list,
    track_eta_list,
    track_phi_list,
    track_mass_list,
    jets_list,
    indices_list,
    annulus_edges,
    compute_psi,
    only_leading_jet,
    only_associated,
    jet_pt_range,
    jet_y_max,
    use_rho_old,
    ):
    n_events = len(track_pt_list)
    n_bins = len(annulus_edges) - 1

    profiles = np.zeros((n_events, n_bins))
    n_jets_arr = np.zeros(n_events)

    for ievt in nb.prange(n_events):
        jets_evt = jets_list[ievt]

        if jets_evt.shape[0] == 0:
            continue

        profile_evt, n_jets = _profile_event(
            track_pt_list[ievt],
            track_eta_list[ievt],
            track_phi_list[ievt],
            track_mass_list[ievt],
            indices_list[ievt],
            jets_evt,
            annulus_edges,
            compute_psi,
            only_leading_jet,
            only_associated,
            jet_pt_range,
            jet_y_max,
            use_rho_old,
        )

        profiles[ievt] = profile_evt
        n_jets_arr[ievt] = n_jets

    return profiles, n_jets_arr

# ---------------- User-facing function ----------------

def jet_radial_profile_numba(
    jets,
    event_jet_indices,
    track_pt,
    track_eta,
    track_phi,
    track_mass,
    annulus_edges,
    weights_dict,
    n_jobs=4,
    compute_psi=False,
    only_leading_jet=False,
    only_associated=True,
    jet_pt_range=None,
    jet_y_max=None,
    use_rho_old=False,
    max_events=-1,
    show_progress=True,
    ):
    """
    Numba-based radial profile computation with:
    - max_events sampling
    - optional tqdm progress bar
    - vectorized reduction like jet_radial_profile_parallel
    """
    import numpy as np
    import awkward as ak
    import numba as nb

    print("Running Numba parallel version with", n_jobs, "threads")
    nb.set_num_threads(n_jobs)

    annulus_edges = np.asarray(annulus_edges)
    n_bins = len(annulus_edges) - 1

    # -------- Convert Awkward → NumPy --------
    track_pt_list   = [np.asarray(ak.to_numpy(x)) for x in track_pt]
    track_eta_list  = [np.asarray(ak.to_numpy(x)) for x in track_eta]
    track_phi_list  = [np.asarray(ak.to_numpy(x)) for x in track_phi]
    track_mass_list = [np.asarray(ak.to_numpy(x)) for x in track_mass]

    jets_list = [np.asarray(ak.to_numpy(x)) for x in jets]
    indices_list = [np.asarray(ak.to_numpy(x), dtype=np.int64) for x in event_jet_indices]

    # -------- Handle max_events sampling --------
    n_events = len(event_jet_indices)
    if max_events > 0:
        n_events = min(n_events, max_events)
        sample_indices = np.random.choice(len(event_jet_indices), size=n_events, replace=False)
    else:
        sample_indices = np.arange(n_events)

    # -------- Run Numba kernel per event with optional progress bar --------
    profiles = []
    n_jets_arr = []

    if show_progress and n_jobs == 1:
        iterable = tqdm(sample_indices, desc="Computing radial profile numba", unit="event")
    else:
        iterable = sample_indices

    for ievt in iterable:
        p, n = _profile_events_nb(
            [track_pt_list[ievt]],
            [track_eta_list[ievt]],
            [track_phi_list[ievt]],
            [track_mass_list[ievt]],
            [jets_list[ievt]],
            [indices_list[ievt]],
            annulus_edges,
            compute_psi,
            only_leading_jet,
            only_associated,
            jet_pt_range,
            jet_y_max,
            use_rho_old,
        )
        profiles.append(p[0])
        n_jets_arr.append(n[0])

    profiles = np.asarray(profiles)    # (n_events, n_bins)
    n_jets_arr = np.asarray(n_jets_arr)  # (n_events,)
    ievt_arr = np.arange(len(profiles))

    # -------- Convert weights dict to NumPy arrays --------
    weights_np = {k: np.asarray(v) for k, v in weights_dict.items()}
    nominal_w = weights_np["nominal"][sample_indices]  # use sampled events

    # -------- Identify weight categories --------
    ensemble = [k for k in weights_dict if k.startswith("ensemble_")]
    bootstrap_mc = [k for k in weights_dict if k.startswith("bootstrap_mc_")]
    bootstrap_data = [k for k in weights_dict if k.startswith("bootstrap_data_")]
    special = set(["nominal"] + ensemble + bootstrap_mc + bootstrap_data)
    syst_weights = [k for k in weights_dict if k not in special]

    profile_dict = {}

    # -------- Nominal (vectorized) --------
    w = nominal_w
    total_profile_nom = np.sum(profiles * w[:, None], axis=0)
    total_profile_nom_var = np.sum(profiles * (w**2)[:, None], axis=0)
    n_jets_counter = np.sum(n_jets_arr * w)
    if n_jets_counter > 0:
        total_profile_nom /= n_jets_counter
        total_profile_nom_var /= n_jets_counter ** 2
    profile_dict["nominal"] = (total_profile_nom.copy(), total_profile_nom_var.copy(), annulus_edges)

    # -------- Systematics (vectorized) --------
    if syst_weights:
        W = np.stack([weights_np[name][sample_indices] for name in syst_weights])
        star_mask = np.array([name.startswith("*") for name in syst_weights])
        if np.any(star_mask):
            W[star_mask] *= w[:, None]
        total = W @ profiles
        total_var = (W**2) @ profiles
        n_jets_w = W @ n_jets_arr
        for i, name in enumerate(syst_weights):
            if n_jets_w[i] > 0:
                mean = total[i] / n_jets_w[i]
                var = total_var[i] / n_jets_w[i] ** 2
            else:
                mean = np.zeros(n_bins)
                var = np.zeros(n_bins)
            key = name[1:] if name.startswith("*") else name
            profile_dict[key] = (mean.copy(), var.copy(), annulus_edges)

    # -------- Replica weights (vectorized) --------
    replica_keys = ensemble + bootstrap_mc + bootstrap_data
    if replica_keys:
        W = np.stack([weights_np[k][sample_indices] for k in replica_keys])
        total = W @ profiles
        total_var = (W**2) @ profiles
        n_jets_w = W @ n_jets_arr
        for i, key in enumerate(replica_keys):
            if n_jets_w[i] > 0:
                mean = total[i] / n_jets_w[i]
                var = total_var[i] / n_jets_w[i] ** 2
            else:
                mean = np.zeros(n_bins)
                var = np.zeros(n_bins)
            profile_dict[key] = (mean.copy(), var.copy(), annulus_edges)

    return profile_dict


def validate_single_bin_from_scratch(
    jets,
    event_jet_indices,
    track_pt,
    track_eta,
    track_phi,
    track_mass,
    annulus_edges,
    weights,
    only_associated=True,
    jet_pt_range=None,
    jet_y_max=None,
    use_rho_old=False,
    compute_psi=False,
    only_leading_jet=False,
    bin_idx=-1,     
    make_plots=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import awkward as ak

    pi = np.pi
    n_events = len(jets)
    n_bins = len(annulus_edges) - 1

    # bin normalization
    if use_rho_old:
        annulus_norm = annulus_edges[1:] - annulus_edges[:-1]
    else:
        annulus_norm = np.pi * (annulus_edges[1:]**2 - annulus_edges[:-1]**2)

    jets_list = ak.to_list(jets)
    indices_list = ak.to_list(event_jet_indices)

    weights_np = {k: np.asarray(v) for k, v in weights.items()}
    nominal_w = weights_np["nominal"]

    # per-event storage
    X  = []   # shape: (events, bins)
    nJ = []   # jets per event
    w  = []

    # ===============================
    # EVENT LOOP
    # ===============================
    for ievt in range(n_events):
        if ievt % max(1, n_events // 20) == 0:
            print(f"{100*ievt/n_events:.0f}%")

        t_pt   = track_pt[ievt].to_numpy()
        t_eta  = track_eta[ievt].to_numpy()
        t_phi  = track_phi[ievt].to_numpy()
        t_mass = track_mass[ievt].to_numpy()

        evt_jets = np.array(jets_list[ievt], dtype=np.float64)
        indices  = np.asarray(indices_list[ievt], dtype=np.int64)

        evt_weight = nominal_w[ievt]

        if len(evt_jets) == 0:
            continue

        rho_sum_evt = np.zeros(n_bins)
        n_selected_jets = 0

        # -------- jet loop --------
        for jidx, jet in enumerate(evt_jets):

            jet_pt, jet_y, jet_phi, jet_mass = jet

            if only_leading_jet and jidx != 0:
                continue

            if jet_pt_range is not None:
                if jet_pt < jet_pt_range[0] or jet_pt > jet_pt_range[1]:
                    continue

            if jet_y_max is not None:
                if abs(jet_y) > jet_y_max:
                    continue

            n_selected_jets += 1

            px_sum = np.zeros(n_bins)
            py_sum = np.zeros(n_bins)

            # -------- track loop --------
            for t in range(len(t_pt)):

                if only_associated and indices[t] != jidx:
                    continue

                pt  = t_pt[t]
                eta = t_eta[t]
                phi = t_phi[t]
                m   = t_mass[t]

                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                pz = pt * np.sinh(eta)
                E  = np.sqrt(px*px + py*py + pz*pz + m*m)
                y  = 0.5 * np.log((E + pz) / (E - pz))

                dphi = (phi - jet_phi + pi) % (2*pi) - pi
                dy   = y - jet_y
                dR   = np.sqrt(dy*dy + dphi*dphi)

                b = np.digitize(dR, annulus_edges) - 1
                if b < 0 or b >= n_bins:
                    continue

                px_sum[b] += px
                py_sum[b] += py

            pt_vec = np.sqrt(px_sum**2 + py_sum**2)

            if compute_psi:
                raise NotImplementedError("psi validation not implemented here")
            else:
                if use_rho_old:
                    rho_sum_evt += pt_vec / (annulus_norm * jet_pt)
                else:
                    rho_sum_evt += pt_vec / annulus_norm

        if n_selected_jets > 0:
            X.append(rho_sum_evt)
            nJ.append(n_selected_jets)
            w.append(evt_weight)

    X  = np.array(X)   # (events, bins)
    nJ = np.array(nJ)
    w  = np.array(w)

    # ===============================
    # OUTPUT PER BIN
    # ===============================
    N = np.sum(w * nJ)
    if N == 0:
        return np.zeros(n_bins), np.zeros(n_bins), X, nJ, w

    mean = np.sum(w[:, None] * X, axis=0) / N
    diff = X - nJ[:, None] * mean
    var  = np.sum((w[:, None]**2) * diff**2, axis=0) / N
    err  = np.sqrt(var)  / np.sqrt(N)
    
    # ===============================
    # OPTIONAL PLOTTING
    # ===============================
    if make_plots:
        bins_to_plot = range(n_bins) if bin_idx == -1 else [bin_idx]
        for b in bins_to_plot:
            plt.figure(figsize=(8,5))
            plt.scatter(np.arange(len(X)), X[:, b], alpha=0.7, label="per-event value")
            plt.hlines(mean[b], 0, len(X)-1, colors='r', label=f"mean = {mean[b]:.3f}")
            plt.fill_between(range(len(X)), mean[b]-err[b], mean[b]+err[b], color='r', alpha=0.2, label=f"error = {err[b]:.3f}")
            plt.title(f"Bin {b}: N={N}, variance={var[b]:.5f}")
            plt.xlabel("Event index")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

    return mean, err