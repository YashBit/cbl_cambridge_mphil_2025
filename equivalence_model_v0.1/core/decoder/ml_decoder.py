"""
Maximum Likelihood Decoding with Factorised Marginalisation
===========================================================

Implements the efficient decoder derived in §5 of the paper.

The full Poisson log-likelihood (Eq. 16) simplifies in three steps:

    Step 2  Activity Cap eliminates the rate-sum term:
            L(θ_S) = Σ_i n_i log r^{pre}_i(S,θ)  −  (Σ_i n_i) log D(S,θ)  +  const

    Step 4  Denominator concentrates → penalty absorbed as θ-independent constant.

    Step 5  Multiplicative separability (Assumptions 2.1 & 2.2) decomposes:
            L(θ_S) = Σ_{k∈S} L_k(θ_k)   where  L_k(θ_k) = Σ_i n_i f_{i,k}(θ_k)   [Eq. 23]

    Step 6  Marginalisation over non-cued locations factorises:
            L_M(θ_c) = L_c(θ_c) + Σ_{k≠c} logsumexp(L_k)                          [Eq. 26]

    Estimate:  θ̂_c = argmax_{θ_c} L_M(θ_c)                                         [Eq. 28]

Complexity: O(N · l · n_θ)  vs.  naive O(N · n_θ^l).

Note: Since Σ_{k≠c} logsumexp(L_k) is constant w.r.t. θ_c, the ML point
estimate depends only on L_c(θ_c) = Σ_i n_i f_{i,c}(θ_c).  The non-cued
items affect the estimate *only* through their impact on spike counts via
divisive normalization (reduced rates → noisier data), not through the
decoding computation itself.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.special import logsumexp

from divisive_normalization import dn_pointwise, compute_r_pre_at_config


# =============================================================================
# CORE DECODER  (§5, Eqs. 23 → 26 → 28)
# =============================================================================

def decode(
    spike_counts: np.ndarray,
    f_per_location: List[np.ndarray],
    theta_grid: np.ndarray,
    cued_location: int
) -> Tuple[float, np.ndarray]:
    """
    Efficient ML decoding with factorised marginalisation.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (N,)
        Observed spike counts from all neurons.
    f_per_location : list of np.ndarray, each shape (N, n_θ)
        f_per_location[k][i, :] = f_{i,k}(θ) — log-rate tuning of neuron i
        at the k-th active location, evaluated on the orientation grid.
    theta_grid : np.ndarray, shape (n_θ,)
        Orientation grid values.
    cued_location : int
        Index (into f_per_location) of the cued location.

    Returns
    -------
    theta_hat : float
        ML orientation estimate at the cued location.
    L_marginal : np.ndarray, shape (n_θ,)
        Marginal log-likelihood curve over the grid.
    """
    # Eq. 23:  L_k(θ) = Σ_i n_i f_{i,k}(θ)   for each location k
    L_list = [spike_counts @ f_k for f_k in f_per_location]

    # Eq. 26:  L_M(θ_c) = L_c(θ_c) + Σ_{k≠c} logsumexp(L_k)
    L_marginal = L_list[cued_location].copy()
    for k, L_k in enumerate(L_list):
        if k != cued_location:
            L_marginal += logsumexp(L_k)       # scalar, constant w.r.t. θ_c

    # Eq. 28
    theta_hat = theta_grid[np.argmax(L_marginal)]
    return theta_hat, L_marginal


# =============================================================================
# CIRCULAR STATISTICS
# =============================================================================

def circular_error(
    theta_true: float,
    theta_est: float,
    period: float = 2 * np.pi
) -> float:
    """Signed circular error, wrapped to [-period/2, period/2)."""
    d = theta_est - theta_true
    return (d + period / 2) % period - period / 2


def circular_std(errors: np.ndarray, period: float = 2 * np.pi) -> float:
    """
    Circular standard deviation via mean resultant length.

    σ_circ = √(−2 ln R̄)  ·  (period / 2π)

    where R̄ = |mean(exp(i · 2π · errors / period))|.
    """
    phases = errors * (2 * np.pi / period)
    R = np.abs(np.mean(np.exp(1j * phases)))
    R = np.clip(R, 1e-10, 1.0 - 1e-10)
    return np.sqrt(-2 * np.log(R)) * (period / (2 * np.pi))


# =============================================================================
# GRID HELPERS
# =============================================================================

def snap_to_grid(theta: float, grid: np.ndarray) -> int:
    """
    Return the grid index closest to theta under circular distance.

    This is used to discretise continuous orientations for both encoding
    (evaluating tuning curves) and ground-truth comparison.
    """
    d = np.abs(grid - theta)
    return int(np.argmin(np.minimum(d, 2 * np.pi - d)))


# =============================================================================
# SINGLE-TRIAL PIPELINE
# =============================================================================

def run_trial(
    f_population: np.ndarray,
    theta_grid: np.ndarray,
    active_locations: Tuple[int, ...],
    true_orientations: np.ndarray,
    cued_index: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    rng: np.random.RandomState
) -> Dict:
    """
    Run one encode → spike → decode trial.

    Pipeline
    --------
    1. Snap true orientations to the grid (discrete simulation).
    2. Compute pre-normalised rates at the snapped configuration (Eq. 13).
    3. Apply divisive normalisation (Eq. 6 via dn_pointwise).
    4. Generate Poisson spikes (Def. 4.5).
    5. Decode cued orientation via factorised ML (Eqs. 23–28).
    6. Compute circular error relative to the *snapped* true value,
       so that reported error reflects decoding noise only, not grid
       quantisation.

    Parameters
    ----------
    f_population : np.ndarray, shape (N, n_locations, n_θ)
        Log-rate tuning functions for the full population.
    theta_grid : np.ndarray, shape (n_θ,)
        Orientation grid.
    active_locations : tuple of int, length l
        Which spatial locations carry items.
    true_orientations : np.ndarray, length l
        Continuous true orientations (will be snapped).
    cued_index : int
        Index (into active_locations) of the probed item.
    gamma, sigma_sq, T_d : float
        DN gain, semi-saturation, decoding window.
    rng : np.random.RandomState

    Returns
    -------
    dict with keys: theta_true, theta_est, error, spike_counts, L_marginal
    """
    l = len(active_locations)
    N = f_population.shape[0]
    n_theta = len(theta_grid)

    # --- 1. Build per-location tuning matrices & snap orientations ---
    f_per_loc = [f_population[:, loc, :] for loc in active_locations]   # l × (N, n_θ)
    theta_idx = [snap_to_grid(t, theta_grid) for t in true_orientations]
    theta_true_snapped = theta_grid[theta_idx[cued_index]]

    # --- 2–3. Encode: r_pre → DN → r_post ---
    r_pre = compute_r_pre_at_config(f_population, active_locations, theta_idx)
    r_post = dn_pointwise(r_pre, gamma, sigma_sq)

    # --- 4. Spike ---
    spike_counts = rng.poisson(r_post * T_d)

    # --- 5. Decode ---
    theta_hat, L_marginal = decode(spike_counts, f_per_loc, theta_grid, cued_index)

    # --- 6. Error (against snapped ground truth) ---
    error = circular_error(theta_true_snapped, theta_hat)

    return {
        'theta_true': theta_true_snapped,
        'theta_est': theta_hat,
        'error': error,
        'spike_counts': spike_counts,
        'L_marginal': L_marginal,
    }


# =============================================================================
# BATCH EXPERIMENT
# =============================================================================

def run_experiment(
    f_population: np.ndarray,
    theta_grid: np.ndarray,
    set_sizes: Tuple[int, ...],
    n_locations: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    n_trials: int,
    seed: int = 42
) -> Dict[int, Dict]:
    """
    Run decoding experiment across set sizes.

    Parameters
    ----------
    f_population : np.ndarray, shape (N, n_locations, n_θ)
    theta_grid   : np.ndarray, shape (n_θ,)
    set_sizes    : tuple of int
    n_locations  : int — total spatial locations available
    gamma, sigma_sq, T_d : float
    n_trials     : int — trials per set size
    seed         : int

    Returns
    -------
    results : dict[int, dict]
        Keyed by set size.  Each value contains 'errors' (array),
        'circular_std', 'circular_std_deg'.
    """
    rng = np.random.RandomState(seed)
    results = {}

    for l in set_sizes:
        errors = np.empty(n_trials)

        for t in range(n_trials):
            locs = tuple(rng.choice(n_locations, size=l, replace=False))
            orientations = rng.uniform(-np.pi, np.pi, size=l)
            cued = rng.randint(l)

            trial = run_trial(
                f_population, theta_grid, locs, orientations,
                cued, gamma, sigma_sq, T_d, rng
            )
            errors[t] = trial['error']

        std = circular_std(errors)
        results[l] = {
            'errors': errors,
            'circular_std': std,
            'circular_std_deg': np.degrees(std),
        }

    return results