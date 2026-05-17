"""
Population Class
================

Thin OOP orchestration layer over the existing core modules:
    - core.encoder.gaussian_process        (tuning curve generation)
    - core.encoder.divisive_normalization  (DN encoding)
    - core.decoder.ml_decoder              (ML decoding)

Usage:
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='bays')
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='gp', n_locations=8)
    pop = Population(M=1000, n_theta=256, omega=0.5, tuning_type='gp',
                     n_locations=8, lengthscale_variability=0.5, method='gamma')

    counts = pop.encode(active_locations, theta_indices, gamma, T_d)
    theta_hat, L_marg = pop.decode(counts, active_locations, cued_location)
    errors = pop.run_trials(n_trials, gamma, T_d, set_size=4)

encode() and decode() are the single source of truth for the pipeline.
run_trials() is a vectorised convenience wrapper that implements the same
    logic in batch for performance (~50-100× faster than looping).
"""

import numpy as np
from typing import Tuple, Optional

from core.encoder.gaussian_process import generate_neuron_population
from core.encoder.divisive_normalization import dn_pointwise, compute_r_pre_at_config
from core.decoder.ml_decoder import decode as ml_decode, circular_error


class Population:
    """
    A population of N neurons with tuning curves over L spatial locations
    and n_theta orientation bins.

    Parameters
    ----------
    M : int
        Number of neurons.
    n_theta : int
        Number of orientation bins.
    omega : float
        Tuning width parameter (Bays's omega). For GP mode, converted
        to lengthscale via lambda = sqrt(omega).
    tuning_type : str, 'bays' or 'gp'
        'bays' — Bays (2014) Eq. 1 parametric tuning, homogeneous,
                 evenly spaced preferred orientations, single location.
        'gp'   — Gaussian Process tuning curves with location-dependent
                 lengthscales.
    n_locations : int
        Number of spatial locations (only used when tuning_type='gp').
    seed : int
        Random seed (only used when tuning_type='gp').
    lengthscale_variability : float
        sigma_lambda for heterogeneous tuning widths across locations
        (only used when tuning_type='gp', 0 = homogeneous).
    gain_variability : float
        Amplitude variability across locations
        (only used when tuning_type='gp', 0 = homogeneous).
    method : str, optional
        Lengthscale sampling method for GP mode:
            'folded_normal'  — λ_i = λ_base · |1 + σ_λ · z_i|, z_i ~ N(0,1)
            'gamma'          — λ_i ~ Gamma with mean=λ_base, CV=σ_λ
            'random_vector'  — two-component scheme (see gaussian_process.py)
        Default 'folded_normal' preserves prior behaviour. Only used
        when tuning_type='gp'.

    Attributes
    ----------
    f : np.ndarray, shape (N, L, n_theta)
        Log-rate tuning functions.
    g : np.ndarray, shape (N, L, n_theta)
        Driving inputs exp(f).
    log_g : np.ndarray, shape (N, L, n_theta)
        log(max(g, eps)).
    theta_grid : np.ndarray, shape (n_theta,)
        Orientation grid in radians.
    N, L, n_theta : int
        Population size, number of locations, orientation resolution.
    tuning_type : str
        'bays' or 'gp'.
    omega : float
        Tuning width parameter used to construct the population.
    """

    def __init__(
        self,
        M: int,
        n_theta: int,
        omega: float,
        tuning_type: str = "bays",
        n_locations: int = 1,
        seed: int = 42,
        lengthscale_variability: float = 0.0,
        gain_variability: float = 0.0,
        method: str = "folded_normal",
    ):
        if tuning_type not in ("bays", "gp"):
            raise ValueError(f"tuning_type must be 'bays' or 'gp', got '{tuning_type}'")

        self.tuning_type = tuning_type
        self.omega = omega

        if tuning_type == "bays":
            f, theta_grid = self._build_bays(M, n_theta, omega)
        else:
            f, theta_grid = self._build_gp(
                M, n_theta, omega, n_locations, seed,
                lengthscale_variability, gain_variability, method,
            )

        self.f = f
        self.theta_grid = theta_grid
        self.N, self.L, self.n_theta = f.shape

        self.g = np.exp(self.f)
        self.log_g = np.log(np.maximum(self.g, 1e-30))

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bays(M, n_theta, omega):
        """Bays (2014) Eq. 1: f_i(theta) = (1/omega)(cos(phi_i - theta) - 1)"""
        theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        phi = np.linspace(-np.pi, np.pi, M, endpoint=False)
        diff = phi[:, None] - theta_grid[None, :]
        f = (1.0 / omega) * (np.cos(diff) - 1.0)
        return f[:, np.newaxis, :], theta_grid     # (M, 1, n_theta)

    @staticmethod
    def _build_gp(M, n_theta, omega, n_locations, seed,
                  lengthscale_variability, gain_variability,
                  method="folded_normal"):
        """GP tuning curves with lambda = sqrt(omega)."""
        lengthscale = np.sqrt(omega)
        raw = generate_neuron_population(
            n_neurons=M,
            n_orientations=n_theta,
            n_locations=n_locations,
            base_lengthscale=lengthscale,
            lengthscale_variability=lengthscale_variability,
            seed=seed,
            gain_variability=gain_variability,
            method=method,
        )
        theta_grid = raw[0]["orientations"]
        f = np.zeros((M, n_locations, n_theta))
        for n in range(M):
            f[n] = raw[n]["f_samples"]
        return f, theta_grid

    # ------------------------------------------------------------------
    # Core methods — the single source of truth
    # ------------------------------------------------------------------

    def encode(
        self,
        active_locations: Tuple[int, ...],
        theta_indices: np.ndarray,
        gamma: float,
        T_d: float,
        sigma_sq: float = 1e-6,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Encode a stimulus configuration into Poisson spike counts.

        Pipeline:  r_pre (Eq. 13) -> DN (Eq. 6) -> Poisson (Def. 4.5)

        Parameters
        ----------
        active_locations : tuple of int, length l
            Which spatial locations carry items.
        theta_indices : array-like of int, length l
            Orientation grid index at each active location.
        gamma : float
            Mean per-neuron gain (Hz).
        T_d : float
            Decoding time window (seconds).
        sigma_sq : float
            Semi-saturation constant.
        rng : RandomState, optional

        Returns
        -------
        spike_counts : np.ndarray, shape (N,)
        """
        if rng is None:
            rng = np.random.RandomState()

        r_pre = compute_r_pre_at_config(self.f, active_locations, theta_indices)
        r_post = dn_pointwise(r_pre, gamma, sigma_sq)
        return rng.poisson(r_post * T_d)

    def decode(
        self,
        spike_counts: np.ndarray,
        active_locations: Tuple[int, ...],
        cued_location: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Decode spike counts via factorised ML (Eqs. 23-28).

        Parameters
        ----------
        spike_counts : np.ndarray, shape (N,)
        active_locations : tuple of int, length l
            Which spatial locations are active.
        cued_location : int
            Index *into active_locations* of the cued item.

        Returns
        -------
        theta_hat : float
            ML orientation estimate.
        L_marginal : np.ndarray, shape (n_theta,)
            Marginal log-likelihood curve.
        """
        f_per_loc = [self.f[:, loc, :] for loc in active_locations]
        return ml_decode(spike_counts, f_per_loc, self.theta_grid, cued_location)

    # ------------------------------------------------------------------
    # Vectorised batch trial runner
    # ------------------------------------------------------------------

    def run_trials(
        self,
        n_trials: int,
        gamma: float,
        T_d: float,
        set_size: int = 1,
        sigma_sq: float = 1e-6,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Run n_trials of encode → decode, return circular errors.

        Vectorised implementation of the same pipeline as calling
        encode() and decode() in a loop.  ~50-100× faster.

        Encode (vectorised over trials):
            r_pre[n, t] = exp( Σ_k  f[n, locs[t,k], θ_idxs[t,k]] )   (Eq. 13)
            D[t]        = σ² + mean_n( r_pre[n, t] )
            r_post[n,t] = γ · r_pre[n,t] / D[t]                       (Eq. 6)
            counts[n,t] ~ Poisson( r_post[n,t] · T_d )                (Def. 4.5)

        Decode (vectorised over trials):
            L_c[t, θ]  = Σ_n  counts[n,t] · f[n, cued_loc[t], θ]     (Eq. 23)
            θ_hat[t]   = argmax_θ  L_c[t, θ]                          (Eq. 28)

        Note: The ML point estimate depends only on L_c — the logsumexp
        terms over non-cued locations are constant w.r.t. θ_c and do not
        affect the argmax (see ml_decoder.py docstring).

        Parameters
        ----------
        n_trials : int
        gamma : float
            Mean per-neuron gain (Hz).
        T_d : float
            Decoding time window (seconds).
        set_size : int
            Number of items (l).
        sigma_sq : float
        rng : RandomState, optional

        Returns
        -------
        errors : np.ndarray, shape (n_trials,)
            Signed circular errors in [-pi, pi).
        """
        if rng is None:
            rng = np.random.RandomState()

        l = set_size
        N, L, n_theta = self.N, self.L, self.n_theta

        # ---- 1. Sample all configurations upfront ----
        if l == 1:
            locs = rng.randint(L, size=(n_trials, 1))
        elif l >= L:
            locs = np.tile(np.arange(L), (n_trials, 1))
        else:
            # Vectorised sampling without replacement:
            # argsort of uniform randoms gives a random permutation per row
            locs = rng.random((n_trials, L)).argsort(axis=1)[:, :l]

        theta_idxs = rng.randint(n_theta, size=(n_trials, l))
        cued = (np.zeros(n_trials, dtype=int) if l == 1
                else rng.randint(l, size=n_trials))

        # ---- 2. Vectorised encode ----
        # r_pre[n, t] = exp( Σ_k f[n, locs[t,k], θ_idxs[t,k]] )
        # f[:, locs[:, k], theta_idxs[:, k]] gathers shape (N, n_trials)
        log_r_pre = np.zeros((N, n_trials))
        for k in range(l):
            log_r_pre += self.f[:, locs[:, k], theta_idxs[:, k]]

        r_pre = np.exp(log_r_pre)                       # (N, n_trials)

        # DN: r_post = γ · r_pre / D,  D = σ² + mean_n(r_pre)
        D = sigma_sq + r_pre.mean(axis=0)                # (n_trials,)
        r_post = gamma * r_pre / D[np.newaxis, :]        # (N, n_trials)

        # Poisson spikes
        counts = rng.poisson(r_post * T_d)               # (N, n_trials)

        # ---- 3. Vectorised decode (point estimate only) ----
        # L_c[t, θ] = Σ_n counts[n,t] · f[n, cued_loc[t], θ]
        # Group trials by cued location → one matmul per location
        cued_locs = locs[np.arange(n_trials), cued]      # (n_trials,)

        L_c = np.empty((n_trials, n_theta))
        for loc in range(L):
            mask = (cued_locs == loc)
            if mask.any():
                # (n_mask, N) @ (N, n_theta) → (n_mask, n_theta)
                L_c[mask] = counts[:, mask].T @ self.f[:, loc, :]

        theta_hat_idx = np.argmax(L_c, axis=1)           # (n_trials,)

        # ---- 4. Circular errors ----
        theta_true = self.theta_grid[theta_idxs[np.arange(n_trials), cued]]
        theta_hat = self.theta_grid[theta_hat_idx]

        d = theta_hat - theta_true
        errors = (d + np.pi) % (2.0 * np.pi) - np.pi

        return errors

    # ------------------------------------------------------------------
    # Sequential reference implementation (for validation / debugging)
    # ------------------------------------------------------------------

    def _run_trials_sequential(
        self,
        n_trials: int,
        gamma: float,
        T_d: float,
        set_size: int = 1,
        sigma_sq: float = 1e-6,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Sequential (non-vectorised) trial loop — kept for validation.

        Calls encode() and decode() per trial, exactly matching the
        original implementation.  Use to verify that run_trials() gives
        statistically equivalent results.
        """
        if rng is None:
            rng = np.random.RandomState()

        errors = np.empty(n_trials)

        for t in range(n_trials):
            locs = tuple(rng.choice(self.L, size=set_size, replace=False))
            theta_idxs = rng.randint(self.n_theta, size=set_size)
            cued = rng.randint(set_size)
            theta_true = self.theta_grid[theta_idxs[cued]]

            counts = self.encode(locs, theta_idxs, gamma, T_d, sigma_sq, rng)
            theta_hat, _ = self.decode(counts, locs, cued)

            errors[t] = circular_error(theta_true, theta_hat)

        return errors

    def __repr__(self) -> str:
        return (
            f"Population(N={self.N}, L={self.L}, n_theta={self.n_theta}, "
            f"tuning='{self.tuning_type}', omega={self.omega})"
        )