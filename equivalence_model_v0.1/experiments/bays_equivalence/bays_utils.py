"""
Shared Utilities for Bays (2014) Equivalence Experiments
========================================================

Centralises code that was duplicated across figure_1 through figure_5:

    1. Circular statistics  (variance, kurtosis, trigonometric moments)
    2. Deviation from circular normal  (histogram vs matched von Mises)
    3. GP population generation  (single- and multi-location)

Each figure file imports from here instead of defining its own copy.
"""

import numpy as np
from scipy.special import ive, logsumexp
from scipy.optimize import brentq
from typing import Dict, List, Tuple

from core.encoder.gaussian_process import periodic_rbf_kernel, sample_gp_function


# =============================================================================
# CIRCULAR STATISTICS  (Fisher 1995; Bays 2014 Methods)
# =============================================================================

def circular_moments(errors: np.ndarray) -> Tuple[complex, complex]:
    """1st and 2nd uncentred trigonometric moments."""
    return np.mean(np.exp(1j * errors)), np.mean(np.exp(2j * errors))


def circular_variance(errors: np.ndarray) -> float:
    """
    Circular variance: σ² = −2 · log(ρ₁).

    ρ₁ = |m₁| is the mean resultant length.
    """
    m1, _ = circular_moments(errors)
    rho1 = np.clip(np.abs(m1), 1e-15, 1.0 - 1e-10)
    return -2.0 * np.log(rho1)


def circular_kurtosis(errors: np.ndarray) -> float:
    """
    Circular kurtosis: κ = (ρ₂ cos(μ₂ − 2μ₁) − ρ₁⁴) / (1 − ρ₁)².
    """
    m1, m2 = circular_moments(errors)
    rho1, rho2 = np.abs(m1), np.abs(m2)
    mu1, mu2 = np.angle(m1), np.angle(m2)
    denom = (1.0 - rho1) ** 2
    if denom < 1e-15:
        return 0.0
    return (rho2 * np.cos(mu2 - 2 * mu1) - rho1 ** 4) / denom


# =============================================================================
# DEVIATION FROM CIRCULAR NORMAL  (Bays 2014 Fig 2e)
# =============================================================================

def _bessel_ratio(kappa: float) -> float:
    """I₁(κ) / I₀(κ), overflow-safe via exponentially scaled Bessels."""
    return ive(1, kappa) / ive(0, kappa)


def _rho_to_kappa(rho: float) -> float:
    """Invert A(κ) = I₁(κ)/I₀(κ) = ρ to find von Mises concentration κ."""
    if rho < 1e-10:
        return 0.0
    if rho > 1 - 1e-10:
        return 1e4
    return brentq(lambda k: _bessel_ratio(k) - rho, 1e-8, 1e4)


def _von_mises_pdf(theta: np.ndarray, kappa: float) -> np.ndarray:
    """Von Mises PDF, overflow-safe."""
    log_p = kappa * np.cos(theta) - np.log(2 * np.pi) - kappa - np.log(ive(0, kappa))
    return np.exp(log_p)


def compute_deviation_from_normal(
    errors: np.ndarray, n_bins: int = 50
) -> Dict:
    """
    Compute deviation of error histogram from a matched circular normal.

    Returns dict with bin_centers, empirical density, von Mises fit,
    deviation (empirical − von Mises), and fitted κ.
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts, _ = np.histogram(errors, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    empirical = counts / (len(errors) * bin_width)

    m1, _ = circular_moments(errors)
    kappa = _rho_to_kappa(np.abs(m1))
    von_mises = _von_mises_pdf(bin_centers, kappa)

    return {
        'bin_centers': bin_centers,
        'empirical': empirical,
        'von_mises': von_mises,
        'deviation': empirical - von_mises,
        'kappa': kappa,
    }


# =============================================================================
# GP POPULATION GENERATION
# =============================================================================

def generate_population(
    M: int,
    n_theta: int,
    lengthscale: float,
    n_locations: int,
    seed: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate M neurons with independent GP tuning at each location.

    Each neuron gets an independent GP sample per location — this is the
    source of mixed selectivity when n_locations > 1.

    Parameters
    ----------
    M : int
        Number of neurons.
    n_theta : int
        Orientation grid resolution.
    lengthscale : float
        Periodic RBF kernel lengthscale.
    n_locations : int
        Number of spatial locations (1 for single-location experiments).
    seed : int
        Random seed.

    Returns
    -------
    theta_grid : np.ndarray, shape (n_theta,)
    f_all : list of n_locations arrays, each shape (M, n_theta)
        f_all[k][i, :] = f_{i,k}(θ) — log-driving input at location k.
    """
    rng = np.random.RandomState(seed)
    theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    K = periodic_rbf_kernel(theta_grid, lengthscale)

    f_all = []
    for _ in range(n_locations):
        f_loc = np.zeros((M, n_theta))
        for i in range(M):
            f_loc[i] = sample_gp_function(K, rng)
        f_all.append(f_loc)

    return theta_grid, f_all