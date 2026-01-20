"""
Poisson Spiking Noise - Core Module

=============================================================================
THEORETICAL FOUNDATION
=============================================================================

THE POISSON DISTRIBUTION
------------------------
For a neuron with firing rate r (Hz) observed for time T_d (seconds):
    - Expected spike count: λ = r × T_d
    - Actual count: n ~ Poisson(λ)
    - P(n | λ) = (λⁿ × e^{-λ}) / n!

KEY PROPERTIES:
    - E[n] = λ              (mean)
    - Var[n] = λ            (variance = mean)
    - SNR = √λ              (signal-to-noise ratio)

THE CAPACITY LIMIT CONNECTION:
    Under DN: rate ∝ 1/l  →  λ ∝ 1/l  →  SNR ∝ 1/√l  →  Error ∝ √l

=============================================================================
"""

import numpy as np
from typing import Optional


# =============================================================================
# CORE SPIKE GENERATION
# =============================================================================

def generate_spikes(
    rates: np.ndarray,
    T_d: float,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Generate spike counts from Poisson distribution.
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates in Hz, shape (N,)
    T_d : float
        Decoding time window in seconds
    rng : np.random.RandomState, optional
        Random number generator for reproducibility
        
    Returns
    -------
    counts : np.ndarray
        Spike counts, shape (N,), dtype int
    """
    if rng is None:
        rng = np.random.RandomState()
    
    rates = np.maximum(np.asarray(rates, dtype=np.float64), 0.0)
    expected_counts = rates * T_d
    
    return rng.poisson(expected_counts)


def generate_spikes_multi_trial(
    rates: np.ndarray,
    T_d: float,
    n_trials: int,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Generate spike counts for multiple trials (vectorized).
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates in Hz, shape (N,)
    T_d : float
        Decoding time window in seconds
    n_trials : int
        Number of independent trials
    rng : np.random.RandomState, optional
        Random number generator
        
    Returns
    -------
    counts : np.ndarray
        Spike counts, shape (n_trials, N), dtype int
    """
    if rng is None:
        rng = np.random.RandomState()
    
    rates = np.maximum(np.asarray(rates, dtype=np.float64), 0.0)
    expected_counts = rates * T_d
    
    return rng.poisson(expected_counts, size=(n_trials, len(rates)))


# =============================================================================
# SIGNAL-TO-NOISE RATIO
# =============================================================================

def compute_snr(rates: np.ndarray, T_d: float) -> np.ndarray:
    """
    Compute theoretical SNR for Poisson neurons.
    
    SNR = E[n] / Std[n] = λ / √λ = √λ = √(r × T_d)
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates in Hz, shape (N,)
    T_d : float
        Decoding time window in seconds
        
    Returns
    -------
    snr : np.ndarray
        Signal-to-noise ratio, shape (N,)
    """
    expected_counts = np.maximum(rates * T_d, 1e-10)
    return np.sqrt(expected_counts)


# =============================================================================
# FISHER INFORMATION
# =============================================================================

def compute_fisher_information(
    rates: np.ndarray,
    rate_derivatives: np.ndarray,
    T_d: float
) -> float:
    """
    Compute Fisher Information for Poisson neurons.
    
    I_F(θ) = T_d × Σᵢ [r'ᵢ(θ)]² / rᵢ(θ)
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates r_i(θ), shape (N,)
    rate_derivatives : np.ndarray
        Tuning curve slopes dr_i/dθ, shape (N,)
    T_d : float
        Decoding time window
        
    Returns
    -------
    I_F : float
        Fisher Information
    """
    rates_safe = np.maximum(rates, 1e-10)
    return float(T_d * np.sum(rate_derivatives**2 / rates_safe))


def compute_cramer_rao_bound(fisher_information: float) -> float:
    """
    Compute Cramér-Rao lower bound on estimation variance.
    
    Var[θ̂] ≥ 1 / I_F(θ)
    
    Parameters
    ----------
    fisher_information : float
        Fisher Information I_F(θ)
        
    Returns
    -------
    min_variance : float
        Minimum achievable variance
    """
    return 1.0 / (fisher_information + 1e-10)