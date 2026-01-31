"""
Maximum Likelihood Decoding - Core Module

=============================================================================
THEORETICAL FOUNDATION
=============================================================================

THE DECODING PROBLEM
--------------------
Given observed spike counts n = (n₁, n₂, ..., n_N) from N neurons,
estimate the stimulus θ that caused this pattern.

THE LIKELIHOOD FUNCTION
-----------------------
For independent Poisson neurons with tuning curves f_i(θ):

    P(n | θ) = ∏ᵢ [f_i(θ)·T_d]^{n_i} · exp(-f_i(θ)·T_d) / n_i!

THE LOG-LIKELIHOOD
------------------
    ℓ(θ) = Σᵢ [ n_i · log(f_i(θ)) - f_i(θ)·T_d ]

    (ignoring constant terms that don't depend on θ)

THE ML ESTIMATE
---------------
    θ̂_ML = argmax_θ ℓ(θ)

OPTIMALITY PROPERTIES (asymptotic)
----------------------------------
    1. Consistent: θ̂ → θ_true as N → ∞
    2. Efficient: Var[θ̂] → 1/I_F (achieves Cramér-Rao bound)
    3. Normal: θ̂ - θ ~ N(0, 1/I_F) for large N

=============================================================================
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DecodingResult:
    """Result of ML decoding for a single trial."""
    theta_true: float           # True stimulus value
    theta_estimate: float       # ML estimate
    error: float                # θ̂ - θ (signed error)
    log_likelihood_max: float   # Log-likelihood at estimate
    log_likelihood_true: float  # Log-likelihood at true value


@dataclass 
class PopulationDecodingResult:
    """Results of ML decoding across multiple trials."""
    theta_true: np.ndarray          # Shape (n_trials,)
    theta_estimates: np.ndarray     # Shape (n_trials,)
    errors: np.ndarray              # Shape (n_trials,)
    mean_absolute_error: float
    std_error: float
    circular_std: float             # For circular variables
    

# =============================================================================
# CORE LOG-LIKELIHOOD COMPUTATION
# =============================================================================

def compute_log_likelihood(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    T_d: float
) -> np.ndarray:
    """
    Compute log-likelihood for all candidate stimulus values.
    
    ℓ(θ) = Σᵢ [ n_i · log(f_i(θ)) - f_i(θ)·T_d ]
    
    Parameters
    ----------
    spike_counts : np.ndarray
        Observed spike counts, shape (N,) where N = number of neurons
    tuning_curves : np.ndarray
        Tuning curves f_i(θ), shape (N, n_theta) where n_theta = number of 
        stimulus values to evaluate
    T_d : float
        Decoding time window in seconds
        
    Returns
    -------
    log_likelihood : np.ndarray
        Log-likelihood at each stimulus value, shape (n_theta,)
    """
    # Ensure numerical stability
    tuning_curves_safe = np.maximum(tuning_curves, 1e-10)
    
    # ℓ(θ) = Σᵢ [ n_i · log(f_i(θ)) - f_i(θ)·T_d ]
    # spike_counts: (N,) -> (N, 1) for broadcasting
    # tuning_curves_safe: (N, n_theta)
    
    term1 = spike_counts[:, np.newaxis] * np.log(tuning_curves_safe)  # (N, n_theta)
    term2 = tuning_curves_safe * T_d  # (N, n_theta)
    
    # Sum over neurons
    log_likelihood = np.sum(term1 - term2, axis=0)  # (n_theta,)
    
    return log_likelihood


def compute_log_likelihood_single(
    spike_counts: np.ndarray,
    rates: np.ndarray,
    T_d: float
) -> float:
    """
    Compute log-likelihood for a single stimulus value.
    
    Parameters
    ----------
    spike_counts : np.ndarray
        Observed spike counts, shape (N,)
    rates : np.ndarray
        Firing rates at the candidate stimulus, shape (N,)
    T_d : float
        Decoding time window
        
    Returns
    -------
    log_likelihood : float
        Log-likelihood value
    """
    rates_safe = np.maximum(rates, 1e-10)
    return float(np.sum(spike_counts * np.log(rates_safe) - rates_safe * T_d))


# =============================================================================
# MAXIMUM LIKELIHOOD ESTIMATION
# =============================================================================

def decode_ml(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    T_d: float
) -> Tuple[float, float, np.ndarray]:
    """
    Decode stimulus using maximum likelihood (grid search).
    
    θ̂_ML = argmax_θ ℓ(θ)
    
    Parameters
    ----------
    spike_counts : np.ndarray
        Observed spike counts, shape (N,)
    tuning_curves : np.ndarray
        Tuning curves f_i(θ), shape (N, n_theta)
    theta_values : np.ndarray
        Stimulus values corresponding to tuning curve columns, shape (n_theta,)
    T_d : float
        Decoding time window
        
    Returns
    -------
    theta_ml : float
        Maximum likelihood estimate
    log_likelihood_max : float
        Log-likelihood at the ML estimate
    log_likelihood_curve : np.ndarray
        Full log-likelihood curve, shape (n_theta,)
    """
    log_likelihood = compute_log_likelihood(spike_counts, tuning_curves, T_d)
    
    idx_max = np.argmax(log_likelihood)
    theta_ml = theta_values[idx_max]
    log_likelihood_max = log_likelihood[idx_max]
    
    return theta_ml, log_likelihood_max, log_likelihood


def decode_ml_circular(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    T_d: float
) -> Tuple[float, float, np.ndarray]:
    """
    Decode circular stimulus (e.g., orientation) using ML.
    
    Same as decode_ml but handles circular variable properly.
    Theta values assumed to be in radians, spanning [0, 2π) or [-π, π).
    
    Parameters
    ----------
    spike_counts : np.ndarray
        Observed spike counts, shape (N,)
    tuning_curves : np.ndarray
        Tuning curves f_i(θ), shape (N, n_theta)
    theta_values : np.ndarray
        Stimulus values in radians, shape (n_theta,)
    T_d : float
        Decoding time window
        
    Returns
    -------
    theta_ml : float
        Maximum likelihood estimate (in radians)
    log_likelihood_max : float
        Log-likelihood at the ML estimate
    log_likelihood_curve : np.ndarray
        Full log-likelihood curve
    """
    # For circular variables, the grid search is the same
    # The difference is in how we compute errors (use circular_error)
    return decode_ml(spike_counts, tuning_curves, theta_values, T_d)


# =============================================================================
# ERROR COMPUTATION
# =============================================================================

def compute_error(theta_true: float, theta_estimate: float) -> float:
    """
    Compute signed decoding error (linear variable).
    
    Parameters
    ----------
    theta_true : float
        True stimulus value
    theta_estimate : float
        Estimated stimulus value
        
    Returns
    -------
    error : float
        Signed error (estimate - true)
    """
    return theta_estimate - theta_true


def compute_circular_error(
    theta_true: float, 
    theta_estimate: float,
    period: float = 2 * np.pi
) -> float:
    """
    Compute signed decoding error for circular variable.
    
    Wraps error to [-period/2, period/2).
    
    Parameters
    ----------
    theta_true : float
        True stimulus value (radians)
    theta_estimate : float
        Estimated stimulus value (radians)
    period : float
        Period of the circular variable (default: 2π)
        
    Returns
    -------
    error : float
        Signed circular error
    """
    error = theta_estimate - theta_true
    # Wrap to [-period/2, period/2)
    error = (error + period/2) % period - period/2
    return error


def compute_circular_std(errors: np.ndarray, period: float = 2 * np.pi) -> float:
    """
    Compute circular standard deviation.
    
    Uses the resultant vector length method:
    R = |mean(exp(i·θ))|
    circular_std = sqrt(-2·log(R)) · (period / 2π)
    
    Parameters
    ----------
    errors : np.ndarray
        Array of circular errors
    period : float
        Period of the circular variable
        
    Returns
    -------
    circ_std : float
        Circular standard deviation (same units as period)
    """
    # Convert to unit circle
    phases = 2 * np.pi * errors / period
    
    # Resultant vector
    R = np.abs(np.mean(np.exp(1j * phases)))
    
    # Circular std (in radians on unit circle)
    if R > 1e-10:
        circ_std_radians = np.sqrt(-2 * np.log(R))
    else:
        circ_std_radians = np.pi  # Maximum dispersion
    
    # Convert back to original units
    circ_std = circ_std_radians * period / (2 * np.pi)
    
    return circ_std


# =============================================================================
# BATCH DECODING (MULTIPLE TRIALS)
# =============================================================================

def decode_batch(
    spike_counts_batch: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    theta_true_batch: np.ndarray,
    T_d: float,
    circular: bool = True
) -> PopulationDecodingResult:
    """
    Decode multiple trials and compute error statistics.
    
    Parameters
    ----------
    spike_counts_batch : np.ndarray
        Spike counts for multiple trials, shape (n_trials, N)
    tuning_curves : np.ndarray
        Tuning curves, shape (N, n_theta)
    theta_values : np.ndarray
        Stimulus values, shape (n_theta,)
    theta_true_batch : np.ndarray
        True stimulus for each trial, shape (n_trials,)
    T_d : float
        Decoding time window
    circular : bool
        Whether the stimulus variable is circular
        
    Returns
    -------
    result : PopulationDecodingResult
        Decoding results and statistics
    """
    n_trials = spike_counts_batch.shape[0]
    theta_estimates = np.zeros(n_trials)
    errors = np.zeros(n_trials)
    
    period = theta_values[-1] - theta_values[0] + (theta_values[1] - theta_values[0])
    
    for trial in range(n_trials):
        theta_ml, _, _ = decode_ml(
            spike_counts_batch[trial],
            tuning_curves,
            theta_values,
            T_d
        )
        theta_estimates[trial] = theta_ml
        
        if circular:
            errors[trial] = compute_circular_error(
                theta_true_batch[trial], theta_ml, period
            )
        else:
            errors[trial] = compute_error(theta_true_batch[trial], theta_ml)
    
    # Compute statistics
    mean_abs_error = np.mean(np.abs(errors))
    std_error = np.std(errors)
    circ_std = compute_circular_std(errors, period) if circular else std_error
    
    return PopulationDecodingResult(
        theta_true=theta_true_batch,
        theta_estimates=theta_estimates,
        errors=errors,
        mean_absolute_error=mean_abs_error,
        std_error=std_error,
        circular_std=circ_std
    )


# =============================================================================
# FISHER INFORMATION & CRAMÉR-RAO BOUND
# =============================================================================

def compute_fisher_information(
    tuning_curves: np.ndarray,
    tuning_curve_derivatives: np.ndarray,
    theta_idx: int,
    T_d: float
) -> float:
    """
    Compute Fisher Information at a specific stimulus value.
    
    I_F(θ) = T_d · Σᵢ [f'_i(θ)]² / f_i(θ)
    
    Parameters
    ----------
    tuning_curves : np.ndarray
        Tuning curves f_i(θ), shape (N, n_theta)
    tuning_curve_derivatives : np.ndarray
        Derivatives f'_i(θ), shape (N, n_theta)
    theta_idx : int
        Index of stimulus value at which to compute I_F
    T_d : float
        Decoding time window
        
    Returns
    -------
    I_F : float
        Fisher Information
    """
    f = tuning_curves[:, theta_idx]
    f_prime = tuning_curve_derivatives[:, theta_idx]
    
    f_safe = np.maximum(f, 1e-10)
    
    I_F = T_d * np.sum(f_prime**2 / f_safe)
    
    return float(I_F)


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
    return 1.0 / max(fisher_information, 1e-10)


def compute_tuning_curve_derivative(
    tuning_curves: np.ndarray,
    theta_values: np.ndarray
) -> np.ndarray:
    """
    Compute numerical derivative of tuning curves.
    
    Parameters
    ----------
    tuning_curves : np.ndarray
        Tuning curves, shape (N, n_theta)
    theta_values : np.ndarray
        Stimulus values, shape (n_theta,)
        
    Returns
    -------
    derivatives : np.ndarray
        Tuning curve derivatives, shape (N, n_theta)
    """
    d_theta = theta_values[1] - theta_values[0]
    
    # Central difference with periodic boundary (for circular variables)
    derivatives = np.zeros_like(tuning_curves)
    derivatives[:, 1:-1] = (tuning_curves[:, 2:] - tuning_curves[:, :-2]) / (2 * d_theta)
    
    # Periodic boundaries
    derivatives[:, 0] = (tuning_curves[:, 1] - tuning_curves[:, -1]) / (2 * d_theta)
    derivatives[:, -1] = (tuning_curves[:, 0] - tuning_curves[:, -2]) / (2 * d_theta)
    
    return derivatives


# =============================================================================
# TUNING CURVE UTILITIES
# =============================================================================

def create_von_mises_tuning_curves(
    preferred_orientations: np.ndarray,
    theta_values: np.ndarray,
    kappa: float,
    peak_rate: float
) -> np.ndarray:
    """
    Create von Mises (circular Gaussian) tuning curves.
    
    f_i(θ) = peak_rate · exp(κ · (cos(θ - θ_pref_i) - 1))
    
    Parameters
    ----------
    preferred_orientations : np.ndarray
        Preferred orientation for each neuron, shape (N,)
    theta_values : np.ndarray
        Stimulus values to evaluate, shape (n_theta,)
    kappa : float
        Concentration parameter (higher = narrower tuning)
    peak_rate : float
        Maximum firing rate (Hz)
        
    Returns
    -------
    tuning_curves : np.ndarray
        Tuning curves, shape (N, n_theta)
    """
    N = len(preferred_orientations)
    n_theta = len(theta_values)
    
    # Broadcast: (N, 1) - (1, n_theta) = (N, n_theta)
    theta_diff = preferred_orientations[:, np.newaxis] - theta_values[np.newaxis, :]
    
    tuning_curves = peak_rate * np.exp(kappa * (np.cos(theta_diff) - 1))
    
    return tuning_curves


def create_uniform_population(
    N: int,
    theta_values: np.ndarray,
    kappa: float,
    peak_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a population with uniformly distributed preferred orientations.
    
    Parameters
    ----------
    N : int
        Number of neurons
    theta_values : np.ndarray
        Stimulus values, shape (n_theta,)
    kappa : float
        Tuning concentration
    peak_rate : float
        Peak firing rate
        
    Returns
    -------
    tuning_curves : np.ndarray
        Tuning curves, shape (N, n_theta)
    preferred_orientations : np.ndarray
        Preferred orientations, shape (N,)
    """
    # Uniformly tile the stimulus space
    theta_range = theta_values[-1] - theta_values[0] + (theta_values[1] - theta_values[0])
    preferred_orientations = np.linspace(
        theta_values[0], 
        theta_values[0] + theta_range * (N-1) / N, 
        N
    )
    
    tuning_curves = create_von_mises_tuning_curves(
        preferred_orientations, theta_values, kappa, peak_rate
    )
    
    return tuning_curves, preferred_orientations


# =============================================================================
# DIVISIVE NORMALIZATION INTEGRATION
# =============================================================================

def apply_divisive_normalization(
    tuning_curves: np.ndarray,
    gamma: float,
    sigma_sq: float = 1e-6
) -> np.ndarray:
    """
    Apply divisive normalization to tuning curves.
    
    r_i^post(θ) = γ · f_i(θ) / (σ² + N⁻¹ · Σⱼ f_j(θ))
    
    Parameters
    ----------
    tuning_curves : np.ndarray
        Pre-normalized tuning curves, shape (N, n_theta)
    gamma : float
        Gain constant
    sigma_sq : float
        Semi-saturation constant
        
    Returns
    -------
    normalized_curves : np.ndarray
        Post-DN tuning curves, shape (N, n_theta)
    """
    N = tuning_curves.shape[0]
    
    # Population mean at each stimulus
    population_mean = np.mean(tuning_curves, axis=0, keepdims=True)  # (1, n_theta)
    
    # Divisive normalization
    denominator = sigma_sq + population_mean
    normalized_curves = gamma * tuning_curves / denominator
    
    return normalized_curves


def scale_tuning_curves_for_set_size(
    tuning_curves: np.ndarray,
    set_size: int,
    gamma: float,
    N: int
) -> np.ndarray:
    """
    Scale tuning curves to reflect per-item activity under DN.
    
    Under DN with l items, total activity = γN, so per-item = γN/l.
    This scales the tuning curves accordingly.
    
    Parameters
    ----------
    tuning_curves : np.ndarray
        Base tuning curves (for set size 1), shape (N, n_theta)
    set_size : int
        Number of items in memory
    gamma : float
        Gain constant
    N : int
        Number of neurons
        
    Returns
    -------
    scaled_curves : np.ndarray
        Tuning curves scaled for set size
    """
    # Total activity budget per item
    scaling_factor = 1.0 / set_size
    
    return tuning_curves * scaling_factor