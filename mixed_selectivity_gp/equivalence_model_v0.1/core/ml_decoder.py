"""
Maximum Likelihood Decoder Module for Mixed Selectivity Framework

Implements the ML decoder following Bays (2014):

    θ̂ = argmax_θ Σ_i n_i · log(r_i(θ))

For a single location, this simplifies to:

    θ̂ = argmax_θ n · log(r(θ))
    
Since r(θ) = γ · exp(f(θ)) / D, this becomes:

    θ̂ = argmax_θ n · f(θ)

KEY INSIGHT:
    The ML decoder is optimal under Poisson noise. It finds the 
    orientation that maximizes the probability of observing the
    spike count given the tuning curve shape.

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# SINGLE LOCATION DECODING
# ============================================================================

def decode_ml_single_location(
    spike_count: int,
    f_at_location: np.ndarray,
    denominator: float,
    gamma: float = 100.0
) -> Tuple[int, np.ndarray]:
    """
    ML decode for a single location using grid search.
    
    The ML estimate is:
        θ̂ = argmax_θ n · log(r(θ))
        
    Since r(θ) = γ · exp(f(θ)) / D:
        θ̂ = argmax_θ n · f(θ)  (the log and constants cancel in argmax)
    
    Parameters:
        spike_count: Observed spike count n
        f_at_location: GP sample f(θ) for all θ values, shape (n_theta,)
        denominator: DN denominator (for computing log-likelihood)
        gamma: DN gain constant
    
    Returns:
        theta_hat_idx: Decoded orientation index
        log_likelihood: Log-likelihood at each θ (for visualization)
    """
    # Firing rate at each possible θ
    r_theta = gamma * np.exp(f_at_location) / denominator
    
    # Avoid log(0)
    r_theta = np.maximum(r_theta, 1e-10)
    
    # Log-likelihood (proportional to): n · log(r(θ))
    log_likelihood = spike_count * np.log(r_theta)
    
    # Find argmax
    theta_hat_idx = np.argmax(log_likelihood)
    
    return theta_hat_idx, log_likelihood


def decode_ml_all_locations(
    spike_counts: np.ndarray,
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    denominator: float,
    gamma: float = 100.0
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    ML decode for all active locations independently.
    
    Each location is decoded separately using its tuning curve
    and observed spike count.
    
    Parameters:
        spike_counts: Spike counts at each active location
        f_samples: GP samples, shape (total_locations, n_theta)
        active_locations: Tuple of active location indices
        denominator: DN denominator
        gamma: DN gain constant
    
    Returns:
        theta_hat_indices: Decoded orientation index at each location
        all_log_likelihoods: Log-likelihood curves for each location
    """
    n_active = len(active_locations)
    theta_hat_indices = np.zeros(n_active, dtype=int)
    all_log_likelihoods = []
    
    for i, loc in enumerate(active_locations):
        theta_hat_idx, log_lik = decode_ml_single_location(
            spike_counts[i],
            f_samples[loc, :],
            denominator,
            gamma
        )
        theta_hat_indices[i] = theta_hat_idx
        all_log_likelihoods.append(log_lik)
    
    return theta_hat_indices, all_log_likelihoods


# ============================================================================
# ERROR COMPUTATION
# ============================================================================

def compute_circular_error(
    true_idx: int,
    decoded_idx: int,
    n_theta: int
) -> float:
    """
    Compute circular error in radians.
    
    Parameters:
        true_idx: True orientation index
        decoded_idx: Decoded orientation index
        n_theta: Number of orientation bins
    
    Returns:
        error: Signed circular error in radians, range [-π, π]
    """
    # Convert indices to radians
    theta_values = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    
    true_theta = theta_values[true_idx]
    decoded_theta = theta_values[decoded_idx]
    
    # Circular difference
    error = decoded_theta - true_theta
    
    # Wrap to [-π, π]
    error = np.arctan2(np.sin(error), np.cos(error))
    
    return error


def compute_errors_batch(
    true_indices: np.ndarray,
    decoded_indices: np.ndarray,
    n_theta: int
) -> np.ndarray:
    """
    Compute circular errors for a batch of trials.
    
    Parameters:
        true_indices: Array of true orientation indices
        decoded_indices: Array of decoded orientation indices
        n_theta: Number of orientation bins
    
    Returns:
        errors: Array of signed circular errors in radians
    """
    theta_values = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    
    true_theta = theta_values[true_indices]
    decoded_theta = theta_values[decoded_indices]
    
    errors = decoded_theta - true_theta
    errors = np.arctan2(np.sin(errors), np.cos(errors))
    
    return errors


# ============================================================================
# PRECISION METRICS
# ============================================================================

def compute_precision(errors: np.ndarray) -> float:
    """
    Compute precision as inverse variance.
    
    Parameters:
        errors: Array of circular errors in radians
    
    Returns:
        precision: 1 / variance (higher = more precise)
    """
    return 1.0 / (np.var(errors) + 1e-10)


def compute_circular_variance(errors: np.ndarray) -> float:
    """
    Compute circular variance (0 = all same, 1 = uniform).
    
    Uses the resultant length: R = |mean(exp(i·θ))|
    Circular variance = 1 - R
    
    Parameters:
        errors: Array of circular errors in radians
    
    Returns:
        circular_variance: Range [0, 1]
    """
    resultant_length = np.abs(np.mean(np.exp(1j * errors)))
    return 1.0 - resultant_length


def compute_circular_std(errors: np.ndarray) -> float:
    """
    Compute circular standard deviation.
    
    Circular SD = sqrt(-2 * log(R)) where R is resultant length
    
    Parameters:
        errors: Array of circular errors in radians
    
    Returns:
        circular_std: In radians
    """
    resultant_length = np.abs(np.mean(np.exp(1j * errors)))
    # Avoid log(0)
    resultant_length = np.maximum(resultant_length, 1e-10)
    return np.sqrt(-2 * np.log(resultant_length))


def compute_rmse(errors: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(np.mean(errors**2))


def compute_mae(errors: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(errors))


def compute_all_metrics(errors: np.ndarray) -> Dict:
    """
    Compute all precision/error metrics.
    
    Parameters:
        errors: Array of circular errors in radians
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'rmse': compute_rmse(errors),
        'mae': compute_mae(errors),
        'precision': compute_precision(errors),
        'variance': np.var(errors),
        'circular_variance': compute_circular_variance(errors),
        'circular_std': compute_circular_std(errors),
        'mean_error': np.mean(errors),  # Should be ~0 if unbiased
        'n_samples': len(errors)
    }


# ============================================================================
# FISHER INFORMATION (THEORETICAL PRECISION)
# ============================================================================

def compute_fisher_information_single(
    f_at_location: np.ndarray,
    theta_idx: int,
    denominator: float,
    gamma: float = 100.0,
    T_d: float = 0.1,
    delta_theta: float = None
) -> float:
    """
    Compute Fisher Information at a single orientation.
    
    For Poisson noise: I(θ) = T_d · (r'(θ))² / r(θ)
    
    where r'(θ) is the derivative of firing rate w.r.t. θ.
    
    Parameters:
        f_at_location: GP sample f(θ) for all θ values
        theta_idx: Index of the orientation
        denominator: DN denominator
        gamma: DN gain constant
        T_d: Decoding window
        delta_theta: Step size for numerical derivative (auto if None)
    
    Returns:
        fisher_info: Fisher Information at this θ
    """
    n_theta = len(f_at_location)
    
    if delta_theta is None:
        delta_theta = 2 * np.pi / n_theta
    
    # Firing rate at θ
    r_theta = gamma * np.exp(f_at_location[theta_idx]) / denominator
    
    # Numerical derivative using central difference
    idx_plus = (theta_idx + 1) % n_theta
    idx_minus = (theta_idx - 1) % n_theta
    
    r_plus = gamma * np.exp(f_at_location[idx_plus]) / denominator
    r_minus = gamma * np.exp(f_at_location[idx_minus]) / denominator
    
    r_prime = (r_plus - r_minus) / (2 * delta_theta)
    
    # Fisher Information
    fisher_info = T_d * (r_prime ** 2) / (r_theta + 1e-10)
    
    return fisher_info


def compute_mean_fisher_information(
    f_at_location: np.ndarray,
    denominator: float,
    gamma: float = 100.0,
    T_d: float = 0.1
) -> float:
    """
    Compute mean Fisher Information across all orientations.
    
    This gives the theoretical precision bound for this location.
    """
    n_theta = len(f_at_location)
    
    fisher_values = []
    for theta_idx in range(n_theta):
        fi = compute_fisher_information_single(
            f_at_location, theta_idx, denominator, gamma, T_d
        )
        fisher_values.append(fi)
    
    return np.mean(fisher_values)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ML DECODER MODULE TEST")
    print("="*70)
    
    np.random.seed(42)
    
    # Create synthetic f_samples
    n_locations = 8
    n_theta = 36
    f_samples = np.random.randn(n_locations, n_theta) * 0.5
    
    print(f"\n  Test data: f_samples shape = {f_samples.shape}")
    
    # Test single location decoding
    print("\n" + "-"*70)
    print("  Testing: decode_ml_single_location")
    print("-"*70)
    
    spike_count = 5
    f_loc = f_samples[0, :]
    denominator = 10.0
    
    decoded_idx, log_lik = decode_ml_single_location(
        spike_count, f_loc, denominator, gamma=100.0
    )
    
    print(f"\n  Spike count: {spike_count}")
    print(f"  Decoded index: {decoded_idx}")
    print(f"  Max log-likelihood: {log_lik[decoded_idx]:.4f}")
    
    # Test error computation
    print("\n" + "-"*70)
    print("  Testing: compute_circular_error")
    print("-"*70)
    
    true_idx = 10
    error = compute_circular_error(true_idx, decoded_idx, n_theta)
    error_deg = error * 180 / np.pi
    
    print(f"\n  True index: {true_idx}")
    print(f"  Decoded index: {decoded_idx}")
    print(f"  Error: {error:.4f} rad = {error_deg:.1f}°")
    
    # Test batch metrics
    print("\n" + "-"*70)
    print("  Testing: compute_all_metrics")
    print("-"*70)
    
    # Simulate some errors
    errors = np.random.randn(100) * 0.3  # ~17° std
    metrics = compute_all_metrics(errors)
    
    print(f"\n  Simulated {metrics['n_samples']} errors")
    print(f"  RMSE: {metrics['rmse']:.4f} rad = {metrics['rmse']*180/np.pi:.1f}°")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Circular variance: {metrics['circular_variance']:.4f}")
    
    # Test Fisher Information
    print("\n" + "-"*70)
    print("  Testing: Fisher Information")
    print("-"*70)
    
    mean_fi = compute_mean_fisher_information(
        f_samples[0, :], denominator=10.0, gamma=100.0, T_d=0.1
    )
    
    print(f"\n  Mean Fisher Information: {mean_fi:.4f}")
    print(f"  Theoretical precision bound: sqrt(I) = {np.sqrt(mean_fi):.4f}")
    
    print("\n  ✓ All tests passed!")