"""
Poisson Spiking Module for Mixed Selectivity Framework

Implements the neural noise layer following Bays (2014):

    n_i ~ Poisson(r_i(θ) · T_d)

Where:
    - r_i(θ) is the post-DN firing rate at location i for stimulus θ
    - T_d is the decoding window duration (typically 100ms)
    - n_i is the spike count

KEY INSIGHT:
    Poisson noise is the source of behavioral errors. With DN limiting
    total activity, more items means less activity per item, which means
    more Poisson noise relative to signal, which means lower precision.

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================================
# SPIKE GENERATION
# ============================================================================

def generate_spikes(
    firing_rates: np.ndarray,
    T_d: float = 0.1,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Generate Poisson spike counts from firing rates.
    
    Implements: n_i ~ Poisson(r_i · T_d)
    
    Parameters:
        firing_rates: Array of firing rates in Hz (any shape)
        T_d: Decoding window duration in seconds (default: 100ms)
        random_state: Random state for reproducibility
    
    Returns:
        spike_counts: Integer spike counts (same shape as firing_rates)
    
    Example:
        For rate=50 Hz and T_d=0.1s, expected count = 5 spikes
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Poisson parameter λ = rate × time
    lambda_param = firing_rates * T_d
    
    # Ensure non-negative (rates should already be positive after DN)
    lambda_param = np.maximum(lambda_param, 0)
    
    # Generate Poisson counts
    spike_counts = random_state.poisson(lambda_param)
    
    return spike_counts


def compute_dn_firing_rates(
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    theta_indices: Tuple[int, ...],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Compute post-DN firing rates for a given stimulus configuration.
    
    Parameters:
        f_samples: GP samples, shape (total_locations, n_theta)
        active_locations: Tuple of active location indices
        theta_indices: True orientation index at each active location
        gamma: DN gain constant (Hz)
        sigma_sq: DN semi-saturation constant
    
    Returns:
        firing_rates: Post-DN firing rates at each active location
        denominator: The DN denominator used
    """
    # Get f values at the true stimulus orientations
    f_at_theta = np.array([
        f_samples[loc, theta_idx] 
        for loc, theta_idx in zip(active_locations, theta_indices)
    ])
    
    # Pre-normalized: g = exp(f)
    g_at_theta = np.exp(f_at_theta)
    
    # Compute global DN denominator over active locations
    g_all = np.exp(f_samples[list(active_locations), :])
    g_bar = np.mean(g_all, axis=1)
    denominator = np.sum(g_bar) + sigma_sq
    
    # Post-normalized firing rates
    firing_rates = gamma * g_at_theta / denominator
    
    return firing_rates, denominator


def generate_spikes_for_stimulus(
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    theta_indices: Tuple[int, ...],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6,
    T_d: float = 0.1,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate spikes for a specific stimulus configuration.
    
    Full pipeline: stimulus → DN firing rates → Poisson spikes
    
    Parameters:
        f_samples: GP samples, shape (total_locations, n_theta)
        active_locations: Which locations have items
        theta_indices: True orientation index at each active location
        gamma: DN gain constant
        sigma_sq: DN semi-saturation
        T_d: Decoding window
        random_state: For reproducibility
    
    Returns:
        spike_counts: Spike counts at each active location
        firing_rates: The underlying firing rates (for analysis)
        denominator: The DN denominator used
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Compute DN firing rates
    firing_rates, denominator = compute_dn_firing_rates(
        f_samples, active_locations, theta_indices, gamma, sigma_sq
    )
    
    # Generate spikes
    spike_counts = generate_spikes(firing_rates, T_d, random_state)
    
    return spike_counts, firing_rates, denominator


# ============================================================================
# SPIKE STATISTICS
# ============================================================================

def compute_expected_spikes(firing_rate: float, T_d: float = 0.1) -> float:
    """Expected spike count: E[n] = r · T_d"""
    return firing_rate * T_d


def compute_spike_variance(firing_rate: float, T_d: float = 0.1) -> float:
    """Variance of spike count (Poisson): Var[n] = r · T_d"""
    return firing_rate * T_d


def compute_fano_factor(spike_counts: np.ndarray) -> float:
    """
    Fano factor: Var/Mean (should be ~1 for Poisson).
    
    Useful for verifying Poisson statistics.
    """
    mean = np.mean(spike_counts)
    var = np.var(spike_counts)
    return var / (mean + 1e-10)


def compute_snr(firing_rate: float, T_d: float = 0.1) -> float:
    """
    Signal-to-noise ratio for Poisson process.
    
    SNR = E[n] / std[n] = sqrt(r · T_d)
    
    This is why precision scales with sqrt(activity).
    """
    expected = firing_rate * T_d
    return np.sqrt(expected)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  POISSON SPIKING MODULE TEST")
    print("="*70)
    
    # Test basic spike generation
    print("\n" + "-"*70)
    print("  Testing: generate_spikes")
    print("-"*70)
    
    rng = np.random.RandomState(42)
    
    # Test with known rates
    rates = np.array([10, 50, 100, 200])  # Hz
    T_d = 0.1  # 100ms
    
    print(f"\n  Firing rates: {rates} Hz")
    print(f"  T_d: {T_d*1000:.0f} ms")
    print(f"\n  Expected counts: {rates * T_d}")
    
    # Generate many samples to check statistics
    n_samples = 10000
    all_spikes = np.zeros((n_samples, len(rates)))
    for i in range(n_samples):
        all_spikes[i] = generate_spikes(rates, T_d, rng)
    
    print(f"\n  Observed mean: {np.mean(all_spikes, axis=0)}")
    print(f"  Observed var:  {np.var(all_spikes, axis=0)}")
    print(f"  Fano factors:  {np.var(all_spikes, axis=0) / np.mean(all_spikes, axis=0)}")
    print(f"  (Fano ≈ 1 confirms Poisson)")
    
    # Test with DN firing rates
    print("\n" + "-"*70)
    print("  Testing: generate_spikes_for_stimulus")
    print("-"*70)
    
    # Create synthetic f_samples
    n_locations = 8
    n_theta = 20
    f_samples = rng.randn(n_locations, n_theta) * 0.5
    
    active_locations = (0, 1, 2, 3)
    theta_indices = (5, 10, 15, 3)
    
    spikes, rates, denom = generate_spikes_for_stimulus(
        f_samples, active_locations, theta_indices,
        gamma=100.0, T_d=0.1, random_state=rng
    )
    
    print(f"\n  Active locations: {active_locations}")
    print(f"  Theta indices: {theta_indices}")
    print(f"  Firing rates: {rates}")
    print(f"  Spike counts: {spikes}")
    print(f"  DN denominator: {denom:.4f}")
    
    print("\n  ✓ All tests passed!")