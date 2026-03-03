"""
Efficient Maximum Likelihood Decoding with Factorization

=============================================================================
THEORETICAL FOUNDATION: WHY THIS IS EFFICIENT
=============================================================================

The naive approach builds a joint tuning tensor of shape (N, n_theta, ..., n_theta)
with l dimensions, requiring O(N * n_theta^l) memory - EXPONENTIAL in set size.

This module exploits TWO mathematical properties to achieve O(N * l * n_theta):

PROPERTY 1: ACTIVITY CAP THEOREM
--------------------------------
Under divisive normalization with sigma^2 -> 0:

    Sum_i r_i(theta_1, ..., theta_l) = gamma*N    for ALL stimulus configurations!

This means the log-likelihood simplifies:

    l(theta) = Sum_i [n_i log r_i(theta) - r_i(theta) T_d]
             = Sum_i n_i log r_i(theta) - T_d * gamma*N
                                          ^
                                      CONSTANT (drops out!)

So:  theta_ML = argmax_theta Sum_i n_i log r_i(theta)

PROPERTY 2: FACTORIZED LOG-RATES
--------------------------------
Pre-normalized rates are multiplicatively separable:

    r_i^pre(theta_1, ..., theta_l) = Prod_k g_i,k(theta_k) = exp(Sum_k f_i,k(theta_k))

Taking logs:
    log r_i^pre(theta) = Sum_k f_i,k(theta_k)

The spike-weighted log-likelihood SEPARATES:
    Sum_i n_i log r_i^pre(theta) = Sum_i n_i Sum_k f_i,k(theta_k)
                                 = Sum_k [Sum_i n_i f_i,k(theta_k)]
                                 = Sum_k L_k(theta_k)

Where L_k(theta_k) = Sum_i n_i f_i,k(theta_k) is a 1D function for each location!

EFFICIENT MARGINALISATION
-------------------------
For the cued location c, we need:

    l_marginal(theta_c) = log Sum_{theta\\c} exp(l(theta))

Using factorization, the sum-of-products becomes product-of-sums:

    l_marginal(theta_c) = Lc(theta_c) + Sum_{k!=c} logsumexp(L_k)

COMPLEXITY COMPARISON
---------------------
Naive:     O(N * n_theta^l) memory and time - EXPONENTIAL
Efficient: O(N * l * n_theta) memory and time - LINEAR in set size!

Example (N=100, n_theta=32, l=8):
    Naive:     100 * 32^8 * 8 bytes = 1.1 PETABYTES
    Efficient: 100 * 8 * 32 * 8 bytes = 200 KB

=============================================================================
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.special import logsumexp


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EfficientDecodingResult:
    """Result of efficient ML decoding."""
    theta_true: float
    theta_estimate: float
    error: float
    log_likelihood_marginal: np.ndarray  # Shape (n_theta,)
    cued_location: int
    set_size: int
    method: str = 'efficient_factorized'


@dataclass 
class PopulationDecodingStats:
    """Statistics from batch decoding."""
    errors: np.ndarray
    circular_std: float
    circular_std_deg: float
    mean_absolute_error: float
    mae_deg: float


# =============================================================================
# SINGLE-ITEM LOG-LIKELIHOOD (for baseline comparisons)
# =============================================================================

def compute_log_likelihood(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    T_d: float
) -> np.ndarray:
    """
    Compute log-likelihood for all candidate stimulus values.
    
    l(theta) = Sum_i [ n_i * log(f_i(theta)) - f_i(theta)*T_d ]
    
    Parameters
    ----------
    spike_counts : np.ndarray, shape (N,)
    tuning_curves : np.ndarray, shape (N, n_theta)
    T_d : float
        
    Returns
    -------
    log_likelihood : np.ndarray, shape (n_theta,)
    """
    tuning_safe = np.maximum(tuning_curves, 1e-10)
    term1 = spike_counts[:, np.newaxis] * np.log(tuning_safe)
    term2 = tuning_safe * T_d
    return np.sum(term1 - term2, axis=0)


def decode_ml(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    T_d: float
) -> Tuple[float, float, np.ndarray]:
    """
    Decode stimulus using maximum likelihood (grid search).
    
    Used for single-item baseline comparisons.
    
    Returns
    -------
    theta_ml, log_likelihood_max, log_likelihood_curve
    """
    log_likelihood = compute_log_likelihood(spike_counts, tuning_curves, T_d)
    idx_max = np.argmax(log_likelihood)
    return theta_values[idx_max], log_likelihood[idx_max], log_likelihood


# =============================================================================
# CORE EFFICIENT COMPUTATION
# =============================================================================

def compute_spike_weighted_log_tuning(
    spike_counts: np.ndarray,
    f_samples_per_location: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Compute spike-weighted log-tuning functions for each location.
    
    L_k(theta) = Sum_i n_i f_i,k(theta)
    
    This is the KEY factorization step that reduces O(n_theta^l) to O(l * n_theta).
    
    Parameters
    ----------
    spike_counts : np.ndarray, shape (N,)
        Observed spike counts
    f_samples_per_location : List[np.ndarray]
        List of l arrays, each shape (N, n_theta)
        f_samples_per_location[k][i, :] = f_i,k(theta) for neuron i at location k
        
    Returns
    -------
    L_per_location : List[np.ndarray]
        List of l arrays, each shape (n_theta,)
        L_per_location[k] = L_k(theta) = Sum_i n_i f_i,k(theta)
    """
    L_list = []
    for f_k in f_samples_per_location:
        # f_k has shape (N, n_theta)
        # spike_counts has shape (N,)
        # L_k = Sum_i n_i f_i,k(theta) has shape (n_theta,)
        L_k = np.dot(spike_counts, f_k)  # (N,) @ (N, n_theta) = (n_theta,)
        L_list.append(L_k)
    return L_list


def compute_marginal_log_likelihood_efficient(
    L_per_location: List[np.ndarray],
    cued_location: int
) -> np.ndarray:
    """
    Compute marginal log-likelihood using factorization.
    
    l_marginal(theta_c) = Lc(theta_c) + Sum_{k!=c} logsumexp(L_k)
    
    The sum-of-products over non-cued locations factorizes into
    a product of sums (which becomes sum of logsumexp in log space).
    
    Parameters
    ----------
    L_per_location : List[np.ndarray]
        List of l arrays, each shape (n_theta,)
    cued_location : int
        Index of the cued location (0-indexed)
        
    Returns
    -------
    log_likelihood_marginal : np.ndarray, shape (n_theta,)
    """
    ell = len(L_per_location)
    
    if ell == 1:
        # Single item - no marginalisation needed
        return L_per_location[0].copy()
    
    # Get the cued location's contribution (varies with theta_c)
    L_cued = L_per_location[cued_location]
    
    # Sum logsumexp over non-cued locations (constant w.r.t. theta_c)
    non_cued_contribution = 0.0
    for k, L_k in enumerate(L_per_location):
        if k != cued_location:
            # logsumexp(L_k) = log Sum_{theta_k} exp(L_k(theta_k))
            non_cued_contribution += logsumexp(L_k)
    
    return L_cued + non_cued_contribution


def decode_ml_efficient(
    spike_counts: np.ndarray,
    f_samples_per_location: List[np.ndarray],
    theta_values: np.ndarray,
    cued_location: int
) -> Tuple[float, float, np.ndarray]:
    """
    Efficient ML decoding with marginalisation.
    
    Complexity: O(N * l * n_theta) instead of O(N * n_theta^l)
    
    Parameters
    ----------
    spike_counts : np.ndarray, shape (N,)
    f_samples_per_location : List[np.ndarray]
        List of l arrays, each shape (N, n_theta)
    theta_values : np.ndarray, shape (n_theta,)
    cued_location : int
        
    Returns
    -------
    theta_ml : float
        ML estimate of orientation at cued location
    ll_max : float
        Maximum marginal log-likelihood
    ll_marginal : np.ndarray
        Full marginal log-likelihood curve, shape (n_theta,)
    """
    # Step 1: Compute spike-weighted log-tuning for each location
    L_list = compute_spike_weighted_log_tuning(spike_counts, f_samples_per_location)
    
    # Step 2: Compute marginal log-likelihood efficiently
    ll_marginal = compute_marginal_log_likelihood_efficient(L_list, cued_location)
    
    # Step 3: Find ML estimate
    idx_max = np.argmax(ll_marginal)
    theta_ml = theta_values[idx_max]
    ll_max = ll_marginal[idx_max]
    
    return theta_ml, ll_max, ll_marginal


# =============================================================================
# ERROR COMPUTATION
# =============================================================================

def compute_circular_error(
    theta_true: float, 
    theta_estimate: float,
    period: float = 2 * np.pi
) -> float:
    """Compute signed decoding error for circular variable."""
    error = theta_estimate - theta_true
    return (error + period/2) % period - period/2


def compute_circular_std(errors: np.ndarray, period: float = 2 * np.pi) -> float:
    """
    Compute circular standard deviation using resultant vector length.
    """
    scaled = errors * (2 * np.pi / period)
    z = np.exp(1j * scaled)
    R = np.abs(np.mean(z))
    R = np.clip(R, 1e-10, 1.0 - 1e-10)
    return np.sqrt(-2 * np.log(R)) * (period / (2 * np.pi))


# =============================================================================
# COMPLEXITY COMPARISON UTILITY
# =============================================================================

def compare_complexity(n_neurons: int, n_theta: int, set_sizes: List[int]) -> None:
    """
    Print complexity comparison between naive and efficient methods.
    """
    print("\n" + "=" * 70)
    print("COMPLEXITY COMPARISON: Naive vs Efficient")
    print("=" * 70)
    print(f"\nParameters: N={n_neurons}, n_theta={n_theta}")
    print(f"\n{'Set Size':<10} {'Naive Memory':<20} {'Efficient Memory':<20} {'Speedup':<15}")
    print("-" * 65)
    
    for l in set_sizes:
        naive_bytes = n_neurons * (n_theta ** l) * 8
        efficient_bytes = n_neurons * l * n_theta * 8
        
        if naive_bytes < 1e6:
            naive_str = f"{naive_bytes/1e3:.1f} KB"
        elif naive_bytes < 1e9:
            naive_str = f"{naive_bytes/1e6:.1f} MB"
        elif naive_bytes < 1e12:
            naive_str = f"{naive_bytes/1e9:.1f} GB"
        else:
            naive_str = f"{naive_bytes/1e12:.1f} TB"
        
        efficient_str = f"{efficient_bytes/1e3:.1f} KB"
        speedup = naive_bytes / efficient_bytes
        
        print(f"{l:<10} {naive_str:<20} {efficient_str:<20} {speedup:.0e}x")
    
    print()


# =============================================================================
# SINGLE TRIAL HELPER
# =============================================================================

def run_efficient_trial(
    population: List[Dict],
    theta_values: np.ndarray,
    active_locations: Tuple[int, ...],
    true_orientations: np.ndarray,
    cued_index: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    rng: np.random.RandomState
) -> EfficientDecodingResult:
    """
    Run a single trial using efficient decoding.
    
    Full pipeline:
    1. Extract f_samples for active locations
    2. Compute firing rates at true configuration (with DN)
    3. Generate Poisson spikes
    4. Decode using efficient factorized method
    """
    ell = len(active_locations)
    N = len(population)
    n_theta = len(theta_values)
    
    # Extract f_samples for active locations
    f_samples_list = []
    for loc in active_locations:
        f_k = np.zeros((N, n_theta))
        for i, neuron in enumerate(population):
            f_k[i, :] = neuron['f_samples'][loc, :]
        f_samples_list.append(f_k)
    
    # Get true orientation indices
    theta_indices = [np.argmin(np.abs(theta_values - t)) for t in true_orientations]
    
    # Compute firing rates at true configuration with DN
    log_r_pre = np.zeros(N)
    for k, f_k in enumerate(f_samples_list):
        log_r_pre += f_k[:, theta_indices[k]]
    r_pre = np.exp(log_r_pre)
    
    D = sigma_sq + np.mean(r_pre)
    rates = gamma * r_pre / D
    
    # Generate Poisson spikes
    spike_counts = rng.poisson(rates * T_d)
    
    # Decode using efficient method
    theta_ml, ll_max, ll_marginal = decode_ml_efficient(
        spike_counts, f_samples_list, theta_values, cued_index
    )
    
    # Compute error
    theta_true = true_orientations[cued_index]
    error = compute_circular_error(theta_true, theta_ml)
    
    return EfficientDecodingResult(
        theta_true=theta_true,
        theta_estimate=theta_ml,
        error=error,
        log_likelihood_marginal=ll_marginal,
        cued_location=cued_index,
        set_size=ell
    )


# =============================================================================
# BATCH EXPERIMENT
# =============================================================================

def run_efficient_experiment(
    population: List[Dict],
    theta_values: np.ndarray,
    set_sizes: Tuple[int, ...],
    n_locations: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    n_trials: int,
    seed: int = 42
) -> Dict:
    """
    Run decoding experiment across set sizes using efficient method.
    
    This can handle ANY set size without memory issues!
    """
    rng = np.random.RandomState(seed)
    
    results = {l: {'errors': [], 'theta_true': [], 'theta_est': []} 
               for l in set_sizes}
    
    print(f"Running efficient decoding (complexity: O(N * l * n_theta))")
    print(f"Set sizes: {set_sizes}")
    
    for l in set_sizes:
        print(f"\n  Set size l={l}...")
        
        for trial in range(n_trials):
            # Sample active locations
            active_locations = tuple(rng.choice(n_locations, size=l, replace=False))
            
            # Sample true orientations
            true_orientations = rng.uniform(0, 2*np.pi, size=l)
            
            # Cue random location
            cued_index = rng.randint(l)
            
            # Run trial
            result = run_efficient_trial(
                population, theta_values, active_locations,
                true_orientations, cued_index,
                gamma, sigma_sq, T_d, rng
            )
            
            results[l]['errors'].append(result.error)
            results[l]['theta_true'].append(result.theta_true)
            results[l]['theta_est'].append(result.theta_estimate)
        
        # Compute statistics
        errors = np.array(results[l]['errors'])
        results[l]['errors'] = errors
        results[l]['circular_std'] = compute_circular_std(errors)
        results[l]['circular_std_deg'] = np.degrees(results[l]['circular_std'])
        results[l]['mean_absolute_error'] = np.mean(np.abs(errors))
        
        print(f"    sigma = {results[l]['circular_std_deg']:.2f} deg")
    
    return results


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    # Show complexity comparison
    compare_complexity(
        n_neurons=100, 
        n_theta=32, 
        set_sizes=[2, 4, 6, 8, 10, 12]
    )
    
    print("The efficient method makes ALL set sizes feasible!")
    print("Set size 12: Naive needs 1.2 PB, Efficient needs 300 KB")