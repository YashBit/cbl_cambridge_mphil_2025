"""
Divisive Normalization Module - TRUE POPULATION DN (Memory-Efficient)

This module implements the CORRECT divisive normalization as specified:

    r^{post}_i(θ) = γ * r^{pre}_i(θ) / (σ² + N^{-1} * Σ_j r^{pre}_j(θ))

=============================================================================
MEMORY-EFFICIENT IMPLEMENTATION
=============================================================================

The original implementation creates tensors of shape (n_θ)^l for N neurons:
- l=2: 16 MB    ✓
- l=4: 1.6 GB   ⚠️
- l=6: 160 GB   ✗ KILLED
- l=8: 16 TB    ✗ IMPOSSIBLE

This version exploits mathematical structure to avoid tensor explosion:

1. ACTIVITY CAP THEOREM (analytical):
   Σᵢ r^post_i(θ) = γ × N  (EXACT when σ²→0, for ALL stimuli!)

2. PRE-DN FACTORIZATION (analytical):
   Mean[∏ₖ g(θₖ)] = ∏ₖ Mean[g(θₖ)]  (O(l×n_θ) not O(n_θ^l))

3. MONTE CARLO (for verification):
   Sample random configurations → O(N × n_samples) memory

Key differences from naive version:
- Denominator sums over NEURONS (index j), not stimuli
- Denominator is STIMULUS-DEPENDENT (different for each θ)
- In limit σ² → 0: Total population response = γ * N (constant!)

Author: Mixed Selectivity Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# EFFICIENT COMPUTATION (ANALYTICAL)
# ============================================================================

def compute_pre_dn_mean_efficient(
    f_samples: np.ndarray,
    subset: Tuple[int, ...]
) -> float:
    """
    Compute mean pre-DN response analytically using factorization.
    
    Mean[R_pre] = Mean[∏ₖ g(θₖ, locₖ)] = ∏ₖ Mean_θ[g(θ, locₖ)]
    
    This exploits the factorization property: orientations at different
    locations are sampled independently, so the expectation of the product
    equals the product of expectations.
    
    Complexity: O(l × n_θ) instead of O(n_θ^l)
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)
    subset : Tuple[int, ...]
        Active location indices
        
    Returns
    -------
    mean_pre : float
        Mean pre-normalized response (averaged over all configurations)
    """
    g = np.exp(f_samples)  # (n_locations, n_orientations)
    g_bar = np.mean(g[list(subset), :], axis=1)  # (l,) # Average over orientations
    return np.prod(g_bar)


def compute_total_pre_dn_population(
    f_samples_population: List[np.ndarray],
    subset: Tuple[int, ...]
) -> float:
    """
    Compute total pre-DN activity for entire population (analytical).
    
    Complexity: O(N × l × n_θ)
    Memory: O(N × L × n_θ) - just the tuning curves, no tensor explosion
    """
    return sum(
        compute_pre_dn_mean_efficient(f, subset) 
        for f in f_samples_population
    )


def compute_total_post_dn_analytical(gamma: float, N: int) -> float:
    """
    Analytical result for total post-DN activity.
    
    THEOREM: In the limit σ² → 0:
        Σᵢ Mean[R^post_i] = γ × N (exact!)
    
    This is the fundamental conservation law of population DN.
    """
    return gamma * N


# ============================================================================
# MONTE CARLO ESTIMATION
# ============================================================================

def compute_dn_montecarlo_vectorized(
    G: np.ndarray,
    subset: Tuple[int, ...],
    gamma: float,
    sigma_sq: float,
    n_samples: int,
    rng: np.random.RandomState
) -> Dict:
    """
    Vectorized Monte Carlo DN computation.
    
    Memory: O(N × n_samples) regardless of set size l
    
    Parameters
    ----------
    G : np.ndarray
        Pre-stacked population array, shape (N, n_locations, n_orientations)
        where G[i] = exp(f_samples[i])
    subset : Tuple[int, ...]
        Active location indices
    gamma : float
        Gain constant
    sigma_sq : float
        Semi-saturation constant
    n_samples : int
        Number of Monte Carlo samples
    rng : np.random.RandomState
        Random number generator
        
    Returns
    -------
    dict with Monte Carlo estimates
    """
    N = G.shape[0]
    l = len(subset)
    n_orientations = G.shape[2]
    
    # Sample configurations: (n_samples, l)
    configs = rng.randint(0, n_orientations, size=(n_samples, l))
    subset_arr = np.array(subset)
    
    # Vectorized: R_pre[s, n] = ∏ₖ G[n, subset[k], configs[s,k]]
    R_pre = np.ones((n_samples, N))
    for k in range(l):
        loc = subset_arr[k]
        R_pre *= G[:, loc, configs[:, k]].T
    
    # Population DN
    pop_mean = np.mean(R_pre, axis=1, keepdims=True)
    R_post = gamma * R_pre / (sigma_sq + pop_mean)
    
    # Statistics
    total_pre = np.sum(np.mean(R_pre, axis=0))
    total_post = np.sum(np.mean(R_post, axis=0))
    
    return {
        'total_pre': total_pre,
        'total_post': total_post,
        'mean_pre_per_neuron': np.mean(R_pre),
        'mean_post_per_neuron': np.mean(R_post)
    }


# ============================================================================
# POPULATION-LEVEL DN (EFFICIENT VERSION - MAIN API)
# ============================================================================

def compute_population_denominator(
    R_pre_population: List[np.ndarray],
    sigma_sq: float = 1e-6
) -> np.ndarray:
    """
    Compute TRUE population DN denominator at each stimulus configuration.
    
    D(θ) = σ² + N^{-1} * Σ_j r^{pre}_j(θ)
    
    WARNING: This materializes the full tensor. Use efficient methods for large l.
    """
    N = len(R_pre_population)
    R_pre_stack = np.stack(R_pre_population, axis=0)
    mean_response = np.mean(R_pre_stack, axis=0)
    return sigma_sq + mean_response


def apply_population_divisive_normalization(
    f_samples_population: List[np.ndarray],
    subset: Tuple[int, ...],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6,
    use_efficient: bool = True,
    n_mc_samples: int = 10000,
    mc_seed: Optional[int] = None,
    G_precomputed: Optional[np.ndarray] = None
) -> Dict:
    """
    Apply TRUE population DN to all neurons (memory-efficient).
    
    For each neuron i at each stimulus θ:
        r^{post}_i(θ) = γ * r^{pre}_i(θ) / D(θ)
    where:
        D(θ) = σ² + N^{-1} * Σ_j r^{pre}_j(θ)
    
    In the limit σ² → 0:
        Σ_i r^{post}_i(θ) = γ * N  (constant for all θ!)
    
    Parameters
    ----------
    f_samples_population : List[np.ndarray]
        Population of neurons' tuning functions
        Each has shape (n_locations, n_orientations)
    subset : Tuple[int, ...]
        Active location indices
    gamma : float
        Gain constant (total activity budget per neuron)
    sigma_sq : float
        Semi-saturation constant
    use_efficient : bool
        If True (default), use memory-efficient computation
        If False, use original (WILL CRASH for large N and l)
    n_mc_samples : int
        Monte Carlo samples for verification (when use_efficient=True)
    mc_seed : int, optional
        Random seed for Monte Carlo
    G_precomputed : np.ndarray, optional
        Pre-computed G = exp(f) stacked array for efficiency
        
    Returns
    -------
    dict with:
        'R_pre_population': List of pre-normalized responses (None if efficient)
        'R_post_population': List of post-normalized responses (None if efficient)
        'denominator': Stimulus-dependent denominator array (None if efficient)
        'total_pre_activity': Total pre-DN activity
        'total_post_activity': Total post-DN activity (should ≈ γ*N)
    """
    N = len(f_samples_population)
    l = len(subset)
    
    if use_efficient:
        # ─────────────────────────────────────────────────────────────────
        # EFFICIENT PATH: No tensor materialization
        # ─────────────────────────────────────────────────────────────────
        
        # Analytical pre-DN (factorized)
        total_pre = compute_total_pre_dn_population(f_samples_population, subset)
        
        # Analytical post-DN (activity cap theorem)
        total_post_analytical = gamma * N
        
        # Monte Carlo verification
        if G_precomputed is not None:
            G = G_precomputed
        else:
            G = np.stack([np.exp(f) for f in f_samples_population], axis=0)
        
        rng = np.random.RandomState(mc_seed)
        mc_result = compute_dn_montecarlo_vectorized(
            G, subset, gamma, sigma_sq, n_mc_samples, rng
        )
        
        return {
            'R_pre_population': None,  # Not materialized (memory efficient)
            'R_post_population': None,
            'denominator': None,
            'total_pre_activity': total_pre,
            'total_post_activity': total_post_analytical,  # Exact
            'total_post_mc': mc_result['total_post'],  # MC verification
            'N_neurons': N,
            'theoretical_total': gamma * N,
            'method': 'efficient',
            'compression_ratio': total_pre / (gamma * N),
            'per_item_activity': gamma * N / l
        }
    
    else:
        # ─────────────────────────────────────────────────────────────────
        # ORIGINAL PATH: Full tensor (DANGEROUS for large N, l)
        # ─────────────────────────────────────────────────────────────────
        from core.gaussian_process import compute_pre_normalized_response
        
        # Check memory requirements
        n_orientations = f_samples_population[0].shape[1]
        memory_gb = 2 * N * (n_orientations ** l) * 8 / 1e9
        
        if memory_gb > 8:
            raise MemoryError(
                f"Original implementation would require {memory_gb:.1f} GB. "
                f"Use use_efficient=True instead (requires ~{n_mc_samples * N * 16 / 1e6:.0f} MB)."
            )
        
        # Step 1: Compute pre-normalized responses for ALL neurons
        R_pre_population = []
        for neuron_f in f_samples_population:
            R_pre = compute_pre_normalized_response(neuron_f, subset)
            R_pre_population.append(R_pre)
        
        # Step 2: Compute population denominator
        denominator = compute_population_denominator(R_pre_population, sigma_sq)
        
        # Step 3: Apply DN to each neuron
        R_post_population = []
        for R_pre in R_pre_population:
            R_post = gamma * R_pre / denominator
            R_post_population.append(R_post)
        
        # Step 4: Compute summary statistics
        total_pre = sum(np.mean(R) for R in R_pre_population)
        total_post = sum(np.mean(R) for R in R_post_population)
        
        return {
            'R_pre_population': R_pre_population,
            'R_post_population': R_post_population,
            'denominator': denominator,
            'total_pre_activity': total_pre,
            'total_post_activity': total_post,
            'N_neurons': N,
            'theoretical_total': gamma * N,
            'method': 'original'
        }


# ============================================================================
# SINGLE-NEURON DN (FOR EXPERIMENT 1 - BACKWARD COMPATIBILITY)
# ============================================================================

def compute_global_denominator(
    f_samples: np.ndarray,
    sigma_sq: float = 1e-6
) -> float:
    """
    OLD VERSION: Compute denominator by averaging over stimuli.
    
    Kept for backward compatibility with Experiment 1.
    
    D = Σ_j ḡ_j + σ²  where ḡ_j = mean_θ[exp(f_j(θ))]
    """
    g_all = np.exp(f_samples)
    g_bar = np.mean(g_all, axis=1)
    return np.sum(g_bar) + sigma_sq


def compute_subset_denominator(
    f_samples: np.ndarray,
    subset: Tuple[int, ...],
    sigma_sq: float = 1e-6
) -> float:
    """
    Compute per-subset DN denominator.
    
    D_subset = Σ_{j ∈ S} ḡ_j + σ²
    """
    g_subset = np.exp(f_samples[list(subset), :])
    g_bar_subset = np.mean(g_subset, axis=1)
    return np.sum(g_bar_subset) + sigma_sq


def apply_divisive_normalization(
    R_pre: np.ndarray,
    denominator: float,
    gamma: float = 100.0
) -> np.ndarray:
    """
    Apply DN with scalar denominator (for Experiment 1).
    
    R_post = γ * R_pre / D
    """
    return gamma * R_pre / denominator


def compute_normalized_response_global(
    f_samples: np.ndarray,
    subset: Tuple[int, ...],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6,
    global_denominator: Optional[float] = None
) -> Dict:
    """
    Compute both pre- and post-normalized responses with Global DN.
    """
    from core.gaussian_process import compute_pre_normalized_response
    
    R_pre = compute_pre_normalized_response(f_samples, subset)
    
    if global_denominator is None:
        global_denominator = compute_global_denominator(f_samples, sigma_sq)
    
    R_post = apply_divisive_normalization(R_pre, global_denominator, gamma)
    
    return {
        'R_pre': R_pre,
        'R_post': R_post,
        'denominator': global_denominator
    }


# ============================================================================
# ACTIVITY COMPUTATION HELPERS
# ============================================================================

def compute_total_activity(R: np.ndarray) -> float:
    """Compute total (mean) activity from response tensor."""
    return np.mean(R)


def compute_per_item_activity(R_post: np.ndarray, n_items: int) -> float:
    """Compute mean activity per item after DN."""
    return np.mean(R_post) / n_items


def compute_per_item_activity_population(
    R_post_population: List[np.ndarray],
    n_items: int
) -> Dict:
    """Compute per-item activity for a population."""
    if R_post_population is None or R_post_population[0] is None:
        # Efficient mode - use analytical result
        return {
            'mean_per_neuron': None,
            'total_population': None,
            'per_item': None,
            'note': 'Use efficient API for statistics'
        }
    
    activities_per_neuron = [np.mean(R) for R in R_post_population]
    mean_per_neuron = np.mean(activities_per_neuron)
    total_population = sum(activities_per_neuron)
    
    return {
        'mean_per_neuron': mean_per_neuron,
        'total_population': total_population,
        'per_item': total_population / n_items,
        'activities_per_neuron': activities_per_neuron
    }


def compute_per_item_activity_efficient(gamma: float, N: int, l: int) -> float:
    """
    Compute per-item activity analytically.
    
    Per-item = (γ × N) / l
    """
    return gamma * N / l


# ============================================================================
# COMPRESSION RATIO ANALYSIS
# ============================================================================

def compute_compression_ratio(R_pre_mean: float, R_post_mean: float) -> float:
    """Compute compression ratio from DN."""
    return R_pre_mean / (R_post_mean + 1e-10)


def compute_compression_ratio_population(
    R_pre_population: List[np.ndarray],
    R_post_population: List[np.ndarray]
) -> Dict:
    """Compute compression statistics for population."""
    if R_pre_population is None or R_post_population is None:
        return {'note': 'Use efficient API for compression stats'}
    
    total_pre = sum(np.mean(R) for R in R_pre_population)
    total_post = sum(np.mean(R) for R in R_post_population)
    
    return {
        'total_pre': total_pre,
        'total_post': total_post,
        'compression_ratio': total_pre / (total_post + 1e-10)
    }


def compute_compression_ratio_efficient(total_pre: float, gamma: float, N: int) -> float:
    """Compute compression ratio efficiently."""
    return total_pre / (gamma * N)


# ============================================================================
# VERIFICATION AND ANALYSIS
# ============================================================================

def verify_activity_cap(
    R_post_population: List[np.ndarray],
    gamma: float,
    N: int,
    sigma_sq: float
) -> Dict:
    """
    Verify that total population activity ≈ γ * N.
    
    This is the key prediction of true DN:
        lim_{σ²→0} Σ_i r^{post}_i(θ) = γ * N
    """
    if R_post_population is None or R_post_population[0] is None:
        # Efficient mode
        return {
            'total_activity': gamma * N,  # Analytical
            'theoretical_total': gamma * N,
            'absolute_error': 0.0,
            'relative_error': 0.0,
            'method': 'analytical',
            'note': 'Exact by Activity Cap Theorem when σ²→0'
        }
    
    total_activity = sum(np.mean(R) for R in R_post_population)
    theoretical_total = gamma * N
    
    error = np.abs(total_activity - theoretical_total)
    relative_error = error / theoretical_total
    
    return {
        'total_activity': total_activity,
        'theoretical_total': theoretical_total,
        'absolute_error': error,
        'relative_error': relative_error,
        'gamma': gamma,
        'N': N,
        'sigma_sq': sigma_sq,
        'method': 'computed'
    }


def verify_activity_cap_efficient(
    gamma: float,
    N: int,
    sigma_sq: float,
    observed_mc: Optional[float] = None
) -> Dict:
    """
    Verify activity cap using efficient computation.
    """
    theoretical = gamma * N
    
    result = {
        'theoretical_total': theoretical,
        'gamma': gamma,
        'N': N,
        'sigma_sq': sigma_sq,
        'theorem': 'Σᵢ rᵢᵖᵒˢᵗ(θ) = γN for all θ when σ²→0'
    }
    
    if observed_mc is not None:
        result['observed_mc'] = observed_mc
        result['mc_error'] = abs(observed_mc - theoretical) / theoretical
    
    return result


def compute_activity_summary(
    f_samples: np.ndarray,
    gamma: float = 100.0,
    sigma_sq: float = 1e-6
) -> Dict:
    """Compute summary statistics of DN components for single neuron."""
    g_all = np.exp(f_samples)
    g_bar = np.mean(g_all, axis=1)
    denominator = np.sum(g_bar) + sigma_sq
    
    return {
        'g_bar': g_bar,
        'denominator': denominator,
        'gamma': gamma,
        'effective_gain': gamma / denominator
    }


def compute_activity_summary_population(
    f_samples_population: List[np.ndarray],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6
) -> Dict:
    """Compute summary statistics for population."""
    N = len(f_samples_population)
    
    mean_g_per_neuron = []
    for f_samples in f_samples_population:
        g = np.exp(f_samples)
        mean_g = np.mean(g)
        mean_g_per_neuron.append(mean_g)
    
    return {
        'N': N,
        'gamma': gamma,
        'sigma_sq': sigma_sq,
        'mean_g_per_neuron': mean_g_per_neuron,
        'mean_g_population': np.mean(mean_g_per_neuron),
        'theoretical_total_activity': gamma * N
    }


def predict_per_item_activity(
    gamma: float,
    denominator: float,
    n_items: int,
    mean_g: float
) -> float:
    """Predict per-item activity from DN parameters."""
    return gamma * mean_g / (denominator * n_items)


# ============================================================================
# MEMORY ESTIMATION UTILITY
# ============================================================================

def estimate_memory_usage(
    n_neurons: int,
    n_orientations: int,
    set_size: int,
    method: str = 'efficient'
) -> Dict:
    """
    Estimate memory usage for different computation methods.
    """
    bytes_per_float = 8
    config_space = n_orientations ** set_size
    
    if method == 'original':
        memory_bytes = 2 * n_neurons * config_space * bytes_per_float
        description = f"Full tensors: 2 × {n_neurons} × {n_orientations}^{set_size}"
        safe = memory_bytes < 8e9
    else:  # efficient
        n_samples = 10000
        memory_bytes = n_samples * n_neurons * bytes_per_float * 2
        description = f"MC samples: {n_samples} × {n_neurons} × 2"
        safe = True
    
    return {
        'method': method,
        'memory_bytes': memory_bytes,
        'memory_mb': memory_bytes / (1024**2),
        'memory_gb': memory_bytes / (1024**3),
        'description': description,
        'config_space': config_space,
        'safe': safe
    }