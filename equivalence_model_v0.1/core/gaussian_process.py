"""
Core Gaussian Process Module for Mixed Selectivity Framework

This module contains all GP-related functions for generating neural tuning curves
with location-dependent lengthscales (the source of mixed selectivity).

The key innovation is that different spatial locations have different tuning widths,
creating non-separable tuning: R(θ, L) ≠ f(θ) · g(L)

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

def periodic_rbf_kernel(
    orientations: np.ndarray,
    lengthscale: float,
    jitter: float = 1e-6
) -> np.ndarray:
    """
    Compute periodic RBF (squared exponential) kernel for circular orientation space.
    
    The kernel is: k(θ_i, θ_j) = exp(-d²/(2λ²))
    where d is the circular distance: d = min(|θ_i - θ_j|, 2π - |θ_i - θ_j|)
    
    Parameters
    ----------
    orientations : np.ndarray
        Array of orientation values in radians, shape (n_theta,)
    lengthscale : float
        Lengthscale λ controlling tuning width (smaller = sharper tuning)
    jitter : float
        Small value added to diagonal for numerical stability
        
    Returns
    -------
    K : np.ndarray
        Covariance matrix, shape (n_theta, n_theta)
    """
    n_theta = len(orientations)
    K = np.zeros((n_theta, n_theta))
    
    for i in range(n_theta):
        for j in range(n_theta):
            # Circular distance on [-π, π]
            dist = np.abs(orientations[i] - orientations[j])
            dist = np.minimum(dist, 2 * np.pi - dist)
            K[i, j] = np.exp(-dist**2 / (2 * lengthscale**2))
    
    # Add jitter for numerical stability
    K += jitter * np.eye(n_theta)
    
    return K


def sample_gp_function(
    K: np.ndarray,
    random_state: np.random.RandomState
) -> np.ndarray:
    """
    Sample a function from a Gaussian Process with covariance K.
    
    Uses Cholesky decomposition: f = L @ z where L = chol(K), z ~ N(0, I)
    
    Parameters
    ----------
    K : np.ndarray
        Covariance matrix, shape (n_theta, n_theta)
    random_state : np.random.RandomState
        Random state for reproducibility
        
    Returns
    -------
    f : np.ndarray
        GP sample, shape (n_theta,)
    """
    L = np.linalg.cholesky(K)
    z = random_state.randn(K.shape[0])
    return L @ z


# ============================================================================
# LENGTHSCALE GENERATION
# ============================================================================

def generate_location_dependent_lengthscales(
    n_locations: int,
    base_lengthscale: float,
    variability: float,
    random_state: np.random.RandomState
) -> np.ndarray:
    """
    Generate location-dependent lengthscales (source of mixed selectivity).
    
    λ_i = λ_base × |1 + σ_λ × z_i| where z_i ~ N(0, 1)
    
    This creates heterogeneous tuning widths across locations, breaking
    separability and creating conjunctive (mixed) selectivity.
    
    Parameters
    ----------
    n_locations : int
        Number of spatial locations
    base_lengthscale : float
        Base lengthscale λ_base
    variability : float
        Standard deviation σ_λ controlling heterogeneity
    random_state : np.random.RandomState
        Random state for reproducibility
        
    Returns
    -------
    lengthscales : np.ndarray
        Location-dependent lengthscales, shape (n_locations,)
    """
    random_factors = 1.0 + variability * random_state.randn(n_locations)
    random_factors = np.abs(random_factors)  # Ensure positive
    return base_lengthscale * random_factors


# ============================================================================
# NEURON GENERATION
# ============================================================================

def generate_neuron_tuning_curves(
    n_orientations: int,
    n_locations: int,
    base_lengthscale: float,
    lengthscale_variability: float,
    random_state: np.random.RandomState,
    gain_variability: float = 0.2
) -> Dict:
    """
    Generate tuning curves for a single neuron across all locations.
    
    For each location, we:
    1. Generate a location-specific lengthscale
    2. Build the covariance matrix with that lengthscale
    3. Sample a GP function (log-rate)
    4. Apply random gain modulation
    
    Parameters
    ----------
    n_orientations : int
        Number of orientation bins (n_θ)
    n_locations : int
        Number of spatial locations (L)
    base_lengthscale : float
        Base lengthscale λ_base
    lengthscale_variability : float
        Variability σ_λ for heterogeneous tuning
    random_state : np.random.RandomState
        Random state for reproducibility
    gain_variability : float
        Variability in gain modulation across locations
        
    Returns
    -------
    dict with:
        - 'f_samples': np.ndarray, shape (n_locations, n_orientations)
            Log-rate tuning functions for each location
        - 'lengthscales': np.ndarray, shape (n_locations,)
            Location-dependent lengthscales
        - 'orientations': np.ndarray, shape (n_orientations,)
            Orientation values in radians
        - 'gains': np.ndarray, shape (n_locations,)
            Gain factors applied to each location
    """
    # Create orientation grid
    orientations = np.linspace(-np.pi, np.pi, n_orientations)
    
    # Generate location-dependent lengthscales
    lengthscales = generate_location_dependent_lengthscales(
        n_locations, base_lengthscale, lengthscale_variability, random_state
    )
    
    # Sample GP functions for each location
    f_samples = np.zeros((n_locations, n_orientations))
    gains = np.zeros(n_locations)
    
    for loc in range(n_locations):
        # Build kernel with this location's lengthscale
        K = periodic_rbf_kernel(orientations, lengthscales[loc])
        
        # Sample GP function
        f_loc = sample_gp_function(K, random_state)
        
        # Apply gain modulation
        gain = np.abs(1.0 + gain_variability * random_state.randn())
        gains[loc] = gain
        f_samples[loc, :] = f_loc * gain
    
    return {
        'f_samples': f_samples,
        'lengthscales': lengthscales,
        'orientations': orientations,
        'gains': gains
    }


def generate_neuron_population(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    base_lengthscale: float,
    lengthscale_variability: float,
    seed: int,
    gain_variability: float = 0.2
) -> List[Dict]:
    """
    Generate a population of neurons with mixed selectivity.
    
    Each neuron gets its own random lengthscales, creating population
    heterogeneity that is characteristic of prefrontal cortex.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons to generate
    n_orientations : int
        Number of orientation bins
    n_locations : int
        Number of spatial locations
    base_lengthscale : float
        Base lengthscale
    lengthscale_variability : float
        Lengthscale variability
    seed : int
        Master random seed
    gain_variability : float
        Gain variability
        
    Returns
    -------
    population : List[Dict]
        List of neuron dictionaries, each containing tuning curve data
    """
    master_rng = np.random.RandomState(seed)
    population = []
    
    for neuron_idx in range(n_neurons):
        # Each neuron gets its own random state (derived from master)
        neuron_seed = master_rng.randint(0, 2**31)
        neuron_rng = np.random.RandomState(neuron_seed)
        
        neuron_data = generate_neuron_tuning_curves(
            n_orientations=n_orientations,
            n_locations=n_locations,
            base_lengthscale=base_lengthscale,
            lengthscale_variability=lengthscale_variability,
            random_state=neuron_rng,
            gain_variability=gain_variability
        )
        neuron_data['neuron_idx'] = neuron_idx
        neuron_data['seed'] = neuron_seed
        
        population.append(neuron_data)
    
    return population


# ============================================================================
# RESPONSE COMPUTATION
# ============================================================================

def compute_log_rate_tensor(
    f_samples: np.ndarray,
    subset: Tuple[int, ...]
) -> np.ndarray:
    """
    Compute the log-rate tensor G for a subset of locations.
    
    G(θ_1, ..., θ_l) = Σ_k f_k(θ_k)
    
    This is the additive combination in log-space, which becomes
    multiplicative in rate-space after exponentiation.
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)
    subset : Tuple[int, ...]
        Indices of active locations
        
    Returns
    -------
    G : np.ndarray
        Log-rate tensor, shape (n_theta,) * l where l = len(subset)
    """
    n_theta = f_samples.shape[1]
    l = len(subset)
    
    # Initialize l-dimensional tensor
    G = np.zeros([n_theta] * l)
    
    # Add each location's contribution along its dimension
    for dim_idx, loc in enumerate(subset):
        shape = [1] * l
        shape[dim_idx] = n_theta
        G = G + f_samples[loc, :].reshape(shape)
    
    return G


def compute_pre_normalized_response(
    f_samples: np.ndarray,
    subset: Tuple[int, ...]
) -> np.ndarray:
    """
    Compute pre-normalized response R = exp(G) for a subset.
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions
    subset : Tuple[int, ...]
        Active location indices
        
    Returns
    -------
    R_pre : np.ndarray
        Pre-normalized response tensor
    """
    G = compute_log_rate_tensor(f_samples, subset)
    return np.exp(G)


def compute_driving_input(f_samples: np.ndarray) -> np.ndarray:
    """
    Compute driving input g = exp(f) for all locations.
    
    This is the quantity that enters the DN denominator.
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)
        
    Returns
    -------
    g : np.ndarray
        Driving inputs, shape (n_locations, n_orientations)
    """
    return np.exp(f_samples)


def compute_mean_driving_input(f_samples: np.ndarray) -> np.ndarray:
    """
    Compute mean driving input ḡ_j = mean_θ[exp(f_j(θ))] for each location.
    
    This is what enters the DN denominator in Bays (2014).
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)
        
    Returns
    -------
    g_bar : np.ndarray
        Mean driving inputs, shape (n_locations,)
    """
    g = compute_driving_input(f_samples)
    return np.mean(g, axis=1)