"""
Separability Analysis Module

This module provides tools for analyzing whether neural tuning functions
exhibit pure selectivity (separable) or mixed selectivity (non-separable).

Key Concept:
    A tuning function R(θ, L) is SEPARABLE if it can be factored:
        R(θ, L) = f(θ) · g(L)
    
    This is equivalent to the tuning matrix being RANK-1.

Analysis Method:
    We use SVD to decompose the tuning matrix:
        M = U @ S @ V^T
    
    Separability Index = σ₁² / Σᵢ σᵢ² (variance explained by first component)
    
    - Index ≈ 1: Perfectly separable (pure selectivity)
    - Index < 0.8: Mixed selectivity (conjunctive coding)

This is important because:
    - Pure selectivity: Neurons encode θ OR location, independently
    - Mixed selectivity: Neurons encode θ AND location conjunctively
    
Mixed selectivity is characteristic of prefrontal cortex and enables
flexible, context-dependent computation (Rigotti et al., 2013).

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CORE SEPARABILITY COMPUTATION
# ============================================================================

def compute_separability_index(
    tuning_matrix: np.ndarray,
    return_components: bool = False
) -> float | Tuple[float, Dict]:
    """
    Compute separability index using SVD.
    
    SVD decomposes M = U @ S @ V^T
    Separability = σ₁² / Σᵢ σᵢ² (fraction of variance in first component)
    
    Parameters
    ----------
    tuning_matrix : np.ndarray
        2D array of shape (n_orientations, n_locations) or vice versa
    return_components : bool
        If True, also return SVD components and detailed statistics
        
    Returns
    -------
    separability : float
        Value between 0 and 1
        - 1.0: Perfectly separable (rank-1, pure selectivity)
        - < 0.8: Mixed selectivity (conjunctive coding)
    components : dict (only if return_components=True)
        SVD components and additional statistics
    """
    # Perform SVD
    U, s, Vt = np.linalg.svd(tuning_matrix, full_matrices=False)
    
    # Compute separability (variance explained by first component)
    total_variance = np.sum(s**2)
    separability = s[0]**2 / total_variance if total_variance > 0 else 1.0
    
    if return_components:
        components = {
            'singular_values': s,
            'U': U,
            'Vt': Vt,
            'variance_explained': s**2 / total_variance,
            'cumulative_variance': np.cumsum(s**2) / total_variance,
            'effective_rank': np.sum(s > 1e-10 * s[0]),  # Number of significant components
            'condition_number': s[0] / (s[-1] + 1e-10)
        }
        return separability, components
    
    return separability


def classify_selectivity(separability: float, threshold: float = 0.8) -> str:
    """
    Classify neuron as having pure or mixed selectivity.
    
    Parameters
    ----------
    separability : float
        Separability index
    threshold : float
        Threshold for classification (default 0.8)
        
    Returns
    -------
    classification : str
        'pure' or 'mixed'
    """
    return 'pure' if separability >= threshold else 'mixed'


# ============================================================================
# POPULATION ANALYSIS
# ============================================================================

def analyze_neuron_separability(
    f_samples: np.ndarray,
    use_rates: bool = True
) -> Dict:
    """
    Analyze separability of a single neuron's tuning functions.
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions, shape (n_locations, n_orientations)
    use_rates : bool
        If True, analyze exp(f) (rates); if False, analyze f (log-rates)
        
    Returns
    -------
    dict with separability analysis results
    """
    # Choose representation
    if use_rates:
        tuning_matrix = np.exp(f_samples).T  # (n_orientations, n_locations)
    else:
        tuning_matrix = f_samples.T
    
    # Compute separability with components
    sep, components = compute_separability_index(tuning_matrix, return_components=True)
    
    return {
        'separability_index': sep,
        'classification': classify_selectivity(sep),
        'singular_values': components['singular_values'],
        'variance_explained': components['variance_explained'],
        'effective_rank': components['effective_rank'],
        'is_mixed': sep < 0.8
    }


def analyze_population_separability(
    population: List[Dict],
    use_rates: bool = True
) -> Dict:
    """
    Analyze separability statistics for an entire population.
    
    Parameters
    ----------
    population : List[Dict]
        List of neuron dictionaries (from generate_neuron_population)
    use_rates : bool
        Whether to analyze rates or log-rates
        
    Returns
    -------
    dict with population-level statistics
    """
    separabilities = []
    effective_ranks = []
    
    for neuron in population:
        analysis = analyze_neuron_separability(neuron['f_samples'], use_rates)
        separabilities.append(analysis['separability_index'])
        effective_ranks.append(analysis['effective_rank'])
    
    separabilities = np.array(separabilities)
    effective_ranks = np.array(effective_ranks)
    
    return {
        'separabilities': separabilities,
        'mean': np.mean(separabilities),
        'std': np.std(separabilities),
        'median': np.median(separabilities),
        'min': np.min(separabilities),
        'max': np.max(separabilities),
        'percent_mixed': np.mean(separabilities < 0.8) * 100,
        'percent_pure': np.mean(separabilities >= 0.8) * 100,
        'effective_ranks': effective_ranks,
        'mean_effective_rank': np.mean(effective_ranks),
        'n_neurons': len(population)
    }


# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

def compute_lengthscale_separability_correlation(
    population: List[Dict]
) -> Dict:
    """
    Analyze relationship between lengthscale heterogeneity and separability.
    
    Hypothesis: More variable lengthscales → lower separability (more mixed)
    
    Parameters
    ----------
    population : List[Dict]
        List of neuron dictionaries
        
    Returns
    -------
    dict with correlation analysis
    """
    lengthscale_cvs = []  # Coefficient of variation
    separabilities = []
    
    for neuron in population:
        ls = neuron['lengthscales']
        cv = np.std(ls) / np.mean(ls)
        lengthscale_cvs.append(cv)
        
        sep = analyze_neuron_separability(neuron['f_samples'])['separability_index']
        separabilities.append(sep)
    
    lengthscale_cvs = np.array(lengthscale_cvs)
    separabilities = np.array(separabilities)
    
    # Compute correlation
    correlation = np.corrcoef(lengthscale_cvs, separabilities)[0, 1]
    
    return {
        'lengthscale_cvs': lengthscale_cvs,
        'separabilities': separabilities,
        'correlation': correlation,
        'negative_correlation_expected': True,  # More CV → less separable
        'confirms_hypothesis': correlation < 0
    }


def decompose_tuning_structure(
    f_samples: np.ndarray,
    n_components: int = 3
) -> Dict:
    """
    Decompose tuning structure into separable and non-separable parts.
    
    Parameters
    ----------
    f_samples : np.ndarray
        Log-rate tuning functions
    n_components : int
        Number of SVD components to return
        
    Returns
    -------
    dict with decomposition
    """
    rates = np.exp(f_samples).T  # (n_orientations, n_locations)
    U, s, Vt = np.linalg.svd(rates, full_matrices=False)
    
    # Rank-1 approximation (separable part)
    separable_part = s[0] * np.outer(U[:, 0], Vt[0, :])
    
    # Residual (non-separable part)
    non_separable_part = rates - separable_part
    
    # Reconstruct with top k components
    reconstructions = {}
    for k in range(1, min(n_components + 1, len(s) + 1)):
        recon = np.zeros_like(rates)
        for i in range(k):
            recon += s[i] * np.outer(U[:, i], Vt[i, :])
        reconstructions[f'rank_{k}'] = recon
    
    return {
        'separable_part': separable_part,
        'non_separable_part': non_separable_part,
        'reconstructions': reconstructions,
        'singular_values': s[:n_components],
        'variance_explained': (s[:n_components]**2) / np.sum(s**2),
        'residual_norm': np.linalg.norm(non_separable_part) / np.linalg.norm(rates)
    }


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def summarize_separability(
    population: List[Dict],
    verbose: bool = False
) -> Dict:
    """
    Generate comprehensive separability summary for reporting.
    
    Parameters
    ----------
    population : List[Dict]
        List of neuron dictionaries
    verbose : bool
        If True, print summary
        
    Returns
    -------
    dict with comprehensive summary
    """
    pop_analysis = analyze_population_separability(population)
    
    summary = {
        'n_neurons': pop_analysis['n_neurons'],
        'separability': {
            'mean': pop_analysis['mean'],
            'std': pop_analysis['std'],
            'median': pop_analysis['median'],
            'range': (pop_analysis['min'], pop_analysis['max'])
        },
        'classification': {
            'percent_mixed': pop_analysis['percent_mixed'],
            'percent_pure': pop_analysis['percent_pure'],
            'n_mixed': int(pop_analysis['percent_mixed'] * pop_analysis['n_neurons'] / 100),
            'n_pure': int(pop_analysis['percent_pure'] * pop_analysis['n_neurons'] / 100)
        },
        'effective_rank': {
            'mean': pop_analysis['mean_effective_rank'],
            'values': pop_analysis['effective_ranks']
        },
        'all_separabilities': pop_analysis['separabilities']
    }
    
    if verbose:
        print("\n" + "="*60)
        print("  SEPARABILITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"\n  Population size: {summary['n_neurons']} neurons")
        print(f"\n  Separability Index:")
        print(f"    Mean:   {summary['separability']['mean']:.3f}")
        print(f"    Std:    {summary['separability']['std']:.3f}")
        print(f"    Median: {summary['separability']['median']:.3f}")
        print(f"    Range:  [{summary['separability']['range'][0]:.3f}, "
              f"{summary['separability']['range'][1]:.3f}]")
        print(f"\n  Classification (threshold = 0.8):")
        print(f"    Mixed selectivity:  {summary['classification']['percent_mixed']:.1f}% "
              f"({summary['classification']['n_mixed']} neurons)")
        print(f"    Pure selectivity:   {summary['classification']['percent_pure']:.1f}% "
              f"({summary['classification']['n_pure']} neurons)")
        print(f"\n  Effective Rank: {summary['effective_rank']['mean']:.2f} (mean)")
    
    return summary