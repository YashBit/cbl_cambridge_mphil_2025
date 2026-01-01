"""
Separability analysis for neural tuning curves.

This tests whether neurons show mixed selectivity (conjunctive coding).
"""

import numpy as np
from typing import Tuple
from tqdm import tqdm


def compute_separability_index(tuning_matrix: np.ndarray, verbose: bool = False) -> float:
    """
    Compute separability index using SVD.
    
    SVD decomposes the tuning matrix into components:
    - If rank-1 (only first singular value large): separable/pure selectivity
    - If multiple singular values large: non-separable/mixed selectivity
    
    Parameters:
        tuning_matrix: 2D array [n_orientations, n_locations]
        verbose: Print decomposition details
    
    Returns:
        separability: float between 0 and 1
                     1 = perfectly separable (pure selectivity)
                     <0.8 = mixed selectivity
    """
    # Perform SVD: M = U @ S @ V^T
    U, s, Vt = np.linalg.svd(tuning_matrix, full_matrices=False)
    
    # Compute separability as variance explained by first component
    separability = s[0]**2 / np.sum(s**2)
    
    if verbose:
        print(f"  Singular values: {s[:5]}")  # First 5
        print(f"  Separability index: {separability:.3f}")
        if separability > 0.8:
            print("  → Pure selectivity (separable)")
        else:
            print("  → Mixed selectivity (conjunctive)")
    
    return separability


def analyze_population_separability(
    tuning_curves: np.ndarray,
    show_progress: bool = True
) -> dict:
    """
    Analyze separability for entire population.
    
    Parameters:
        tuning_curves: [n_neurons, n_orientations, n_locations]
        show_progress: Show progress bar
    
    Returns:
        Dictionary with statistics
    """
    n_neurons = tuning_curves.shape[0]
    separabilities = []
    
    print(f"\nAnalyzing separability for {n_neurons} neurons...")
    
    # Use tqdm for progress if requested
    iterator = tqdm(range(n_neurons), desc="Computing separability") if show_progress else range(n_neurons)
    
    for i in iterator:
        sep = compute_separability_index(tuning_curves[i])
        separabilities.append(sep)
    
    separabilities = np.array(separabilities)
    
    # Print example for first neuron
    print("\nExample (Neuron 1):")
    _ = compute_separability_index(tuning_curves[0], verbose=True)
    
    return {
        'mean': np.mean(separabilities),
        'std': np.std(separabilities),
        'median': np.median(separabilities),
        'percent_mixed': np.mean(separabilities < 0.8) * 100,
        'all_values': separabilities
    }