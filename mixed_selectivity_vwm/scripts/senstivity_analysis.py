#!/usr/bin/env python
"""
Parameter Sensitivity Analysis for Mixed Selectivity

REAL SCIENCE: Test how results change with different parameters
to understand what actually matters vs. what's manufactured.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mixed_selectivity.core.gaussian_process import NeuralGaussianProcess
from mixed_selectivity.experiments.exp1_validation import analyze_population_separability


def test_lengthscale_sensitivity():
    """Test how GP lengthscales affect separability."""
    
    print("="*70)
    print("SENSITIVITY ANALYSIS: GP Lengthscales")
    print("="*70)
    
    theta_scales = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    spatial_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    results = np.zeros((len(theta_scales), len(spatial_scales)))
    
    for i, theta_ls in enumerate(theta_scales):
        for j, spatial_ls in enumerate(spatial_scales):
            print(f"\nTesting: theta={theta_ls:.2f}, spatial={spatial_ls:.2f}")
            
            gp = NeuralGaussianProcess(
                n_orientations=20,
                n_locations=4,
                theta_lengthscale=theta_ls,
                spatial_lengthscale=spatial_ls,
                method='gp_interaction',
                seed=42
            )
            
            tuning = gp.sample_neurons(20)
            sep_stats = analyze_population_separability(tuning, show_progress=False)
            
            results[i, j] = sep_stats['mean']
            print(f"  → Mean separability: {sep_stats['mean']:.3f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(results, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(spatial_scales)))
    ax.set_yticks(range(len(theta_scales)))
    ax.set_xticklabels([f'{s:.1f}' for s in spatial_scales])
    ax.set_yticklabels([f'{t:.1f}' for t in theta_scales])
    
    ax.set_xlabel('Spatial Lengthscale', fontsize=12)
    ax.set_ylabel('Theta Lengthscale', fontsize=12)
    ax.set_title('Mean Separability vs. GP Lengthscales\n(Green=Mixed, Red=Separable)', 
                fontsize=14, fontweight='bold')
    
    # Add values
    for i in range(len(theta_scales)):
        for j in range(len(spatial_scales)):
            text = ax.text(j, i, f'{results[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Mean Separability')
    
    # Add threshold line
    ax.contour(results, levels=[0.8], colors='blue', linewidths=3)
    
    Path('figures/analysis').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/analysis/lengthscale_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: figures/analysis/lengthscale_sensitivity.png")
    
    return results


def test_population_size_effect():
    """Test if results are stable across population sizes."""
    
    print("\n" + "="*70)
    print("STABILITY ANALYSIS: Population Size")
    print("="*70)
    
    pop_sizes = [5, 10, 20, 50, 100, 200]
    n_repeats = 5
    
    results = {}
    
    for method in ['direct', 'gp_interaction', 'simple_conjunctive']:
        print(f"\n--- Testing method: {method} ---")
        
        method_results = []
        
        for n in pop_sizes:
            sep_values = []
            
            for rep in range(n_repeats):
                gp = NeuralGaussianProcess(
                    n_orientations=20,
                    n_locations=4,
                    method=method,
                    seed=42 + rep
                )
                
                tuning = gp.sample_neurons(n)
                sep_stats = analyze_population_separability(tuning, show_progress=False)
                sep_values.append(sep_stats['mean'])
            
            mean_sep = np.mean(sep_values)
            std_sep = np.std(sep_values)
            
            method_results.append({
                'n': n,
                'mean': mean_sep,
                'std': std_sep
            })
            
            print(f"  n={n:3d}: {mean_sep:.3f} ± {std_sep:.3f}")
        
        results[method] = method_results
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'direct': 'red', 'gp_interaction': 'green', 'simple_conjunctive': 'blue'}
    
    for method, method_results in results.items():
        n_vals = [r['n'] for r in method_results]
        means = [r['mean'] for r in method_results]
        stds = [r['std'] for r in method_results]
        
        ax.errorbar(n_vals, means, yerr=stds, 
                   label=method, marker='o', linewidth=2,
                   color=colors[method], capsize=5)
    
    ax.axhline(0.8, color='black', linestyle='--', linewidth=2, 
              label='Threshold (0.8)')
    ax.set_xlabel('Population Size', fontsize=12)
    ax.set_ylabel('Mean Separability', fontsize=12)
    ax.set_title('Stability of Separability Across Population Sizes', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.savefig('figures/analysis/population_stability.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: figures/analysis/population_stability.png")
    
    return results


def test_stimulus_space_effect():
    """Test how stimulus space dimensionality affects results."""
    
    print("\n" + "="*70)
    print("DIMENSIONALITY ANALYSIS: Stimulus Space")
    print("="*70)
    
    configs = [
        (10, 2),   # Low dimensional
        (20, 4),   # Standard
        (30, 6),   # High dimensional
        (40, 8),   # Very high dimensional
    ]
    
    results = {}
    
    for method in ['gp_interaction']:  # Focus on working method
        print(f"\n--- Testing method: {method} ---")
        
        method_results = []
        
        for n_ori, n_loc in configs:
            print(f"\n  Testing: {n_ori} orientations × {n_loc} locations")
            
            gp = NeuralGaussianProcess(
                n_orientations=n_ori,
                n_locations=n_loc,
                method=method,
                seed=42
            )
            
            tuning = gp.sample_neurons(20)
            sep_stats = analyze_population_separability(tuning, show_progress=False)
            
            method_results.append({
                'n_ori': n_ori,
                'n_loc': n_loc,
                'total': n_ori * n_loc,
                'mean_sep': sep_stats['mean'],
                'percent_mixed': sep_stats['percent_mixed']
            })
            
            print(f"    → Mean sep: {sep_stats['mean']:.3f}")
            print(f"    → Mixed %: {sep_stats['percent_mixed']:.1f}%")
        
        results[method] = method_results
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    r = results['gp_interaction']
    dims = [res['total'] for res in r]
    labels = [f"{res['n_ori']}×{res['n_loc']}" for res in r]
    
    # Plot 1: Separability
    ax1.plot(dims, [res['mean_sep'] for res in r], 
            marker='o', linewidth=2, markersize=10, color='steelblue')
    ax1.axhline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Total Stimulus Conditions', fontsize=12)
    ax1.set_ylabel('Mean Separability', fontsize=12)
    ax1.set_title('Separability vs. Stimulus Dimensionality', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(dims)
    ax1.set_xticklabels(labels, rotation=45)
    
    # Plot 2: Mixed percentage
    ax2.bar(range(len(r)), [res['percent_mixed'] for res in r], 
           color='coral', edgecolor='black', linewidth=2)
    ax2.set_xlabel('Stimulus Space Configuration', fontsize=12)
    ax2.set_ylabel('% Mixed Selectivity Neurons', fontsize=12)
    ax2.set_title('Mixed Selectivity vs. Dimensionality', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(len(r)))
    ax2.set_xticklabels(labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('figures/analysis/dimensionality_effect.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: figures/analysis/dimensionality_effect.png")
    
    return results


def test_random_seed_stability():
    """Test if results are reproducible across random seeds."""
    
    print("\n" + "="*70)
    print("REPRODUCIBILITY ANALYSIS: Random Seeds")
    print("="*70)
    
    n_seeds = 20
    
    results = {}
    
    for method in ['direct', 'gp_interaction', 'simple_conjunctive']:
        print(f"\n--- Testing method: {method} ---")
        
        sep_values = []
        
        for seed in range(n_seeds):
            gp = NeuralGaussianProcess(
                n_orientations=20,
                n_locations=4,
                method=method,
                seed=seed
            )
            
            tuning = gp.sample_neurons(20)
            sep_stats = analyze_population_separability(tuning, show_progress=False)
            sep_values.append(sep_stats['mean'])
        
        results[method] = {
            'values': sep_values,
            'mean': np.mean(sep_values),
            'std': np.std(sep_values),
            'min': np.min(sep_values),
            'max': np.max(sep_values)
        }
        
        print(f"  Mean: {results[method]['mean']:.3f} ± {results[method]['std']:.3f}")
        print(f"  Range: [{results[method]['min']:.3f}, {results[method]['max']:.3f}]")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    colors = {'direct': 'red', 'gp_interaction': 'green', 'simple_conjunctive': 'blue'}
    
    for i, method in enumerate(methods):
        values = results[method]['values']
        x = np.random.normal(i, 0.1, len(values))  # Jitter for visibility
        ax.scatter(x, values, alpha=0.6, s=50, color=colors[method], label=method)
        
        # Add mean and std
        mean = results[method]['mean']
        std = results[method]['std']
        ax.errorbar(i, mean, yerr=std, fmt='D', markersize=10, 
                   color='black', capsize=10, linewidth=3)
    
    ax.axhline(0.8, color='black', linestyle='--', linewidth=2, 
              label='Threshold (0.8)', alpha=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Mean Separability', fontsize=12)
    ax.set_title(f'Reproducibility Across {n_seeds} Random Seeds\n(Black diamonds = mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('figures/analysis/seed_reproducibility.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: figures/analysis/seed_reproducibility.png")
    
    return results


if __name__ == '__main__':
    print("\n" + "="*70)
    print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nThis will test if results are:")
    print("  1. Sensitive to parameter choices")
    print("  2. Stable across population sizes")
    print("  3. Affected by stimulus dimensionality")
    print("  4. Reproducible across random seeds")
    print("\nRunning all tests...\n")
    
    # Run all tests
    lengthscale_results = test_lengthscale_sensitivity()
    population_results = test_population_size_effect()
    dimensionality_results = test_stimulus_space_effect()
    seed_results = test_random_seed_stability()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nAll figures saved to: figures/analysis/")
    print("\nKey Questions Answered:")
    print("  ✓ How do lengthscales affect separability?")
    print("  ✓ Are results stable across population sizes?")
    print("  ✓ Does stimulus dimensionality matter?")
    print("  ✓ Are results reproducible?")
