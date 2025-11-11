"""
Experiment 1: Validation of mixed selectivity in synthetic neural populations.

This experiment tests whether the generated neural tuning curves exhibit genuine
mixed selectivity - i.e., non-separable conjunctive responses to orientation and
spatial location.

Scientific rationale:
    Mixed selectivity is a hallmark of flexible neural computation in prefrontal
    cortex and other brain regions. Neurons with mixed selectivity respond to
    conjunctions of features rather than single features in isolation, enabling
    high-dimensional representations that support complex cognitive operations.

Validation approach:
    We use Singular Value Decomposition (SVD) to quantify separability:
        Separability = σ₁² / Σᵢ σᵢ²
    
    - Separability ≈ 1.0: Response is separable (r(θ,L) ≈ f(θ)·g(L))
    - Separability < 0.8: True mixed selectivity (non-separable)
    
    Target: Mean population separability < 0.8

References:
    Rigotti et al. (2013) Nature
    Fusi et al. (2016) Neuron
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from typing import Dict, Literal, Optional
import os
from pathlib import Path

# FIXED: Use proper package imports for your project structure
from mixed_selectivity.core.gaussian_process import NeuralGaussianProcess


def run_experiment1(
    n_neurons: int = 20,
    n_orientations: int = 20,
    n_locations: int = 4,
    theta_lengthscale: float = 0.3,
    spatial_lengthscale: float = 1.5,
    plot: bool = True,
    seed: int = 42,
    save_dir: str = 'figures/exp1',
    method: Literal['direct', 'gp_interaction', 'simple_conjunctive', 'compare'] = 'direct'
) -> Dict:
    """
    Run Experiment 1: Generate neural population and validate mixed selectivity.
    
    Experimental pipeline:
        1. Generate population with specified method
        2. Compute separability index for each neuron via SVD
        3. Test hypothesis: mean separability < 0.8
        4. Visualize results and save figures
    
    Args:
        n_neurons: Population size
        n_orientations: Number of orientation values in [-π, π]
        n_locations: Number of spatial locations
        theta_lengthscale: Orientation kernel width (GP method only)
        spatial_lengthscale: Spatial kernel width (GP method only)
        plot: Whether to generate and save figures
        seed: Random seed for reproducibility
        save_dir: Directory to save results
        method: Generation method or 'compare' to test both
    
    Returns:
        Dictionary containing:
            - tuning_curves: Neural responses (n_neurons, n_orientations, n_locations)
            - separability_stats: Statistical summary of separability indices
            - success: Boolean indicating if hypothesis test passed
            - method: Method used for generation
    """
    print("=" * 70)
    print("EXPERIMENT 1: VALIDATION OF MIXED SELECTIVITY")
    print("=" * 70)
    print(f"\nExperimental parameters:")
    print(f"  Population size: {n_neurons} neurons")
    print(f"  Stimulus space: {n_orientations} orientations × {n_locations} locations")
    print(f"  Generation method: {method}")
    print(f"  Random seed: {seed}")
    print(f"  Hypothesis: Mean separability < 0.8")
    
    if method == 'compare':
        # Compare both methods
        return _compare_methods(
            n_neurons, n_orientations, n_locations,
            theta_lengthscale, spatial_lengthscale,
            plot, seed, save_dir
        )
    
    # Standard single-method experiment
    return _run_single_method(
        n_neurons, n_orientations, n_locations,
        theta_lengthscale, spatial_lengthscale,
        method, plot, seed, save_dir
    )


def _run_single_method(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    theta_lengthscale: float,
    spatial_lengthscale: float,
    method: str,
    plot: bool,
    seed: int,
    save_dir: str
) -> Dict:
    """Execute experiment with a single generation method."""
    
    # ========================================
    # PHASE 1: GENERATE NEURAL POPULATION
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: NEURAL POPULATION GENERATION")
    print("=" * 70)
    
    gp = NeuralGaussianProcess(
        n_orientations=n_orientations,
        n_locations=n_locations,
        theta_lengthscale=theta_lengthscale,
        spatial_lengthscale=spatial_lengthscale,
        seed=seed,
        method=method
    )
    
    tuning_curves = gp.sample_neurons(n_neurons)
    
    print(f"\n✓ Generated population:")
    print(f"  Shape: {tuning_curves.shape}")
    print(f"  Mean activity: {tuning_curves.mean():.3f}")
    print(f"  Activity range: [{tuning_curves.min():.3f}, {tuning_curves.max():.3f}]")
    
    # ========================================
    # PHASE 2: SEPARABILITY ANALYSIS
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: SEPARABILITY ANALYSIS")
    print("=" * 70)
    print("\nComputing SVD-based separability for each neuron...")
    
    sep_results = analyze_population_separability(tuning_curves, show_progress=True)
    
    # Report statistics
    print(f"\nPopulation statistics:")
    print(f"  Mean separability:   {sep_results['mean']:.3f} ± {sep_results['std']:.3f}")
    print(f"  Median separability: {sep_results['median']:.3f}")
    print(f"  Range: [{sep_results['min']:.3f}, {sep_results['max']:.3f}]")
    print(f"  Neurons with mixed selectivity (<0.8): {sep_results['percent_mixed']:.1f}%")
    
    # ========================================
    # PHASE 3: HYPOTHESIS TEST
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 3: HYPOTHESIS TEST")
    print("=" * 70)
    
    threshold = 0.8
    success = sep_results['mean'] < threshold
    
    print(f"\nH₀: Mean separability < {threshold} (mixed selectivity)")
    print(f"Result: Mean separability = {sep_results['mean']:.3f}")
    
    if success:
        print(f"\n✓✓✓ HYPOTHESIS CONFIRMED")
        print(f"    Population exhibits strong mixed selectivity!")
        print(f"    {sep_results['percent_mixed']:.1f}% of neurons are non-separable")
    else:
        print(f"\n✗✗✗ HYPOTHESIS REJECTED")
        print(f"    Population shows predominantly separable tuning")
        print(f"    Only {sep_results['percent_mixed']:.1f}% of neurons are non-separable")
        
        # Provide diagnostic feedback
        if method in ['gp_interaction', 'simple_conjunctive']:
            print(f"\n    💡 Suggestion: Try method='direct' for guaranteed mixed selectivity")
    
    # ========================================
    # PHASE 4: VISUALIZATION
    # ========================================
    if plot:
        print("\n" + "=" * 70)
        print("PHASE 4: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _create_result_figures(tuning_curves, sep_results, save_dir, method)
        print(f"✓ Figures saved to: {save_dir}/")
    
    # ========================================
    # RETURN RESULTS
    # ========================================
    return {
        'tuning_curves': tuning_curves,
        'separability_stats': sep_results,
        'success': success,
        'method': method,
        'threshold': threshold
    }


def _compare_methods(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    theta_lengthscale: float,
    spatial_lengthscale: float,
    plot: bool,
    seed: int,
    save_dir: str
) -> Dict:
    """Compare all available generation methods."""
    
    print("\n" + "=" * 70)
    print("COMPARISON MODE: TESTING ALL METHODS")
    print("=" * 70)
    
    methods = ['direct', 'gp_interaction', 'simple_conjunctive']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"  TESTING METHOD: {method.upper()}")
        print(f"{'='*70}")
        
        # Run with this method
        result = _run_single_method(
            n_neurons, n_orientations, n_locations,
            theta_lengthscale, spatial_lengthscale,
            method, plot=False, seed=seed + len(all_results),
            save_dir=f"{save_dir}/{method}"
        )
        
        all_results[method] = {
            'tuning_curves': result['tuning_curves'],
            'separability_stats': result['separability_stats'],
            'success': result['success'],
            'mean_sep': result['separability_stats']['mean'],
            'percent_mixed': result['separability_stats']['percent_mixed']
        }
    
    # Create comparison visualization
    if plot:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _create_comparison_figures(all_results, save_dir)
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    sorted_methods = sorted(all_results.items(), key=lambda x: x[1]['mean_sep'])
    
    print(f"\n{'Rank':<6} {'Method':<20} {'Mean Sep':<12} {'Mixed %':<12} {'Status'}")
    print("-" * 70)
    
    for rank, (method, r) in enumerate(sorted_methods, 1):
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{rank:<6} {method:<20} {r['mean_sep']:<12.3f} {r['percent_mixed']:<12.1f} {status}")
    
    best_method = sorted_methods[0][0]
    print(f"\n🏆 BEST METHOD: {best_method}")
    print(f"   Mean separability: {sorted_methods[0][1]['mean_sep']:.3f}")
    
    return {
        'all_results': all_results,
        'best_method': best_method,
        'success': sorted_methods[0][1]['success']
    }


def _create_result_figures(
    tuning_curves: np.ndarray,
    sep_results: Dict,
    save_dir: str,
    method: str
) -> None:
    """Create and save result figures."""
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Find best (lowest separability) and worst neurons
    sep_values = sep_results['all_values']
    best_idx = np.argmin(sep_values)
    worst_idx = np.argmax(sep_values)
    
    # Plot best neuron
    ax1 = fig.add_subplot(gs[0, 0:2])
    im1 = ax1.imshow(tuning_curves[best_idx], aspect='auto', cmap='hot', interpolation='nearest')
    ax1.set_title(f'Best Neuron (Lowest Sep = {sep_values[best_idx]:.3f})', fontweight='bold')
    ax1.set_xlabel('Spatial Location')
    ax1.set_ylabel('Orientation')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot worst neuron
    ax2 = fig.add_subplot(gs[0, 2:4])
    im2 = ax2.imshow(tuning_curves[worst_idx], aspect='auto', cmap='hot', interpolation='nearest')
    ax2.set_title(f'Worst Neuron (Highest Sep = {sep_values[worst_idx]:.3f})', fontweight='bold')
    ax2.set_xlabel('Spatial Location')
    ax2.set_ylabel('Orientation')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Histogram of separability
    ax3 = fig.add_subplot(gs[1, :])
    ax3.hist(sep_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(0.8, color='red', linestyle='--', linewidth=2, label='Mixed threshold (0.8)')
    ax3.axvline(sep_results['mean'], color='green', linestyle='-', linewidth=2, 
               label=f'Mean ({sep_results["mean"]:.3f})')
    ax3.set_xlabel('Separability Index', fontsize=11)
    ax3.set_ylabel('Number of Neurons', fontsize=11)
    ax3.set_title('Distribution of Separability Across Population', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.axis('off')
    
    stats_text = f"""
POPULATION STATISTICS

Mean Separability:    {sep_results['mean']:.3f} ± {sep_results['std']:.3f}
Median Separability:  {sep_results['median']:.3f}
Range:                [{sep_results['min']:.3f}, {sep_results['max']:.3f}]

Mixed Neurons:        {sep_results['percent_mixed']:.1f}%
(Separability < 0.8)

INTERPRETATION:
{_get_interpretation(sep_results['mean'], sep_results['percent_mixed'])}
"""
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Example neurons grid
    ax5 = fig.add_subplot(gs[2, 2:])
    n_examples = min(6, len(tuning_curves))
    example_indices = np.random.choice(len(tuning_curves), n_examples, replace=False)
    
    for i, idx in enumerate(example_indices):
        ax_small = plt.subplot(3, 2, i + 1)
        ax_small.imshow(tuning_curves[idx], aspect='auto', cmap='hot', interpolation='nearest')
        ax_small.set_title(f'N{idx} (S={sep_values[idx]:.2f})', fontsize=8)
        ax_small.axis('off')
    
    # Main title
    fig.suptitle(f'Experiment 1: Mixed Selectivity Validation ({method})\n'
                f'Mean Separability = {sep_results["mean"]:.3f}',
                fontsize=14, fontweight='bold')
    
    # Save figure
    filepath = Path(save_dir) / f'exp1_results_{method}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filepath.name}")


def _create_comparison_figures(results: Dict, save_dir: str) -> None:
    """Create side-by-side comparison of all methods."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    methods = list(results.keys())
    colors = ['green', 'blue', 'purple']
    
    # Plot histograms for each method
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax = fig.add_subplot(gs[0, i])
        r = results[method]
        sep_vals = r['separability_stats']['all_values']
        
        ax.hist(sep_vals, bins=15, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(0.8, color='red', linestyle='--', linewidth=2)
        ax.axvline(r['mean_sep'], color='darkgreen', linestyle='-', linewidth=2)
        ax.set_title(f'{method}\nMean Sep = {r["mean_sep"]:.3f}', fontweight='bold')
        ax.set_xlabel('Separability')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    # Comparison bar chart
    ax_comp = fig.add_subplot(gs[1, :])
    x = np.arange(len(methods))
    mean_seps = [results[m]['mean_sep'] for m in methods]
    percent_mixed = [results[m]['percent_mixed'] for m in methods]
    
    ax_comp.bar(x - 0.2, mean_seps, 0.4, label='Mean Separability', alpha=0.7, color='steelblue')
    ax_comp.bar(x + 0.2, [p/100 for p in percent_mixed], 0.4, label='% Mixed (normalized)', 
               alpha=0.7, color='coral')
    ax_comp.axhline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold (0.8)')
    
    ax_comp.set_xlabel('Method', fontsize=12)
    ax_comp.set_ylabel('Value', fontsize=12)
    ax_comp.set_title('Method Comparison', fontsize=14, fontweight='bold')
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(methods, rotation=15)
    ax_comp.legend(fontsize=10)
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Save
    filepath = Path(save_dir) / 'method_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filepath.name}")


def _get_interpretation(mean_sep: float, percent_mixed: float) -> str:
    """Generate interpretation text based on results."""
    if mean_sep < 0.5:
        return (f"Strong non-separability across population.\n"
                f"      Neurons exhibit robust mixed selectivity,\n"
                f"      suitable for flexible computation.")
    elif mean_sep < 0.8:
        return (f"Moderate mixed selectivity detected.\n"
                f"      Population shows conjunctive encoding\n"
                f"      of orientation and location.")
    else:
        return (f"Population shows predominantly separable tuning.\n"
                f"      Limited evidence of mixed selectivity.\n"
                f"      Consider using 'direct' method.")


# ========================================
# UTILITY: Separability analysis function
# ========================================

def analyze_population_separability(
    tuning_curves: np.ndarray,
    show_progress: bool = False
) -> Dict:
    """
    Analyze separability of neural tuning curves using SVD.
    
    For each neuron, compute:
        Separability = σ₁² / Σᵢ σᵢ²
    
    where σᵢ are singular values of the (orientations × locations) matrix.
    
    Args:
        tuning_curves: Array of shape (n_neurons, n_orientations, n_locations)
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with statistics: mean, std, median, min, max, 
        percent_mixed, all_values
    """
    n_neurons = tuning_curves.shape[0]
    separability_values = np.zeros(n_neurons)
    
    iterator = range(n_neurons)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Computing separability")
        except ImportError:
            pass
    
    for i in iterator:
        # Compute SVD
        U, S, Vt = np.linalg.svd(tuning_curves[i], full_matrices=False)
        
        # Separability = variance explained by first component
        separability_values[i] = (S[0]**2) / (np.sum(S**2) + 1e-10)
    
    # Compute statistics
    threshold = 0.8
    percent_mixed = 100 * np.mean(separability_values < threshold)
    
    return {
        'mean': np.mean(separability_values),
        'std': np.std(separability_values),
        'median': np.median(separability_values),
        'min': np.min(separability_values),
        'max': np.max(separability_values),
        'percent_mixed': percent_mixed,
        'all_values': separability_values
    }


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == '__main__':
    """
    Example usage:
    
    # Test single method
    results = run_experiment1(
        n_neurons=50,
        n_orientations=20,
        n_locations=4,
        method='direct',
        seed=42
    )
    
    # Compare methods
    results = run_experiment1(
        n_neurons=50,
        method='compare',
        seed=42
    )
    """
    
    print("\nExample: Running experiment with direct method...")
    results = run_experiment1(
        n_neurons=20,
        n_orientations=20,
        n_locations=4,
        method='direct',
        plot=True,
        seed=42
    )
    
    print(f"\n✓ Experiment complete!")
    print(f"  Success: {results['success']}")
    print(f"  Mean separability: {results['separability_stats']['mean']:.3f}")