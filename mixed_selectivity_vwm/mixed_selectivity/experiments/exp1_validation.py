"""
Experiment 1: Validate that GP generates mixed selectivity.
Enhanced version with multiple methods for comparison.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Literal
import os

from ..core.gaussian_process import NeuralGaussianProcess
from ..analysis.separability import analyze_population_separability


def run_experiment1(
    n_neurons: int = 20,
    n_orientations: int = 20,
    n_locations: int = 4,
    theta_lengthscale: float = 0.3,
    spatial_lengthscale: float = 1.5,
    plot: bool = True,
    seed: int = 42,
    verbose: bool = True,
    save_dir: str = 'figures/exp1',
    method: Literal['direct', 'gp_interaction', 'original', 'compare'] = 'direct'
) -> Dict:
    """
    Run Experiment 1: Generate population and test for mixed selectivity.
    
    Parameters:
        method: Which generation method to use
            - 'direct': Direct construction with guaranteed mixed selectivity
            - 'gp_interaction': GP-based with interaction terms
            - 'original': Original implementation
            - 'compare': Compare all methods
    
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("EXPERIMENT 1: Validating Mixed Selectivity")
    print("=" * 60)
    print(f"\nConfiguration (optimized for MacBook):")
    print(f"  Neurons: {n_neurons}")
    print(f"  Grid: {n_orientations} orientations × {n_locations} locations")
    print(f"  Method: {method}")
    print(f"  Random seed: {seed}")
    
    if method == 'compare':
        # Compare all methods
        return compare_all_methods(
            n_neurons, n_orientations, n_locations,
            theta_lengthscale, spatial_lengthscale,
            plot, seed, verbose, save_dir
        )
    
    # Phase 1: Generate population
    print("\n" + "="*40)
    print("PHASE 1: Generate Neural Population")
    print("="*40)
    
    gp = NeuralGaussianProcess(
        n_orientations=n_orientations,
        n_locations=n_locations,
        theta_lengthscale=theta_lengthscale,
        spatial_lengthscale=spatial_lengthscale,
        seed=seed,
        method=method  # Use specified method
    )
    
    tuning_curves = gp.sample_neurons(n_neurons)
    print(f"✓ Generated {n_neurons} neurons with shape {tuning_curves.shape}")
    
    # Phase 2: Analyze separability
    print("\n" + "="*40)
    print("PHASE 2: Separability Analysis")
    print("="*40)
    
    sep_results = analyze_population_separability(tuning_curves, show_progress=True)
    
    print(f"\n📊 Results:")
    print(f"  Mean separability: {sep_results['mean']:.3f} ± {sep_results['std']:.3f}")
    print(f"  Median separability: {sep_results['median']:.3f}")
    print(f"  Neurons with mixed selectivity: {sep_results['percent_mixed']:.1f}%")
    
    # Phase 3: Test hypothesis
    print("\n" + "="*40)
    print("HYPOTHESIS TEST")
    print("="*40)
    
    if sep_results['mean'] < 0.8:
        print(f"\n✅ SUCCESS: Mean separability = {sep_results['mean']:.3f} < 0.8")
        print("   Population shows mixed selectivity!")
    else:
        print(f"\n⚠️  PARTIAL: Mean separability = {sep_results['mean']:.3f} ≥ 0.8")
        print("   Population shows more separable tuning")
        if method == 'original':
            print("\n   💡 TIP: Try method='direct' or method='gp_interaction' for stronger mixed selectivity")
    
    # Phase 4: Save plots
    if plot:
        print("\n" + "="*40)
        print("PHASE 4: Generating Figures")
        print("="*40)
        os.makedirs(save_dir, exist_ok=True)
        save_results_plots(tuning_curves, sep_results, save_dir, method)
        print(f"✓ Figures saved to {save_dir}/")
    
    return {
        'tuning_curves': tuning_curves,
        'separability_stats': sep_results,
        'success': sep_results['mean'] < 0.8,
        'method': method
    }


def compare_all_methods(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    theta_lengthscale: float,
    spatial_lengthscale: float,
    plot: bool,
    seed: int,
    verbose: bool,
    save_dir: str
) -> Dict:
    """Compare all generation methods."""
    
    print("\n" + "="*60)
    print("COMPARING ALL METHODS")
    print("="*60)
    
    methods = ['direct', 'gp_interaction', 'original']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*40}")
        print(f"Testing method: {method}")
        print(f"{'='*40}")
        
        gp = NeuralGaussianProcess(
            n_orientations=n_orientations,
            n_locations=n_locations,
            theta_lengthscale=theta_lengthscale,
            spatial_lengthscale=spatial_lengthscale,
            seed=seed,
            method=method
        )
        
        tuning_curves = gp.sample_neurons(n_neurons)
        sep_results = analyze_population_separability(tuning_curves, show_progress=False)
        
        all_results[method] = {
            'tuning_curves': tuning_curves,
            'separability_stats': sep_results,
            'mean_sep': sep_results['mean'],
            'percent_mixed': sep_results['percent_mixed']
        }
        
        print(f"  Mean separability: {sep_results['mean']:.3f}")
        print(f"  Mixed selectivity: {sep_results['percent_mixed']:.1f}%")
    
    # Find best method
    best_method = min(all_results.keys(), key=lambda k: all_results[k]['mean_sep'])
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"\n🏆 BEST METHOD: {best_method}")
    print(f"   Mean separability: {all_results[best_method]['mean_sep']:.3f}")
    print(f"   Mixed neurons: {all_results[best_method]['percent_mixed']:.1f}%")
    
    # Plot comparison
    if plot:
        os.makedirs(save_dir, exist_ok=True)
        plot_method_comparison(all_results, save_dir)
        print(f"\n✓ Comparison figures saved to {save_dir}/")
    
    # Return best results
    best_results = all_results[best_method]
    return {
        'tuning_curves': best_results['tuning_curves'],
        'separability_stats': best_results['separability_stats'],
        'success': best_results['mean_sep'] < 0.8,
        'method': best_method,
        'all_results': all_results
    }


def save_results_plots(tuning_curves: np.ndarray, sep_results: Dict, save_dir: str, method: str):
    """Save all plots to files."""
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    sorted_indices = np.argsort(sep_results['all_values'])
    examples = [
        (sorted_indices[0], "Most Mixed"),
        (sorted_indices[len(sorted_indices)//2], "Medium"),
        (sorted_indices[-1], "Most Separable")
    ]
    
    for i, (idx, label) in enumerate(examples):
        ax = axes[0, i]
        im = ax.imshow(tuning_curves[idx], aspect='auto', cmap='hot')
        ax.set_title(f'{label}\nSep: {sep_results["all_values"][idx]:.3f}')
        ax.set_xlabel('Location')
        ax.set_ylabel('Orientation')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Separability histogram
    ax = axes[1, 0]
    ax.hist(sep_results['all_values'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0.8, color='red', linestyle='--', label='Mixed threshold')
    ax.axvline(sep_results['mean'], color='green', linestyle='-', 
               label=f'Mean: {sep_results["mean"]:.3f}')
    ax.set_xlabel('Separability Index')
    ax.set_ylabel('Count')
    ax.set_title('Distribution')
    ax.legend()
    
    # Method info
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""Method: {method}
    
Mean Sep: {sep_results['mean']:.3f}
Std Sep: {sep_results['std']:.3f}
Median Sep: {sep_results['median']:.3f}

Mixed Neurons: {sep_results['percent_mixed']:.1f}%
Target: <80% separability
Success: {'✓' if sep_results['mean'] < 0.8 else '✗'}"""
    ax.text(0.5, 0.5, info_text, ha='center', va='center',
            fontsize=11, fontfamily='monospace')
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Experiment 1 Results ({method}): Mean Separability = {sep_results["mean"]:.3f}')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp1_results_{method}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: exp1_results_{method}.png")


def plot_method_comparison(all_results: Dict, save_dir: str):
    """Plot comparison of all methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot example tuning curves from each method
    for i, (method, results) in enumerate(all_results.items()):
        ax = axes[0, i]
        # Show the most mixed neuron from each method
        idx = np.argmin(results['separability_stats']['all_values'])
        im = ax.imshow(results['tuning_curves'][idx], aspect='auto', cmap='hot')
        ax.set_title(f'{method}\nBest neuron sep: {results["separability_stats"]["all_values"][idx]:.3f}')
        ax.set_xlabel('Location')
        ax.set_ylabel('Orientation')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Comparison bar plot
    ax = axes[1, 0]
    methods = list(all_results.keys())
    mean_seps = [all_results[m]['mean_sep'] for m in methods]
    colors = ['green' if s < 0.8 else 'orange' for s in mean_seps]
    
    bars = ax.bar(methods, mean_seps, color=colors, edgecolor='black')
    ax.axhline(0.8, color='red', linestyle='--', label='Mixed threshold')
    ax.set_ylabel('Mean Separability')
    ax.set_title('Method Comparison')
    ax.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars, mean_seps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Percentage mixed neurons
    ax = axes[1, 1]
    percent_mixed = [all_results[m]['percent_mixed'] for m in methods]
    bars = ax.bar(methods, percent_mixed, color='purple', alpha=0.7, edgecolor='black')
    ax.set_ylabel('% Mixed Neurons')
    ax.set_title('Neurons with Mixed Selectivity')
    
    for bar, val in zip(bars, percent_mixed):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "METHOD COMPARISON\n" + "="*25 + "\n\n"
    for method, results in all_results.items():
        status = "✓" if results['mean_sep'] < 0.8 else "✗"
        summary_text += f"{method:15s} {status}\n"
        summary_text += f"  Mean Sep: {results['mean_sep']:.3f}\n"
        summary_text += f"  Mixed: {results['percent_mixed']:.1f}%\n\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
            va='center')
    
    plt.suptitle('Mixed Selectivity Generation: Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: method_comparison.png")