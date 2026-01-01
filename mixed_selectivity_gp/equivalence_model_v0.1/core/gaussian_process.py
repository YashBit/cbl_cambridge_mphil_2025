"""
Multi-Neuron Gaussian Process Framework for Mixed Selectivity

This module extends the single neuron GP framework to sample multiple neurons,
computing R.mean (instead of R.sum) for each neuron across all combinations of
a given set size l, then summing across all neurons for population-level analysis.

KEY CHANGES FROM SINGLE NEURON VERSION:
1. n_neurons is now a variable (sample as many neurons as desired)
2. For each neuron and each set size l: compute R.mean across all subsets
3. For each l: sum R.mean values across all neurons to get population total
4. Each neuron has its own random lengthscales (maintained separately)
5. Progress monitoring via TQDM for neuron-by-neuron tracking
6. Timing information: per-neuron time and total experiment time

OUTPUT STRUCTURE:
================
{
    'n_neurons': int,
    'neuron_data': [
        {
            'neuron_idx': int,
            'lengthscale_vector': (8,) array,
            'f_samples': (8, n_theta) array,
            'R_mean_by_l': {2: float, 4: float, 6: float, 8: float},
            'time_seconds': float
        },
        ...
    ],
    'population_summary': {
        2: {'total_R_mean': float, 'all_R_means': list},
        4: {'total_R_mean': float, 'all_R_means': list},
        6: {'total_R_mean': float, 'all_R_means': list},
        8: {'total_R_mean': float, 'all_R_means': list}
    },
    'timing': {
        'total_seconds': float,
        'mean_per_neuron': float,
        'std_per_neuron': float
    },
    'config': {...}
}

Author: Based on original single neuron GP framework
Date: December 2025
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm
import time
from typing import Dict, List, Optional


def sample_single_neuron_silent(
    n_orientations: int,
    total_locations: int,
    subset_sizes: List[int],
    theta_lengthscale: float,
    lengthscale_variability: float,
    random_state: np.random.RandomState
) -> Dict:
    """
    Generate a single neuron with mixed selectivity (silent version for batch processing).
    
    This is an optimized version that doesn't print output, designed for batch generation.
    
    Args:
        n_orientations: Number of orientation bins
        total_locations: Number of spatial locations (typically 8)
        subset_sizes: List of set sizes to enumerate [2, 4, 6, 8]
        theta_lengthscale: Base lengthscale for GP kernel
        lengthscale_variability: σ_λ for lengthscale heterogeneity
        random_state: numpy RandomState for reproducibility
    
    Returns:
        Dictionary with lengthscales, GP samples, and R.mean for each l
    """
    n_theta = n_orientations
    orientations = np.linspace(-np.pi, np.pi, n_theta)
    
    # STAGE 1: Generate location-dependent lengthscales
    random_factors = 1.0 + lengthscale_variability * random_state.randn(total_locations)
    random_factors = np.abs(random_factors)
    lengthscale_vector = theta_lengthscale * random_factors
    
    # STAGE 2: Sample 8 GP functions with location-specific lengthscales
    f_samples = np.zeros((total_locations, n_theta))
    
    for loc in range(total_locations):
        lengthscale = lengthscale_vector[loc]
        
        # Build covariance matrix with THIS location's lengthscale
        K = np.zeros((n_theta, n_theta))
        for i in range(n_theta):
            for j in range(n_theta):
                dist = np.abs(orientations[i] - orientations[j])
                dist = np.minimum(dist, 2*np.pi - dist)  # Circular distance
                K[i, j] = np.exp(-dist**2 / (2 * lengthscale**2))
        
        K += 1e-6 * np.eye(n_theta)  # Numerical stability
        L = np.linalg.cholesky(K)
        
        # Sample from GP
        z = random_state.randn(n_theta)
        f_loc = L @ z
        
        # Apply random gain modulation
        gain = 1.0 + 0.2 * random_state.randn()
        f_loc = f_loc * np.abs(gain)
        
        f_samples[loc, :] = f_loc
    
    # STAGE 3: Build response surfaces and compute R.mean for each l
    R_mean_by_l = {}
    
    for l in subset_sizes:
        subsets = list(combinations(range(total_locations), l))
        
        # Accumulate R values to compute mean across all subsets
        all_subset_means = []
        
        for subset in subsets:
            # Get f functions for this subset
            f_subset = [f_samples[loc, :] for loc in subset]
            
            # Build l-dimensional log-rate tensor via broadcasting
            log_rate_sum = np.zeros([n_theta] * l)
            
            for dim_idx, f_loc in enumerate(f_subset):
                shape = [1] * l
                shape[dim_idx] = n_theta
                f_reshaped = f_loc.reshape(shape)
                log_rate_sum = log_rate_sum + f_reshaped
            
            # Compute response: R = exp(G)
            R = np.exp(log_rate_sum)
            
            # Compute mean for this subset
            all_subset_means.append(R.mean())
        
        # R.mean for this l = mean of all subset means
        R_mean_by_l[l] = np.mean(all_subset_means)
    
    return {
        'lengthscale_vector': lengthscale_vector,
        'f_samples': f_samples,
        'R_mean_by_l': R_mean_by_l
    }


def sample_multi_neuron_population(
    n_neurons: int = 100,
    n_orientations: int = 10,
    total_locations: int = 8,
    subset_sizes: List[int] = [2, 4, 6, 8],
    theta_lengthscale: float = 0.3,
    lengthscale_variability: float = 0.5,
    seed: int = 22,
    verbose: bool = True
) -> Dict:
    """
    Generate a population of neurons with mixed selectivity using GP framework.
    
    For each neuron:
    - Generate unique random lengthscales (source of mixed selectivity)
    - Sample 8 independent GP functions
    - For each set size l, compute R.mean across all C(8,l) subsets
    
    For the population:
    - Sum R.mean values across all neurons for each l
    
    Args:
        n_neurons: Number of neurons to generate
        n_orientations: Number of orientation bins (n_theta)
        total_locations: Number of spatial locations (typically 8)
        subset_sizes: List of set sizes to enumerate [2, 4, 6, 8]
        theta_lengthscale: Base lengthscale for GP kernel (λ_base)
        lengthscale_variability: σ_λ for lengthscale heterogeneity
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with neuron data, population summary, and timing info
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🧠 MULTI-NEURON GP FRAMEWORK")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"   Number of neurons: {n_neurons}")
        print(f"   Number of orientations (n_θ): {n_orientations}")
        print(f"   Total locations: {total_locations}")
        print(f"   Subset sizes: {subset_sizes}")
        print(f"   Base lengthscale (λ_base): {theta_lengthscale}")
        print(f"   Lengthscale variability (σ_λ): {lengthscale_variability}")
        print(f"   Random seed: {seed}")
        
        # Show combinatorics
        print(f"\n📐 Combinatorics per neuron:")
        for l in subset_sizes:
            n_subsets = len(list(combinations(range(total_locations), l)))
            print(f"   l={l}: C({total_locations},{l}) = {n_subsets} subsets")
    
    # Initialize random state
    master_rng = np.random.RandomState(seed)
    
    # Storage for results
    neuron_data = []
    neuron_times = []
    
    # Population-level accumulators
    population_R_means = {l: [] for l in subset_sizes}
    
    # Timer for total experiment
    total_start_time = time.time()
    
    if verbose:
        print(f"\n{'='*80}")
        print("🔄 GENERATING NEURONS")
        print("="*80)
    
    # Generate each neuron with progress bar
    neuron_iterator = tqdm(range(n_neurons), 
                          desc="Generating neurons", 
                          unit="neuron",
                          ncols=100)
    
    for neuron_idx in neuron_iterator:
        neuron_start_time = time.time()
        
        # Generate unique seed for this neuron (ensures different lengthscales)
        neuron_seed = master_rng.randint(0, 2**31)
        neuron_rng = np.random.RandomState(neuron_seed)
        
        # Sample this neuron
        neuron_result = sample_single_neuron_silent(
            n_orientations=n_orientations,
            total_locations=total_locations,
            subset_sizes=subset_sizes,
            theta_lengthscale=theta_lengthscale,
            lengthscale_variability=lengthscale_variability,
            random_state=neuron_rng
        )
        
        neuron_end_time = time.time()
        neuron_time = neuron_end_time - neuron_start_time
        neuron_times.append(neuron_time)
        
        # Store neuron data
        neuron_data.append({
            'neuron_idx': neuron_idx,
            'lengthscale_vector': neuron_result['lengthscale_vector'],
            'f_samples': neuron_result['f_samples'],
            'R_mean_by_l': neuron_result['R_mean_by_l'],
            'time_seconds': neuron_time
        })
        
        # Add to population accumulators
        for l in subset_sizes:
            population_R_means[l].append(neuron_result['R_mean_by_l'][l])
        
        # Update progress bar description with timing
        neuron_iterator.set_postfix({
            'time': f'{neuron_time:.3f}s',
            'avg': f'{np.mean(neuron_times):.3f}s'
        })
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Compute population summaries
    population_summary = {}
    for l in subset_sizes:
        all_R_means = population_R_means[l]
        population_summary[l] = {
            'total_R_mean': np.sum(all_R_means),  # Sum across all neurons
            'mean_R_mean': np.mean(all_R_means),  # Average R.mean per neuron
            'std_R_mean': np.std(all_R_means),    # Variation across neurons
            'all_R_means': all_R_means            # Individual neuron values
        }
    
    # Timing statistics
    timing = {
        'total_seconds': total_time,
        'mean_per_neuron': np.mean(neuron_times),
        'std_per_neuron': np.std(neuron_times),
        'min_per_neuron': np.min(neuron_times),
        'max_per_neuron': np.max(neuron_times),
        'all_neuron_times': neuron_times
    }
    
    # Configuration for reproducibility
    config = {
        'n_neurons': n_neurons,
        'n_orientations': n_orientations,
        'total_locations': total_locations,
        'subset_sizes': subset_sizes,
        'theta_lengthscale': theta_lengthscale,
        'lengthscale_variability': lengthscale_variability,
        'seed': seed
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("✅ GENERATION COMPLETE")
        print("="*80)
        
        print(f"\n⏱️  TIMING SUMMARY:")
        print(f"   Total experiment time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"   Mean time per neuron: {timing['mean_per_neuron']:.3f} ± {timing['std_per_neuron']:.3f} seconds")
        print(f"   Range: [{timing['min_per_neuron']:.3f}, {timing['max_per_neuron']:.3f}] seconds")
        
        print(f"\n📊 POPULATION SUMMARY (R.mean summed across {n_neurons} neurons):")
        print(f"   {'Set Size (l)':<15} {'Total R.mean':<20} {'Mean per neuron':<20} {'Std':<15}")
        print(f"   {'-'*70}")
        for l in subset_sizes:
            ps = population_summary[l]
            print(f"   l = {l:<10} {ps['total_R_mean']:<20.4e} {ps['mean_R_mean']:<20.4e} {ps['std_R_mean']:<15.4e}")
    
    return {
        'n_neurons': n_neurons,
        'neuron_data': neuron_data,
        'population_summary': population_summary,
        'timing': timing,
        'config': config
    }


def plot_population_results(
    results: Dict,
    save_dir: str = 'figures/multi_neuron',
    show_plot: bool = True
) -> None:
    """
    Create publication-quality plots for multi-neuron population results.
    
    Creates two plots:
    1. Set size (l) vs Total R.mean (summed across all neurons) - LOG SCALE
    2. Distribution of R.mean across neurons for each l
    
    Args:
        results: Output dictionary from sample_multi_neuron_population
        save_dir: Directory to save figures
        show_plot: Whether to display plots interactively
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_neurons = results['n_neurons']
    population_summary = results['population_summary']
    timing = results['timing']
    
    # Extract data for plotting
    set_sizes = sorted(population_summary.keys())
    total_R_means = [population_summary[l]['total_R_mean'] for l in set_sizes]
    
    print("\n" + "="*70)
    print("📈 CREATING PLOTS")
    print("="*70)
    
    # ========================================
    # PLOT 1: Set Size vs Total R.mean (Log Scale)
    # ========================================
    sns.set_style("whitegrid")
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # Use log scale for y-axis
    ax1.set_yscale('log')
    
    # Main line plot
    ax1.plot(set_sizes, total_R_means, 'o-', 
             linewidth=2.5, markersize=12, 
             color='#2E86AB', label='Total R.mean (summed across neurons)')
    
    # Add scatter for emphasis
    ax1.scatter(set_sizes, total_R_means, s=200, c='#A23B72', 
                alpha=0.7, edgecolors='white', linewidths=2, zorder=5)
    
    # Add value annotations
    for l, val in zip(set_sizes, total_R_means):
        # Position label above the point
        ax1.annotate(f'{val:.2e}', 
                    xy=(l, val), 
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', 
                             facecolor='white', 
                             edgecolor='gray', 
                             alpha=0.9))
    
    # Labels and title
    ax1.set_xlabel('Set Size (l) - Number of Active Locations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total R.mean (Summed Across Neurons)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Population Neural Activity vs Set Size\n'
                 f'({n_neurons} neurons, log scale)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-ticks
    ax1.set_xticks(set_sizes)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Add grid for log scale
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add timing info as text box
    textstr = f'Total time: {timing["total_seconds"]:.1f}s\nMean/neuron: {timing["mean_per_neuron"]:.3f}s'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    filepath1 = Path(save_dir) / f'set_size_vs_total_R_mean_{n_neurons}neurons.png'
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath1}")
    
    filepath1_pdf = Path(save_dir) / f'set_size_vs_total_R_mean_{n_neurons}neurons.pdf'
    plt.savefig(filepath1_pdf, bbox_inches='tight')
    print(f"✓ Saved: {filepath1_pdf}")
    
    # ========================================
    # PLOT 2: Distribution of R.mean across neurons
    # ========================================
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, l in enumerate(set_sizes):
        ax = axes[idx]
        all_R_means = population_summary[l]['all_R_means']
        
        # Histogram with KDE
        sns.histplot(all_R_means, kde=True, ax=ax, color='#2E86AB', 
                    edgecolor='white', linewidth=1.5, alpha=0.7)
        
        # Add vertical line for mean
        mean_val = np.mean(all_R_means)
        ax.axvline(mean_val, color='#A23B72', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2e}')
        
        ax.set_xlabel('R.mean per neuron', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'l = {l} (C(8,{l}) = {len(list(combinations(range(8), l)))} subsets)', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.tick_params(axis='both', labelsize=10)
    
    fig2.suptitle(f'Distribution of R.mean Across {n_neurons} Neurons\nby Set Size', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    filepath2 = Path(save_dir) / f'R_mean_distributions_{n_neurons}neurons.png'
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath2}")
    
    filepath2_pdf = Path(save_dir) / f'R_mean_distributions_{n_neurons}neurons.pdf'
    plt.savefig(filepath2_pdf, bbox_inches='tight')
    print(f"✓ Saved: {filepath2_pdf}")
    
    # ========================================
    # PLOT 3: Linear Scale Version (for comparison)
    # ========================================
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    
    # Main line plot (linear scale)
    ax3.plot(set_sizes, total_R_means, 'o-', 
             linewidth=2.5, markersize=12, 
             color='#2E86AB', label='Total R.mean')
    
    ax3.scatter(set_sizes, total_R_means, s=200, c='#A23B72', 
                alpha=0.7, edgecolors='white', linewidths=2, zorder=5)
    
    # Add value annotations
    for l, val in zip(set_sizes, total_R_means):
        ax3.annotate(f'{val:.2e}', 
                    xy=(l, val), 
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', 
                             facecolor='white', 
                             edgecolor='gray', 
                             alpha=0.9))
    
    ax3.set_xlabel('Set Size (l) - Number of Active Locations', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Total R.mean (Summed Across Neurons)', fontsize=14, fontweight='bold')
    ax3.set_title(f'Population Neural Activity vs Set Size\n'
                 f'({n_neurons} neurons, linear scale)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax3.set_xticks(set_sizes)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis with scientific notation
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    
    filepath3 = Path(save_dir) / f'set_size_vs_total_R_mean_{n_neurons}neurons_linear.png'
    plt.savefig(filepath3, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath3}")
    
    if show_plot:
        plt.show()
    
    print(f"\n✅ All plots saved to: {save_dir}/")


def print_detailed_summary(results: Dict) -> None:
    """
    Print a detailed summary of the multi-neuron experiment results.
    
    Args:
        results: Output dictionary from sample_multi_neuron_population
    """
    n_neurons = results['n_neurons']
    population_summary = results['population_summary']
    timing = results['timing']
    config = results['config']
    
    print("\n" + "="*80)
    print("📋 DETAILED EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n🔧 CONFIGURATION:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n⏱️  TIMING:")
    print(f"   Total experiment time: {timing['total_seconds']:.2f} seconds ({timing['total_seconds']/60:.2f} minutes)")
    print(f"   Mean time per neuron: {timing['mean_per_neuron']:.4f} seconds")
    print(f"   Std time per neuron: {timing['std_per_neuron']:.4f} seconds")
    print(f"   Min time per neuron: {timing['min_per_neuron']:.4f} seconds")
    print(f"   Max time per neuron: {timing['max_per_neuron']:.4f} seconds")
    
    print(f"\n📊 RESULTS BY SET SIZE:")
    print(f"\n   {'l':<5} {'Total R.mean':<18} {'Mean/neuron':<18} {'Std/neuron':<18} {'# Subsets':<12}")
    print(f"   {'-'*75}")
    
    set_sizes = sorted(population_summary.keys())
    for l in set_sizes:
        ps = population_summary[l]
        n_subsets = len(list(combinations(range(8), l)))
        print(f"   {l:<5} {ps['total_R_mean']:<18.4e} {ps['mean_R_mean']:<18.4e} {ps['std_R_mean']:<18.4e} {n_subsets:<12}")
    
    print(f"\n📈 SCALING ANALYSIS:")
    if len(set_sizes) > 1:
        for i in range(len(set_sizes) - 1):
            l1, l2 = set_sizes[i], set_sizes[i+1]
            r1 = population_summary[l1]['total_R_mean']
            r2 = population_summary[l2]['total_R_mean']
            fold_change = r2 / r1 if r1 > 0 else float('inf')
            print(f"   l={l1} → l={l2}: {fold_change:.2f}× increase")
    
    # Overall fold change
    r_first = population_summary[set_sizes[0]]['total_R_mean']
    r_last = population_summary[set_sizes[-1]]['total_R_mean']
    total_fold = r_last / r_first if r_first > 0 else float('inf')
    print(f"\n   Overall (l={set_sizes[0]} → l={set_sizes[-1]}): {total_fold:.2f}× increase")
    
    print(f"\n🔬 LENGTHSCALE HETEROGENEITY (first 5 neurons):")
    for i, neuron in enumerate(results['neuron_data'][:5]):
        ls_vec = neuron['lengthscale_vector']
        print(f"   Neuron {i}: [{', '.join([f'{v:.3f}' for v in ls_vec])}]")
        print(f"            Range: [{ls_vec.min():.3f}, {ls_vec.max():.3f}], Ratio: {ls_vec.max()/ls_vec.min():.2f}×")


if __name__ == "__main__":
    # Example usage: Generate 100 neurons
    print("\n" + "🚀 "*20)
    print("RUNNING MULTI-NEURON GP EXPERIMENT")
    print("🚀 "*20)
    
    # Run the experiment
    results = sample_multi_neuron_population(
        n_neurons=100,
        n_orientations=10,
        total_locations=8,
        subset_sizes=[2, 4, 6, 8],
        theta_lengthscale=0.3,
        lengthscale_variability=0.5,
        seed=22,
        verbose=True
    )
    
    # Print detailed summary
    print_detailed_summary(results)
    
    # Create plots
    plot_population_results(results, save_dir='figures/multi_neuron', show_plot=True)
    
    print("\n" + "✅ "*20)
    print("EXPERIMENT COMPLETE")
    print("✅ "*20)