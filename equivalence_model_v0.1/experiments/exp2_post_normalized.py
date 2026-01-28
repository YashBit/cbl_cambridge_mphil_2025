"""
Experiment 2: Post-Normalized Response Analysis with TRUE POPULATION DN

This experiment implements the CORRECT divisive normalization formula:
    r^{post}_i(θ) = γ * r^{pre}_i(θ) / (σ² + N^{-1} * Σ_j r^{pre}_j(θ))

=============================================================================
KEY INSIGHT (Graph Blurb)
=============================================================================

Divisive normalization enforces a constant activity budget (γN) across the 
population, but redistributes resources as set size increases—neurons with 
strong tuning at all locations gain disproportionately ("winners"), while 
those with weak tuning anywhere are suppressed ("losers"). The mean stays 
fixed at γ, but variance grows ~5× from l=2 to l=8.

=============================================================================
MEMORY-EFFICIENT IMPLEMENTATION
=============================================================================

1. ACTIVITY CAP THEOREM (analytical):
   Σᵢ r^post_i(θ) = γ × N  (EXACT when σ²→0, for ALL stimuli!)
   
2. PRE-DN FACTORIZATION (analytical):
   Mean[∏ₖ g(θₖ)] = ∏ₖ Mean[g(θₖ)]  (O(l×n_θ) not O(n_θ^l))

Author: Mixed Selectivity Project
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import time

from core.gaussian_process import generate_neuron_population
from core.divisive_normalization import estimate_memory_usage


# ============================================================================
# EXPERIMENT DESIGN
# ============================================================================

def sample_fixed_orientations(
    n_locations: int,
    n_orientations: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample one fixed orientation for each location.
    
    These remain fixed throughout the experiment - we only vary which
    subset of locations is active.
    """
    return rng.integers(0, n_orientations, size=n_locations)


def compute_single_neuron_responses_fixed_theta(
    G_stacked: np.ndarray,
    subset: Tuple[int, ...],
    fixed_thetas: np.ndarray,
    gamma: float,
    sigma_sq: float
) -> np.ndarray:
    """
    Compute post-DN response for each neuron at a fixed stimulus configuration.
    
    Parameters
    ----------
    G_stacked : np.ndarray
        Pre-computed exp(f) for all neurons, shape (N, n_locations, n_orientations)
    subset : Tuple[int, ...]
        Active location indices
    fixed_thetas : np.ndarray
        Fixed orientation index for each location, shape (n_locations,)
    gamma : float
        Gain constant
    sigma_sq : float
        Semi-saturation constant
        
    Returns
    -------
    R_post : np.ndarray
        Shape (N,) - post-DN response for each neuron
    """
    N = G_stacked.shape[0]
    subset_arr = np.array(subset)
    l = len(subset)
    
    # Compute pre-DN response for each neuron: R_pre[n] = ∏_k G[n, loc_k, θ_k]
    R_pre = np.ones(N)
    for k in range(l):
        loc = subset_arr[k]
        theta_idx = fixed_thetas[loc]
        R_pre *= G_stacked[:, loc, theta_idx]
    
    # Population mean for denominator
    pop_mean = np.mean(R_pre)
    
    # Apply divisive normalization
    R_post = gamma * R_pre / (sigma_sq + pop_mean)
    
    return R_post


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_experiment_2(config: Dict) -> Dict:
    """
    Run Experiment 2: Population-level DN analysis.
    
    For each set size:
    1. Generate population of N neurons
    2. Enumerate all location subsets
    3. Compute pre-DN and post-DN responses for population
    4. Verify total activity cap: Σ_i r^{post}_i ≈ γ * N
    5. Compute single-neuron average responses
    
    Parameters
    ----------
    config : Dict
        Required keys:
        - n_neurons: int
        - n_orientations: int
        - n_locations: int
        - set_sizes: list[int]
        - seed: int
        - gamma: float
        - sigma_sq: float
        - lambda_base: float
        - sigma_lambda: float
        
    Returns
    -------
    results : Dict
        Complete experimental results
    """
    print("="*80)
    print("EXPERIMENT 2: POST-NORMALIZED RESPONSE (TRUE POPULATION DN)")
    print("="*80)
    print(f"Configuration:")
    print(f"  N neurons: {config['n_neurons']}")
    print(f"  n_θ: {config['n_orientations']}")
    print(f"  L: {config['n_locations']}")
    print(f"  γ: {config['gamma']} Hz")
    print(f"  σ²: {config['sigma_sq']}")
    print(f"  Set sizes: {config['set_sizes']}")
    print(f"  Theoretical total activity: {config['gamma'] * config['n_neurons']} Hz")
    print()
    
    # Memory comparison
    print("Memory Comparison (Original vs Efficient):")
    for l in config['set_sizes']:
        if l > 1:
            orig = estimate_memory_usage(config['n_neurons'], config['n_orientations'], l, 'original')
            eff = estimate_memory_usage(config['n_neurons'], config['n_orientations'], l, 'efficient')
            status = "✗ CRASH" if not orig['safe'] else "⚠️ SLOW" if orig['memory_gb'] > 1 else "✓"
            print(f"  l={l}: Original={orig['memory_gb']:>8.1f} GB {status:<10} Efficient={eff['memory_mb']:>6.0f} MB ✓")
    print()
    
    # Generate population
    print(f"Generating population of {config['n_neurons']} neurons...")
    start_time = time.time()
    population = generate_neuron_population(
        n_neurons=config['n_neurons'],
        n_orientations=config['n_orientations'],
        n_locations=config['n_locations'],
        base_lengthscale=config['lambda_base'],
        lengthscale_variability=config['sigma_lambda'],
        seed=config['seed']
    )
    gen_time = time.time() - start_time
    print(f"✓ Population generated ({gen_time:.1f}s)")
    
    # Extract f_samples for all neurons
    f_samples_population = [neuron['f_samples'] for neuron in population]
    
    # Pre-compute G = exp(f) for vectorized computation
    print("Pre-computing G = exp(f)...")
    G_stacked = np.stack([np.exp(f) for f in f_samples_population], axis=0)
    print(f"✓ G matrix shape: {G_stacked.shape}")
    
    # Setup RNG
    rng = np.random.default_rng(config['seed'] + 1000)
    
    # Sample fixed orientations for each location
    fixed_thetas = sample_fixed_orientations(
        n_locations=config['n_locations'],
        n_orientations=config['n_orientations'],
        rng=rng
    )
    print(f"Fixed orientations per location: {fixed_thetas}")
    
    # Storage for results
    results = {
        'config': config,
        'G_stacked': G_stacked,
        'fixed_thetas': fixed_thetas,
        'set_size_results': {},
        'single_neuron_responses': {}
    }
    
    # Run for each set size
    for set_size in config['set_sizes']:
        print(f"\n{'='*80}")
        print(f"SET SIZE l = {set_size}")
        print(f"{'='*80}")
        
        # Generate all subsets
        all_subsets = list(combinations(range(config['n_locations']), set_size))
        n_subsets = len(all_subsets)
        print(f"Number of subsets: {n_subsets}")
        
        # Storage
        pre_totals = []
        post_totals = []
        N = config['n_neurons']
        neuron_response_accumulator = np.zeros(N)
        
        # Process each subset
        start_time = time.time()
        for subset in tqdm(all_subsets, desc=f"Processing l={set_size}", unit="subset"):
            # Compute post-DN responses
            R_post = compute_single_neuron_responses_fixed_theta(
                G_stacked=G_stacked,
                subset=subset,
                fixed_thetas=fixed_thetas,
                gamma=config['gamma'],
                sigma_sq=config['sigma_sq']
            )
            
            # Compute pre-DN for comparison
            subset_arr = np.array(subset)
            R_pre = np.ones(N)
            for k in range(len(subset)):
                loc = subset_arr[k]
                theta_idx = fixed_thetas[loc]
                R_pre *= G_stacked[:, loc, theta_idx]
            
            # Accumulate
            neuron_response_accumulator += R_post
            pre_totals.append(np.sum(R_pre))
            post_totals.append(np.sum(R_post))
        
        elapsed = time.time() - start_time
        
        # Average across subsets
        neuron_avg_responses = neuron_response_accumulator / n_subsets
        results['single_neuron_responses'][set_size] = neuron_avg_responses
        
        # Store statistics
        theoretical_total = config['gamma'] * config['n_neurons']
        results['set_size_results'][set_size] = {
            'n_subsets': n_subsets,
            'pre_mean': np.mean(pre_totals),
            'pre_std': np.std(pre_totals),
            'pre_min': np.min(pre_totals),
            'pre_max': np.max(pre_totals),
            'post_mean': np.mean(post_totals),
            'post_std': np.std(post_totals),
            'theoretical_total': theoretical_total,
            'elapsed_seconds': elapsed,
            'neuron_avg_mean': np.mean(neuron_avg_responses),
            'neuron_avg_std': np.std(neuron_avg_responses),
            'neuron_avg_min': np.min(neuron_avg_responses),
            'neuron_avg_max': np.max(neuron_avg_responses),
            'neuron_avg_q05': np.percentile(neuron_avg_responses, 5),
            'neuron_avg_q25': np.percentile(neuron_avg_responses, 25),
            'neuron_avg_q75': np.percentile(neuron_avg_responses, 75),
            'neuron_avg_q95': np.percentile(neuron_avg_responses, 95),
        }
        
        # Print summary
        r = results['set_size_results'][set_size]
        print(f"\nResults for l = {set_size}:")
        print(f"  Pre-DN total:  {r['pre_mean']:.2f} ± {r['pre_std']:.2f}")
        print(f"  Post-DN total: {r['post_mean']:.2f} ± {r['post_std']:.6f}")
        print(f"  Theoretical:   {r['theoretical_total']:.2f}")
        print(f"  Neuron mean:   {r['neuron_avg_mean']:.2f} Hz, std: {r['neuron_avg_std']:.2f} Hz")
        print(f"  Time: {elapsed:.1f}s")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Create visualization plots for Experiment 2."""
    import matplotlib.cm as cm
    
    config = results['config']
    set_sizes = config['set_sizes']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    N = config['n_neurons']
    gamma = config['gamma']
    theoretical = gamma * N
    
    # Extract data
    pre_means = [results['set_size_results'][l]['pre_mean'] for l in set_sizes]
    pre_stds = [results['set_size_results'][l]['pre_std'] for l in set_sizes]
    post_means = [results['set_size_results'][l]['post_mean'] for l in set_sizes]
    
    # =========================================================================
    # Plot 1: Pre-DN vs Post-DN comparison with embedded blurbs
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.errorbar(set_sizes, pre_means, yerr=pre_stds, fmt='o-', 
                 color='#E74C3C', linewidth=3, markersize=12, capsize=6, capthick=2,
                 markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Set Size (l)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Population Activity (Hz)', fontsize=13, fontweight='bold')
    ax1.set_title('Pre-DN: GROWS with Set Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(set_sizes)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Embedded blurb for pre-DN
    blurb_pre = "Without normalization:\nactivity explodes as\nmore items are added"
    ax1.text(0.05, 0.95, blurb_pre, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='mistyrose', 
                       alpha=0.95, edgecolor='red', linewidth=1.5))
    
    ax2 = axes[1]
    ax2.plot(set_sizes, post_means, 'o-', color='#2ECC71', linewidth=3, markersize=12,
             markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2ECC71')
    ax2.axhline(theoretical, color='#E74C3C', linestyle='--', linewidth=2.5, 
                label=f'Theoretical: γN = {theoretical:,.0f} Hz')
    ax2.set_xlabel('Set Size (l)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total Population Activity (Hz)', fontsize=13, fontweight='bold')
    ax2.set_title('Post-DN: CONSTANT (Activity Cap)', fontsize=14, fontweight='bold')
    ax2.set_xticks(set_sizes)
    ax2.set_ylim([theoretical * 0.99, theoretical * 1.01])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Embedded blurb for post-DN (Activity Cap Theorem)
    blurb_post = (
        "ACTIVITY CAP THEOREM\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Σᵢ rᵢᵖᵒˢᵗ = γN (constant!)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "DN enforces fixed budget\n"
        "regardless of set size"
    )
    ax2.text(0.02, 0.98, blurb_post, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', 
                       alpha=0.95, edgecolor='green', linewidth=2),
             family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path / f'exp2_pre_vs_post_N{N}.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp2_pre_vs_post_N{N}.png")
    
    # =========================================================================
    # Plot 2: Single-neuron band visualization
    # =========================================================================
    _plot_single_neuron_responses(results, output_path, show_plot)


def _plot_single_neuron_responses(results: Dict, output_path: Path, show_plot: bool = False):
    """Plot single-neuron average responses across set sizes with colors and markers."""
    import matplotlib.cm as cm
    
    config = results['config']
    set_sizes = config['set_sizes']
    N = config['n_neurons']
    gamma = config['gamma']
    
    neuron_responses = results['single_neuron_responses']
    
    # Build matrix: (N, n_set_sizes)
    response_matrix = np.zeros((N, len(set_sizes)))
    for j, l in enumerate(set_sizes):
        response_matrix[:, j] = neuron_responses[l]
    
    # Statistics
    pop_mean = np.mean(response_matrix, axis=0)
    pop_std = np.std(response_matrix, axis=0)
    
    # Sort neurons by response at final set size for color mapping
    sort_idx = np.argsort(response_matrix[:, -1])
    response_sorted = response_matrix[sort_idx, :]
    
    # Create color gradient: blue (losers) -> red (winners)
    cmap = cm.coolwarm
    colors = [cmap(i / (N - 1)) for i in range(N)]
    
    # -------------------------------------------------------------------------
    # Main plot: Colored neurons with markers
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Plot each neuron with color and MARKERS for y-axis coordination
    for i in range(N):
        ax.plot(set_sizes, response_sorted[i, :], 
                color=colors[i], alpha=0.7, linewidth=1.5,
                marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.3,
                zorder=i+1)
    
    # Population mean (black, prominent)
    ax.plot(set_sizes, pop_mean, 'o-', color='black', linewidth=4, markersize=12,
            markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='black',
            zorder=N+100, label=f'Population mean = γ = {gamma:.0f} Hz')
    
    # Reference line
    ax.axhline(gamma, color='gray', linestyle=':', linewidth=2, alpha=0.5, zorder=0)
    
    ax.set_yscale('log')
    ax.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Response (Hz) — Log Scale', fontsize=14, fontweight='bold')
    ax.set_title(f'Single-Neuron Response vs Set Size (N = {N} neurons)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(set_sizes)
    ax.set_xlim([set_sizes[0] - 0.5, set_sizes[-1] + 0.5])
    ax.set_ylim([0.05, 5000])
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', which='both')
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.legend(loc='upper right', fontsize=11)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=25, pad=0.02)
    cbar.set_label('Neuron rank at l=8\n(blue=suppressed, red=enhanced)', fontsize=10)
    cbar.set_ticks([0, N//2, N-1])
    cbar.set_ticklabels(['Losers', 'Middle', 'Winners'])
    
    # EMBEDDED BLURB (key insight box)
    blurb = (
        "Winner-take-all under DN:\n"
        f"• Total budget fixed at γN = {gamma*N:,.0f} Hz\n"
        f"• Mean stays constant at γ = {gamma:.0f} Hz\n"
        f"• But variance grows {pop_std[-1]/pop_std[0]:.1f}× (l={set_sizes[0]}→l={set_sizes[-1]})\n"
        "• Winners (red): high tuning everywhere\n"
        "• Losers (blue): one bad location ruins all"
    )
    ax.text(0.02, 0.98, blurb, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                      edgecolor='orange', alpha=0.95, linewidth=2),
            fontfamily='sans-serif', linespacing=1.4)
    
    plt.tight_layout()
    plt.savefig(output_path / f'exp2_single_neuron_band_N{N}.png', dpi=150, bbox_inches='tight', facecolor='white')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp2_single_neuron_band_N{N}.png")
    
    # -------------------------------------------------------------------------
    # Summary statistics plot with colors and markers
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Individual neurons with colors and markers
    ax1 = axes[0]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i in range(N):
        ax1.plot(set_sizes, response_sorted[i, :], 
                 color=colors[i], alpha=0.7, linewidth=1.2,
                 marker='s', markersize=4, markeredgecolor='white', markeredgewidth=0.2)
    
    # Population mean
    ax1.plot(set_sizes, pop_mean, 'o-', color='black', linewidth=3.5, markersize=11,
             markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='black',
             label='Population mean', zorder=100)
    
    ax1.axhline(gamma, color='#2ECC71', linestyle='--', linewidth=2.5, alpha=0.8,
                label=f'γ = {gamma:.0f} Hz')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Set Size (l)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Response (Hz) — Log Scale', fontsize=13, fontweight='bold')
    ax1.set_title('Individual Neurons (colored by rank)', fontsize=13, fontweight='bold')
    ax1.set_xticks(set_sizes)
    ax1.set_xlim([set_sizes[0] - 0.5, set_sizes[-1] + 0.5])
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Embedded blurb for left panel
    blurb_left = (
        "Each line = one neuron\n"
        "Markers show exact values\n"
        "Blue→Red = Low→High response"
    )
    ax1.text(0.02, 0.02, blurb_left, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Right: Heterogeneity bar chart
    ax2 = axes[1]
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    bars = ax2.bar(set_sizes, pop_std, color='#9B59B6', alpha=0.8, edgecolor='black', 
                   linewidth=1.5, width=1.2)
    
    ax2.set_xlabel('Set Size (l)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Std Dev Across Neurons (Hz)', fontsize=13, fontweight='bold')
    ax2.set_title('Response Heterogeneity INCREASES', fontsize=13, fontweight='bold')
    ax2.set_xticks(set_sizes)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (l, s) in enumerate(zip(set_sizes, pop_std)):
        ax2.text(l, s + pop_std.max() * 0.02, f'{s:.1f}', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')
    
    # Embedded blurb for right panel
    blurb_right = (
        "Fixed budget → unequal shares\n"
        f"Std grows {pop_std[-1]/pop_std[0]:.1f}× from l={set_sizes[0]} to l={set_sizes[-1]}"
    )
    ax2.text(0.98, 0.98, blurb_right, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                       alpha=0.95, edgecolor='orange', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path / f'exp2_response_summary_N{N}.png', dpi=150, bbox_inches='tight', facecolor='white')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp2_response_summary_N{N}.png")