"""
Experiment 2: Population Divisive Normalization Analysis

=============================================================================
THEORETICAL FRAMEWORK
=============================================================================

This experiment validates the core predictions of population-level divisive 
normalization for visual working memory encoding:

    r^{post}_i(θ) = γ · r^{pre}_i(θ) / [σ² + N⁻¹ Σⱼ r^{pre}_j(θ)]

Three key empirical predictions:

1. PER-ITEM ACTIVITY DECREASES: As set size l increases, the resource 
   allocated per item decreases as 1/l (inverse relationship).

2. ACTIVITY CAP: Total population activity is bounded at γN regardless 
   of set size—this is the "metabolic budget" constraint.

3. RESPONSE HETEROGENEITY: Individual neurons diverge from the mean as 
   set size increases—some neurons become "winners" (high response at 
   all active locations) while others become "losers" (suppressed).

=============================================================================
EXPERIMENTAL DESIGN
=============================================================================

- 8 spatial locations with fixed orientations (sampled once, held constant)
- Average responses over all C(8, l) location subsets per set size
- Each neuron's response is tracked across set sizes
- Clean Seaborn visualizations with statistical bands

Author: Mixed Selectivity Project
Date: January 2026
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import time

# Import from core modules
from core.gaussian_process import generate_neuron_population
from core.divisive_normalization import (
    compute_total_post_dn_analytical,
    compute_per_item_activity_efficient,
    verify_activity_cap_efficient,
)

# Set Seaborn style globally
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment_2(config: Dict) -> Dict:
    """
    Run Experiment 2: Population DN analysis.
    
    Design:
    - 8 locations with fixed orientations (sampled once)
    - All C(8, l) subsets averaged per set size
    - Track each neuron's response across set sizes
    
    Parameters
    ----------
    config : Dict
        Required keys: n_neurons, n_orientations, n_locations, set_sizes,
                       seed, gamma, sigma_sq, lambda_base, sigma_lambda
    """
    print("=" * 70)
    print("EXPERIMENT 2: POPULATION DIVISIVE NORMALIZATION")
    print("=" * 70)
    print(f"  N neurons:    {config['n_neurons']}")
    print(f"  Orientations: {config['n_orientations']}")
    print(f"  Locations:    {config['n_locations']}")
    print(f"  Set sizes:    {config['set_sizes']}")
    print(f"  γ (gain):     {config['gamma']} Hz")
    print(f"  σ²:           {config['sigma_sq']}")
    print(f"  Theoretical total activity: γN = {config['gamma'] * config['n_neurons']} Hz")
    print()
    
    # Generate neuron population using core module
    print("Generating neuron population...")
    start = time.time()
    population = generate_neuron_population(
        n_neurons=config['n_neurons'],
        n_orientations=config['n_orientations'],
        n_locations=config['n_locations'],
        base_lengthscale=config['lambda_base'],
        lengthscale_variability=config['sigma_lambda'],
        seed=config['seed']
    )
    
    # Extract f_samples and pre-compute G = exp(f) for vectorized computation
    f_samples_population = [neuron['f_samples'] for neuron in population]
    G_stacked = np.stack([np.exp(f) for f in f_samples_population], axis=0)
    print(f"  Done in {time.time() - start:.1f}s")
    print(f"  G matrix shape: {G_stacked.shape}")
    
    # Sample fixed orientations (held constant throughout experiment)
    rng = np.random.default_rng(config['seed'] + 1000)
    fixed_thetas = rng.integers(0, config['n_orientations'], size=config['n_locations'])
    print(f"  Fixed orientations: {fixed_thetas}")
    print()
    
    # Storage
    N = config['n_neurons']
    gamma = config['gamma']
    sigma_sq = config['sigma_sq']
    
    results = {
        'config': config,
        'fixed_thetas': fixed_thetas,
        'set_size_data': {},
        'neuron_responses': {}  # {set_size: (N,) array of avg responses}
    }
    
    # Process each set size
    for l in config['set_sizes']:
        print(f"Processing set size l = {l}...")
        
        all_subsets = list(combinations(range(config['n_locations']), l))
        n_subsets = len(all_subsets)
        print(f"  Subsets: C({config['n_locations']}, {l}) = {n_subsets}")
        
        # Accumulators
        pre_totals = []
        post_totals = []
        neuron_responses_sum = np.zeros(N)
        
        for subset in tqdm(all_subsets, desc=f"  l={l}", leave=False):
            # Compute pre-DN response for each neuron: R_pre[n] = ∏_{k∈S} G[n, loc_k, θ_k]
            R_pre = np.ones(N)
            for loc in subset:
                theta_idx = fixed_thetas[loc]
                R_pre *= G_stacked[:, loc, theta_idx]
            
            # Apply population divisive normalization
            # r^{post}_i = γ · r^{pre}_i / [σ² + mean(r^{pre})]
            pop_mean = np.mean(R_pre)
            R_post = gamma * R_pre / (sigma_sq + pop_mean)
            
            # Accumulate statistics
            pre_totals.append(np.sum(R_pre))
            post_totals.append(np.sum(R_post))
            neuron_responses_sum += R_post
        
        # Average across all subsets
        neuron_avg = neuron_responses_sum / n_subsets
        results['neuron_responses'][l] = neuron_avg
        
        # Compute theoretical values using core module functions
        theoretical_total = compute_total_post_dn_analytical(gamma, N)
        per_item = compute_per_item_activity_efficient(gamma, N, l)
        
        # Verify activity cap
        activity_cap_check = verify_activity_cap_efficient(
            gamma=gamma, N=N, sigma_sq=sigma_sq,
            observed_mc=np.mean(post_totals)
        )
        
        results['set_size_data'][l] = {
            'n_subsets': n_subsets,
            'pre_total_mean': np.mean(pre_totals),
            'pre_total_std': np.std(pre_totals),
            'post_total_mean': np.mean(post_totals),
            'post_total_std': np.std(post_totals),
            'theoretical_total': theoretical_total,
            'per_item_activity': per_item,
            'activity_cap_error': activity_cap_check.get('mc_error', 0.0),
            'neuron_mean': np.mean(neuron_avg),
            'neuron_std': np.std(neuron_avg),
            'neuron_q05': np.percentile(neuron_avg, 5),
            'neuron_q25': np.percentile(neuron_avg, 25),
            'neuron_q75': np.percentile(neuron_avg, 75),
            'neuron_q95': np.percentile(neuron_avg, 95),
        }
        
        r = results['set_size_data'][l]
        print(f"  Pre-DN total:  {r['pre_total_mean']:.1f} ± {r['pre_total_std']:.1f} Hz")
        print(f"  Post-DN total: {r['post_total_mean']:.1f} Hz (theory: {theoretical_total:.0f})")
        print(f"  Per-item:      {per_item:.1f} Hz")
        print()
    
    print("=" * 70)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# VISUALIZATION (Clean Seaborn Plots)
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """
    Create three clean Seaborn visualizations:
    
    1. Per-Item Activity vs Set Size (shows 1/l decrease)
    2. Activity Cap Empirical Verification (total activity constant)
    3. Single-Neuron Response Bands (heterogeneity across neurons)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = results['config']
    set_sizes = config['set_sizes']
    N = config['n_neurons']
    gamma = config['gamma']
    
    # Set consistent style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    palette = sns.color_palette("deep")
    
    # =========================================================================
    # PLOT 1: Per-Item Activity Decreases with Set Size
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    per_item = [results['set_size_data'][l]['per_item_activity'] for l in set_sizes]
    theoretical_per_item = [gamma * N / l for l in set_sizes]
    
    # Empirical points
    sns.lineplot(x=set_sizes, y=per_item, marker='o', markersize=10, 
                 linewidth=2.5, color=palette[0], label='Empirical', ax=ax)
    
    # Theoretical 1/l curve
    ax.plot(set_sizes, theoretical_per_item, '--', color=palette[3], 
            linewidth=2, label=r'Theory: $\gamma N / l$')
    
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Per-Item Activity (Hz)')
    ax.set_title('Per-Item Activity Decreases with Set Size')
    ax.set_xticks(set_sizes)
    ax.legend(frameon=True, loc='upper right')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path / f'exp2_per_item_activity_N{N}.png', 
                bbox_inches='tight', facecolor='white')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp2_per_item_activity_N{N}.png")
    
    # =========================================================================
    # PLOT 2: Activity Cap (Total Population Activity is Constant)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Pre-DN (grows with set size)
    pre_means = [results['set_size_data'][l]['pre_total_mean'] for l in set_sizes]
    pre_stds = [results['set_size_data'][l]['pre_total_std'] for l in set_sizes]
    
    ax1 = axes[0]
    ax1.errorbar(set_sizes, pre_means, yerr=pre_stds, fmt='o-', 
                 color=palette[3], linewidth=2.5, markersize=10, 
                 capsize=5, capthick=2, label='Pre-DN')
    ax1.set_xlabel('Set Size (l)')
    ax1.set_ylabel('Total Population Activity (Hz)')
    ax1.set_title('Pre-DN: Activity GROWS')
    ax1.set_xticks(set_sizes)
    sns.despine(ax=ax1)
    
    # Right: Post-DN (constant at γN)
    post_means = [results['set_size_data'][l]['post_total_mean'] for l in set_sizes]
    theoretical = gamma * N
    
    ax2 = axes[1]
    sns.lineplot(x=set_sizes, y=post_means, marker='o', markersize=10,
                 linewidth=2.5, color=palette[2], label='Post-DN', ax=ax2)
    ax2.axhline(theoretical, color=palette[3], linestyle='--', linewidth=2,
                label=f'Theory: γN = {theoretical:,.0f} Hz')
    ax2.set_xlabel('Set Size (l)')
    ax2.set_ylabel('Total Population Activity (Hz)')
    ax2.set_title('Post-DN: Activity CAPPED at γN')
    ax2.set_xticks(set_sizes)
    ax2.set_ylim([theoretical * 0.98, theoretical * 1.02])
    ax2.legend(frameon=True, loc='lower right')
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_path / f'exp2_activity_cap_N{N}.png',
                bbox_inches='tight', facecolor='white')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp2_activity_cap_N{N}.png")
    
    # =========================================================================
    # PLOT 3: Single-Neuron Response Bands
    # 
    # - 8 locations with fixed orientations
    # - Average over all C(8, l) subsets per set size
    # - Each line = one neuron's average response
    # - Band = spread across N neurons (5th-95th percentile)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Build response matrix: (N, n_set_sizes)
    response_matrix = np.column_stack([
        results['neuron_responses'][l] for l in set_sizes
    ])
    
    # Compute statistics
    pop_mean = np.mean(response_matrix, axis=0)
    pop_q05 = np.percentile(response_matrix, 5, axis=0)
    pop_q25 = np.percentile(response_matrix, 25, axis=0)
    pop_q75 = np.percentile(response_matrix, 75, axis=0)
    pop_q95 = np.percentile(response_matrix, 95, axis=0)
    
    # Sort neurons by final response for coloring
    sort_idx = np.argsort(response_matrix[:, -1])
    response_sorted = response_matrix[sort_idx, :]
    
    # Plot individual neuron lines (thin, colored by rank)
    cmap = plt.cm.coolwarm
    for i in range(N):
        color = cmap(i / (N - 1))
        ax.plot(set_sizes, response_sorted[i, :], color=color, 
                alpha=0.5, linewidth=0.8, zorder=1)
    
    # Plot bands (5-95 percentile and 25-75 percentile)
    ax.fill_between(set_sizes, pop_q05, pop_q95, alpha=0.2, 
                    color=palette[0], label='5th–95th percentile')
    ax.fill_between(set_sizes, pop_q25, pop_q75, alpha=0.3,
                    color=palette[0], label='25th–75th percentile')
    
    # Population mean
    ax.plot(set_sizes, pop_mean, 'o-', color='black', linewidth=3, 
            markersize=10, markerfacecolor='gold', markeredgewidth=2,
            label=f'Population mean (≈ γ = {gamma:.0f} Hz)', zorder=100)
    
    # Reference line at gamma
    ax.axhline(gamma, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_yscale('log')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Average Post-DN Response (Hz)')
    ax.set_title(f'Single-Neuron Responses Across Set Sizes (N = {N} neurons)\n'
                 f'8 locations, fixed orientations, averaged over C(8,l) subsets')
    ax.set_xticks(set_sizes)
    ax.set_xlim([set_sizes[0] - 0.3, set_sizes[-1] + 0.3])
    ax.legend(frameon=True, loc='upper right')
    sns.despine()
    
    # Add colorbar for neuron ranking
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label(f'Neuron rank at l={set_sizes[-1]}\n(blue=suppressed, red=enhanced)')
    cbar.set_ticks([0, N//2, N-1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    
    plt.tight_layout()
    plt.savefig(output_path / f'exp2_neuron_response_bands_N{N}.png',
                bbox_inches='tight', facecolor='white')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp2_neuron_response_bands_N{N}.png")
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Set Size':<12} {'Pre-DN Total':<18} {'Post-DN Total':<18} {'Per-Item':<12}")
    print("-" * 70)
    for l in set_sizes:
        r = results['set_size_data'][l]
        print(f"{l:<12} {r['pre_total_mean']:>12,.1f} Hz     "
              f"{r['post_total_mean']:>12,.1f} Hz     {r['per_item_activity']:>8,.1f} Hz")
    print("-" * 70)
    print(f"Theoretical activity cap: γN = {gamma * N:,.0f} Hz")
    print("=" * 70)


# =============================================================================
# ENTRY POINT (for standalone testing)
# =============================================================================

if __name__ == '__main__':
    # Default configuration for standalone run
    config = {
        'n_neurons': 100,
        'n_orientations': 10,
        'n_locations': 8,
        'set_sizes': [2, 4, 6, 8],
        'seed': 42,
        'gamma': 100.0,
        'sigma_sq': 1e-6,
        'lambda_base': 0.3,
        'sigma_lambda': 0.5,
    }
    
    results = run_experiment_2(config)
    plot_results(results, 'results/exp2', show_plot=True)