"""
Experiment 1: Pre-Normalized Response Analysis (ENHANCED PLOTTING)

This experiment analyzes RAW (pre-normalized) neural responses from the
GP-based mixed selectivity framework, BEFORE divisive normalization.

PURPOSE:
    - Establish baseline scaling of neural activity with set size
    - Demonstrate exponential growth: R ∝ ḡ^l
    - Show why DN is necessary (activity explodes without it)

KEY OUTPUT:
    - R.mean vs Set Size (exponential growth)
    - Per-item activity measures
    - Separability analysis (mixed selectivity verification)

MECHANISM TESTED:
    Pre-normalized response R = exp(Σ_k f_k(θ_k)) grows exponentially
    because more locations → more terms in sum → larger exponent

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time

# Import from core modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gaussian_process import (
    generate_neuron_population,
    compute_pre_normalized_response
)
from core.divisive_normalization import (
    compute_total_activity,
    compute_per_item_activity
)
from analysis.separability import (
    analyze_population_separability,
    summarize_separability
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'n_orientations': 10,
    'total_locations': 8,
    'subset_sizes': [2, 4, 6, 8],
    'base_lengthscale': 0.3,
    'lengthscale_variability': 0.5,
    'gain_variability': 0.2
}


# ============================================================================
# CORE EXPERIMENT FUNCTIONS
# ============================================================================

def compute_pre_normalized_statistics(
    neuron: Dict,
    subset_sizes: List[int],
    show_progress: bool = False
) -> Dict:
    """
    Compute pre-normalized response statistics for a single neuron.
    
    Parameters
    ----------
    neuron : Dict
        Neuron data from generate_neuron_tuning_curves
    subset_sizes : List[int]
        List of set sizes to analyze
    show_progress : bool
        Show progress bar for subsets
        
    Returns
    -------
    dict with statistics for each set size
    """
    f_samples = neuron['f_samples']
    total_locations = f_samples.shape[0]
    n_theta = f_samples.shape[1]
    
    results = {}
    
    for l in subset_sizes:
        subsets = list(combinations(range(total_locations), l))
        
        subset_means = []
        subset_per_item = []
        
        iterator = tqdm(subsets, desc=f"l={l}", leave=False) if show_progress else subsets
        
        for subset in iterator:
            # Compute pre-normalized response
            R_pre = compute_pre_normalized_response(f_samples, subset)
            
            # Statistics
            total = compute_total_activity(R_pre)
            per_item = compute_per_item_activity(R_pre, l)
            
            subset_means.append(total)
            subset_per_item.append(per_item)
        
        results[l] = {
            'R_mean': np.mean(subset_means),
            'R_std': np.std(subset_means),
            'R_all': np.array(subset_means),
            'per_item_mean': np.mean(subset_per_item),
            'per_item_std': np.std(subset_per_item),
            'per_item_all': np.array(subset_per_item),
            'n_subsets': len(subsets)
        }
    
    return results


def run_experiment1(
    n_neurons: int = 1,
    seed: int = 22,
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Run Experiment 1: Pre-Normalized Response Analysis.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons to generate
    seed : int
        Random seed for reproducibility
    config : Dict, optional
        Configuration overrides
    verbose : bool
        Print progress and results
        
    Returns
    -------
    dict with complete experiment results
    """
    # Merge config
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    if verbose:
        print("\n" + "="*70)
        print("  EXPERIMENT 1: PRE-NORMALIZED RESPONSE ANALYSIS")
        print("="*70)
        print(f"\n  Configuration:")
        print(f"    n_neurons:       {n_neurons}")
        print(f"    n_orientations:  {cfg['n_orientations']}")
        print(f"    total_locations: {cfg['total_locations']}")
        print(f"    subset_sizes:    {cfg['subset_sizes']}")
        print(f"    λ_base:          {cfg['base_lengthscale']}")
        print(f"    σ_λ:             {cfg['lengthscale_variability']}")
        print(f"    seed:            {seed}")
    
    start_time = time.time()
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: Generate Neuron Population
    # ────────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  STEP 1: Generating {n_neurons} neuron(s)")
        print(f"  {'─'*60}")
    
    population = generate_neuron_population(
        n_neurons=n_neurons,
        n_orientations=cfg['n_orientations'],
        n_locations=cfg['total_locations'],
        base_lengthscale=cfg['base_lengthscale'],
        lengthscale_variability=cfg['lengthscale_variability'],
        seed=seed,
        gain_variability=cfg['gain_variability']
    )
    
    if verbose and n_neurons == 1:
        print(f"\n    Lengthscales: {population[0]['lengthscales']}")
        ls = population[0]['lengthscales']
        print(f"    Range: [{ls.min():.3f}, {ls.max():.3f}], Ratio: {ls.max()/ls.min():.2f}×")
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: Compute Pre-Normalized Responses
    # ────────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  STEP 2: Computing pre-normalized responses")
        print(f"  {'─'*60}")
    
    all_neuron_results = []
    population_stats = {l: {'R_means': [], 'per_item_means': []} for l in cfg['subset_sizes']}
    
    neuron_iter = tqdm(population, desc="  Neurons", unit="neuron") if (n_neurons > 1 and verbose) else population
    
    for neuron in neuron_iter:
        neuron_stats = compute_pre_normalized_statistics(
            neuron,
            cfg['subset_sizes'],
            show_progress=(n_neurons == 1 and verbose)
        )
        
        all_neuron_results.append({
            'neuron_idx': neuron['neuron_idx'],
            'lengthscales': neuron['lengthscales'],
            'statistics': neuron_stats
        })
        
        # Accumulate population statistics
        for l in cfg['subset_sizes']:
            population_stats[l]['R_means'].append(neuron_stats[l]['R_mean'])
            population_stats[l]['per_item_means'].append(neuron_stats[l]['per_item_mean'])
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: Separability Analysis
    # ────────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  STEP 3: Analyzing mixed selectivity")
        print(f"  {'─'*60}")
    
    # FIX: summarize_separability expects population, not separability_results
    separability_summary = summarize_separability(population)
    
    if verbose:
        # FIX: Access nested structure correctly
        print(f"\n    Mean separability: {separability_summary['separability']['mean']:.3f}")
        print(f"    Mixed selectivity: {separability_summary['classification']['percent_mixed']:.1f}%")
    
    elapsed = time.time() - start_time
    
    # Compute population summary
    population_summary = {}
    for l in cfg['subset_sizes']:
        R_means = np.array(population_stats[l]['R_means'])
        per_item_means = np.array(population_stats[l]['per_item_means'])
        
        population_summary[l] = {
            'R_mean': np.mean(R_means) if n_neurons > 1 else R_means[0],
            'R_std': np.std(R_means) if n_neurons > 1 else 0.0,
            'R_all': R_means,
            'per_item_mean': np.mean(per_item_means) if n_neurons > 1 else per_item_means[0],
            'per_item_std': np.std(per_item_means) if n_neurons > 1 else 0.0,
            'per_item_all': per_item_means,
            'n_subsets': len(list(combinations(range(cfg['total_locations']), l)))
        }
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: Print Results
    # ────────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  RESULTS: Pre-Normalized Response")
        print(f"  {'─'*60}")
        
        header = f"\n  {'l':<5} {'R.mean':<18} {'Per-Item':<18} {'# Subsets':<12}"
        if n_neurons > 1:
            header = f"\n  {'l':<5} {'R.mean (avg)':<18} {'R.std':<15} {'Per-Item':<18} {'# Subsets':<12}"
        print(header)
        print("  " + "-"*65)
        
        for l in cfg['subset_sizes']:
            ps = population_summary[l]
            if n_neurons == 1:
                print(f"  {l:<5} {ps['R_mean']:<18.4e} {ps['per_item_mean']:<18.4e} {ps['n_subsets']:<12}")
            else:
                print(f"  {l:<5} {ps['R_mean']:<18.4e} {ps['R_std']:<15.4e} {ps['per_item_mean']:<18.4e} {ps['n_subsets']:<12}")
        
        # Scaling analysis
        print(f"\n  SCALING ANALYSIS:")
        for i in range(len(cfg['subset_sizes']) - 1):
            l1, l2 = cfg['subset_sizes'][i], cfg['subset_sizes'][i+1]
            r1 = population_summary[l1]['R_mean']
            r2 = population_summary[l2]['R_mean']
            fold = r2 / (r1 + 1e-10)
            print(f"    l={l1} → l={l2}: {fold:.2f}× increase")
        
        # Overall
        r_first = population_summary[cfg['subset_sizes'][0]]['R_mean']
        r_last = population_summary[cfg['subset_sizes'][-1]]['R_mean']
        print(f"\n    Overall (l={cfg['subset_sizes'][0]} → l={cfg['subset_sizes'][-1]}): "
              f"{r_last/r_first:.2f}× increase")
        
        print(f"\n  ⏱️  Time: {elapsed:.2f}s")
    
    # ────────────────────────────────────────────────────────────────────────
    # Return Results
    # ────────────────────────────────────────────────────────────────────────
    return {
        'experiment': 'pre_normalized',
        'n_neurons': n_neurons,
        'seed': seed,
        'config': cfg,
        'population': population,
        'neuron_results': all_neuron_results,
        'population_summary': population_summary,
        'separability': separability_summary,
        'timing': {
            'total_seconds': elapsed,
            'per_neuron': elapsed / n_neurons
        }
    }


# ============================================================================
# ENHANCED PLOTTING
# ============================================================================

def plot_experiment1(
    results: Dict,
    save_dir: str = 'figures/exp1_pre_norm',
    show_plot: bool = True
) -> Dict:
    """
    Create ENHANCED plots for Experiment 1 with detailed annotations.
    
    Plots:
    1. R.mean vs Set Size (log scale) - with scaling factors, theory, config
    2. Per-Item Activity vs Set Size - with scaling factors
    3. Separability distribution
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_neurons = results['n_neurons']
    seed = results['seed']
    cfg = results['config']
    pop_summary = results['population_summary']
    subset_sizes = results['config']['subset_sizes']
    separability = results['separability']
    
    R_means = [pop_summary[l]['R_mean'] for l in subset_sizes]
    R_stds = [pop_summary[l]['R_std'] for l in subset_sizes]
    per_item_means = [pop_summary[l]['per_item_mean'] for l in subset_sizes]
    per_item_stds = [pop_summary[l]['per_item_std'] for l in subset_sizes]
    
    # Compute scaling factors
    scaling_factors = []
    for i in range(len(subset_sizes) - 1):
        fold = R_means[i+1] / (R_means[i] + 1e-10)
        scaling_factors.append(fold)
    overall_scaling = R_means[-1] / (R_means[0] + 1e-10)
    
    # Estimate effective gain (ḡ) from exponential fit: R ≈ ḡ^l
    # log(R) = l * log(ḡ) => fit linear regression
    log_R = np.log(R_means)
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset_sizes, log_R)
    g_bar_estimated = np.exp(slope)  # effective per-location gain
    
    figures = {}
    sns.set_style("whitegrid")
    neuron_str = "1 neuron" if n_neurons == 1 else f"{n_neurons} neurons (avg)"
    
    print(f"\n  {'='*60}")
    print(f"  CREATING ENHANCED PLOTS")
    print(f"  {'='*60}")
    
    # ────────────────────────────────────────────────────────────────────────
    # PLOT 1: R.mean vs Set Size (ENHANCED)
    # ────────────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    ax1.set_yscale('log')
    
    # Main data line
    ax1.plot(subset_sizes, R_means, 'o-', lw=3, ms=14, color='#E74C3C', 
             label='R.mean (observed)', zorder=5)
    
    # Error bars if multiple neurons
    if n_neurons > 1 and any(s > 0 for s in R_stds):
        ax1.errorbar(subset_sizes, R_means, yerr=R_stds, fmt='none', 
                     color='#E74C3C', capsize=5, capthick=2, alpha=0.7)
    
    # Theoretical exponential fit line
    l_fine = np.linspace(min(subset_sizes)-0.5, max(subset_sizes)+0.5, 100)
    R_fit = np.exp(intercept + slope * l_fine)
    ax1.plot(l_fine, R_fit, '--', lw=2, color='#7F8C8D', alpha=0.8,
             label=f'Exponential fit: R ≈ {np.exp(intercept):.2f} × {g_bar_estimated:.2f}$^l$')
    
    # Annotate each point with value AND scaling factor
    for i, (l, val) in enumerate(zip(subset_sizes, R_means)):
        # Value annotation
        ax1.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#E74C3C', alpha=0.95))
        
        # Scaling factor annotation (between points)
        if i < len(scaling_factors):
            mid_x = (subset_sizes[i] + subset_sizes[i+1]) / 2
            mid_y = np.sqrt(R_means[i] * R_means[i+1])  # geometric mean for log scale
            ax1.annotate(f'×{scaling_factors[i]:.2f}', xy=(mid_x, mid_y),
                        fontsize=10, ha='center', va='center', color='#8E44AD',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8', 
                                 edgecolor='#8E44AD', alpha=0.9))
    
    ax1.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pre-Normalized R.mean (log scale)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Pre-Normalized Response vs Set Size\n({neuron_str})',
                  fontsize=16, fontweight='bold')
    ax1.set_xticks(subset_sizes)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # ── Configuration & Statistics Box ──
    config_text = (
        f"Configuration\n"
        f"─────────────────\n"
        f"seed: {seed}\n"
        f"n_orientations: {cfg['n_orientations']}\n"
        f"n_locations: {cfg['total_locations']}\n"
        f"λ_base: {cfg['base_lengthscale']}\n"
        f"σ_λ: {cfg['lengthscale_variability']}"
    )
    ax1.text(0.98, 0.02, config_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', 
                      edgecolor='#E67E22', alpha=0.95))
    
    # ── Scaling Summary Box ──
    scaling_text = (
        f"Scaling Summary\n"
        f"─────────────────────\n"
        f"Overall: ×{overall_scaling:.2f}\n"
        f"(l={subset_sizes[0]}→{subset_sizes[-1]})\n"
        f"─────────────────────\n"
        f"Estimated ḡ: {g_bar_estimated:.3f}\n"
        f"R² fit: {r_value**2:.4f}"
    )
    ax1.text(0.02, 0.98, scaling_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F5E3', 
                      edgecolor='#27AE60', alpha=0.95))
    
    # ── Theory Box ──
    theory_text = (
        f"Theory: R = exp(Σ fₖ(θₖ)) ≈ ḡˡ\n"
        f"More items → larger exponent\n"
        f"→ Exponential explosion!"
    )
    ax1.text(0.98, 0.98, theory_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FADBD8', 
                      edgecolor='#E74C3C', alpha=0.95))
    
    plt.tight_layout()
    filepath = Path(save_dir) / f'exp1_R_mean_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    figures['R_mean'] = fig1
    
    # ────────────────────────────────────────────────────────────────────────
    # PLOT 2: Per-Item Activity vs Set Size (ENHANCED)
    # ────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    ax2.set_yscale('log')
    
    # Main data line
    ax2.plot(subset_sizes, per_item_means, 's-', lw=3, ms=14, color='#27AE60', 
             label='Per-Item Activity', zorder=5)
    
    # Error bars if multiple neurons
    if n_neurons > 1 and any(s > 0 for s in per_item_stds):
        ax2.errorbar(subset_sizes, per_item_means, yerr=per_item_stds, fmt='none',
                     color='#27AE60', capsize=5, capthick=2, alpha=0.7)
    
    # Per-item scaling factors
    per_item_scaling = []
    for i in range(len(subset_sizes) - 1):
        fold = per_item_means[i+1] / (per_item_means[i] + 1e-10)
        per_item_scaling.append(fold)
    per_item_overall = per_item_means[-1] / (per_item_means[0] + 1e-10)
    
    # Annotate each point with value AND scaling factor
    for i, (l, val) in enumerate(zip(subset_sizes, per_item_means)):
        ax2.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#27AE60', alpha=0.95))
        
        # Scaling factor annotation
        if i < len(per_item_scaling):
            mid_x = (subset_sizes[i] + subset_sizes[i+1]) / 2
            mid_y = np.sqrt(per_item_means[i] * per_item_means[i+1])
            ax2.annotate(f'×{per_item_scaling[i]:.2f}', xy=(mid_x, mid_y),
                        fontsize=10, ha='center', va='center', color='#8E44AD',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8', 
                                 edgecolor='#8E44AD', alpha=0.9))
    
    ax2.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Per-Item Activity (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Per-Item Activity vs Set Size (Pre-DN)\n({neuron_str})',
                  fontsize=16, fontweight='bold')
    ax2.set_xticks(subset_sizes)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    # ── Configuration Box ──
    ax2.text(0.98, 0.02, config_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', 
                      edgecolor='#E67E22', alpha=0.95))
    
    # ── Scaling Summary Box ──
    per_item_text = (
        f"Per-Item Scaling\n"
        f"─────────────────────\n"
        f"Overall: ×{per_item_overall:.2f}\n"
        f"(l={subset_sizes[0]}→{subset_sizes[-1]})\n"
        f"─────────────────────\n"
        f"Note: Per-item = R/l\n"
        f"Still grows (pre-DN)!"
    )
    ax2.text(0.02, 0.98, per_item_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F5E3', 
                      edgecolor='#27AE60', alpha=0.95))
    
    # ── Interpretation Box ──
    interp_text = (
        f"Pre-DN: Per-item activity\n"
        f"GROWS with set size!\n"
        f"(R grows faster than l)"
    )
    ax2.text(0.98, 0.98, interp_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D6EAF8', 
                      edgecolor='#2E86AB', alpha=0.95))
    
    plt.tight_layout()
    filepath = Path(save_dir) / f'exp1_per_item_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    figures['per_item'] = fig2
    
    # ────────────────────────────────────────────────────────────────────────
    # PLOT 3: Separability Distribution (if multiple neurons)
    # ────────────────────────────────────────────────────────────────────────
    if n_neurons > 1:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        seps = separability['all_separabilities']
        sns.histplot(seps, kde=True, ax=ax3, color='#9B59B6', alpha=0.7)
        ax3.axvline(0.8, color='red', linestyle='--', lw=2, label='Mixed/Pure threshold')
        ax3.axvline(np.mean(seps), color='#2E86AB', linestyle='-', lw=2,
                   label=f'Mean: {np.mean(seps):.3f}')
        
        ax3.set_xlabel('Separability Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax3.set_title(f'Separability Distribution\n({n_neurons} neurons, '
                      f'{separability["classification"]["percent_mixed"]:.1f}% mixed)',
                      fontsize=16, fontweight='bold')
        ax3.legend(fontsize=11)
        
        plt.tight_layout()
        filepath = Path(save_dir) / f'exp1_separability_{n_neurons}neurons.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        figures['separability'] = fig3
    
    if show_plot:
        plt.show()
    
    return figures


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment 1: Pre-Normalized Response')
    parser.add_argument('--n_neurons', type=int, default=1)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--save_dir', type=str, default='figures/exp1_pre_norm')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()
    
    results = run_experiment1(n_neurons=args.n_neurons, seed=args.seed)
    
    if not args.no_plot:
        plot_experiment1(results, save_dir=args.save_dir)