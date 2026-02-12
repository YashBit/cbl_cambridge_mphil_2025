"""
Experiment 1: Pre-Normalized Response Analysis

=============================================================================
WHAT THIS EXPERIMENT DOES
=============================================================================

A single call to run_experiment_1(config) runs TWO complementary analyses:

Part A — EXHAUSTIVE ENUMERATION:
    - Enumerates ALL C(L, l) subsets for each set size
    - Computes population-averaged R.mean and per-item activity
    - Includes SVD-based separability analysis
    - Demonstrates exponential growth R ~ g_bar^l

Part B — RANDOM (S, theta) SAMPLING:
    - Samples random subsets AND random orientations
    - Shows per-neuron dot-band at each set size
    - Demonstrates that R scales with l regardless of specific (S, theta)
    - Foundation for Activity Cap Theorem simplification in
      Marginalised Log-Likelihood

=============================================================================
PLOTS PRODUCED (5 total)
=============================================================================

1. R.mean vs Set Size (exhaustive, log scale)
2. Per-Item Activity vs Set Size (exhaustive)
3. Separability Distribution (if N > 1)
4. Dot-Band (random sampling)
5. Exponential Scaling + Normalised Spread (random)

Author: Mixed Selectivity Project
Date: December 2025 (Part A), January 2026 (Part B)
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
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'n_neurons': 100,
    'n_orientations': 10,
    'n_locations': 8,
    'set_sizes': [2, 4, 6, 8],
    'seed': 42,
    'lambda_base': 0.3,
    'sigma_lambda': 0.5,
    'n_random_subsets': 50,
    'n_theta_draws': 5,
}


# ============================================================================
# PART A HELPERS — EXHAUSTIVE ENUMERATION
# ============================================================================

def compute_pre_normalized_statistics(neuron, subset_sizes, show_progress=False):
    """Compute pre-normalized response statistics for a single neuron."""
    f_samples = neuron['f_samples']
    results = {}
    for l in subset_sizes:
        subsets = list(combinations(range(f_samples.shape[0]), l))
        subset_means = []
        subset_per_item = []
        iterator = tqdm(subsets, desc=f"l={l}", leave=False) if show_progress else subsets
        for subset in iterator:
            R_pre = compute_pre_normalized_response(f_samples, subset)
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


def _run_part_a(population, cfg, n_neurons, seed, verbose=True):
    """Part A: Exhaustive enumeration over all C(L, l) subsets."""
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART A: EXHAUSTIVE ENUMERATION")
        print(f"  {'─'*60}")

    all_neuron_results = []
    population_stats = {l: {'R_means': [], 'per_item_means': []} for l in cfg['set_sizes']}
    neuron_iter = tqdm(population, desc="  Neurons (Part A)", unit="neuron") if (n_neurons > 1 and verbose) else population

    for neuron in neuron_iter:
        neuron_stats = compute_pre_normalized_statistics(
            neuron, cfg['set_sizes'],
            show_progress=(n_neurons == 1 and verbose))
        all_neuron_results.append({
            'neuron_idx': neuron['neuron_idx'],
            'lengthscales': neuron['lengthscales'],
            'statistics': neuron_stats
        })
        for l in cfg['set_sizes']:
            population_stats[l]['R_means'].append(neuron_stats[l]['R_mean'])
            population_stats[l]['per_item_means'].append(neuron_stats[l]['per_item_mean'])

    if verbose:
        print(f"\n  Analyzing mixed selectivity (SVD)...")
    separability_summary = summarize_separability(population)
    if verbose:
        print(f"    Mean separability: {separability_summary['separability']['mean']:.3f}")
        print(f"    Mixed selectivity: {separability_summary['classification']['percent_mixed']:.1f}%")

    population_summary = {}
    for l in cfg['set_sizes']:
        R_means = np.array(population_stats[l]['R_means'])
        per_item_means = np.array(population_stats[l]['per_item_means'])
        population_summary[l] = {
            'R_mean': np.mean(R_means) if n_neurons > 1 else R_means[0],
            'R_std': np.std(R_means) if n_neurons > 1 else 0.0,
            'R_all': R_means,
            'per_item_mean': np.mean(per_item_means) if n_neurons > 1 else per_item_means[0],
            'per_item_std': np.std(per_item_means) if n_neurons > 1 else 0.0,
            'per_item_all': per_item_means,
            'n_subsets': len(list(combinations(range(cfg['n_locations']), l)))
        }

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART A RESULTS: Pre-Normalized Response")
        print(f"  {'─'*60}")
        header = f"\n  {'l':<5} {'R.mean':<18} {'Per-Item':<18} {'# Subsets':<12}"
        if n_neurons > 1:
            header = f"\n  {'l':<5} {'R.mean (avg)':<18} {'R.std':<15} {'Per-Item':<18} {'# Subsets':<12}"
        print(header)
        print("  " + "-"*65)
        for l in cfg['set_sizes']:
            ps = population_summary[l]
            if n_neurons == 1:
                print(f"  {l:<5} {ps['R_mean']:<18.4e} {ps['per_item_mean']:<18.4e} {ps['n_subsets']:<12}")
            else:
                print(f"  {l:<5} {ps['R_mean']:<18.4e} {ps['R_std']:<15.4e} {ps['per_item_mean']:<18.4e} {ps['n_subsets']:<12}")
        print(f"\n  SCALING ANALYSIS:")
        for i in range(len(cfg['set_sizes']) - 1):
            l1, l2 = cfg['set_sizes'][i], cfg['set_sizes'][i+1]
            r1, r2 = population_summary[l1]['R_mean'], population_summary[l2]['R_mean']
            print(f"    l={l1} -> l={l2}: {r2/(r1+1e-10):.2f}x increase")
        r_first = population_summary[cfg['set_sizes'][0]]['R_mean']
        r_last = population_summary[cfg['set_sizes'][-1]]['R_mean']
        print(f"\n    Overall (l={cfg['set_sizes'][0]} -> l={cfg['set_sizes'][-1]}): "
              f"{r_last/r_first:.2f}x increase")

    return {
        'neuron_results': all_neuron_results,
        'population_summary': population_summary,
        'separability': separability_summary,
    }


# ============================================================================
# PART B HELPERS — RANDOM (S, theta) SAMPLING
# ============================================================================

def compute_neuron_pre_dn(G_stacked, subset, theta_indices):
    """Pre-DN response for all neurons: r^pre_n(S,theta) = prod_{k in S} G[n,k,theta_k]"""
    R_pre = np.ones(G_stacked.shape[0])
    for loc in subset:
        R_pre *= G_stacked[:, loc, theta_indices[loc]]
    return R_pre


def sample_random_subsets(n_locations, l, n_subsets, rng):
    """Sample random subsets of size l."""
    all_locs = np.arange(n_locations)
    return [tuple(sorted(rng.choice(all_locs, size=l, replace=False))) for _ in range(n_subsets)]


def _run_part_b(population, G_stacked, cfg, verbose=True):
    """Part B: Random (S, theta) sampling of pre-normalized responses."""
    set_sizes = cfg['set_sizes']
    N = cfg['n_neurons']
    n_subsets = cfg.get('n_random_subsets', 50)
    n_theta_draws = cfg.get('n_theta_draws', 5)

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART B: RANDOM (S, theta) SAMPLING")
        print(f"  {'─'*60}")
        print(f"    Subsets/set size:  {n_subsets}")
        print(f"    theta draws:      {n_theta_draws}")
        print(f"    Total samples:    {n_subsets * n_theta_draws}")

    rng = np.random.default_rng(cfg['seed'] + 2000)
    per_neuron_avg = {}
    per_neuron_all = {}

    for l in set_sizes:
        all_responses = []
        for t in range(n_theta_draws):
            theta_indices = rng.integers(0, cfg['n_orientations'], size=cfg['n_locations'])
            subsets = sample_random_subsets(cfg['n_locations'], l, n_subsets, rng)
            for subset in subsets:
                all_responses.append(compute_neuron_pre_dn(G_stacked, subset, theta_indices))
        all_responses = np.stack(all_responses, axis=0)
        neuron_avg = np.mean(all_responses, axis=0)
        per_neuron_avg[l] = neuron_avg
        per_neuron_all[l] = all_responses.T
        if verbose:
            print(f"    l={l}: {all_responses.shape[0]} samples | "
                  f"median = {np.median(neuron_avg):.4f} | "
                  f"IQR = [{np.percentile(neuron_avg, 25):.4f}, {np.percentile(neuron_avg, 75):.4f}]")

    summary = {}
    for l in set_sizes:
        vals = per_neuron_avg[l]
        q25, q75 = np.percentile(vals, [25, 75])
        summary[l] = {
            'mean': np.mean(vals), 'median': np.median(vals),
            'std': np.std(vals), 'cv': np.std(vals) / (np.mean(vals) + 1e-30),
            'iqr': q75 - q25, 'q25': q25, 'q75': q75,
            'q05': np.percentile(vals, 5), 'q95': np.percentile(vals, 95),
            'min': np.min(vals), 'max': np.max(vals),
        }

    from scipy import stats as sp_stats
    medians = np.array([summary[l]['median'] for l in set_sizes])
    slope, intercept, r_value, _, _ = sp_stats.linregress(set_sizes, np.log(medians + 1e-30))
    fit = {'g_bar': np.exp(slope), 'intercept': intercept, 'slope': slope, 'r_squared': r_value**2}

    if verbose:
        print(f"\n    Fit (median): R ~ {np.exp(intercept):.4f} x {np.exp(slope):.3f}^l  (R2={r_value**2:.6f})")

    return {'per_neuron_avg': per_neuron_avg, 'per_neuron_all': per_neuron_all, 'summary': summary, 'fit': fit}


# ============================================================================
# MAIN EXPERIMENT — called by run_experiments.py
# ============================================================================

def run_experiment_1(config):
    """
    Run Experiment 1: Pre-Normalized Response Analysis.
    Runs BOTH Part A (exhaustive) and Part B (random sampling).
    """
    cfg = {**DEFAULT_CONFIG, **config}
    N = cfg['n_neurons']
    seed = cfg['seed']

    print("\n" + "="*70)
    print("  EXPERIMENT 1: PRE-NORMALIZED RESPONSE ANALYSIS")
    print("="*70)
    print(f"    n_neurons: {N}  |  orientations: {cfg['n_orientations']}  |  "
          f"locations: {cfg['n_locations']}  |  seed: {seed}")

    start_time = time.time()

    population = generate_neuron_population(
        n_neurons=N, n_orientations=cfg['n_orientations'],
        n_locations=cfg['n_locations'], base_lengthscale=cfg['lambda_base'],
        lengthscale_variability=cfg['sigma_lambda'], seed=seed,
        gain_variability=cfg.get('gain_variability', 0.2))

    if N == 1:
        ls = population[0]['lengthscales']
        print(f"    Lengthscales range: [{ls.min():.3f}, {ls.max():.3f}]")

    G_stacked = np.stack([np.exp(neuron['f_samples']) for neuron in population])

    part_a = _run_part_a(population, cfg, N, seed, verbose=True)
    part_b = _run_part_b(population, G_stacked, cfg, verbose=True)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.2f}s")
    print("="*70)

    return {
        'experiment': 'pre_normalized', 'n_neurons': N, 'seed': seed,
        'config': cfg, 'population': population,
        # Part A
        'neuron_results': part_a['neuron_results'],
        'population_summary': part_a['population_summary'],
        'separability': part_a['separability'],
        # Part B
        'per_neuron_avg': part_b['per_neuron_avg'],
        'per_neuron_all': part_b['per_neuron_all'],
        'random_sampling_summary': part_b['summary'],
        'fit': part_b['fit'],
        'timing': {'total_seconds': elapsed, 'per_neuron': elapsed / N}
    }


# ============================================================================
# PLOTTING — called by run_experiments.py
# ============================================================================

def plot_results(results, output_dir, show_plot=False):
    """Create ALL 5 plots for Experiment 1 (Part A + Part B)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    N = results['n_neurons']
    seed = results['seed']
    cfg = results['config']
    set_sizes = cfg['set_sizes']
    sns.set_style("whitegrid")
    neuron_str = "1 neuron" if N == 1 else f"{N} neurons (avg)"

    print(f"\n  {'='*60}")
    print(f"  CREATING EXPERIMENT 1 PLOTS")
    print(f"  {'='*60}")

    # ── Shared Part A data ──
    pop_summary = results['population_summary']
    R_means = [pop_summary[l]['R_mean'] for l in set_sizes]
    R_stds = [pop_summary[l]['R_std'] for l in set_sizes]
    per_item_means = [pop_summary[l]['per_item_mean'] for l in set_sizes]
    per_item_stds = [pop_summary[l]['per_item_std'] for l in set_sizes]

    scaling_factors = [R_means[i+1]/(R_means[i]+1e-10) for i in range(len(set_sizes)-1)]
    overall_scaling = R_means[-1] / (R_means[0] + 1e-10)

    log_R = np.log(R_means)
    slope, intercept, r_value, _, _ = stats.linregress(set_sizes, log_R)
    g_bar_estimated = np.exp(slope)

    config_text = (f"Configuration\n{'─'*17}\nseed: {seed}\n"
                   f"n_orientations: {cfg['n_orientations']}\n"
                   f"n_locations: {cfg['n_locations']}\n"
                   f"lambda_base: {cfg['lambda_base']}\nsigma_lambda: {cfg['sigma_lambda']}")

    # ================================================================
    # PLOT 1: R.mean vs Set Size (Part A)
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    ax1.set_yscale('log')
    ax1.plot(set_sizes, R_means, 'o-', lw=3, ms=14, color='#E74C3C',
             label='R.mean (observed)', zorder=5)
    if N > 1 and any(s > 0 for s in R_stds):
        ax1.errorbar(set_sizes, R_means, yerr=R_stds, fmt='none',
                     color='#E74C3C', capsize=5, capthick=2, alpha=0.7)
    l_fine = np.linspace(min(set_sizes)-0.5, max(set_sizes)+0.5, 100)
    R_fit = np.exp(intercept + slope * l_fine)
    ax1.plot(l_fine, R_fit, '--', lw=2, color='#7F8C8D', alpha=0.8,
             label=f'Fit: R ~ {np.exp(intercept):.2f} x {g_bar_estimated:.2f}^l')
    for i, (l, val) in enumerate(zip(set_sizes, R_means)):
        ax1.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='bottom', fontsize=11,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='#E74C3C', alpha=0.95))
        if i < len(scaling_factors):
            mid_x = (set_sizes[i] + set_sizes[i+1]) / 2
            mid_y = np.sqrt(R_means[i] * R_means[i+1])
            ax1.annotate(f'x{scaling_factors[i]:.2f}', xy=(mid_x, mid_y), fontsize=10,
                        ha='center', va='center', color='#8E44AD', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8',
                        edgecolor='#8E44AD', alpha=0.9))
    ax1.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pre-Normalized R.mean (log scale)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Pre-Normalized Response vs Set Size\n({neuron_str})', fontsize=16, fontweight='bold')
    ax1.set_xticks(set_sizes)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.text(0.98, 0.02, config_text, transform=ax1.transAxes, fontsize=9,
             va='bottom', ha='right', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', edgecolor='#E67E22', alpha=0.95))
    ax1.text(0.02, 0.98, f"Scaling Summary\n{'─'*21}\nOverall: x{overall_scaling:.2f}\n"
             f"(l={set_sizes[0]}->{set_sizes[-1]})\n{'─'*21}\n"
             f"Estimated g_bar: {g_bar_estimated:.3f}\nR2 fit: {r_value**2:.4f}",
             transform=ax1.transAxes, fontsize=10, va='top', ha='left', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F5E3', edgecolor='#27AE60', alpha=0.95))
    ax1.text(0.98, 0.98, "Theory: R = exp(sum f_k(theta_k)) ~ g_bar^l\nMore items -> larger exponent\n-> Exponential explosion!",
             transform=ax1.transAxes, fontsize=10, va='top', ha='right', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FADBD8', edgecolor='#E74C3C', alpha=0.95))
    plt.tight_layout()
    plt.savefig(out / f'exp1_R_mean_{N}neurons.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: exp1_R_mean_{N}neurons.png")
    if show_plot: plt.show()
    plt.close()

    # ================================================================
    # PLOT 2: Per-Item Activity (Part A)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    ax2.set_yscale('log')
    ax2.plot(set_sizes, per_item_means, 's-', lw=3, ms=14, color='#27AE60', label='Per-Item Activity', zorder=5)
    if N > 1 and any(s > 0 for s in per_item_stds):
        ax2.errorbar(set_sizes, per_item_means, yerr=per_item_stds, fmt='none',
                     color='#27AE60', capsize=5, capthick=2, alpha=0.7)
    pi_scaling = [per_item_means[i+1]/(per_item_means[i]+1e-10) for i in range(len(set_sizes)-1)]
    pi_overall = per_item_means[-1] / (per_item_means[0] + 1e-10)
    for i, (l, val) in enumerate(zip(set_sizes, per_item_means)):
        ax2.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 20), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#27AE60', alpha=0.95))
        if i < len(pi_scaling):
            mid_x = (set_sizes[i] + set_sizes[i+1]) / 2
            mid_y = np.sqrt(per_item_means[i] * per_item_means[i+1])
            ax2.annotate(f'x{pi_scaling[i]:.2f}', xy=(mid_x, mid_y), fontsize=10,
                        ha='center', va='center', color='#8E44AD', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8', edgecolor='#8E44AD', alpha=0.9))
    ax2.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Per-Item Activity (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Per-Item Activity vs Set Size (Pre-DN)\n({neuron_str})', fontsize=16, fontweight='bold')
    ax2.set_xticks(set_sizes)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(0.98, 0.02, config_text, transform=ax2.transAxes, fontsize=9,
             va='bottom', ha='right', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', edgecolor='#E67E22', alpha=0.95))
    ax2.text(0.02, 0.98, f"Per-Item Scaling\n{'─'*21}\nOverall: x{pi_overall:.2f}\n"
             f"(l={set_sizes[0]}->{set_sizes[-1]})\n{'─'*21}\nNote: Per-item = R/l\nStill grows (pre-DN)!",
             transform=ax2.transAxes, fontsize=10, va='top', ha='left', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F5E3', edgecolor='#27AE60', alpha=0.95))
    ax2.text(0.98, 0.98, "Pre-DN: Per-item activity\nGROWS with set size!\n(R grows faster than l)",
             transform=ax2.transAxes, fontsize=10, va='top', ha='right', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D6EAF8', edgecolor='#2E86AB', alpha=0.95))
    plt.tight_layout()
    plt.savefig(out / f'exp1_per_item_{N}neurons.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: exp1_per_item_{N}neurons.png")
    if show_plot: plt.show()
    plt.close()

    # ================================================================
    # PLOT 3: Separability Distribution (Part A, if N > 1)
    # ================================================================
    if N > 1:
        separability = results['separability']
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        seps = separability['all_separabilities']
        sns.histplot(seps, kde=True, ax=ax3, color='#9B59B6', alpha=0.7)
        ax3.axvline(0.8, color='red', linestyle='--', lw=2, label='Mixed/Pure threshold')
        ax3.axvline(np.mean(seps), color='#2E86AB', linestyle='-', lw=2, label=f'Mean: {np.mean(seps):.3f}')
        ax3.set_xlabel('Separability Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax3.set_title(f'Separability Distribution\n({N} neurons, '
                      f'{separability["classification"]["percent_mixed"]:.1f}% mixed)',
                      fontsize=16, fontweight='bold')
        ax3.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(out / f'exp1_separability_{N}neurons.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: exp1_separability_{N}neurons.png")
        if show_plot: plt.show()
        plt.close()

    # ================================================================
    # PLOT 4: DOT-BAND (Part B)
    # ================================================================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    per_neuron_avg = results['per_neuron_avg']
    summary = results['random_sampling_summary']
    fit = results['fit']
    palette = sns.color_palette("deep")

    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.set_yscale('log')
    jitter_width = 0.25
    rng_plot = np.random.default_rng(0)
    for i, l in enumerate(set_sizes):
        vals = per_neuron_avg[l]
        x_jitter = l + rng_plot.uniform(-jitter_width, jitter_width, size=len(vals))
        ax4.scatter(x_jitter, vals, s=18, alpha=0.55, color=palette[i],
                   edgecolors='none', zorder=3, label=f'l={l} (N={N})')
        q25, q75 = summary[l]['q25'], summary[l]['q75']
        ax4.plot([l-0.35, l+0.35], [summary[l]['median']]*2, color='black', lw=2.5, zorder=5)
        ax4.plot([l, l], [q25, q75], color='black', lw=2, zorder=4)
        ax4.plot([l-0.15, l+0.15], [q25]*2, color='black', lw=1.5, zorder=4)
        ax4.plot([l-0.15, l+0.15], [q75]*2, color='black', lw=1.5, zorder=4)
    l_fine_b = np.linspace(min(set_sizes)-0.5, max(set_sizes)+0.5, 200)
    R_fit_b = np.exp(fit['intercept'] + fit['slope'] * l_fine_b)
    ax4.plot(l_fine_b, R_fit_b, '--', color='grey', lw=2, alpha=0.8,
            label=f'Fit: R ~ {np.exp(fit["intercept"]):.2f} x {fit["g_bar"]:.2f}^l  (R2={fit["r_squared"]:.4f})')
    ax4.set_xlabel('Set Size (l)', fontsize=14)
    ax4.set_ylabel('Mean Pre-DN Response per Neuron (log)', fontsize=14)
    ax4.set_title(f'Pre-DN Response — Random (S, theta) Sampling\n'
                  f'N={N}, {cfg.get("n_random_subsets",50)} subsets x {cfg.get("n_theta_draws",5)} theta-draws',
                  fontsize=13)
    ax4.set_xticks(set_sizes)
    ax4.legend(fontsize=10, loc='upper left', frameon=True)
    sns.despine()
    stats_lines = ["Band Statistics (IQR)", "─"*28]
    for l in set_sizes:
        s = summary[l]
        stats_lines.append(f"l={l}:  IQR={s['iqr']:.3f}  CV={s['cv']:.2f}")
    ax4.text(0.98, 0.02, "\n".join(stats_lines), transform=ax4.transAxes,
            fontsize=8.5, va='bottom', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#FFF9E6', ec='#E6A817', alpha=0.92))
    plt.tight_layout()
    plt.savefig(out / f'exp1_dot_band_N{N}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: exp1_dot_band_N{N}.png")
    if show_plot: plt.show()
    plt.close()

    # ================================================================
    # PLOT 5: Exponential Scaling + Normalised Spread (Part B)
    # ================================================================
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
    medians = [summary[l]['median'] for l in set_sizes]
    q25s = [summary[l]['q25'] for l in set_sizes]
    q75s = [summary[l]['q75'] for l in set_sizes]
    q05s = [summary[l]['q05'] for l in set_sizes]
    q95s = [summary[l]['q95'] for l in set_sizes]

    ax5a.set_yscale('log')
    ax5a.fill_between(set_sizes, q05s, q95s, alpha=0.15, color=palette[0], label='5th-95th %ile')
    ax5a.fill_between(set_sizes, q25s, q75s, alpha=0.3, color=palette[0], label='IQR (25th-75th)')
    ax5a.plot(set_sizes, medians, 'o-', color=palette[0], lw=2.5, ms=9, label='Median', zorder=5)
    ax5a.plot(l_fine_b, R_fit_b, '--', color='grey', lw=1.8, alpha=0.7,
             label=f'Fit: g_bar={fit["g_bar"]:.3f}')
    for i in range(len(set_sizes)-1):
        fold = medians[i+1] / (medians[i] + 1e-30)
        mid_x = (set_sizes[i] + set_sizes[i+1]) / 2
        mid_y = np.sqrt(medians[i] * medians[i+1])
        ax5a.annotate(f'x{fold:.1f}', xy=(mid_x, mid_y), fontsize=10,
                     ha='center', va='center', color='#8E44AD', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='#F5EEF8', ec='#8E44AD', alpha=0.9))
    ax5a.set_xlabel('Set Size (l)', fontsize=13)
    ax5a.set_ylabel('Pre-DN Response (log)', fontsize=13)
    ax5a.set_title('Exponential Growth of Pre-DN Response', fontsize=13)
    ax5a.set_xticks(set_sizes)
    ax5a.legend(fontsize=9, loc='upper left', frameon=True)
    sns.despine(ax=ax5a)

    cvs = [summary[l]['cv'] for l in set_sizes]
    iqr_ratio = [summary[l]['iqr']/(summary[l]['median']+1e-30) for l in set_sizes]
    ax5b.plot(set_sizes, cvs, 's-', color=palette[3], lw=2, ms=9, label='CV (std/mean)')
    ax5b.plot(set_sizes, iqr_ratio, 'D-', color=palette[2], lw=2, ms=9, label='IQR/median')
    ax5b.set_xlabel('Set Size (l)', fontsize=13)
    ax5b.set_ylabel('Normalised Spread', fontsize=13)
    ax5b.set_title('Band Width (Normalised)', fontsize=13)
    ax5b.set_xticks(set_sizes)
    ax5b.legend(fontsize=10, frameon=True)
    sns.despine(ax=ax5b)

    plt.tight_layout()
    plt.savefig(out / f'exp1_scaling_N{N}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: exp1_scaling_N{N}.png")
    if show_plot: plt.show()
    plt.close()

    # ── Summary ──
    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT 1 COMPLETE — {5 if N > 1 else 4} plots saved to {out}")
    print(f"  {'='*60}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1: Pre-Normalized Response')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/exp1')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    config = {
        'n_neurons': args.n_neurons, 'n_orientations': 10, 'n_locations': 8,
        'set_sizes': [2, 4, 6, 8], 'seed': args.seed,
        'lambda_base': 0.3, 'sigma_lambda': 0.5,
    }
    results = run_experiment_1(config)
    if not args.no_plot:
        plot_results(results, args.save_dir, show_plot=True)