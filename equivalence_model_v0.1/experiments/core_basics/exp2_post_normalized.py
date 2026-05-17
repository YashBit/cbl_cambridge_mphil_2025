"""
Experiment 2: Population Divisive Normalisation Analysis
========================================================

Validates the three core predictions of population-level divisive
normalisation (Eq. 6/14, Activity Cap Theorem Eq. 15):

    r^{post}_i(S,θ) = γ · r^{pre}_i(S,θ) / D(S,θ)
    D(S,θ) = σ² + N⁻¹ Σ_j r^{pre}_j(S,θ)

Three predictions tested:

    Plot 1 — PER-ITEM ACTIVITY DECREASES (§4.5)
        Per-item share of the activity budget = γN / l.
        As set size grows, each item gets less resource.

    Plot 2 — ACTIVITY CAP (Eq. 15)
        Σ_i r^{post}_i = γN  for all (S, θ)  when σ² → 0.
        Pre-DN total grows exponentially; post-DN is flat.

    Plot 3 — RESPONSE HETEROGENEITY (Fig. 8 in paper)
        Individual neurons diverge from the population mean γ as
        set size increases.  Some become "winners" (high response
        at all active locations), others "losers" (suppressed).
        Population mean stays locked at γ = 100 Hz.

Design:
    8 locations with fixed orientations (sampled once).
    All C(8, l) subsets averaged per set size.
    Each neuron tracked across set sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import time

from core.encoder.gaussian_process import generate_neuron_population
from core.encoder.divisive_normalization import (
    dn_pointwise,
    total_post_activity,
    per_item_activity,
    verify_activity_cap,
)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment_2(config: Dict) -> Dict:
    """
    Run Experiment 2: Population DN analysis.

    For each set size l and each of the C(L, l) location subsets:
        1. Compute r^pre_n = ∏_{k∈S} G[n, k, θ_k]           [Eq. 13]
        2. Apply DN: r^post = dn_pointwise(r^pre, γ, σ²)     [Eq. 6]
        3. Record total pre-DN, total post-DN, per-neuron post-DN
    """
    N = config['n_neurons']
    gamma = config['gamma']
    sigma_sq = config['sigma_sq']
    L = config['n_locations']
    set_sizes = config['set_sizes']

    print("=" * 70)
    print("EXPERIMENT 2: POPULATION DIVISIVE NORMALISATION")
    print("=" * 70)
    print(f"  N={N}  L={L}  γ={gamma}  σ²={sigma_sq}")
    print(f"  Theoretical total: γN = {gamma * N} Hz")
    print()

    # Generate population
    population = generate_neuron_population(
        n_neurons=N, n_orientations=config['n_orientations'],
        n_locations=L, base_lengthscale=config['lambda_base'],
        lengthscale_variability=config['sigma_lambda'],
        seed=config['seed'],
    )
    G_stacked = np.stack([np.exp(neuron['f_samples']) for neuron in population])

    # Fixed orientations (held constant throughout)
    rng = np.random.default_rng(config['seed'] + 1000)
    fixed_thetas = rng.integers(0, config['n_orientations'], size=L)
    print(f"  Fixed θ indices: {fixed_thetas}\n")

    results = {
        'config': config,
        'fixed_thetas': fixed_thetas,
        'set_size_data': {},
        'neuron_responses': {},
    }

    for l in set_sizes:
        all_subsets = list(combinations(range(L), l))
        n_subsets = len(all_subsets)
        print(f"  l={l}: C({L},{l}) = {n_subsets} subsets...", end=" ")

        pre_totals = []
        post_totals = []
        neuron_sum = np.zeros(N)

        for subset in all_subsets:
            # ── Eq. 13: r^pre_n = ∏_{k∈S} G[n, k, θ_k] ──
            R_pre = np.ones(N)
            for loc in subset:
                R_pre *= G_stacked[:, loc, fixed_thetas[loc]]

            # ── Eq. 6: DN via dn_pointwise ──
            R_post = dn_pointwise(R_pre, gamma, sigma_sq)

            pre_totals.append(np.sum(R_pre))
            post_totals.append(np.sum(R_post))
            neuron_sum += R_post

        neuron_avg = neuron_sum / n_subsets
        results['neuron_responses'][l] = neuron_avg

        # Theoretical values from refactored DN module
        theo_total = total_post_activity(gamma, N)
        theo_per_item = per_item_activity(gamma, N, l)
        cap_check = verify_activity_cap(np.mean(post_totals), gamma, N)

        # Empirical per-item: measured from actual post-DN totals
        empirical_per_item = np.mean(post_totals) / l

        results['set_size_data'][l] = {
            'n_subsets': n_subsets,
            'pre_total_mean': np.mean(pre_totals),
            'pre_total_std': np.std(pre_totals),
            'post_total_mean': np.mean(post_totals),
            'post_total_std': np.std(post_totals),
            'theoretical_total': theo_total,
            'per_item_theoretical': theo_per_item,
            'per_item_empirical': empirical_per_item,
            'activity_cap_error': cap_check['relative_error'],
            'neuron_mean': np.mean(neuron_avg),
            'neuron_std': np.std(neuron_avg),
        }

        print(f"post-DN total={np.mean(post_totals):.1f} Hz  "
              f"(theory: {theo_total:.0f})  "
              f"per-item: empirical={empirical_per_item:.1f} theory={theo_per_item:.1f} Hz")

    print("\n" + "=" * 70)
    return results


# =============================================================================
# PLOTTING
# =============================================================================
#
# Three plots, each validating one paper prediction:
#   1. Per-item activity = γN/l  (§4.5)
#   2. Activity Cap: Σ r^post = γN  (Eq. 15)
#   3. Neuron heterogeneity: individual responses diverge (Fig. 8)
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = results['config']
    set_sizes = config['set_sizes']
    N = config['n_neurons']
    gamma = config['gamma']

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # ================================================================
    # PLOT 1: Per-Item Activity = γN/l
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))

    empirical = [results['set_size_data'][l]['per_item_empirical'] for l in set_sizes]
    theoretical = [results['set_size_data'][l]['per_item_theoretical'] for l in set_sizes]

    ax1.plot(set_sizes, empirical, 'o-', color='#2E86AB', lw=2, ms=8,
             label=r'Empirical: $\Sigma\, r_i^{\mathrm{post}} \,/\, l$')
    ax1.plot(set_sizes, theoretical, '--', color='#E74C3C', lw=2, label=r'Theory: $\gamma N / l$')

    ax1.set_xlabel('Set size $l$')
    ax1.set_ylabel('Per-item activity (Hz)')
    ax1.set_title('Per-Item Activity Decreases with Set Size')
    ax1.set_xticks(set_sizes)
    ax1.legend(fontsize=10)

    ax1.text(0.98, 0.98,
             "§4.5: Activity Cap forces\n"
             "per-item share = γN / l\n"
             "→ resource competition",
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#2E86AB', alpha=0.9))

    fig1.tight_layout()
    fig1.savefig(output_path / f'exp2_per_item_N{N}.png')
    print(f"  Saved: exp2_per_item_N{N}.png")
    if show_plot: plt.show()
    plt.close(fig1)

    # ================================================================
    # PLOT 2: Activity Cap (pre-DN grows, post-DN flat)
    # ================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Pre-DN grows exponentially
    pre_means = [results['set_size_data'][l]['pre_total_mean'] for l in set_sizes]
    pre_stds = [results['set_size_data'][l]['pre_total_std'] for l in set_sizes]

    ax2a.errorbar(set_sizes, pre_means, yerr=pre_stds, fmt='o-',
                  color='#E74C3C', lw=2, ms=8, capsize=4)
    ax2a.set_xlabel('Set size $l$')
    ax2a.set_ylabel('Total population activity (Hz)')
    ax2a.set_title('Pre-DN: Activity GROWS')
    ax2a.set_xticks(set_sizes)

    # Right: Post-DN is constant at γN
    post_means = [results['set_size_data'][l]['post_total_mean'] for l in set_sizes]
    theo = gamma * N

    ax2b.plot(set_sizes, post_means, 'o-', color='#27AE60', lw=2, ms=8, label='Post-DN')
    ax2b.axhline(theo, color='gray', ls='--', lw=1.5, label=f'Theory: γN = {theo:,.0f} Hz')
    ax2b.set_xlabel('Set size $l$')
    ax2b.set_ylabel('Total population activity (Hz)')
    ax2b.set_title('Post-DN: Activity CAPPED at γN')
    ax2b.set_xticks(set_sizes)
    ax2b.set_ylim([theo * 0.98, theo * 1.02])
    ax2b.legend(fontsize=9)

    ax2b.text(0.02, 0.02,
              "Eq. 15: Σᵢ rᵢᵖᵒˢᵗ = γN\n"
              f"Error < {max(r['activity_cap_error'] for r in results['set_size_data'].values()):.4%}",
              transform=ax2b.transAxes, fontsize=9, va='bottom', ha='left',
              bbox=dict(boxstyle='round,pad=0.4', fc='#D5F5E3', ec='#27AE60', alpha=0.9))

    fig2.tight_layout()
    fig2.savefig(output_path / f'exp2_activity_cap_N{N}.png')
    print(f"  Saved: exp2_activity_cap_N{N}.png")
    if show_plot: plt.show()
    plt.close(fig2)

    # ================================================================
    # PLOT 3: Single-Neuron Response Heterogeneity
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(9, 6))

    response_matrix = np.column_stack([results['neuron_responses'][l] for l in set_sizes])

    # Sort neurons by response at largest set size (for colour coding)
    sort_idx = np.argsort(response_matrix[:, -1])
    response_sorted = response_matrix[sort_idx, :]

    cmap = plt.cm.coolwarm
    for i in range(N):
        ax3.plot(set_sizes, response_sorted[i, :], color=cmap(i / (N - 1)),
                 alpha=0.5, lw=0.8)

    # Percentile bands
    q05 = np.percentile(response_matrix, 5, axis=0)
    q25 = np.percentile(response_matrix, 25, axis=0)
    q75 = np.percentile(response_matrix, 75, axis=0)
    q95 = np.percentile(response_matrix, 95, axis=0)
    pop_mean = np.mean(response_matrix, axis=0)

    ax3.fill_between(set_sizes, q05, q95, alpha=0.15, color='#3498DB', label='5th–95th %ile')
    ax3.fill_between(set_sizes, q25, q75, alpha=0.25, color='#3498DB', label='25th–75th %ile')
    ax3.plot(set_sizes, pop_mean, 'o-', color='black', lw=3, ms=8,
             markerfacecolor='gold', markeredgewidth=1.5,
             label=f'Population mean (≈ γ = {gamma:.0f} Hz)', zorder=100)
    ax3.axhline(gamma, color='gray', ls=':', lw=1.5, alpha=0.7)

    ax3.set_yscale('log')
    ax3.set_xlabel('Set size $l$')
    ax3.set_ylabel('Average post-DN response (Hz)')
    ax3.set_title(f'Single-Neuron Heterogeneity Under DN  (N={N})')
    ax3.set_xticks(set_sizes)
    ax3.legend(fontsize=8, loc='upper right')

    ax3.text(0.02, 0.02,
             "Population mean locked at γ.\n"
             "Individual neurons diverge\n"
             "by > 4 orders of magnitude\n"
             "as set size increases.",
             transform=ax3.transAxes, fontsize=9, va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='#FFF3E0', ec='#E67E22', alpha=0.9))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label(f'Neuron rank at l={set_sizes[-1]}')
    cbar.set_ticks([0, N // 2, N - 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    fig3.tight_layout()
    fig3.savefig(output_path / f'exp2_neuron_heterogeneity_N{N}.png')
    print(f"  Saved: exp2_neuron_heterogeneity_N{N}.png")
    if show_plot: plt.show()
    plt.close(fig3)

    # Summary table
    print(f"\n  {'l':<6} {'Pre-DN':>12} {'Post-DN':>12} {'Per-Item(emp)':>14} {'Per-Item(thy)':>14} {'Cap Error':>12}")
    print("  " + "-" * 74)
    for l in set_sizes:
        r = results['set_size_data'][l]
        print(f"  {l:<6} {r['pre_total_mean']:>10,.1f}Hz {r['post_total_mean']:>10,.1f}Hz "
              f"{r['per_item_empirical']:>12,.1f}Hz {r['per_item_theoretical']:>12,.1f}Hz "
              f"{r['activity_cap_error']:>10.4%}")

    print(f"\n  Experiment 2 plots saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_neurons': 100, 'n_orientations': 10, 'n_locations': 8,
        'set_sizes': [2, 4, 6, 8], 'seed': 42,
        'gamma': 100.0, 'sigma_sq': 1e-6,
        'lambda_base': 0.3, 'sigma_lambda': 0.5,
    }
    results = run_experiment_2(config)
    plot_results(results, 'results/exp2', show_plot=True)