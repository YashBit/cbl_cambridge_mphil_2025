"""
Experiment 3: Poisson Noise and the 1/√l Capacity Limit
========================================================

Validates how Poisson spiking noise converts DN's rate reduction
into the working memory capacity limit (§3.7, §4.5).

The causal chain:
    DN caps total activity at γN  →  per-item rate = γN/l
    →  per-item spike count λ = γN·T_d/l  →  SNR = √λ ∝ 1/√l

Three sub-experiments:

    3A — SNR SCALING (§3.7)
        Total SNR = √(γN·T_d) is CONSTANT across set sizes.
        Per-item SNR = √(γN·T_d/l) DEGRADES as 1/√l.
        This is the capacity limit.

    3B — TIME-ACCURACY TRADE-OFF
        Per-item SNR = √(γN·T_d/l).
        Doubling T_d improves SNR by √2, but the 1/√l penalty
        is fundamental — every curve follows the same shape.

    3C — SPIKE COUNT DISTRIBUTIONS
        Total spike count ~ Poisson(γN·T_d) is identical for all l.
        Per-item spike count ~ Poisson(γN·T_d/l) shifts left and
        narrows as l increases.

Paper equations:
    Per-item rate:  r_item = γN / l                          [§4.5]
    Expected spikes: λ_item = γN·T_d / l                     [Def. 4.5]
    SNR: √λ = √(γN·T_d/l) ∝ 1/√l                           [§3.7]
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple
import time

from core.gaussian_process import generate_neuron_population
from core.poisson_spike import generate_spikes
from core.divisive_normalisation import dn_pointwise, total_post_activity, per_item_activity


# =============================================================================
# SHARED SETUP
# =============================================================================

def _setup_population(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Generate GP population and fixed orientations. Returns (G_stacked, fixed_thetas)."""
    population = generate_neuron_population(
        n_neurons=config['n_neurons'],
        n_orientations=config['n_orientations'],
        n_locations=config['n_locations'],
        base_lengthscale=config['lambda_base'],
        lengthscale_variability=config['sigma_lambda'],
        seed=config['seed'],
    )
    G_stacked = np.stack([np.exp(n['f_samples']) for n in population])
    rng = np.random.default_rng(config['seed'] + 1000)
    fixed_thetas = rng.integers(0, config['n_orientations'], size=config['n_locations'])
    return G_stacked, fixed_thetas


def _compute_rates_at_subset(G_stacked, subset, fixed_thetas, gamma, sigma_sq):
    """
    Compute post-DN rates at one (S, θ) configuration.

        r^pre_n = ∏_{k∈S} G[n, k, θ_k]    [Eq. 13]
        r^post = dn_pointwise(r^pre, γ, σ²) [Eq. 6]
    """
    N = G_stacked.shape[0]
    R_pre = np.ones(N)
    for loc in subset:
        R_pre *= G_stacked[:, loc, fixed_thetas[loc]]
    return dn_pointwise(R_pre, gamma, sigma_sq)


# =============================================================================
# 3A — SNR SCALING  (§3.7)
# =============================================================================
#
# Theoretical:
#   λ_total = γN·T_d          → SNR_total = √(γN·T_d)       (constant)
#   λ_item  = γN·T_d / l      → SNR_item  = √(γN·T_d / l)   (∝ 1/√l)
#
# Empirical:
#   Generate Poisson spikes at each (S,θ), measure total and per-item
#   spike count statistics.
# =============================================================================

def _run_3a(G_stacked, fixed_thetas, config):
    N = config['n_neurons']
    gamma = config['gamma']
    sigma_sq = config['sigma_sq']
    T_d = config['T_d']
    set_sizes = config['set_sizes']
    n_trials = config.get('n_trials', 1000)
    rng = np.random.RandomState(config['seed'] + 2000)

    gamma_N_Td = gamma * N * T_d  # total expected spikes (constant)

    theoretical = {
        'lambda_total': [], 'lambda_per_item': [],
        'snr_total': [], 'snr_per_item': [],
    }
    empirical = {'snr_per_item': []}

    for l in set_sizes:
        # ── Theoretical ──
        lam_total = gamma_N_Td
        lam_item = gamma_N_Td / l
        theoretical['lambda_total'].append(lam_total)
        theoretical['lambda_per_item'].append(lam_item)
        theoretical['snr_total'].append(np.sqrt(lam_total))
        theoretical['snr_per_item'].append(np.sqrt(lam_item))

        # ── Empirical: sample spikes across subsets ──
        all_subsets = list(combinations(range(config['n_locations']), l))
        per_item_spikes = []

        for subset in all_subsets:
            rates = _compute_rates_at_subset(G_stacked, subset, fixed_thetas, gamma, sigma_sq)
            trials_per_subset = max(1, n_trials // len(all_subsets))
            for _ in range(trials_per_subset):
                spikes = generate_spikes(rates, T_d, rng)
                per_item_spikes.append(np.sum(spikes) / l)

        arr = np.array(per_item_spikes)
        empirical['snr_per_item'].append(
            np.mean(arr) / np.std(arr) if np.std(arr) > 0 else 0.0)

    return {'theoretical': theoretical, 'empirical': empirical, 'set_sizes': set_sizes}


# =============================================================================
# 3B — TIME-ACCURACY TRADE-OFF
# =============================================================================
#
# Per-item SNR = √(γN·T_d / l)
# Sweep (l, T_d) grid.  Every curve follows 1/√l; longer T_d shifts up by √T_d.
# =============================================================================

def _run_3b(config):
    N = config['n_neurons']
    gamma = config['gamma']
    set_sizes = config['set_sizes']
    T_d_values = config.get('T_d_values', [0.05, 0.1, 0.2, 0.4])

    grid = np.zeros((len(set_sizes), len(T_d_values)))
    for i, l in enumerate(set_sizes):
        for j, T_d in enumerate(T_d_values):
            grid[i, j] = np.sqrt(gamma * N * T_d / l)

    return {'set_sizes': set_sizes, 'T_d_values': T_d_values, 'snr_grid': grid}


# =============================================================================
# 3C — SPIKE COUNT DISTRIBUTIONS
# =============================================================================
#
# Total spikes ~ Poisson(γN·T_d) — same for all l.
# Per-item spikes ~ Poisson(γN·T_d/l) — shifts left with l.
# =============================================================================

def _run_3c(G_stacked, fixed_thetas, config):
    N = config['n_neurons']
    gamma = config['gamma']
    sigma_sq = config['sigma_sq']
    T_d = config['T_d']
    selected = [l for l in config['set_sizes'] if l >= 2][:4]
    n_trials = config.get('n_trials', 1000)
    rng = np.random.RandomState(config['seed'] + 3000)

    distributions = {}
    for l in selected:
        all_subsets = list(combinations(range(config['n_locations']), l))
        total_spikes = []
        per_item_spikes = []

        for subset in all_subsets:
            rates = _compute_rates_at_subset(G_stacked, subset, fixed_thetas, gamma, sigma_sq)
            trials_per_subset = max(1, n_trials // len(all_subsets))
            for _ in range(trials_per_subset):
                spikes = generate_spikes(rates, T_d, rng)
                s = np.sum(spikes)
                total_spikes.append(s)
                per_item_spikes.append(s / l)

        lam_item = gamma * N * T_d / l
        distributions[l] = {
            'total': np.array(total_spikes),
            'per_item': np.array(per_item_spikes),
            'lambda_total': gamma * N * T_d,
            'lambda_per_item': lam_item,
        }

    return {'set_sizes': selected, 'distributions': distributions}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment_3(config: Dict) -> Dict:
    N = config.get('n_neurons', 100)
    gamma = config.get('gamma', 100.0)
    T_d = config.get('T_d', 0.1)
    config.setdefault('set_sizes', [1, 2, 4, 6, 8])
    config.setdefault('n_locations', 8)
    config.setdefault('lambda_base', 0.3)
    config.setdefault('sigma_lambda', 0.5)
    config.setdefault('n_orientations', 10)
    config.setdefault('seed', 42)

    print("=" * 70)
    print("EXPERIMENT 3: POISSON NOISE — THE 1/√l CAPACITY LIMIT")
    print("=" * 70)
    print(f"  N={N}  γ={gamma}  T_d={T_d}")
    print(f"  γN·T_d = {gamma * N * T_d:.0f} total expected spikes (CONSTANT)")
    print()

    G_stacked, fixed_thetas = _setup_population(config)

    print("  3A: SNR scaling...")
    r3a = _run_3a(G_stacked, fixed_thetas, config)
    for i, l in enumerate(r3a['set_sizes']):
        print(f"    l={l}: SNR_item={r3a['theoretical']['snr_per_item'][i]:.1f} "
              f"(empirical: {r3a['empirical']['snr_per_item'][i]:.1f})")

    print("  3B: Time-accuracy trade-off...")
    r3b = _run_3b(config)

    print("  3C: Spike distributions...")
    r3c = _run_3c(G_stacked, fixed_thetas, config)

    print("\n" + "=" * 70)
    return {'config': config, 'exp3a': r3a, 'exp3b': r3b, 'exp3c': r3c}


# =============================================================================
# PLOTTING — 3 figures (one per sub-experiment)
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    cfg = results['config']
    r3a = results['exp3a']
    r3b = results['exp3b']
    r3c = results['exp3c']
    N = cfg['n_neurons']

    # ================================================================
    # PLOT 1: SNR scaling (3A) — the core capacity limit result
    # ================================================================
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))
    ss = r3a['set_sizes']

    # Left: Total vs per-item SNR
    ax1a.plot(ss, r3a['theoretical']['snr_total'], 'o-', color='#2E86AB',
              lw=2, ms=8, label='Total SNR (constant)')
    ax1a.plot(ss, r3a['theoretical']['snr_per_item'], 's-', color='#E74C3C',
              lw=2, ms=8, label=r'Per-item SNR $\propto 1/\sqrt{l}$')
    ax1a.plot(ss, r3a['empirical']['snr_per_item'], 'x', color='#E74C3C',
              ms=10, mew=2, label='Per-item (empirical)')
    ax1a.fill_between(ss, r3a['theoretical']['snr_per_item'],
                      r3a['theoretical']['snr_total'], alpha=0.15, color='red')
    ax1a.set_xlabel('Set size $l$')
    ax1a.set_ylabel('SNR')
    ax1a.set_title('A. The Capacity Limit')
    ax1a.set_xticks(ss)
    ax1a.legend(fontsize=9)

    ax1a.text(0.98, 0.02,
              "§3.7: SNR = √λ\n"
              "Per-item λ = γN·T_d/l\n"
              "→ SNR ∝ 1/√l",
              transform=ax1a.transAxes, fontsize=9, va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.4', fc='#FADBD8', ec='#E74C3C', alpha=0.9))

    # Right: Resource allocation bars
    x = np.arange(len(ss))
    ax1b.bar(x - 0.18, r3a['theoretical']['lambda_total'], 0.35,
             color='#2E86AB', alpha=0.7, label='λ_total', edgecolor='black', lw=0.5)
    ax1b.bar(x + 0.18, r3a['theoretical']['lambda_per_item'], 0.35,
             color='#E74C3C', alpha=0.7, label='λ_per_item', edgecolor='black', lw=0.5)
    ax1b.set_xlabel('Set size $l$')
    ax1b.set_ylabel('Expected spike count (λ)')
    ax1b.set_title('B. Resource Allocation')
    ax1b.set_xticks(x)
    ax1b.set_xticklabels(ss)
    ax1b.legend(fontsize=9)

    fig1.tight_layout()
    fig1.savefig(output_path / 'exp3_snr_scaling.png')
    print(f"  Saved: exp3_snr_scaling.png")
    if show_plot: plt.show()
    plt.close(fig1)

    # ================================================================
    # PLOT 2: Time-accuracy trade-off (3B)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(r3b['T_d_values'])))

    for j, T_d in enumerate(r3b['T_d_values']):
        ax2.plot(r3b['set_sizes'], r3b['snr_grid'][:, j], 'o-',
                 color=colors[j], lw=2, ms=7, label=f'T_d = {T_d:.2f}s')

    ax2.set_xlabel('Set size $l$')
    ax2.set_ylabel('Per-item SNR')
    ax2.set_title(r'Time-Accuracy Trade-off: SNR = $\sqrt{\gamma N \cdot T_d / l}$')
    ax2.set_xticks(r3b['set_sizes'])
    ax2.legend(title='Integration time', fontsize=9)

    ax2.text(0.98, 0.98,
             "Doubling T_d → SNR × √2\n"
             "But every curve follows\n"
             "1/√l — the penalty is\n"
             "fundamental.",
             transform=ax2.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#3498DB', alpha=0.9))

    fig2.tight_layout()
    fig2.savefig(output_path / 'exp3_time_tradeoff.png')
    print(f"  Saved: exp3_time_tradeoff.png")
    if show_plot: plt.show()
    plt.close(fig2)

    # ================================================================
    # PLOT 3: Spike distributions (3C)
    # ================================================================
    n_sz = len(r3c['set_sizes'])
    fig3, axes = plt.subplots(2, n_sz, figsize=(4 * n_sz, 6.5))
    colors_dist = plt.cm.coolwarm(np.linspace(0.2, 0.8, n_sz))

    for i, l in enumerate(r3c['set_sizes']):
        d = r3c['distributions'][l]

        # Top: total spikes (same for all l)
        ax_t = axes[0, i]
        ax_t.hist(d['total'], bins=30, density=True, color='gray', alpha=0.6, edgecolor='white')
        ax_t.axvline(d['lambda_total'], color='red', ls='--', lw=1.5)
        ax_t.set_title(f'l={l}: Total (λ={d["lambda_total"]:.0f})', fontsize=10)
        if i == 0: ax_t.set_ylabel('Density')

        # Bottom: per-item spikes (shifts left)
        ax_b = axes[1, i]
        ax_b.hist(d['per_item'], bins=30, density=True, color=colors_dist[i],
                  alpha=0.6, edgecolor='white')
        ax_b.axvline(d['lambda_per_item'], color='red', ls='--', lw=1.5)
        ax_b.set_title(f'l={l}: Per-item (λ={d["lambda_per_item"]:.0f})', fontsize=10)
        ax_b.set_xlabel('Spike count')
        if i == 0: ax_b.set_ylabel('Density')

    fig3.suptitle('Total (constant) vs Per-Item (decreasing) Spike Distributions',
                  fontsize=12, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(output_path / 'exp3_distributions.png')
    print(f"  Saved: exp3_distributions.png")
    if show_plot: plt.show()
    plt.close(fig3)

    print(f"\n  Experiment 3 plots saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_neurons': 100, 'n_orientations': 10, 'n_locations': 8,
        'gamma': 100.0, 'sigma_sq': 1e-6, 'T_d': 0.1,
        'n_trials': 1000, 'set_sizes': [1, 2, 4, 6, 8],
        'lambda_base': 0.3, 'sigma_lambda': 0.5, 'seed': 42,
    }
    results = run_experiment_3(config)
    plot_results(results, 'results/exp3', show_plot=True)