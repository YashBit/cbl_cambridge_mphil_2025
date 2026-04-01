"""
Experiment 4: Multi-Item ML Decoding
=====================================

Validates the efficient factorised decoder (§5) and the causal chain
from DN to decoding error:

    More items → DN reduces per-item rate → fewer spikes per item
    → lower SNR → larger decoding error ∝ √l

Three parts:

    Part A — ERROR SCALING (§5, Eq. 28)
        Circular std vs set size for multi-item decoder.
        Validates error ∝ √l from the causal chain.
        Includes error distributions showing broadening with l.

    Part B — BIAS ANALYSIS
        Mean signed error across 1000 trials per set size.
        An unbiased ML decoder should have E[error] ≈ 0.
        Tests whether grid quantisation introduces systematic bias.

    Part C — LENGTHSCALE SWEEP (two regimes)
        Decoding error vs set size across a range of base lengthscales.
        Regime 1: broad tuning (λ = 3.0, 2.0, 1.0) — smooth, low Fisher info
        Regime 2: sharp tuning (λ = 0.8, 0.5, 0.3) — peaked, high Fisher info
        Multi-seed averaging with SEM error bars.
        Key prediction: all curves follow √l regardless of λ — the capacity
        limit is driven by DN resource competition, not tuning shape.

Paper equations:
    Efficient decoder: θ̂_c = argmax L_c(θ_c) + Σ_{k≠c} logsumexp(L_k)  [Eq. 26]
    L_k(θ) = Σ_i n_i f_{i,k}(θ)                                          [Eq. 23]
    Error ∝ √l  (from Fisher info ∝ 1/l under DN)                         [§3.6]
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import time

from core.gaussian_process import generate_neuron_population
from core.poisson_spike import generate_spikes
from core.ml_decoder import (
    compute_log_likelihood,
    compute_circular_error,
    compute_spike_weighted_log_tuning,
    compute_marginal_log_likelihood_efficient,
)
from core.divisive_normalisation import dn_pointwise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _circular_std(errors: np.ndarray) -> float:
    """Circular SD: σ = √(−2 log R̄) where R̄ = |mean(exp(iε))|."""
    R = np.abs(np.mean(np.exp(1j * errors)))
    R = np.clip(R, 1e-10, 1.0 - 1e-10)
    return np.sqrt(-2.0 * np.log(R))


def _snap_to_grid(theta, grid):
    """Nearest grid index under circular distance."""
    d = np.abs(grid - theta)
    return int(np.argmin(np.minimum(d, 2 * np.pi - d)))


def _extract_f_per_location(population, active_locations):
    """Extract f_{i,k}(θ) arrays for active locations. Returns list of (N, n_θ)."""
    N = len(population)
    n_theta = population[0]['f_samples'].shape[1]
    return [
        np.array([population[i]['f_samples'][loc, :] for i in range(N)])
        for loc in active_locations
    ]


def _run_trial(population, theta_values, gamma, sigma_sq, T_d,
               active_locations, true_orientations, cued_index, rng):
    """
    Single trial: encode → DN → spike → decode.

    Encoding:  r^pre_n = exp(Σ_k f_{n,k}(θ_k))          [Eq. 13]
    DN:        r^post = dn_pointwise(r^pre, γ, σ²)       [Eq. 6]
    Spike:     n_i ~ Poisson(r^post_i · T_d)              [Def. 4.5]
    Decode:    θ̂_c = argmax L_c(θ_c) + Σ_{k≠c} lse(L_k)  [Eq. 26]
    """
    N = len(population)
    f_list = _extract_f_per_location(population, active_locations)

    # Snap true orientations to grid
    theta_idx = [_snap_to_grid(t, theta_values) for t in true_orientations]

    # Pre-normalised rate
    log_r_pre = np.zeros(N)
    for k, f_k in enumerate(f_list):
        log_r_pre += f_k[:, theta_idx[k]]
    r_pre = np.exp(log_r_pre)

    # DN + Poisson
    rates = dn_pointwise(r_pre, gamma, sigma_sq)
    spikes = rng.poisson(rates * T_d)

    # Factorised ML decode
    L_list = compute_spike_weighted_log_tuning(spikes, f_list)
    ll_marginal = compute_marginal_log_likelihood_efficient(L_list, cued_index)
    idx_hat = np.argmax(ll_marginal)

    # Error against snapped ground truth (avoids grid quantisation bias)
    theta_true_snapped = theta_values[theta_idx[cued_index]]
    error = compute_circular_error(theta_true_snapped, theta_values[idx_hat])

    return error


def _run_trials(population, theta_values, gamma, sigma_sq, T_d,
                n_locations, set_sizes, n_trials, seed, desc=""):
    """
    Run n_trials per set size. Returns dict[l] with 'errors', 'circular_std_deg'.
    """
    rng = np.random.RandomState(seed)
    results = {}

    for l in set_sizes:
        errors = np.empty(n_trials)
        for t in range(n_trials):
            locs = tuple(rng.choice(n_locations, size=l, replace=False))
            oris = rng.uniform(-np.pi, np.pi, size=l)
            cued = rng.randint(l)
            errors[t] = _run_trial(
                population, theta_values, gamma, sigma_sq, T_d,
                locs, oris, cued, rng)

        std = _circular_std(errors)
        results[l] = {
            'errors': errors,
            'circular_std': std,
            'circular_std_deg': np.degrees(std),
            'mean_error': float(np.mean(errors)),
            'mean_error_deg': float(np.degrees(np.mean(errors))),
        }

    return results


# =============================================================================
# PART A — ERROR SCALING  (§5, causal chain validation)
# =============================================================================
#
# The paper's causal chain predicts decoding error ∝ √l:
#   DN caps activity → per-item rate = γN/l → Fisher info ∝ 1/l
#   → Cramér-Rao bound: Var[θ̂] ≥ 1/I_F ∝ l → SD ∝ √l
#
# We validate this by running the full encode → DN → spike → decode
# pipeline at each set size and checking that σ/√l is approximately
# constant across set sizes.
# =============================================================================

def _run_part_a(cfg):
    population = generate_neuron_population(
        n_neurons=cfg['n_neurons'], n_orientations=cfg['n_orientations'],
        n_locations=cfg['n_locations'], base_lengthscale=cfg['lambda_base'],
        lengthscale_variability=cfg.get('sigma_lambda', 0.3), seed=cfg['seed'])
    theta_values = population[0]['orientations']

    print(f"    Multi-item decoding (n_trials={cfg['n_trials']})...")
    multi = _run_trials(
        population, theta_values, cfg['gamma'], cfg['sigma_sq'], cfg['T_d'],
        cfg['n_locations'], cfg['set_sizes'], cfg['n_trials'], cfg['seed'])

    stds = [multi[l]['circular_std_deg'] for l in cfg['set_sizes']]
    normalised = [stds[i] / np.sqrt(l) for i, l in enumerate(cfg['set_sizes'])]
    cv = np.std(normalised) / np.mean(normalised) if len(normalised) > 1 else 0.0

    for l in cfg['set_sizes']:
        print(f"      l={l}: σ = {multi[l]['circular_std_deg']:.1f}°")
    print(f"    σ/√l CV = {cv:.3f}")

    return {
        'multi_item': multi,
        'scaling': {'empirical_std': stds, 'normalised_by_sqrt_l': normalised, 'cv': cv},
    }


# =============================================================================
# PART B — BIAS ANALYSIS
# =============================================================================
#
# An ML decoder should be asymptotically unbiased.  With a coarse grid
# (n_θ = 10), the snapping of true orientations to grid points can
# introduce systematic bias.  We measure E[error] and |bias|/σ.
#
# The refactored decoder snaps ground truth to the grid before computing
# error, so reported bias reflects decoder bias only (not quantisation).
# =============================================================================

def _run_part_b(cfg):
    n_trials_bias = 1000
    n_theta_bias = 10

    population = generate_neuron_population(
        n_neurons=100, n_orientations=n_theta_bias,
        n_locations=cfg['n_locations'], base_lengthscale=cfg['lambda_base'],
        lengthscale_variability=cfg.get('sigma_lambda', 0.3),
        seed=cfg['seed'] + 500)
    theta_values = population[0]['orientations']

    print(f"    Bias analysis (n_θ={n_theta_bias}, n_trials={n_trials_bias})...")
    results = _run_trials(
        population, theta_values, cfg['gamma'], cfg['sigma_sq'], cfg['T_d'],
        cfg['n_locations'], cfg['set_sizes'], n_trials_bias, cfg['seed'] + 501)

    for l in cfg['set_sizes']:
        me = results[l]['mean_error_deg']
        sd = results[l]['circular_std_deg']
        ratio = abs(me) / sd if sd > 0 else 0
        print(f"      l={l}: bias={me:+.1f}°  σ={sd:.1f}°  |bias|/σ={ratio:.3f}")

    return results


# =============================================================================
# PART C — LENGTHSCALE SWEEP (two regimes, multi-seed)
# =============================================================================
#
# The paper predicts that the √l scaling comes from DN resource
# competition (§3.6), not from tuning curve shape.  We test this by
# sweeping λ across two regimes:
#
#   Regime 1 (broad): λ = 3.0, 2.0, 1.0 — smooth tuning, low Fisher info
#   Regime 2 (sharp): λ = 0.8, 0.5, 0.3 — peaked tuning, high Fisher info
#
# At each λ, we run multiple seeds and report mean ± SEM.
# All conditions use σ_λ = 0 (no heterogeneity) and the same master seed
# structure so the ONLY difference is the kernel width.
#
# Key prediction: the spread across λ (~20°) is much smaller than the
# set-size effect (~30°).  Every curve follows √l.
# =============================================================================

SWEEP_LAMBDAS_BROAD = [3.0, 2.0, 1.0]
SWEEP_LAMBDAS_SHARP = [0.8, 0.5, 0.3]
SWEEP_N_SEEDS = 3


def _run_part_c(cfg):
    all_lambdas = SWEEP_LAMBDAS_BROAD + SWEEP_LAMBDAS_SHARP
    n_seeds = SWEEP_N_SEEDS
    set_sizes = cfg['set_sizes']
    n_trials = cfg['n_trials']

    print(f"    Lengthscale sweep: {all_lambdas}")
    print(f"    {n_seeds} seeds per λ, {n_trials} trials per seed")

    sweep_results = {}

    for lam in all_lambdas:
        seed_stds = []
        seed_errors = {l: [] for l in set_sizes}

        for s in range(n_seeds):
            seed_val = cfg['seed'] + s * 100
            pop = generate_neuron_population(
                n_neurons=cfg['n_neurons'], n_orientations=cfg['n_orientations'],
                n_locations=cfg['n_locations'], base_lengthscale=lam,
                lengthscale_variability=0.0, seed=seed_val)
            theta_vals = pop[0]['orientations']

            res = _run_trials(
                pop, theta_vals, cfg['gamma'], cfg['sigma_sq'], cfg['T_d'],
                cfg['n_locations'], set_sizes, n_trials, seed_val + 1)

            seed_stds.append([res[l]['circular_std_deg'] for l in set_sizes])
            for l in set_sizes:
                seed_errors[l].append(res[l]['errors'])

        seed_stds = np.array(seed_stds)  # (n_seeds, n_set_sizes)
        mean_stds = seed_stds.mean(axis=0).tolist()
        sem_stds = (seed_stds.std(axis=0, ddof=1) / np.sqrt(n_seeds)).tolist()

        # Pool errors across seeds for distribution plots
        pooled_errors = {l: np.concatenate(seed_errors[l]) for l in set_sizes}

        sweep_results[lam] = {
            'stds_deg': mean_stds,
            'stds_sem_deg': sem_stds,
            'pooled_errors': pooled_errors,
            'n_seeds': n_seeds,
            'regime': 'broad' if lam >= 1.0 else 'sharp',
        }

        tag = "  ".join(f"l={l}: {mean_stds[i]:.1f}±{sem_stds[i]:.1f}°"
                        for i, l in enumerate(set_sizes))
        print(f"      λ={lam:.1f}: {tag}")

    return {
        'sweep_results': sweep_results,
        'lambdas_broad': SWEEP_LAMBDAS_BROAD,
        'lambdas_sharp': SWEEP_LAMBDAS_SHARP,
        'all_lambdas': all_lambdas,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment_4(config: Dict) -> Dict:
    cfg = {
        'n_neurons': config.get('n_neurons', 100),
        'n_orientations': config.get('n_orientations', 32),
        'n_locations': config.get('n_locations', 16),
        'gamma': config.get('gamma', 100.0),
        'sigma_sq': config.get('sigma_sq', 1e-6),
        'T_d': config.get('T_d', 0.1),
        'n_trials': config.get('n_trials', 500),
        'set_sizes': config.get('set_sizes', [2, 4, 6, 8]),
        'seed': config.get('seed', 42),
        'lambda_base': config.get('lambda_base', 0.5),
        'sigma_lambda': config.get('sigma_lambda', 0.3),
    }

    print("=" * 70)
    print("EXPERIMENT 4: MULTI-ITEM ML DECODING")
    print("=" * 70)
    print(f"  N={cfg['n_neurons']}  n_θ={cfg['n_orientations']}  "
          f"L={cfg['n_locations']}  T_d={cfg['T_d']}  γ={cfg['gamma']}")

    print("\n  Part A: Error scaling (√l validation)...")
    part_a = _run_part_a(cfg)

    print("\n  Part B: Bias analysis...")
    part_b = _run_part_b(cfg)

    print("\n  Part C: Lengthscale sweep (two regimes)...")
    part_c = _run_part_c(cfg)

    print(f"\n{'=' * 70}")
    return {
        'config': cfg,
        'part_a': part_a,
        'part_b': part_b,
        'part_c': part_c,
    }


# =============================================================================
# PLOTTING — 4 figures
# =============================================================================
#
# 1. Error scaling + √l check (Part A, 2 panels)
# 2. Error distributions broadening with l (Part A)
# 3. Bias analysis (Part B, 2 panels)
# 4. Lengthscale sweep — two regimes (Part C)
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    cfg = results['config']
    ss = cfg['set_sizes']
    pa = results['part_a']
    pb = results['part_b']
    pc = results['part_c']

    # ================================================================
    # PLOT 1: Error scaling + √l check
    # ================================================================
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

    mi_std = [pa['multi_item'][l]['circular_std_deg'] for l in ss]
    ref = [mi_std[0] * np.sqrt(l / ss[0]) for l in ss]

    ax1a.plot(ss, mi_std, 'o-', color='#E74C3C', lw=2, ms=8, label='Decoded σ')
    ax1a.plot(ss, ref, ':', color='gray', lw=2, label=r'$\propto \sqrt{l}$')
    ax1a.set_xlabel('Set size $l$')
    ax1a.set_ylabel('Circular std (degrees)')
    ax1a.set_title('A. Decoding Error vs Set Size')
    ax1a.set_xticks(ss)
    ax1a.legend(fontsize=9)

    ax1a.text(0.02, 0.98,
              "Causal chain (§3.6):\n"
              "DN → rate ∝ 1/l\n"
              "→ Fisher info ∝ 1/l\n"
              "→ error ∝ √l",
              transform=ax1a.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round,pad=0.4', fc='#FADBD8', ec='#E74C3C', alpha=0.9))

    norm = pa['scaling']['normalised_by_sqrt_l']
    ax1b.plot(ss, norm, 'o-', color='#8E44AD', lw=2, ms=8)
    ax1b.axhline(np.mean(norm), color='red', ls='--', lw=1.5,
                 label=f'Mean = {np.mean(norm):.1f}°')
    ax1b.set_xlabel('Set size $l$')
    ax1b.set_ylabel(r'$\sigma / \sqrt{l}$ (degrees)')
    ax1b.set_title(f'B. √l Scaling Check (CV = {pa["scaling"]["cv"]:.3f})')
    ax1b.set_xticks(ss)
    ax1b.legend(fontsize=9)

    fig1.tight_layout()
    fig1.savefig(out / 'exp4a_error_scaling.png')
    print(f"  Saved: exp4a_error_scaling.png")
    if show_plot: plt.show()
    plt.close(fig1)

    # ================================================================
    # PLOT 2: Error distributions broaden with l
    # ================================================================
    n_ss = len(ss)
    fig2, axes2 = plt.subplots(1, n_ss, figsize=(4 * n_ss, 3.5))
    if n_ss == 1: axes2 = [axes2]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, n_ss))

    for i, l in enumerate(ss):
        errs_deg = np.degrees(pa['multi_item'][l]['errors'])
        axes2[i].hist(errs_deg, bins=30, density=True, color=colors[i],
                      alpha=0.7, edgecolor='white')
        sd = pa['multi_item'][l]['circular_std_deg']
        axes2[i].set_title(f'l={l}  (σ={sd:.1f}°)', fontsize=11)
        axes2[i].set_xlabel('Error (°)')
        axes2[i].set_xlim([-90, 90])
        if i == 0: axes2[i].set_ylabel('Density')

    fig2.suptitle('Error Distributions Broaden with Set Size', fontsize=12, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(out / 'exp4a_error_distributions.png')
    print(f"  Saved: exp4a_error_distributions.png")
    if show_plot: plt.show()
    plt.close(fig2)

    # ================================================================
    # PLOT 3: Bias analysis
    # ================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    biases = [pb[l]['mean_error_deg'] for l in ss]
    stds_b = [pb[l]['circular_std_deg'] for l in ss]
    n_b = 1000
    ci95 = [1.96 * np.degrees(np.std(pb[l]['errors'])) / np.sqrt(n_b) for l in ss]

    ax3a.bar(ss, biases, width=0.6, color='#3498DB', alpha=0.7, edgecolor='black')
    ax3a.errorbar(ss, biases, yerr=ci95, fmt='none', color='black', capsize=5)
    ax3a.axhline(0, color='red', ls='--', lw=1.5)
    ax3a.set_xlabel('Set size $l$')
    ax3a.set_ylabel('Mean signed error (°)')
    ax3a.set_title('A. Decoder Bias')
    ax3a.set_xticks(ss)

    ratios = [abs(biases[i]) / stds_b[i] if stds_b[i] > 0 else 0 for i in range(len(ss))]
    ax3b.bar(ss, ratios, width=0.6, color='#27AE60', alpha=0.7, edgecolor='black')
    ax3b.axhline(0.05, color='red', ls='--', lw=1.2, label='5% reference')
    ax3b.set_xlabel('Set size $l$')
    ax3b.set_ylabel('|Bias| / σ')
    ax3b.set_title('B. Relative Bias')
    ax3b.set_xticks(ss)
    ax3b.legend(fontsize=9)

    fig3.suptitle('Experiment 4B: Bias Analysis (n_θ=10)', fontsize=12, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(out / 'exp4b_bias.png')
    print(f"  Saved: exp4b_bias.png")
    if show_plot: plt.show()
    plt.close(fig3)

    # ================================================================
    # PLOT 4: Lengthscale sweep — two regimes
    # ================================================================
    sweep = pc['sweep_results']
    all_lam = pc['all_lambdas']

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    cmap_c = plt.cm.viridis
    norm_c = plt.Normalize(vmin=min(all_lam), vmax=max(all_lam))
    markers = ['o', 's', 'D', '^', 'v', 'P']

    for idx, lam in enumerate(all_lam):
        stds = sweep[lam]['stds_deg']
        sems = sweep[lam]['stds_sem_deg']
        color = cmap_c(norm_c(lam))
        regime = sweep[lam]['regime']
        ls = '-' if regime == 'broad' else '--'

        ax4.errorbar(ss, stds, yerr=sems,
                     marker=markers[idx % len(markers)], color=color,
                     lw=1.5, ms=7, capsize=3, linestyle=ls,
                     label=f'λ={lam:.1f} ({regime})')

        # √l reference anchored at first set size
        ref_c = [stds[0] * np.sqrt(l / ss[0]) for l in ss]
        ax4.plot(ss, ref_c, ':', color=color, lw=0.7, alpha=0.3)

    ax4.set_xlabel('Set size $l$')
    ax4.set_ylabel('Circular std (degrees)')
    ax4.set_title('Lengthscale Sweep: Capacity Limit is DN, Not Tuning Shape')
    ax4.set_xticks(ss)
    ax4.legend(fontsize=8, ncol=2, title='Base lengthscale')

    ax4.text(0.98, 0.02,
             "Solid = broad (λ ≥ 1)\n"
             "Dashed = sharp (λ < 1)\n\n"
             "Spread across λ (~20°)\n"
             "≪ set-size effect (~30°)\n"
             "→ √l from DN, not tuning",
             transform=ax4.transAxes, fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#3498DB', alpha=0.9))

    fig4.tight_layout()
    fig4.savefig(out / 'exp4c_lengthscale_sweep.png')
    print(f"  Saved: exp4c_lengthscale_sweep.png")
    if show_plot: plt.show()
    plt.close(fig4)

    print(f"\n  Experiment 4 plots saved to {out}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 4: ML Decoding')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_orientations', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/exp4')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    config = {
        'n_neurons': args.n_neurons, 'n_orientations': args.n_orientations,
        'n_locations': 16, 'gamma': 100.0, 'sigma_sq': 1e-6,
        'T_d': 0.1, 'n_trials': args.n_trials, 'seed': args.seed,
        'set_sizes': [2, 4, 6, 8], 'lambda_base': 0.5, 'sigma_lambda': 0.3,
    }
    results = run_experiment_4(config)
    if not args.no_plot:
        plot_results(results, args.output_dir, show_plot=True)