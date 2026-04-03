"""
Experiment 1: Pre-Normalised Response Analysis
===============================================

Validates the pre-normalised (before DN) response properties predicted
by the paper's §3.5 and §5.4.

Three parts, each testing a specific theoretical claim:

    Part A — EXPONENTIAL GROWTH (§3.5, Eq. 11)
        Claim:  E[r^pre_n] ≈ ḡ^l  where ḡ = exp(σ²_f / 2)
        Method: Exhaustive enumeration of all C(L,l) subsets at fixed θ.
        Plots:  Mean pre-DN response vs set size (log scale) + exponential fit.

    Part B — DENOMINATOR CONCENTRATION (§5.4)
        Claim:  D(S,θ) = σ² + N⁻¹ Σ_j r^pre_j  concentrates around ḡ^l
                as N → ∞, with CV ∝ 1/√N.
        Method: Random (S,θ) sampling with adaptive design.
        Plots:  D(S,θ) dot-band + CV analysis.

    Part C — MIXED SELECTIVITY (§3.3, Def. 4.3)
        Claim:  Location-dependent lengthscales break separability.
                S = σ₁² / Σσ²_i < 0.8 for most neurons.
        Method: SVD of each neuron's response matrix.
        Plots:  Separability histogram.

Paper equations implemented:
    r^pre_n(S,θ) = exp(Σ_{k∈S} f_{n,k}(θ_k))              [Eq. 13]
    D(S,θ) = σ² + N⁻¹ Σ_j r^pre_j(S,θ)                    [Eq. 14]
    S = σ₁² / Σ_i σ²_i                                      [Def. 4.3]
    E[r^pre] = ḡ^l,  ḡ = exp(σ²_f / 2) ≈ 1.65              [Eq. 11]
"""

import time
import numpy as np
from tqdm import tqdm
from itertools import combinations
from pathlib import Path
from typing import Dict


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.encoder.gaussian_process import (
    generate_neuron_population,
)
from core.encoder.divisive_normalization import compute_r_pre_at_config
from analysis.separability import summarize_separability


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'n_neurons': 100,
    'n_orientations': 200,
    'n_locations': 8,
    'set_sizes': [2, 4, 6, 8],
    'seed': 42,
    'lambda_base': 0.3,
    'sigma_lambda': 0.5,
}


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _exp_fit(set_sizes, values):
    """
    Fit log(values) = intercept + slope · l.

    Returns dict with g_bar = exp(slope), R², and the fit function.
    This fits to MEANS, matching the Jensen's inequality prediction
    E[r^pre] = ḡ^l  (paper §3.5, Eq. 11).
    """
    from scipy import stats as sp
    log_v = np.log(np.maximum(values, 1e-30))
    slope, intercept, r_value, _, _ = sp.linregress(set_sizes, log_v)
    return {
        'g_bar': np.exp(slope),
        'intercept': intercept,
        'slope': slope,
        'r_squared': r_value ** 2,
        'predict': lambda l: np.exp(intercept + slope * np.asarray(l, dtype=float)),
    }


def _band_stats(values):
    """Compute summary statistics for a 1-D array of values."""
    q05, q25, q75, q95 = np.percentile(values, [5, 25, 75, 95])
    mu = np.mean(values)
    return {
        'mean': mu, 'median': np.median(values), 'std': np.std(values),
        'cv': np.std(values) / (mu + 1e-30),
        'q05': q05, 'q25': q25, 'q75': q75, 'q95': q95,
        'iqr': q75 - q25,
    }


# =============================================================================
# PART A — EXPONENTIAL GROWTH  (§3.5)
# =============================================================================
#
# For each neuron, enumerate all C(L,l) location subsets with a fixed
# orientation vector.  Compute r^pre = exp(Σ_k f_{n,k}(θ_k)) and average
# over subsets.  Then average across neurons to get the population mean.
#
# The paper predicts this mean grows as ḡ^l where ḡ = exp(σ²_f / 2).
# We fit to population MEANS (not medians) to match the Jensen's
# inequality prediction.
# =============================================================================

def _run_part_a(population, cfg, F_stacked=None):
    """
    Estimate E[r^pre] at each set size via sampled θ configurations.

    For every C(L,l) location subset we draw n_theta_samples independent
    orientation vectors θ ∈ {0,…,n_θ-1}^l, evaluate r^pre for each
    (subset, θ) pair across all N neurons simultaneously using
    compute_r_pre_at_config, and average.

    This replaces the old call to compute_pre_normalized_response which
    constructed the full n_θ^l tensor product — OOM for l ≥ 4.
    """
    set_sizes = cfg['set_sizes']
    L = cfg['n_locations']
    N = cfg['n_neurons']
    n_theta = cfg['n_orientations']
    n_theta_samples = cfg.get('n_theta_samples_a', 200)
    rng = np.random.default_rng(cfg['seed'] + 1000)

    if F_stacked is None:
        F_stacked = np.stack([neuron['f_samples'] for neuron in population])

    neuron_means = {l: None for l in set_sizes}

    for l in tqdm(set_sizes, desc="  Part A set sizes"):
        # Accumulate r^pre across all (subset, θ) samples — shape (N,)
        running_sum = np.zeros(N)
        n_total = 0
        subsets = list(combinations(range(L), l))
        for subset in tqdm(subsets, desc=f"    l={l} subsets", leave=False):
            for _ in range(n_theta_samples):
                active_theta = rng.integers(0, n_theta, size=l).tolist()
                running_sum += compute_r_pre_at_config(
                    F_stacked, subset, active_theta)
                n_total += 1
        neuron_means[l] = (running_sum / n_total).tolist()

    # Population statistics
    pop_means = {l: np.mean(neuron_means[l]) for l in set_sizes}
    pop_stds = {l: np.std(neuron_means[l]) for l in set_sizes}

    # Exponential fit to population MEANS
    fit = _exp_fit(set_sizes, [pop_means[l] for l in set_sizes])

    return {
        'pop_means': pop_means,
        'pop_stds': pop_stds,
        'neuron_means': neuron_means,
        'fit': fit,
    }


# =============================================================================
# PART B — DENOMINATOR CONCENTRATION  (§5.4)
# =============================================================================
#
# Sample random (S, θ) configurations.  At each configuration, compute
# the denominator D(S,θ) = N⁻¹ Σ_j r^pre_j(S,θ)  (dropping σ² ≈ 0).
#
# The paper predicts D concentrates around ḡ^l with CV ∝ 1/√N.
# We show:
#   - D grows exponentially (same ḡ as Part A)
#   - The scatter (CV) at each l quantifies how well the denominator
#     can be treated as θ-independent — the foundation for efficient
#     decoding (§5, Step 4).
#
# Adaptive sampling: n_subsets × n_theta ≈ n_target at every l,
# so scatter is equally well-estimated across set sizes.
# =============================================================================

def _run_part_b(F_stacked, cfg):
    """Random (S, θ) sampling of the normalisation denominator."""
    from math import comb

    set_sizes = cfg['set_sizes']
    N = cfg['n_neurons']
    L = cfg['n_locations']
    n_target = cfg.get('n_unique_target', 250)
    rng = np.random.default_rng(cfg['seed'] + 2000)

    D_per_l = {}       # D(S,θ) values at each l
    neuron_per_l = {}   # per-neuron averages at each l

    for l in set_sizes:
        n_possible = comb(L, l)
        n_sub = min(n_possible, n_target)
        n_theta = max(1, int(np.ceil(n_target / n_sub)))

        if n_possible <= n_target:
            subsets = list(combinations(range(L), l))
        else:
            all_locs = np.arange(L)
            subsets = [tuple(sorted(rng.choice(all_locs, size=l, replace=False)))
                       for _ in range(n_sub)]

        # Collect r^pre for all (S, θ) samples
        responses = []  # will be (n_samples, N)
        for _ in range(n_theta):
            theta_idx_full = rng.integers(0, cfg['n_orientations'], size=L)
            for subset in subsets:
                active_theta = [theta_idx_full[k] for k in subset]
                responses.append(
                    compute_r_pre_at_config(F_stacked, subset, active_theta))

        responses = np.stack(responses)             # (n_samples, N)
        D_per_l[l] = np.mean(responses, axis=1)     # D at each stimulus
        neuron_per_l[l] = np.mean(responses, axis=0) # per-neuron average

    # Exponential fits to MEANS of D
    D_means = {l: np.mean(D_per_l[l]) for l in set_sizes}
    D_fit = _exp_fit(set_sizes, [D_means[l] for l in set_sizes])

    return {
        'D_per_l': D_per_l,
        'D_stats': {l: _band_stats(D_per_l[l]) for l in set_sizes},
        'D_fit': D_fit,
        'neuron_per_l': neuron_per_l,
        'neuron_stats': {l: _band_stats(neuron_per_l[l]) for l in set_sizes},
    }


# =============================================================================
# PART C — MIXED SELECTIVITY  (§3.3, Def. 4.3)
# =============================================================================
#
# For each neuron, form its response matrix R ∈ R^{n_θ × L} and compute
# the SVD-based separability index S = σ₁² / Σσ²_i.
#
# S = 1  →  perfectly separable (rank-1)
# S < 0.8  →  mixed selective (conjunctive orientation × location tuning)
#
# The paper predicts that location-dependent lengthscales break
# separability, producing S ≪ 1 for most neurons.
# =============================================================================

def _run_part_c(population):
    """SVD-based separability analysis."""
    return summarize_separability(population)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment_1(config):
    cfg = {**DEFAULT_CONFIG, **config}
    N = cfg['n_neurons']
    seed = cfg['seed']

    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: PRE-NORMALISED RESPONSE ANALYSIS")
    print("=" * 70)
    print(f"    N={N}  n_θ={cfg['n_orientations']}  L={cfg['n_locations']}  seed={seed}")

    t0 = time.time()

    population = generate_neuron_population(
        n_neurons=N, n_orientations=cfg['n_orientations'],
        n_locations=cfg['n_locations'], base_lengthscale=cfg['lambda_base'],
        lengthscale_variability=cfg['sigma_lambda'], seed=seed,
        gain_variability=cfg.get('gain_variability', 0.2),
    )
    F_stacked = np.stack([neuron['f_samples'] for neuron in population])

    print("\n  Part A: Exponential growth (sampled θ configurations)...")
    part_a = _run_part_a(population, cfg, F_stacked=F_stacked)
    fit_a = part_a['fit']
    print(f"    ḡ = {fit_a['g_bar']:.3f}  (theory: exp(σ²_f/2) ≈ 1.65)")
    print(f"    R² = {fit_a['r_squared']:.6f}")

    print("\n  Part B: Denominator concentration (random sampling)...")
    part_b = _run_part_b(F_stacked, cfg)
    for l in cfg['set_sizes']:
        cv = part_b['D_stats'][l]['cv']
        print(f"    l={l}: D mean={part_b['D_stats'][l]['mean']:.4f}  CV={cv:.4f}")
    print(f"    D fit: ḡ = {part_b['D_fit']['g_bar']:.3f}  R² = {part_b['D_fit']['r_squared']:.6f}")

    print("\n  Part C: Mixed selectivity (SVD)...")
    part_c = _run_part_c(population)
    print(f"    Mean S = {part_c['separability']['mean']:.3f}")
    print(f"    Mixed: {part_c['classification']['percent_mixed']:.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s")

    return {
        'n_neurons': N, 'seed': seed, 'config': cfg,
        'part_a': part_a,
        'part_b': part_b,
        'part_c': part_c,
        'timing': elapsed,
    }


# =============================================================================
# PLOTTING
# =============================================================================
#
# Three plots, one per validated claim:
#   1. Exponential growth: R_mean vs l with fit (Part A)
#   2. Denominator concentration: D(S,θ) dot-band + CV (Part B)
#   3. Mixed selectivity: separability histogram (Part C)
# =============================================================================

def _setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })


def plot_results(results, output_dir, show_plot=False):
    import matplotlib.pyplot as plt
    _setup_style()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = results['config']
    N = results['n_neurons']
    set_sizes = cfg['set_sizes']
    part_a = results['part_a']
    part_b = results['part_b']
    part_c = results['part_c']

    # ================================================================
    # PLOT 1: Exponential growth of E[r^pre] (Part A)
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.set_yscale('log')

    means = [part_a['pop_means'][l] for l in set_sizes]
    stds = [part_a['pop_stds'][l] for l in set_sizes]
    fit = part_a['fit']

    ax1.errorbar(set_sizes, means, yerr=stds, fmt='o-', color='#E74C3C',
                 lw=2, ms=8, capsize=4, label='Population mean', zorder=5)

    l_fine = np.linspace(min(set_sizes) - 0.5, max(set_sizes) + 0.5, 100)
    ax1.plot(l_fine, fit['predict'](l_fine), '--', color='gray', lw=1.5,
             label=f'Fit: {np.exp(fit["intercept"]):.2f} × {fit["g_bar"]:.3f}$^l$'
                   f'  (R²={fit["r_squared"]:.4f})')

    ax1.set_xlabel('Set size $l$')
    ax1.set_ylabel(r'$\mathbb{E}[r^{\mathrm{pre}}]$  (log scale)')
    ax1.set_title(f'Exponential Growth of Pre-DN Response  (N={N})')
    ax1.set_xticks(set_sizes)
    ax1.legend(fontsize=9)

    # Theory annotation
    ax1.text(0.02, 0.98,
             f"Jensen's inequality (\u00a73.5):\n"
             f"$\\mathbb{{E}}[r^{{\\mathrm{{pre}}}}] = \\bar{{g}}^l$\n"
             f"$\\bar{{g}} = e^{{\\sigma^2_f/2}} \\approx 1.65$\n"
             f"Observed: $\\bar{{g}} = {fit['g_bar']:.3f}$",
             transform=ax1.transAxes, fontsize=9, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='#D5F5E3', ec='#27AE60', alpha=0.9))

    fig1.tight_layout()
    fig1.savefig(out / 'exp1_exponential_growth.png')
    print(f"  Saved: exp1_exponential_growth.png")
    if show_plot: plt.show()
    plt.close(fig1)

    # ================================================================
    # PLOT 2: Denominator concentration (Part B)
    # ================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: D(S,θ) dot-band
    ax2a.set_yscale('log')
    rng_plot = np.random.default_rng(0)
    colors = plt.cm.Set2(np.linspace(0, 1, len(set_sizes)))

    for i, l in enumerate(set_sizes):
        vals = part_b['D_per_l'][l]
        x = l + rng_plot.uniform(-0.2, 0.2, len(vals))
        ax2a.scatter(x, vals, s=12, alpha=0.4, color=colors[i], edgecolors='none')
        stats = part_b['D_stats'][l]
        ax2a.plot([l - 0.3, l + 0.3], [stats['median']] * 2, 'k-', lw=2)
        ax2a.plot([l, l], [stats['q25'], stats['q75']], 'k-', lw=1.5)

    D_fit = part_b['D_fit']
    ax2a.plot(l_fine, D_fit['predict'](l_fine), '--', color='gray', lw=1.5,
              label=f'Fit: \u0121={D_fit["g_bar"]:.3f}  (R\u00b2={D_fit["r_squared"]:.4f})')

    ax2a.set_xlabel('Set size $l$')
    ax2a.set_ylabel('$D(S, \\theta)$  (log scale)')
    ax2a.set_title('Denominator per stimulus')
    ax2a.set_xticks(set_sizes)
    ax2a.legend(fontsize=9)

    # Right: CV of D vs l
    cvs = [part_b['D_stats'][l]['cv'] for l in set_sizes]
    ax2b.bar(set_sizes, cvs, width=0.6, color='#3498DB', alpha=0.7, edgecolor='black')
    for l, cv in zip(set_sizes, cvs):
        ax2b.text(l, cv + 0.005, f'{cv:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax2b.set_xlabel('Set size $l$')
    ax2b.set_ylabel('CV of $D(S, \\theta)$')
    ax2b.set_title('Denominator concentration')

    ax2b.text(0.02, 0.98,
              "\u00a75.4: As N \u2192 \u221e,\n"
              "D \u2192 \u0121$^l$ (deterministic)\n"
              "CV \u221d 1/\u221aN\n\n"
              f"N = {N}: CV range\n"
              f"[{min(cvs):.3f}, {max(cvs):.3f}]",
              transform=ax2b.transAxes, fontsize=9, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#3498DB', alpha=0.9))

    fig2.tight_layout()
    fig2.savefig(out / 'exp1_denominator_concentration.png')
    print(f"  Saved: exp1_denominator_concentration.png")
    if show_plot: plt.show()
    plt.close(fig2)

    # ================================================================
    # PLOT 3: Separability histogram (Part C)
    # ================================================================
    if N > 1:
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        seps = part_c['all_separabilities']
        ax3.hist(seps, bins=20, color='#9B59B6', alpha=0.7, edgecolor='white')
        ax3.axvline(0.8, color='red', ls='--', lw=2, label='S = 0.8 threshold')
        ax3.axvline(np.mean(seps), color='#2E86AB', ls='-', lw=2,
                    label=f'Mean: {np.mean(seps):.3f}')

        ax3.set_xlabel('Separability index $S$')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Mixed Selectivity  ({part_c["classification"]["percent_mixed"]:.0f}% '
                      f'mixed, N={N})')
        ax3.legend(fontsize=9)

        ax3.text(0.02, 0.98,
                 "Def. 4.3: S = \u03c3\u00b9\u00b2/\u03a3\u03c3\u00b2\u1d62\n"
                 "S = 1 \u2192 separable\n"
                 "S < 0.8 \u2192 mixed selective",
                 transform=ax3.transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#F5EEF8', ec='#9B59B6', alpha=0.9))

        fig3.tight_layout()
        fig3.savefig(out / 'exp1_separability.png')
        print(f"  Saved: exp1_separability.png")
        if show_plot: plt.show()
        plt.close(fig3)

    print(f"\n  Experiment 1 plots saved to {out}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1: Pre-Normalised Response')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/exp1')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    config = {
        'n_neurons': args.n_neurons, 'n_orientations': 200, 'n_locations': 8,
        'set_sizes': [2, 4, 6, 8], 'seed': args.seed,
        'lambda_base': 0.3, 'sigma_lambda': 0.5,
    }
    results = run_experiment_1(config)
    if not args.no_plot:
        plot_results(results, args.save_dir, show_plot=True)