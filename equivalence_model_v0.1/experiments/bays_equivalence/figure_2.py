"""
Bays (2014) Figure 2 — GP-Based Model Predictions (no human data)

Recreates the MODEL curves (red lines) from panels c, d, e, f:
    c — Error distributions at set sizes 1, 2, 4, 8
    d — Variance vs set size (power law on log-log)
    e — Deviation from circular normal ("Mexican hat")
    f — Kurtosis vs set size

DN enters here: effective gain per item = γ / N.
Fixed parameters (λ_base, γ) play the role of Bays's (ω, γ).
Multiple seeds → ±1 SE bands (matching Bays's dashed lines).

Usage:
    from experiments.bays_equivalence.figure_2 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import ive
from scipy.optimize import brentq
from pathlib import Path
from typing import Dict
import time

# ── Core module imports ──
from core.gaussian_process import periodic_rbf_kernel, sample_gp_function
from core.poisson_spike import generate_spikes
from core.ml_decoder import compute_circular_error

# ── Shared utilities from figure_1 (DRY) ──
from experiments.bays_equivalence.figure_1 import (
    _generate_population,
    _run_trials_at_gain,
    circular_variance,
    circular_kurtosis,
    _circular_moments,
)


# ═══════════════════════════════════════════════════════════════════════════
# DEVIATION FROM CIRCULAR NORMAL (Figure 2e specific)
# ═══════════════════════════════════════════════════════════════════════════

def _bessel_ratio(kappa: float) -> float:
    """I₁(κ)/I₀(κ) using exponentially scaled Bessels (overflow-safe)."""
    return ive(1, kappa) / ive(0, kappa)


def _rho_to_kappa(rho: float) -> float:
    """Invert A(κ) = I₁(κ)/I₀(κ) = ρ to find von Mises concentration κ."""
    if rho < 1e-10:
        return 0.0
    if rho > 1 - 1e-10:
        return 1e4
    return brentq(lambda k: _bessel_ratio(k) - rho, 1e-8, 1e4)


def _von_mises_pdf(theta: np.ndarray, kappa: float) -> np.ndarray:
    """Von Mises PDF: p(θ; κ) = exp(κ·cos(θ)) / (2π·I₀(κ)), overflow-safe."""
    # log p = κ·cos(θ) − log(2π) − log(I₀(κ))
    # log(I₀(κ)) = κ + log(ive(0, κ))
    log_p = kappa * np.cos(theta) - np.log(2 * np.pi) - kappa - np.log(ive(0, kappa))
    return np.exp(log_p)


def compute_deviation_from_normal(
    errors: np.ndarray, n_bins: int = 50
) -> Dict:
    """
    Compute deviation of error histogram from matched circular normal.

    1. Histogram errors into n_bins bins → empirical density
    2. Compute ρ₁ = |m₁| from errors
    3. Find von Mises κ matching that ρ₁
    4. Evaluate von Mises PDF at bin centers
    5. Deviation = empirical - von Mises

    Returns dict with bin_centers, empirical, von_mises, deviation.
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts, _ = np.histogram(errors, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    empirical = counts / (len(errors) * bin_width)  # density

    m1, _ = _circular_moments(errors)
    rho1 = np.abs(m1)
    kappa = _rho_to_kappa(rho1)
    von_mises = _von_mises_pdf(bin_centers, kappa)

    return {
        'bin_centers': bin_centers,
        'empirical': empirical,
        'von_mises': von_mises,
        'deviation': empirical - von_mises,
        'kappa': kappa,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: Dict) -> Dict:
    """
    Fix (λ_base, γ), sweep set sizes. DN: effective gain = γ/N.

    Config keys
    -----------
    M            : neurons per population          (default 100)
    n_theta      : orientation bins                (default 64)
    n_trials     : trials per condition            (default 10_000)
    T_d          : decoding window (s)             (default 0.1)
    sigma_sq     : semi-saturation constant        (default 1e-6)
    lambda_base  : GP lengthscale (≡ ω in Bays)   (default 0.5)
    gamma        : gain constant (Hz)              (default 100.0)
    set_sizes    : list of N values                (default [1,2,4,8])
    seed         : master seed                     (default 42)
    n_seeds      : seeds for SE bands              (default 5)
    n_bins       : histogram bins                  (default 50)
    """
    M          = config.get('M', 100)
    n_theta    = config.get('n_theta', 64)
    n_trials   = config.get('n_trials', 10_000)
    T_d        = config.get('T_d', 0.1)
    sigma_sq   = config.get('sigma_sq', 1e-6)
    lam        = config.get('lambda_base', 0.5)
    gamma      = config.get('gamma', 100.0)
    set_sizes  = config.get('set_sizes', [1, 2, 4, 8])
    seed       = config.get('seed', 42)
    n_seeds    = config.get('n_seeds', 5)
    n_bins     = config.get('n_bins', 50)

    t0 = time.time()
    all_seeds = []  # list of dicts, one per seed

    for s in range(n_seeds):
        current_seed = seed + s * 1000
        thetas, g, log_g = _generate_population(M, n_theta, lam, current_seed)

        seed_data = {}
        for N in set_sizes:
            effective_gamma = gamma / N  # ← DN consequence
            rng = np.random.RandomState(current_seed + N)
            errors = _run_trials_at_gain(
                g, log_g, thetas, effective_gamma, T_d, sigma_sq, n_trials, rng
            )
            dev = compute_deviation_from_normal(errors, n_bins)
            seed_data[N] = {
                'errors': errors,
                'variance': circular_variance(errors),
                'kurtosis': circular_kurtosis(errors),
                'deviation': dev,
            }
            print(f"  seed={s} N={N}: var={seed_data[N]['variance']:.4f} "
                  f"kurt={seed_data[N]['kurtosis']:.2f}")

        all_seeds.append(seed_data)

    # ── Aggregate across seeds ──
    summary = {}
    for N in set_sizes:
        vars_   = [sd[N]['variance'] for sd in all_seeds]
        kurts_  = [sd[N]['kurtosis'] for sd in all_seeds]
        summary[N] = {
            'variance_mean': np.mean(vars_),
            'variance_se':   np.std(vars_, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0,
            'kurtosis_mean': np.mean(kurts_),
            'kurtosis_se':   np.std(kurts_, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0,
        }
        # Mean deviation curve (average across seeds)
        devs = np.array([sd[N]['deviation']['deviation'] for sd in all_seeds])
        emps = np.array([sd[N]['deviation']['empirical'] for sd in all_seeds])
        summary[N]['deviation_mean'] = np.mean(devs, axis=0)
        summary[N]['deviation_se']   = (np.std(devs, axis=0, ddof=1) / np.sqrt(n_seeds)
                                        if n_seeds > 1 else np.zeros_like(devs[0]))
        summary[N]['empirical_mean'] = np.mean(emps, axis=0)
        summary[N]['empirical_se']   = (np.std(emps, axis=0, ddof=1) / np.sqrt(n_seeds)
                                        if n_seeds > 1 else np.zeros_like(emps[0]))

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({n_seeds} seeds × {len(set_sizes)} set sizes "
          f"× {n_trials} trials)")

    return {
        'set_sizes': set_sizes,
        'summary': summary,
        'all_seeds': all_seeds,
        'bin_centers': all_seeds[0][set_sizes[0]]['deviation']['bin_centers'],
        'config': config,
        'elapsed_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING — Matching Bays (2014) Fig 2 layout (model curves only)
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """
    2-row × 5-column figure:
        Row 1: error distributions at N=1,2,4,8  |  variance vs items
        Row 2: deviation from normal at N=1,2,4,8 |  kurtosis vs items
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_sizes  = results['set_sizes']
    summary    = results['summary']
    bins       = results['bin_centers']
    n_ss       = len(set_sizes)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(2, n_ss + 1, width_ratios=[1]*n_ss + [1.3],
                           hspace=0.4, wspace=0.35,
                           left=0.05, right=0.97, bottom=0.08, top=0.90)

    RED = '#CC2222'

    # ── Row 1: Error distributions + Variance ──
    for i, N in enumerate(set_sizes):
        ax = fig.add_subplot(gs[0, i])
        emp = summary[N]['empirical_mean']
        ax.plot(bins, emp, color=RED, linewidth=1.5)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, max(emp) * 1.15)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel('error')
        if i == 0:
            ax.set_ylabel('probability\ndensity')
        ax.set_title(f'{N}', fontsize=13, fontweight='bold')
        if i == 0:
            ax.text(-0.22, 1.08, r'$\mathbf{c}$', transform=ax.transAxes,
                    fontsize=15, fontweight='bold', va='top')

    # Variance vs items (log-log)
    ax_var = fig.add_subplot(gs[0, n_ss])
    ns = np.array(set_sizes, dtype=float)
    v_mean = np.array([summary[N]['variance_mean'] for N in set_sizes])
    v_se   = np.array([summary[N]['variance_se'] for N in set_sizes])
    ax_var.plot(ns, v_mean, 'o-', color=RED, linewidth=1.5, markersize=5)
    ax_var.fill_between(ns, v_mean - v_se, v_mean + v_se, color=RED, alpha=0.15)
    ax_var.plot(ns, v_mean - v_se, '--', color=RED, linewidth=0.7, alpha=0.6)
    ax_var.plot(ns, v_mean + v_se, '--', color=RED, linewidth=0.7, alpha=0.6)
    ax_var.set_xscale('log', base=2)
    ax_var.set_yscale('log', base=2)
    ax_var.set_xticks(set_sizes)
    ax_var.set_xticklabels([str(n) for n in set_sizes])
    ax_var.set_xlabel('items')
    ax_var.set_ylabel(r'variance ($\sigma^2$)')
    ax_var.text(-0.18, 1.08, r'$\mathbf{d}$', transform=ax_var.transAxes,
                fontsize=15, fontweight='bold', va='top')

    # ── Row 2: Deviation from normal + Kurtosis ──
    for i, N in enumerate(set_sizes):
        ax = fig.add_subplot(gs[1, i])
        dev = summary[N]['deviation_mean']
        dev_se = summary[N]['deviation_se']
        ax.plot(bins, dev, color=RED, linewidth=1.5)
        ax.fill_between(bins, dev - dev_se, dev + dev_se, color=RED, alpha=0.12)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_xlim(-np.pi, np.pi)
        y_lim = max(np.max(np.abs(dev)) * 1.4, 0.05)
        ax.set_ylim(-y_lim * 0.6, y_lim)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel('error')
        if i == 0:
            ax.set_ylabel(r'$\Delta$ probability' + '\ndensity')
        ax.set_title(f'{N}', fontsize=13, fontweight='bold')
        if i == 0:
            ax.text(-0.22, 1.08, r'$\mathbf{e}$', transform=ax.transAxes,
                    fontsize=15, fontweight='bold', va='top')

    # Kurtosis vs items
    ax_kur = fig.add_subplot(gs[1, n_ss])
    k_mean = np.array([summary[N]['kurtosis_mean'] for N in set_sizes])
    k_se   = np.array([summary[N]['kurtosis_se'] for N in set_sizes])
    ax_kur.plot(ns, k_mean, 'o-', color=RED, linewidth=1.5, markersize=5)
    ax_kur.fill_between(ns, k_mean - k_se, k_mean + k_se, color=RED, alpha=0.15)
    ax_kur.plot(ns, k_mean - k_se, '--', color=RED, linewidth=0.7, alpha=0.6)
    ax_kur.plot(ns, k_mean + k_se, '--', color=RED, linewidth=0.7, alpha=0.6)
    ax_kur.set_xscale('log', base=2)
    ax_kur.set_yscale('log', base=2)
    ax_kur.set_xticks(set_sizes)
    ax_kur.set_xticklabels([str(n) for n in set_sizes])
    ax_kur.set_xlabel('items')
    ax_kur.set_ylabel('kurtosis')
    ax_kur.text(-0.18, 1.08, r'$\mathbf{f}$', transform=ax_kur.transAxes,
                fontsize=15, fontweight='bold', va='top')

    fig.suptitle('GP Population Coding — Bays (2014) Fig 2 Model Predictions',
                 fontsize=13, fontweight='bold', y=0.97)

    outpath = Path(output_dir) / 'figure_2_model.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot:
        plt.show()
    plt.close(fig)

    # Save data
    np.savez(
        Path(output_dir) / 'figure_2_data.npz',
        set_sizes=set_sizes, bin_centers=bins,
        **{f'variance_N{N}': summary[N]['variance_mean'] for N in set_sizes},
        **{f'kurtosis_N{N}': summary[N]['kurtosis_mean'] for N in set_sizes},
    )


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = {
        'M': 100, 'n_theta': 64, 'n_trials': 10_000,
        'T_d': 0.1, 'sigma_sq': 1e-6,
        'lambda_base': 0.5, 'gamma': 100.0,
        'set_sizes': [1, 2, 4, 8], 'seed': 42, 'n_seeds': 5,
    }
    print("Running Bays (2014) Figure 2 — GP Model Predictions")
    results = run_experiment(config)
    plot_results(results, 'results/figure_2', show_plot=True)