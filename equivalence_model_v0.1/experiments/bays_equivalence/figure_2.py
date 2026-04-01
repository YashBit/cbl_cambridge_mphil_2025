"""
Bays (2014) Figure 2 — GP-Based Model Predictions (no human data)

Recreates the MODEL curves from panels c, d, e, f:
    c — Error distributions at set sizes 1, 2, 4, 8
    d — Variance vs set size (power law on log-log)
    e — Deviation from circular normal ("Mexican hat")
    f — Kurtosis vs set size

=============================================================================
KEY DESIGN CHOICE: FULL MULTI-LOCATION FACTORISED DECODER
=============================================================================

Neurons have tuning at MULTIPLE locations, spikes carry ENTANGLED
information, and the decoder MARGINALISES over non-cued items.

Pipeline per trial at set size l:
    1. Generate M neurons with GP tuning at l locations
    2. Sample true orientations θ₁, ..., θₗ
    3. Pre-normalised response: r_i^pre = exp(Σ_k f_{i,k}(θ_k))
    4. DN: r_i = γ · r_i^pre / (σ² + M⁻¹ Σ_j r_j^pre)  [dn_pointwise]
    5. Poisson spikes
    6. Efficient factorised ML decode with marginalisation:
         L_k(θ) = Σ_i n_i · f_{i,k}(θ)
         l_marginal(θ_c) = L_c(θ_c) + Σ_{k≠c} logsumexp(L_k)
         θ̂_c = argmax l_marginal(θ_c)

Complexity: O(M · l · n_θ) per trial — linear in set size.

Usage:
    from experiments.bays_equivalence.figure_2 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.special import logsumexp
import time

from core.encoder.gaussian_process import (
    generate_neuron_population, periodic_rbf_kernel, sample_gp_function,
)
from core.encoder.divisive_normalization import dn_pointwise
from core.encoder.poisson_spike import generate_spikes


def compute_log_likelihood(counts, g, T_d):
    log_g = np.log(np.maximum(g, 1e-30))
    return counts @ log_g - T_d * np.sum(g, axis=0)

def compute_circular_error(theta_true, theta_hat):
    return np.angle(np.exp(1j * (theta_hat - theta_true)))

def circular_variance(errors):
    return 1.0 - np.abs(np.mean(np.exp(1j * errors)))

def circular_kurtosis(errors):
    V = circular_variance(errors)
    rho2 = np.abs(np.mean(np.exp(2j * errors)))
    kappa2 = 1.0 - rho2
    return kappa2 / max(V**2, 1e-15) if V > 1e-10 else 0.0

def circular_moments(errors):
    return {'variance': circular_variance(errors), 'kurtosis': circular_kurtosis(errors),
            'mean_resultant': float(np.abs(np.mean(np.exp(1j * errors))))}

def compute_deviation_from_normal(errors, n_bins=50):
    from scipy.stats import vonmises
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    emp, _ = np.histogram(errors, bins=bin_edges, density=True)
    V = circular_variance(errors)
    kappa_fit = max(0.01, 1.0 / V - 1) if V > 0.01 else 100.0
    vm_pdf = vonmises.pdf(centers, kappa_fit)
    return {'bin_centers': centers, 'empirical': emp, 'normal_fit': vm_pdf,
            'deviation': emp - vm_pdf}

def generate_population(M, n_theta, lengthscale, n_locations=1, seed=42):
    population = generate_neuron_population(
        n_neurons=M, n_orientations=n_theta, n_locations=n_locations,
        base_lengthscale=lengthscale, lengthscale_variability=0.0, seed=seed)
    thetas = population[0]['orientations']
    f_all = []
    for loc in range(n_locations):
        f_loc = np.array([population[n]['f_samples'][loc, :] for n in range(M)])
        f_all.append(f_loc)
    return thetas, f_all

def compute_spike_weighted_log_tuning(counts, f_list):
    return [counts @ f_k for f_k in f_list]

def compute_marginal_log_likelihood_efficient(L_list, cued_idx):
    ll = L_list[cued_idx].copy()
    for k in range(len(L_list)):
        if k != cued_idx:
            ll = ll + logsumexp(L_list[k])
    return ll


def _run_multiloc_trials(
    f_all: List[np.ndarray],
    thetas: np.ndarray,
    active_locs: Tuple[int, ...],
    cued_index: int,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    n_trials: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Run n_trials of multi-location encode → spike → factorised decode.

    Returns errors array, shape (n_trials,).
    """
    l = len(active_locs)
    M, n_theta = f_all[0].shape
    errors = np.empty(n_trials)

    f_active = [f_all[loc] for loc in active_locs]

    for t in range(n_trials):
        # 1. Sample true orientation indices for all l locations
        theta_indices = rng.randint(n_theta, size=l)

        # 2. Pre-normalised response: r_i^pre = exp(Σ_k f_{i,k}(θ_k^true))
        log_r_pre = np.zeros(M)
        for k in range(l):
            log_r_pre += f_active[k][:, theta_indices[k]]
        r_pre = np.exp(log_r_pre)

        # 3. Divisive normalisation (Eq. 6)
        rates = dn_pointwise(r_pre, gamma, sigma_sq)

        # 4. Poisson spikes
        counts = generate_spikes(rates, T_d, rng)

        # 5. Efficient factorised ML decode with marginalisation
        L_list = compute_spike_weighted_log_tuning(counts, f_active)
        ll_marginal = compute_marginal_log_likelihood_efficient(L_list, cued_index)
        idx_hat = np.argmax(ll_marginal)

        # 6. Circular error at cued location
        errors[t] = compute_circular_error(
            thetas[theta_indices[cued_index]], thetas[idx_hat]
        )

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: Dict) -> Dict:
    """
    Fix (λ_base, γ), sweep set sizes using FULL multi-location pipeline.

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

    max_locs = max(set_sizes)

    t0 = time.time()
    all_seeds = []

    for s in range(n_seeds):
        current_seed = seed + s * 1000

        thetas, f_all = generate_population(M, n_theta, lam, max_locs, current_seed)

        seed_data = {}
        for N in set_sizes:
            print(f"  seed={s} N={N}...", end=" ", flush=True)

            active_locs = tuple(range(N))
            cued_index = 0

            rng = np.random.RandomState(current_seed + N)
            errors = _run_multiloc_trials(
                f_all, thetas, active_locs, cued_index,
                gamma, T_d, sigma_sq, n_trials, rng,
            )
            dev = compute_deviation_from_normal(errors, n_bins)
            seed_data[N] = {
                'errors': errors,
                'variance': circular_variance(errors),
                'kurtosis': circular_kurtosis(errors),
                'deviation': dev,
            }
            print(f"var={seed_data[N]['variance']:.4f} "
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
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """2-row × 5-column figure matching Bays (2014) Fig 2 layout."""
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
    print("  Decoder: Efficient factorised ML with marginalisation")
    results = run_experiment(config)
    plot_results(results, 'results/figure_2', show_plot=True)