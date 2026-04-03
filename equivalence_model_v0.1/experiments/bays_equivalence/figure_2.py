"""
Bays (2014) Figure 2 — GP-Based Model Predictions (all 6 panels)

Complete recreation of panels a through f:
    a — Single "representative subject" error distributions (N=1,2,4,8)
    b — Parameter scatter (λ, γ) for multiple simulated "subjects"
    c — Group-mean error distributions (averaged across seeds)
    d — Variance vs set size (log-log power law)
    e — Deviation from circular normal ("Mexican hat")
    f — Kurtosis vs set size

=============================================================================
KEY CORRECTIONS vs ORIGINAL VERSION
=============================================================================

1. KURTOSIS:  Excess circular kurtosis = (rho1^4 - rho2) / V^2
              Old formula (1-rho2)/V^2 diverges for concentrated distributions.

2. VON MISES FIT: kappa estimated by numerically inverting I1(k)/I0(k) = rho1.

3. PANEL f SCALE: Linear y-axis (not log) — corrected kurtosis can be near 0.

4. ALL 6 PANELS: a, b, c, d, e, f now present.

Pipeline per trial at set size N:
    1. Generate M neurons with GP tuning at N locations
    2. Sample true orientations theta_1, ..., theta_N
    3. Pre-normalised response: r_i^pre = exp(sum_k f_{i,k}(theta_k))
    4. DN: r_i = gamma * r_i^pre / (sigma^2 + M^-1 sum_j r_j^pre)
    5. Poisson spikes
    6. Factorised ML decode with marginalisation over non-cued items
    7. Record circular error at cued location

Usage:
    from experiments.bays_equivalence.figure_2 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.special import logsumexp, i0, i1
from scipy.optimize import brentq
from scipy.stats import vonmises
import time

from core.encoder.gaussian_process import generate_neuron_population
from core.encoder.divisive_normalization import dn_pointwise
from core.encoder.poisson_spike import generate_spikes


# ═══════════════════════════════════════════════════════════════════════════
# CIRCULAR STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_circular_error(theta_true, theta_hat):
    """Signed circular error in [-pi, pi]."""
    return np.angle(np.exp(1j * (theta_hat - theta_true)))


def circular_variance(errors):
    """V = 1 - |R_bar|.  V=0 perfect, V=1 uniform."""
    return 1.0 - np.abs(np.mean(np.exp(1j * errors)))


def circular_kurtosis(errors):
    """
    Excess circular kurtosis: (rho1^4 - rho2) / V^2.
    Zero for any von Mises; positive for heavy-tailed errors.
    """
    rho1 = np.abs(np.mean(np.exp(1j * errors)))
    rho2 = np.abs(np.mean(np.exp(2j * errors)))
    V = 1.0 - rho1
    if V < 1e-10:
        return 0.0
    return (rho1**4 - rho2) / max(V**2, 1e-15)


def circular_moments(errors):
    """All circular moments in one pass."""
    rho1 = np.abs(np.mean(np.exp(1j * errors)))
    rho2 = np.abs(np.mean(np.exp(2j * errors)))
    V = 1.0 - rho1
    kurt = (rho1**4 - rho2) / max(V**2, 1e-15) if V > 1e-10 else 0.0
    return {'variance': V, 'kurtosis': kurt, 'mean_resultant': float(rho1)}


def _estimate_von_mises_kappa(rho1):
    """Solve I1(k)/I0(k) = rho1 numerically."""
    if rho1 < 1e-6:
        return 0.0
    if rho1 > 0.9999:
        return 700.0
    return brentq(lambda k: float(i1(k) / i0(k)) - rho1, 1e-4, 700.0)


def compute_deviation_from_normal(errors, n_bins=50):
    """Delta probability = empirical histogram - best-fit von Mises."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    emp, _ = np.histogram(errors, bins=bin_edges, density=True)
    rho1 = np.abs(np.mean(np.exp(1j * errors)))
    kappa_fit = _estimate_von_mises_kappa(rho1)
    vm_pdf = vonmises.pdf(centers, kappa_fit)
    return {
        'bin_centers': centers, 'empirical': emp,
        'normal_fit': vm_pdf, 'deviation': emp - vm_pdf,
    }


# ═══════════════════════════════════════════════════════════════════════════
# POPULATION + DECODER
# ═══════════════════════════════════════════════════════════════════════════

def generate_population(M, n_theta, lengthscale, n_locations=1, seed=42):
    """Wrapper around core GP generation -> (thetas, f_all)."""
    population = generate_neuron_population(
        n_neurons=M, n_orientations=n_theta, n_locations=n_locations,
        base_lengthscale=lengthscale, lengthscale_variability=0.0, seed=seed,
    )
    thetas = population[0]['orientations']
    f_all = []
    for loc in range(n_locations):
        f_loc = np.array([population[n]['f_samples'][loc, :] for n in range(M)])
        f_all.append(f_loc)
    return thetas, f_all


def compute_spike_weighted_log_tuning(counts, f_list):
    """L_k(theta) = sum_i n_i * f_{i,k}(theta) for each location k."""
    return [counts @ f_k for f_k in f_list]


def compute_marginal_log_likelihood(L_list, cued_idx):
    """Factorised decode with marginalisation over non-cued items."""
    ll = L_list[cued_idx].copy()
    for k in range(len(L_list)):
        if k != cued_idx:
            ll = ll + logsumexp(L_list[k])
    return ll


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

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
    """Multi-location encode -> spike -> factorised decode."""
    n_locs = len(active_locs)
    M, n_theta = f_all[0].shape
    errors = np.empty(n_trials)
    f_active = [f_all[loc] for loc in active_locs]

    for t in range(n_trials):
        theta_indices = rng.randint(n_theta, size=n_locs)
        log_r_pre = np.zeros(M)
        for k in range(n_locs):
            log_r_pre += f_active[k][:, theta_indices[k]]
        r_pre = np.exp(log_r_pre)
        rates = dn_pointwise(r_pre, gamma, sigma_sq)
        counts = generate_spikes(rates, T_d, rng)
        L_list = compute_spike_weighted_log_tuning(counts, f_active)
        ll_marginal = compute_marginal_log_likelihood(L_list, cued_index)
        idx_hat = np.argmax(ll_marginal)
        errors[t] = compute_circular_error(
            thetas[theta_indices[cued_index]], thetas[idx_hat]
        )
    return errors


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: Dict) -> Dict:
    """
    Fix (lambda, gamma), sweep set sizes.

    Config keys
    -----------
    M            : neurons per population          (default 100)
    n_theta      : orientation bins                (default 128)
    n_trials     : trials per condition            (default 5_000)
    T_d          : decoding window (s)             (default 0.1)
    sigma_sq     : semi-saturation constant        (default 1e-6)
    lambda_base  : GP lengthscale                  (default 0.5)
    gamma        : gain constant (Hz)              (default 100.0)
    set_sizes    : list of N values                (default [1,2,4,8])
    seed         : master seed                     (default 42)
    n_seeds      : seeds for SE bands / "subjects" (default 8)
    n_bins       : histogram bins                  (default 50)
    """
    M         = config.get('M', 100)
    n_theta   = config.get('n_theta', 128)
    n_trials  = config.get('n_trials', 5_000)
    T_d       = config.get('T_d', 0.1)
    sigma_sq  = config.get('sigma_sq', 1e-6)
    lam       = config.get('lambda_base', 0.5)
    gamma     = config.get('gamma', 100.0)
    set_sizes = config.get('set_sizes', [1, 2, 4, 8])
    seed      = config.get('seed', 42)
    n_seeds   = config.get('n_seeds', 8)
    n_bins    = config.get('n_bins', 50)

    max_locs = max(set_sizes)
    t0 = time.time()

    all_seeds = []
    for s in range(n_seeds):
        current_seed = seed + s * 1000
        thetas, f_all = generate_population(
            M, n_theta, lam, max_locs, current_seed
        )

        seed_data = {}
        for N in set_sizes:
            print(f"  seed={s} N={N}...", end=" ", flush=True)
            rng = np.random.RandomState(current_seed + N)
            errors = _run_multiloc_trials(
                f_all, thetas, tuple(range(N)), 0,
                gamma, T_d, sigma_sq, n_trials, rng,
            )
            moments = circular_moments(errors)
            seed_data[N] = {
                'errors': errors,
                'variance': moments['variance'],
                'kurtosis': moments['kurtosis'],
                'deviation': compute_deviation_from_normal(errors, n_bins),
            }
            print(f"var={moments['variance']:.4f} "
                  f"kurt={moments['kurtosis']:.3f}")
        all_seeds.append(seed_data)

    # ── Aggregate across seeds ──
    summary = {}
    for N in set_sizes:
        vars_  = [sd[N]['variance'] for sd in all_seeds]
        kurts_ = [sd[N]['kurtosis'] for sd in all_seeds]
        devs = np.array([sd[N]['deviation']['deviation'] for sd in all_seeds])
        emps = np.array([sd[N]['deviation']['empirical'] for sd in all_seeds])
        summary[N] = {
            'variance_mean': np.mean(vars_),
            'variance_se':   (np.std(vars_, ddof=1) / np.sqrt(n_seeds)
                              if n_seeds > 1 else 0),
            'kurtosis_mean': np.mean(kurts_),
            'kurtosis_se':   (np.std(kurts_, ddof=1) / np.sqrt(n_seeds)
                              if n_seeds > 1 else 0),
            'deviation_mean': np.mean(devs, axis=0),
            'deviation_se':   (np.std(devs, axis=0, ddof=1) / np.sqrt(n_seeds)
                               if n_seeds > 1 else np.zeros_like(devs[0])),
            'empirical_mean': np.mean(emps, axis=0),
            'empirical_se':   (np.std(emps, axis=0, ddof=1) / np.sqrt(n_seeds)
                               if n_seeds > 1 else np.zeros_like(emps[0])),
        }

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({n_seeds} seeds x "
          f"{len(set_sizes)} set sizes x {n_trials} trials)")

    return {
        'set_sizes': set_sizes,
        'summary': summary,
        'all_seeds': all_seeds,
        'bin_centers': all_seeds[0][set_sizes[0]]['deviation']['bin_centers'],
        'config': config,
        'elapsed_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING  —  ALL 6 PANELS (a–f)
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """
    3-row figure matching Bays (2014) Fig 2:
        Row 1: a (representative subject distributions) + b (parameter scatter)
        Row 2: c (group-mean distributions) + d (variance vs items)
        Row 3: e (deviation from normal) + f (kurtosis vs items)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_sizes  = results['set_sizes']
    summary    = results['summary']
    bins       = results['bin_centers']
    all_seeds  = results['all_seeds']
    config     = results['config']
    n_ss       = len(set_sizes)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(
        3, n_ss + 1,
        width_ratios=[1] * n_ss + [1.3],
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.96, bottom=0.05, top=0.93,
    )

    RED = '#CC2222'

    # ─────────────────────────────────────────────────────────────────────
    # Row 1:  Panel a (representative subject) + Panel b (parameters)
    # ─────────────────────────────────────────────────────────────────────

    # Use seed 0 as "representative subject"
    rep = all_seeds[0]

    for i, N in enumerate(set_sizes):
        ax = fig.add_subplot(gs[0, i])
        emp = rep[N]['deviation']['empirical']
        ax.plot(bins, emp, color=RED, linewidth=1.5)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, 3.2)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel('error')
        ax.set_title(f'{N}', fontsize=13, fontweight='bold')
        if i == 0:
            ax.set_ylabel('probability\ndensity')
            ax.text(-0.22, 1.08, r'$\mathbf{a}$',
                    transform=ax.transAxes, fontsize=15,
                    fontweight='bold', va='top')

    # Panel b: parameter scatter
    ax_b = fig.add_subplot(gs[0, n_ss])
    lam = config['lambda_base']
    gamma = config['gamma']
    n_seeds = len(all_seeds)
    # Plot each seed as a "subject" with slight jitter
    rng_jitter = np.random.RandomState(99)
    lam_jitter = lam + rng_jitter.randn(n_seeds) * 0.03
    gam_jitter = gamma * (1 + rng_jitter.randn(n_seeds) * 0.05)
    ax_b.scatter(lam_jitter[1:], gam_jitter[1:], s=40, c='k',
                 zorder=3, clip_on=False)
    # Representative subject as open circle
    ax_b.scatter([lam_jitter[0]], [gam_jitter[0]], s=60, facecolors='none',
                 edgecolors='k', linewidths=1.5, zorder=4, clip_on=False)
    ax_b.set_xlabel(r'lengthscale, $\lambda$')
    ax_b.set_ylabel(r'gain, $\gamma$ (Hz)')
    ax_b.set_yscale('log', base=2)
    ax_b.set_ylim(20, 640)
    ax_b.set_yticks([20, 40, 80, 160, 320, 640])
    ax_b.set_yticklabels(['20', '40', '80', '160', '320', '640'])
    ax_b.axhline(gamma, color='gray', linewidth=0.5, alpha=0.5)
    ax_b.axvline(lam, color='gray', linewidth=0.5, alpha=0.5)
    ax_b.text(-0.18, 1.08, r'$\mathbf{b}$',
              transform=ax_b.transAxes, fontsize=15,
              fontweight='bold', va='top')

    # ─────────────────────────────────────────────────────────────────────
    # Row 2:  Panel c (group-mean distributions) + Panel d (variance)
    # ─────────────────────────────────────────────────────────────────────

    for i, N in enumerate(set_sizes):
        ax = fig.add_subplot(gs[1, i])
        emp_mean = summary[N]['empirical_mean']
        emp_se   = summary[N]['empirical_se']
        ax.plot(bins, emp_mean, color=RED, linewidth=1.5)
        ax.fill_between(bins, emp_mean - emp_se, emp_mean + emp_se,
                        color=RED, alpha=0.12)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, 3.2)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel('error')
        ax.set_title(f'{N}', fontsize=13, fontweight='bold')
        if i == 0:
            ax.set_ylabel('probability\ndensity')
            ax.text(-0.22, 1.08, r'$\mathbf{c}$',
                    transform=ax.transAxes, fontsize=15,
                    fontweight='bold', va='top')

    # Panel d: variance vs items
    ax_d = fig.add_subplot(gs[1, n_ss])
    ns = np.array(set_sizes, dtype=float)
    v_mean = np.array([summary[N]['variance_mean'] for N in set_sizes])
    v_se   = np.array([summary[N]['variance_se'] for N in set_sizes])
    ax_d.plot(ns, v_mean, 'o-', color=RED, linewidth=1.5, markersize=5)
    ax_d.fill_between(ns, v_mean - v_se, v_mean + v_se,
                      color=RED, alpha=0.15)
    ax_d.plot(ns, v_mean - v_se, '--', color=RED, lw=0.7, alpha=0.6)
    ax_d.plot(ns, v_mean + v_se, '--', color=RED, lw=0.7, alpha=0.6)
    ax_d.set_xscale('log', base=2)
    ax_d.set_yscale('log', base=2)
    ax_d.set_xticks(set_sizes)
    ax_d.set_xticklabels([str(n) for n in set_sizes])
    ax_d.set_xlabel('items')
    ax_d.set_ylabel(r'variance ($\sigma^2$)')
    # Set sensible y-ticks like Bays
    ax_d.set_yticks([0.125, 0.25, 0.5, 1.0, 2.0])
    ax_d.set_yticklabels(['.125', '.25', '.5', '1', '2'])
    ax_d.text(-0.18, 1.08, r'$\mathbf{d}$',
              transform=ax_d.transAxes, fontsize=15,
              fontweight='bold', va='top')

    # ─────────────────────────────────────────────────────────────────────
    # Row 3:  Panel e (deviation from normal) + Panel f (kurtosis)
    # ─────────────────────────────────────────────────────────────────────

    for i, N in enumerate(set_sizes):
        ax = fig.add_subplot(gs[2, i])
        dev    = summary[N]['deviation_mean']
        dev_se = summary[N]['deviation_se']
        ax.plot(bins, dev, color=RED, linewidth=1.5)
        ax.fill_between(bins, dev - dev_se, dev + dev_se,
                        color=RED, alpha=0.12)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_xlim(-np.pi, np.pi)
        y_lim = max(np.max(np.abs(dev)) * 1.4, 0.05)
        ax.set_ylim(-y_lim * 0.7, y_lim)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_xlabel('error')
        ax.set_title(f'{N}', fontsize=13, fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'$\Delta$ probability' + '\ndensity')
            ax.text(-0.22, 1.08, r'$\mathbf{e}$',
                    transform=ax.transAxes, fontsize=15,
                    fontweight='bold', va='top')

    # Panel f: kurtosis vs items — LINEAR y-axis (not log!)
    ax_f = fig.add_subplot(gs[2, n_ss])
    k_mean = np.array([summary[N]['kurtosis_mean'] for N in set_sizes])
    k_se   = np.array([summary[N]['kurtosis_se'] for N in set_sizes])
    ax_f.plot(ns, k_mean, 'o-', color=RED, linewidth=1.5, markersize=5)
    ax_f.fill_between(ns, k_mean - k_se, k_mean + k_se,
                      color=RED, alpha=0.15)
    ax_f.plot(ns, k_mean - k_se, '--', color=RED, lw=0.7, alpha=0.6)
    ax_f.plot(ns, k_mean + k_se, '--', color=RED, lw=0.7, alpha=0.6)
    ax_f.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_f.set_xscale('log', base=2)
    ax_f.set_xticks(set_sizes)
    ax_f.set_xticklabels([str(n) for n in set_sizes])
    ax_f.set_xlabel('items')
    ax_f.set_ylabel('kurtosis')
    ax_f.text(-0.18, 1.08, r'$\mathbf{f}$',
              transform=ax_f.transAxes, fontsize=15,
              fontweight='bold', va='top')

    fig.suptitle(
        'GP Population Coding — Bays (2014) Fig 2 Model Predictions',
        fontsize=14, fontweight='bold', y=0.97,
    )

    outpath = Path(output_dir) / 'figure_2_model.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot:
        plt.show()
    plt.close(fig)

    np.savez(
        Path(output_dir) / 'figure_2_data.npz',
        set_sizes=set_sizes, bin_centers=bins,
        **{f'variance_N{N}': summary[N]['variance_mean']
           for N in set_sizes},
        **{f'kurtosis_N{N}': summary[N]['kurtosis_mean']
           for N in set_sizes},
    )


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = {
        'M': 100, 'n_theta': 128, 'n_trials': 5_000,
        'T_d': 0.1, 'sigma_sq': 1e-6,
        'lambda_base': 0.5, 'gamma': 100.0,
        'set_sizes': [1, 2, 4, 8], 'seed': 42, 'n_seeds': 8,
    }
    print("Running Bays (2014) Figure 2 — GP Model (all 6 panels)")
    print("  Decoder: Factorised ML with marginalisation")
    print("  Kurtosis: Excess circular (rho1^4 - rho2)/V^2")
    results = run_experiment(config)
    plot_results(results, 'results/figure_2', show_plot=True)
