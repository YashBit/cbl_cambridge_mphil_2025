"""
Bays (2014) Figure 3 — GP-Based Equivalent

Recreates all four panels:
    a — Error distributions: all trials, cued only, uncued only (N=2,4,8)
    b — Variance vs set size (cued vs uncued)
    c — Kurtosis vs set size (cued vs uncued)
    d — Optimal vs empirical weighting factors

=============================================================================
KEY EXTENSION FROM FIGURE 2: WEIGHTED POPULATION ENCODING
=============================================================================

Figure 2 encoded all locations with equal weight (α_k = 1 for all k).
Figure 3 introduces DIFFERENTIAL WEIGHTING via exponent weighting:

    Encoding:
        log r_i^pre = Σ_k α_k · f_{i,k}(θ_k^true)

    DN normalisation:
        r_i = γ · r_i^pre / (σ² + M⁻¹ Σ_j r_j^pre)   [dn_pointwise]

    Decoding (weighted factorised ML with marginalisation):
        L_k(θ) = Σ_i n_i · f_{i,k}(θ)
        l_marginal(θ_p) = α_p · L_p(θ_p) + Σ_{k≠p} logsumexp(α_k · L_k)

OPTIMAL WEIGHT COMPUTATION (Panel d):
    Sweep α_cued ∈ [1, α_max], find α* = argmin V_total(α)
    where V_total = p_cued · Var_cued + (N-1) · p_uncued · Var_uncued

Usage:
    from experiments.bays_equivalence.figure_3 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import logsumexp
from pathlib import Path
from typing import Dict, List, Tuple
import time

from core.encoder.poisson_spike import generate_spikes
from core.encoder.divisive_normalization import dn_pointwise
from core.decoder.ml_decoder import (
    compute_spike_weighted_log_tuning,
    compute_circular_error,
)

from experiments.bays_equivalence.bays_utils import (
    circular_variance,
    circular_kurtosis,
    compute_deviation_from_normal,
    generate_population,
)


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED MARGINAL LOG-LIKELIHOOD (figure-3 specific)
# ═══════════════════════════════════════════════════════════════════════════

def compute_weighted_marginal_log_likelihood(
    L_per_location: List[np.ndarray],
    alpha_weights: np.ndarray,
    probed_location: int,
) -> np.ndarray:
    """
    Marginal log-likelihood with known location weights.

        l(θ₁,...,θₗ) = Σ_k α_k · L_k(θ_k)

    Marginalising:
        l_marginal(θ_p) = α_p · L_p(θ_p) + Σ_{k≠p} logsumexp(α_k · L_k)
    """
    l = len(L_per_location)
    if l == 1:
        return alpha_weights[0] * L_per_location[0].copy()

    ll_marginal = alpha_weights[probed_location] * L_per_location[probed_location]
    for k in range(l):
        if k != probed_location:
            ll_marginal = ll_marginal + logsumexp(alpha_weights[k] * L_per_location[k])
    return ll_marginal


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED TRIAL ENGINE (figure-3 specific)
# ═══════════════════════════════════════════════════════════════════════════

def _run_weighted_trials(
    f_all: List[np.ndarray],
    thetas: np.ndarray,
    active_locs: Tuple[int, ...],
    cued_loc_in_active: int,
    probed_loc_in_active: int,
    alpha_cued: float,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    n_trials: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Run n_trials of weighted encode → spike → weighted factorised decode.

    Returns errors array, shape (n_trials,).
    """
    l = len(active_locs)
    M, n_theta = f_all[0].shape
    errors = np.empty(n_trials)

    f_active = [f_all[loc] for loc in active_locs]

    # Build weight vector
    alpha_weights = np.ones(l)
    alpha_weights[cued_loc_in_active] = alpha_cued

    for t in range(n_trials):
        theta_indices = rng.randint(n_theta, size=l)

        # Weighted pre-normalised response:
        #    log r_i^pre = Σ_k α_k · f_{i,k}(θ_k^true)
        log_r_pre = np.zeros(M)
        for k in range(l):
            log_r_pre += alpha_weights[k] * f_active[k][:, theta_indices[k]]
        r_pre = np.exp(log_r_pre - np.max(log_r_pre))  # numerical stability

        # Divisive normalisation (Eq. 6)
        rates = dn_pointwise(r_pre, gamma, sigma_sq)

        # Poisson spikes
        counts = generate_spikes(rates, T_d, rng)

        # Spike-weighted log-tuning (standard, unweighted L_k)
        L_list = compute_spike_weighted_log_tuning(counts, f_active)

        # Weighted marginal log-likelihood for the PROBED location
        ll_marginal = compute_weighted_marginal_log_likelihood(
            L_list, alpha_weights, probed_loc_in_active
        )
        idx_hat = np.argmax(ll_marginal)

        errors[t] = compute_circular_error(
            thetas[theta_indices[probed_loc_in_active]], thetas[idx_hat]
        )

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMAL WEIGHT FINDER (Panel d)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_test_probabilities(N: int, cue_ratio: float = 3.0) -> Tuple[float, float]:
    """p_cued = cue_ratio / (cue_ratio + N - 1), p_uncued = 1 / (cue_ratio + N - 1)."""
    p_cued = cue_ratio / (cue_ratio + N - 1)
    p_uncued = 1.0 / (cue_ratio + N - 1)
    return p_cued, p_uncued


def _find_optimal_alpha(
    f_all: List[np.ndarray],
    thetas: np.ndarray,
    N: int,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    cue_ratio: float,
    alpha_values: np.ndarray,
    n_trials_sweep: int,
    rng: np.random.RandomState,
) -> Dict:
    """Find optimal α_cued that minimises total expected error variance."""
    p_cued, p_uncued = _compute_test_probabilities(N, cue_ratio)
    active_locs = tuple(range(N))
    cued_loc = 0

    var_cued_arr = np.empty(len(alpha_values))
    var_uncued_arr = np.empty(len(alpha_values))
    var_total_arr = np.empty(len(alpha_values))

    for ai, alpha in enumerate(alpha_values):
        seed_c = rng.randint(1, 10**7)
        errs_cued = _run_weighted_trials(
            f_all, thetas, active_locs,
            cued_loc_in_active=cued_loc,
            probed_loc_in_active=cued_loc,
            alpha_cued=alpha,
            gamma=gamma, T_d=T_d, sigma_sq=sigma_sq,
            n_trials=n_trials_sweep,
            rng=np.random.RandomState(seed_c),
        )

        uncued_probe = 1 if N > 1 else 0
        seed_u = rng.randint(1, 10**7)
        errs_uncued = _run_weighted_trials(
            f_all, thetas, active_locs,
            cued_loc_in_active=cued_loc,
            probed_loc_in_active=uncued_probe,
            alpha_cued=alpha,
            gamma=gamma, T_d=T_d, sigma_sq=sigma_sq,
            n_trials=n_trials_sweep,
            rng=np.random.RandomState(seed_u),
        )

        var_cued_arr[ai] = circular_variance(errs_cued)
        var_uncued_arr[ai] = circular_variance(errs_uncued)
        var_total_arr[ai] = (p_cued * var_cued_arr[ai]
                             + (N - 1) * p_uncued * var_uncued_arr[ai])

    idx_opt = np.argmin(var_total_arr)

    return {
        'alpha_values': alpha_values,
        'var_cued': var_cued_arr,
        'var_uncued': var_uncued_arr,
        'var_total': var_total_arr,
        'alpha_optimal': alpha_values[idx_opt],
        'p_cued': p_cued,
        'p_uncued': p_uncued,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: Dict) -> Dict:
    """
    Phase 1 — optimal weight search.  Phase 2 — full simulation at α*.

    Config keys
    -----------
    M, n_theta, n_trials, n_trials_sweep, T_d, sigma_sq,
    lambda_base, gamma, set_sizes, cue_ratio,
    alpha_range, n_alpha, seed, n_seeds, n_bins
    """
    M              = config.get('M', 100)
    n_theta        = config.get('n_theta', 64)
    n_trials       = config.get('n_trials', 10_000)
    n_trials_sweep = config.get('n_trials_sweep', 2_000)
    T_d            = config.get('T_d', 0.1)
    sigma_sq       = config.get('sigma_sq', 1e-6)
    lam            = config.get('lambda_base', 0.5)
    gamma          = config.get('gamma', 100.0)
    set_sizes      = config.get('set_sizes', [2, 4, 8])
    cue_ratio      = config.get('cue_ratio', 3.0)
    alpha_lo, alpha_hi = config.get('alpha_range', (1.0, 8.0))
    n_alpha        = config.get('n_alpha', 15)
    seed           = config.get('seed', 42)
    n_seeds        = config.get('n_seeds', 3)
    n_bins         = config.get('n_bins', 50)

    max_locs = max(set_sizes)
    alpha_values = np.linspace(alpha_lo, alpha_hi, n_alpha)

    t0 = time.time()

    # ── PHASE 1: Find optimal α_cued ──
    print("=" * 70)
    print("PHASE 1: Optimal weight search")
    print("=" * 70)

    thetas, f_all = generate_population(M, n_theta, lam, max_locs, seed)

    optimal_results = {}
    for N in set_sizes:
        print(f"\n  N={N}: ", end="", flush=True)
        sweep_rng = np.random.RandomState(seed + N * 100)
        opt = _find_optimal_alpha(
            f_all, thetas, N, gamma, T_d, sigma_sq,
            cue_ratio, alpha_values, n_trials_sweep, sweep_rng,
        )
        optimal_results[N] = opt
        print(f"α* = {opt['alpha_optimal']:.2f}")

    # ── PHASE 2: Full simulation at optimal weights ──
    print("\n" + "=" * 70)
    print("PHASE 2: Full simulation at optimal α*")
    print("=" * 70)

    all_seeds_data = []

    for s in range(n_seeds):
        current_seed = seed + s * 10_000
        print(f"\n  seed {s+1}/{n_seeds}:")

        thetas_s, f_all_s = generate_population(M, n_theta, lam, max_locs, current_seed)

        seed_data = {}
        for N in set_sizes:
            alpha_opt = optimal_results[N]['alpha_optimal']
            active_locs = tuple(range(N))
            cued_loc = 0

            rng_c = np.random.RandomState(current_seed + N)
            errs_cued = _run_weighted_trials(
                f_all_s, thetas_s, active_locs,
                cued_loc_in_active=cued_loc,
                probed_loc_in_active=cued_loc,
                alpha_cued=alpha_opt,
                gamma=gamma, T_d=T_d, sigma_sq=sigma_sq,
                n_trials=n_trials, rng=rng_c,
            )

            uncued_probe = 1 if N > 1 else 0
            rng_u = np.random.RandomState(current_seed + N + 500)
            errs_uncued = _run_weighted_trials(
                f_all_s, thetas_s, active_locs,
                cued_loc_in_active=cued_loc,
                probed_loc_in_active=uncued_probe,
                alpha_cued=alpha_opt,
                gamma=gamma, T_d=T_d, sigma_sq=sigma_sq,
                n_trials=n_trials, rng=rng_u,
            )

            p_cued, p_uncued = _compute_test_probabilities(N, cue_ratio)
            n_cued_mix = int(round(p_cued * n_trials))
            n_uncued_mix = n_trials - n_cued_mix
            errs_all = np.concatenate([errs_cued[:n_cued_mix], errs_uncued[:n_uncued_mix]])

            seed_data[N] = {
                'alpha_opt': alpha_opt,
                'errors_cued': errs_cued,
                'errors_uncued': errs_uncued,
                'errors_all': errs_all,
                'var_cued': circular_variance(errs_cued),
                'var_uncued': circular_variance(errs_uncued),
                'var_all': circular_variance(errs_all),
                'kurt_cued': circular_kurtosis(errs_cued),
                'kurt_uncued': circular_kurtosis(errs_uncued),
                'hist_cued': compute_deviation_from_normal(errs_cued, n_bins),
                'hist_uncued': compute_deviation_from_normal(errs_uncued, n_bins),
                'hist_all': compute_deviation_from_normal(errs_all, n_bins),
            }
            print(f"    N={N}: α*={alpha_opt:.2f}  "
                  f"var_cued={seed_data[N]['var_cued']:.3f}  "
                  f"var_uncued={seed_data[N]['var_uncued']:.3f}")

        all_seeds_data.append(seed_data)

    # ── Aggregate across seeds ──
    summary = {}
    for N in set_sizes:
        vc = [sd[N]['var_cued'] for sd in all_seeds_data]
        vu = [sd[N]['var_uncued'] for sd in all_seeds_data]
        va = [sd[N]['var_all'] for sd in all_seeds_data]
        kc = [sd[N]['kurt_cued'] for sd in all_seeds_data]
        ku = [sd[N]['kurt_uncued'] for sd in all_seeds_data]
        n_s = len(all_seeds_data)
        se = lambda x: np.std(x, ddof=1) / np.sqrt(n_s) if n_s > 1 else 0.0

        summary[N] = {
            'alpha_opt': optimal_results[N]['alpha_optimal'],
            'var_cued_mean': np.mean(vc), 'var_cued_se': se(vc),
            'var_uncued_mean': np.mean(vu), 'var_uncued_se': se(vu),
            'var_all_mean': np.mean(va), 'var_all_se': se(va),
            'kurt_cued_mean': np.mean(kc), 'kurt_cued_se': se(kc),
            'kurt_uncued_mean': np.mean(ku), 'kurt_uncued_se': se(ku),
        }

        for condition in ['cued', 'uncued', 'all']:
            key_h = f'hist_{condition}'
            emps = np.array([sd[N][key_h]['empirical'] for sd in all_seeds_data])
            summary[N][f'emp_{condition}_mean'] = np.mean(emps, axis=0)
            summary[N][f'emp_{condition}_se'] = (
                np.std(emps, axis=0, ddof=1) / np.sqrt(n_s)
                if n_s > 1 else np.zeros_like(emps[0])
            )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return {
        'set_sizes': set_sizes,
        'summary': summary,
        'optimal_results': optimal_results,
        'all_seeds_data': all_seeds_data,
        'bin_centers': all_seeds_data[0][set_sizes[0]]['hist_all']['bin_centers'],
        'config': config,
        'elapsed_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Four-panel figure matching Bays (2014) Fig 3 layout."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_sizes = results['set_sizes']
    summary   = results['summary']
    bins      = results['bin_centers']
    n_ss      = len(set_sizes)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    RED, GREEN, BLACK, BLUE = '#CC2222', '#228B22', '#222222', '#2255AA'

    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1],
                              left=0.06, right=0.96, bottom=0.06, top=0.92, wspace=0.35)
    gs_a = gridspec.GridSpecFromSubplotSpec(3, n_ss, subplot_spec=outer[0],
                                            hspace=0.40, wspace=0.30)
    gs_bcd = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.50)

    # ── Panel a: Error distributions ──
    row_labels = ['all trials', 'cued', 'uncued']
    row_colors = [BLACK, RED, GREEN]
    row_keys   = ['all', 'cued', 'uncued']

    for row in range(3):
        for col, N in enumerate(set_sizes):
            ax = fig.add_subplot(gs_a[row, col])
            emp = summary[N][f'emp_{row_keys[row]}_mean']
            emp_se = summary[N][f'emp_{row_keys[row]}_se']
            ax.plot(bins, emp, color=row_colors[row], linewidth=1.5)
            ax.fill_between(bins, emp - emp_se, emp + emp_se,
                            color=row_colors[row], alpha=0.12)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0, max(emp.max() * 1.15, 0.1))
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
            if row == 2: ax.set_xlabel('error')
            if col == 0: ax.set_ylabel('probability\ndensity')
            if col == 0:
                ax.text(-0.35, 0.5, row_labels[row], transform=ax.transAxes,
                        fontsize=10, fontweight='bold', color=row_colors[row],
                        va='center', ha='right')
            if row == 0: ax.set_title(f'{N}', fontsize=14, fontweight='bold')

    # ── Panel b: Variance ──
    ax_b = fig.add_subplot(gs_bcd[0])
    ns = np.array(set_sizes, dtype=float)
    for key, color, label, marker in [('var_cued', RED, 'cued', 'o'),
                                       ('var_uncued', GREEN, 'uncued', 's')]:
        vals = np.array([summary[N][f'{key}_mean'] for N in set_sizes])
        ses  = np.array([summary[N][f'{key}_se'] for N in set_sizes])
        ax_b.plot(ns, vals, f'{marker}-', color=color, linewidth=1.5, markersize=5, label=label)
        ax_b.fill_between(ns, vals - ses, vals + ses, color=color, alpha=0.12)
    ax_b.set_xticks(set_sizes); ax_b.set_xticklabels([str(n) for n in set_sizes])
    ax_b.set_xlabel('items'); ax_b.set_ylabel(r'variance ($\sigma^2$)')
    ax_b.legend(fontsize=8, frameon=False)
    ax_b.text(-0.20, 1.08, r'$\mathbf{b}$', transform=ax_b.transAxes,
              fontsize=16, fontweight='bold', va='top')

    # ── Panel c: Kurtosis ──
    ax_c = fig.add_subplot(gs_bcd[1])
    for key, color, label, marker in [('kurt_cued', RED, 'cued', 'o'),
                                       ('kurt_uncued', GREEN, 'uncued', 's')]:
        vals = np.array([summary[N][f'{key}_mean'] for N in set_sizes])
        ses  = np.array([summary[N][f'{key}_se'] for N in set_sizes])
        ax_c.plot(ns, vals, f'{marker}-', color=color, linewidth=1.5, markersize=5, label=label)
        ax_c.fill_between(ns, vals - ses, vals + ses, color=color, alpha=0.12)
    ax_c.set_xticks(set_sizes); ax_c.set_xticklabels([str(n) for n in set_sizes])
    ax_c.set_xlabel('items'); ax_c.set_ylabel('kurtosis')
    ax_c.legend(fontsize=8, frameon=False)
    ax_c.text(-0.20, 1.08, r'$\mathbf{c}$', transform=ax_c.transAxes,
              fontsize=16, fontweight='bold', va='top')

    # ── Panel d: Optimal α ──
    ax_d = fig.add_subplot(gs_bcd[2])
    alpha_opts = [summary[N]['alpha_opt'] for N in set_sizes]
    ax_d.bar(np.arange(n_ss), alpha_opts, 0.5,
             color=BLUE, alpha=0.85, edgecolor='black', linewidth=0.8, label='optimal')
    ax_d.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_d.set_xticks(np.arange(n_ss)); ax_d.set_xticklabels([str(n) for n in set_sizes])
    ax_d.set_xlabel('items'); ax_d.set_ylabel(r'weighting factor, $\alpha_{cued}$')
    ax_d.legend(fontsize=8, frameon=False)
    ax_d.text(-0.20, 1.08, r'$\mathbf{d}$', transform=ax_d.transAxes,
              fontsize=16, fontweight='bold', va='top')

    fig.suptitle('GP Population Coding — Bays (2014) Fig 3 Equivalent',
                 fontsize=12, fontweight='bold', y=0.98)

    outpath = Path(output_dir) / 'figure_3_cued.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot: plt.show()
    plt.close(fig)

    np.savez(Path(output_dir) / 'figure_3_data.npz',
             set_sizes=np.array(set_sizes), bin_centers=bins,
             alpha_optimal=np.array(alpha_opts))


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = {
        'M': 100, 'n_theta': 64,
        'n_trials': 10_000, 'n_trials_sweep': 2_000,
        'T_d': 0.1, 'sigma_sq': 1e-6,
        'lambda_base': 0.5, 'gamma': 100.0,
        'set_sizes': [2, 4, 8], 'cue_ratio': 3.0,
        'alpha_range': (1.0, 8.0), 'n_alpha': 15,
        'seed': 42, 'n_seeds': 3, 'n_bins': 50,
    }
    print("Running Bays (2014) Figure 3 — GP Equivalent")
    results = run_experiment(config)
    plot_results(results, 'results/figure_3', show_plot=True)