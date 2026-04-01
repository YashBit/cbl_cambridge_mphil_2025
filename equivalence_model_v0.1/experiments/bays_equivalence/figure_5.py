"""
Bays (2014) Figure 5 — GP-Based Equivalent

Effects of baseline activity on ML parameters and decoding errors.

Panels:
    a — Population gain (ML) vs baseline activation level
    b — Tuning width (ML) vs baseline activation level
    c — Error distributions at different baseline levels (1-8 items)
    d — SNR per neuron vs baseline activation level

=============================================================================
KEY DESIGN CHOICE: SINGLE-LOCATION DECODER + γ/N SHORTCUT
=============================================================================

This figure uses the FULL Poisson ML decoder (single-location, retaining
the rate-penalty term) combined with the γ/N gain-reduction shortcut for
multi-item conditions.  This matches Bays's fitting procedure and is the
correct approach for the baseline-invariance demonstration.

The baseline modifies driving inputs as:
    g_i^baseline(θ) = exp(f_i(θ)) + b_0

Usage:
    from experiments.bays_equivalence.figure_5 import run_experiment, plot_results
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


def _generate_population_with_baseline(
    M: int,
    n_theta: int,
    lengthscale: float,
    baseline_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate M neurons with GP tuning curves plus baseline activity.

    g_i(θ) = exp(f_i(θ)) + b_0

    where b_0 = baseline_frac * mean_peak / (1 - baseline_frac).

    Returns
    -------
    thetas : (n_theta,)
    g_baseline : (M, n_theta) — driving inputs WITH baseline
    g_raw : (M, n_theta) — driving inputs WITHOUT baseline
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    K = periodic_rbf_kernel(thetas, lengthscale)

    f = np.zeros((M, n_theta))
    for i in range(M):
        f[i] = sample_gp_function(K, rng)

    g_raw = np.exp(f)

    if baseline_frac < 1e-10:
        b_0 = 0.0
    else:
        mean_peak = np.mean(np.max(g_raw, axis=1))
        b_0 = baseline_frac * mean_peak / (1.0 - baseline_frac)

    g_baseline = g_raw + b_0
    return thetas, g_baseline, g_raw


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL ENGINE — SINGLE-LOCATION, FULL POISSON ML DECODER (WITH BASELINE)
# ═══════════════════════════════════════════════════════════════════════════

def _run_trials_with_baseline(
    g: np.ndarray,
    thetas: np.ndarray,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    n_trials: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Run n_trials of encode → spike → decode with baseline-augmented
    tuning curves, using full Poisson ML decoder.

    Returns errors array, shape (n_trials,).
    """
    M, n_theta = g.shape
    errors = np.empty(n_trials)

    for t in range(n_trials):
        idx_true = rng.randint(n_theta)

        # DN at true orientation
        rates = dn_pointwise(g[:, idx_true], gamma, sigma_sq)

        # Poisson spikes
        counts = generate_spikes(rates, T_d, rng)

        # Full Poisson ML decode (rate-penalty term retained)
        ll = compute_log_likelihood(counts, g, T_d)
        idx_hat = np.argmax(ll)

        errors[t] = compute_circular_error(thetas[idx_true], thetas[idx_hat])

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# SNR COMPUTATION (figure-5 specific)
# ═══════════════════════════════════════════════════════════════════════════

def compute_snr_per_neuron(
    g: np.ndarray,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    N_items: int = 1,
) -> float:
    """
    Population-averaged SNR per neuron.

    SNR_i = T_d · Var_θ[r_i] / E_θ[r_i]

    Under DN with N_items sharing the population:
        r_i(θ) = γ · g_i(θ) / (σ² + N_items · mean_g(θ))

    Returns population-averaged SNR.
    """
    M, n_theta = g.shape
    mean_g_per_theta = np.mean(g, axis=0)
    denom_per_theta = sigma_sq + N_items * mean_g_per_theta

    rates = gamma * g / denom_per_theta[np.newaxis, :]

    var_rates = np.var(rates, axis=1)
    mean_rates = np.mean(rates, axis=1)

    snr_per_neuron = np.where(
        mean_rates > 1e-15,
        T_d * var_rates / mean_rates,
        0.0
    )
    return np.mean(snr_per_neuron)


# ═══════════════════════════════════════════════════════════════════════════
# ML FITTING (figure-5 specific)
# ═══════════════════════════════════════════════════════════════════════════

def _fit_ml_parameters(
    target_variances: Dict[int, float],
    M: int,
    n_theta: int,
    T_d: float,
    sigma_sq: float,
    baseline_frac: float,
    gamma_values: np.ndarray,
    lambda_values: np.ndarray,
    n_trials_fit: int,
    seed: int,
) -> Dict:
    """Find (γ, λ) that best match target variance profile at given baseline."""
    set_sizes = sorted(target_variances.keys())
    best_cost = np.inf
    best_params = {}

    for li, lam in enumerate(lambda_values):
        thetas, g_bl, g_raw = _generate_population_with_baseline(
            M, n_theta, lam, baseline_frac, seed + li * 1000
        )
        for gi, gam in enumerate(gamma_values):
            cost = 0.0
            variances = {}
            for N in set_sizes:
                effective_gamma = gam / N
                rng = np.random.RandomState(seed + li * 100 + gi * 10 + N)
                errs = _run_trials_with_baseline(
                    g_bl, thetas, effective_gamma, T_d, sigma_sq,
                    n_trials_fit, rng,
                )
                v = circular_variance(errs)
                variances[N] = v
                if v > 1e-10 and target_variances[N] > 1e-10:
                    cost += (np.log(v) - np.log(target_variances[N]))**2

            if cost < best_cost:
                best_cost = cost
                best_params = {
                    'gamma': gam, 'lengthscale': lam,
                    'variances': variances, 'cost': cost,
                    'g_baseline': g_bl, 'g_raw': g_raw, 'thetas': thetas,
                }

    return best_params


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: Dict) -> Dict:
    """
    Three-phase experiment:
        Phase 1 — generate target variances at baseline = 0
        Phase 2 — fit (γ, λ) at each baseline level
        Phase 3 — compute SNR and error distributions at ML parameters
    """
    M              = config.get('M', 100)
    n_theta        = config.get('n_theta', 64)
    T_d            = config.get('T_d', 0.1)
    sigma_sq       = config.get('sigma_sq', 1e-6)
    lam_ref        = config.get('lambda_ref', 0.5)
    gamma_ref      = config.get('gamma_ref', 100.0)
    set_sizes      = config.get('set_sizes', [1, 2, 4, 8])
    baseline_fracs = config.get('baseline_fracs', [0.0, 0.05, 0.20, 0.50, 0.80, 0.95])
    n_trials_fit   = config.get('n_trials_fit', 3000)
    n_trials_final = config.get('n_trials_final', 10_000)
    n_gamma_grid   = config.get('n_gamma_grid', 20)
    n_lambda_grid  = config.get('n_lambda_grid', 8)
    gam_lo, gam_hi = config.get('gamma_range', (10.0, 1e6))
    lam_lo, lam_hi = config.get('lambda_range', (0.3, 1.0))
    seed           = config.get('seed', 42)
    n_seeds        = config.get('n_seeds', 3)

    gamma_grid = np.logspace(np.log10(gam_lo), np.log10(gam_hi), n_gamma_grid)
    lambda_grid = np.linspace(lam_lo, lam_hi, n_lambda_grid)

    t0 = time.time()

    # ── PHASE 1: Target variances at baseline = 0 ──
    print("=" * 70)
    print("PHASE 1: Generating target error variances (baseline = 0)")
    print("=" * 70)

    target_variances = {}
    thetas_ref, g_ref, _ = _generate_population_with_baseline(
        M, n_theta, lam_ref, 0.0, seed
    )
    for N in set_sizes:
        eff_gamma = gamma_ref / N
        rng = np.random.RandomState(seed + N)
        errs = _run_trials_with_baseline(
            g_ref, thetas_ref, eff_gamma, T_d, sigma_sq,
            n_trials_final, rng,
        )
        target_variances[N] = circular_variance(errs)
        print(f"  N={N}: variance = {target_variances[N]:.4f}")

    # ── PHASE 2: Fit at each baseline ──
    print("\n" + "=" * 70)
    print("PHASE 2: ML fitting at each baseline level")
    print("=" * 70)

    fit_results = {}
    for bf in baseline_fracs:
        print(f"\n  baseline = {bf*100:.0f}%: ", end="", flush=True)
        if bf < 1e-10:
            fit_results[bf] = {
                'gamma': gamma_ref, 'lengthscale': lam_ref,
                'variances': target_variances.copy(), 'cost': 0.0,
                'g_baseline': g_ref, 'g_raw': g_ref, 'thetas': thetas_ref,
            }
            print(f"gamma={gamma_ref:.1f}, lambda={lam_ref:.3f} (reference)")
        else:
            fit = _fit_ml_parameters(
                target_variances, M, n_theta, T_d, sigma_sq,
                bf, gamma_grid, lambda_grid, n_trials_fit, seed,
            )
            fit_results[bf] = fit
            print(f"gamma={fit['gamma']:.1f}, lambda={fit['lengthscale']:.3f} "
                  f"(cost={fit['cost']:.4f})")

    # ── PHASE 3: Full simulation + SNR ──
    print("\n" + "=" * 70)
    print("PHASE 3: Full simulation and SNR computation")
    print("=" * 70)

    all_seeds_results = []

    for s in range(n_seeds):
        cseed = seed + s * 50_000
        seed_data = {}

        for bf in baseline_fracs:
            fit = fit_results[bf]
            gam_ml = fit['gamma']
            lam_ml = fit['lengthscale']

            thetas_s, g_bl_s, g_raw_s = _generate_population_with_baseline(
                M, n_theta, lam_ml, bf, cseed + int(bf * 1000)
            )

            snr = compute_snr_per_neuron(g_bl_s, gam_ml, T_d, sigma_sq, N_items=1)

            distributions = {}
            variances_final = {}
            for N in set_sizes:
                eff_gamma = gam_ml / N
                rng = np.random.RandomState(cseed + N + int(bf * 10000))
                errs = _run_trials_with_baseline(
                    g_bl_s, thetas_s, eff_gamma, T_d, sigma_sq,
                    n_trials_final, rng,
                )
                distributions[N] = compute_deviation_from_normal(errs)
                variances_final[N] = circular_variance(errs)

            seed_data[bf] = {
                'gamma': gam_ml, 'lengthscale': lam_ml, 'snr': snr,
                'distributions': distributions, 'variances': variances_final,
            }

        all_seeds_results.append(seed_data)
        print(f"  seed {s+1}/{n_seeds} done ({time.time()-t0:.0f}s)")

    # ── Aggregate ──
    summary = {}
    for bf in baseline_fracs:
        gammas = [sd[bf]['gamma'] for sd in all_seeds_results]
        lams = [sd[bf]['lengthscale'] for sd in all_seeds_results]
        snrs = [sd[bf]['snr'] for sd in all_seeds_results]
        ns = len(all_seeds_results)
        se = lambda x: np.std(x, ddof=1) / np.sqrt(ns) if ns > 1 else 0.0

        summary[bf] = {
            'gamma_mean': np.mean(gammas), 'gamma_se': se(gammas),
            'lambda_mean': np.mean(lams), 'lambda_se': se(lams),
            'snr_mean': np.mean(snrs), 'snr_se': se(snrs),
        }
        for N in set_sizes:
            emps = np.array([sd[bf]['distributions'][N]['empirical']
                             for sd in all_seeds_results])
            summary[bf][f'emp_N{N}_mean'] = np.mean(emps, axis=0)
            summary[bf][f'emp_N{N}_se'] = (
                np.std(emps, axis=0, ddof=1) / np.sqrt(ns)
                if ns > 1 else np.zeros_like(emps[0])
            )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return {
        'baseline_fracs': baseline_fracs, 'set_sizes': set_sizes,
        'target_variances': target_variances, 'fit_results': fit_results,
        'summary': summary, 'all_seeds_results': all_seeds_results,
        'bin_centers': all_seeds_results[0][baseline_fracs[0]]['distributions'][set_sizes[0]]['bin_centers'],
        'config': config, 'elapsed_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Four-panel figure matching Bays (2014) Fig 5 layout."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bfs       = results['baseline_fracs']
    set_sizes = results['set_sizes']
    summary   = results['summary']
    bins      = results['bin_centers']

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    GREY = '#444444'
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    x_pct = [bf * 100 for bf in bfs]

    # Panel a: Gain
    ax_a = axes[0, 0]
    gammas = [summary[bf]['gamma_mean'] for bf in bfs]
    gamma_se = [summary[bf]['gamma_se'] for bf in bfs]
    ax_a.semilogy(x_pct, gammas, 'o-', color=GREY, linewidth=1.5, markersize=5)
    ax_a.fill_between(x_pct, [g-s for g,s in zip(gammas,gamma_se)],
                       [g+s for g,s in zip(gammas,gamma_se)], color=GREY, alpha=0.15)
    ax_a.set_xlabel('baseline (% of peak)'); ax_a.set_ylabel(r'gain, $\gamma$ (Hz)')
    ax_a.text(-0.15, 1.06, r'$\mathbf{a}$', transform=ax_a.transAxes,
              fontsize=16, fontweight='bold', va='top')

    # Panel b: Width
    ax_b = axes[0, 1]
    lams = [summary[bf]['lambda_mean'] for bf in bfs]
    lam_se = [summary[bf]['lambda_se'] for bf in bfs]
    ax_b.plot(x_pct, lams, 'o-', color=GREY, linewidth=1.5, markersize=5)
    ax_b.fill_between(x_pct, [l-s for l,s in zip(lams,lam_se)],
                       [l+s for l,s in zip(lams,lam_se)], color=GREY, alpha=0.15)
    ax_b.set_xlabel('baseline (% of peak)'); ax_b.set_ylabel(r'width, $\lambda$')
    ax_b.text(-0.15, 1.06, r'$\mathbf{b}$', transform=ax_b.transAxes,
              fontsize=16, fontweight='bold', va='top')

    # Panel c: Error distributions
    ax_c = axes[1, 0]
    show_baselines = []
    for tgt in [0, 5, 50, 90]:
        closest = min(bfs, key=lambda bf: abs(bf * 100 - tgt))
        if closest not in show_baselines:
            show_baselines.append(closest)

    colors_c = ['#222222', '#3366BB', '#33AA66', '#CC4444']
    for N in set_sizes:
        for bi, bf in enumerate(show_baselines):
            emp = summary[bf][f'emp_N{N}_mean']
            c = colors_c[bi] if bi < len(colors_c) else GREY
            label = f'{bf*100:.0f}%' if N == set_sizes[0] else None
            ax_c.plot(bins, emp, color=c, linewidth=1.2, alpha=0.8, label=label)

    ax_c.set_xlim(-np.pi, np.pi)
    ax_c.set_xticks([-np.pi, 0, np.pi])
    ax_c.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax_c.set_xlabel('error'); ax_c.set_ylabel('probability\ndensity')
    ax_c.legend(fontsize=7, frameon=False, title='baseline', title_fontsize=7)
    ax_c.text(-0.15, 1.06, r'$\mathbf{c}$', transform=ax_c.transAxes,
              fontsize=16, fontweight='bold', va='top')

    # Panel d: SNR
    ax_d = axes[1, 1]
    snrs = [summary[bf]['snr_mean'] for bf in bfs]
    snr_se = [summary[bf]['snr_se'] for bf in bfs]
    ax_d.plot(x_pct, snrs, 'o-', color=GREY, linewidth=1.5, markersize=5)
    ax_d.fill_between(x_pct, [s-e for s,e in zip(snrs,snr_se)],
                       [s+e for s,e in zip(snrs,snr_se)], color=GREY, alpha=0.15)
    ax_d.set_xlabel('baseline (% of peak)'); ax_d.set_ylabel('SNR')
    ax_d.text(-0.15, 1.06, r'$\mathbf{d}$', transform=ax_d.transAxes,
              fontsize=16, fontweight='bold', va='top')

    fig.suptitle('GP Population Coding — Bays (2014) Fig 5 Equivalent',
                 fontsize=12, fontweight='bold', y=0.98)

    outpath = Path(output_dir) / 'figure_5_baseline.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot: plt.show()
    plt.close(fig)

    np.savez(Path(output_dir) / 'figure_5_data.npz',
             baseline_fracs=np.array(bfs),
             gammas=np.array([summary[bf]['gamma_mean'] for bf in bfs]),
             lambdas=np.array([summary[bf]['lambda_mean'] for bf in bfs]),
             snrs=np.array([summary[bf]['snr_mean'] for bf in bfs]))


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = {
        'M': 100, 'n_theta': 64,
        'T_d': 0.1, 'sigma_sq': 1e-6,
        'lambda_ref': 0.5, 'gamma_ref': 100.0,
        'set_sizes': [1, 2, 4, 8],
        'baseline_fracs': [0.0, 0.05, 0.20, 0.50, 0.80, 0.95],
        'n_trials_fit': 3000, 'n_trials_final': 10_000,
        'n_gamma_grid': 20, 'n_lambda_grid': 8,
        'gamma_range': (10.0, 1e6), 'lambda_range': (0.3, 1.0),
        'seed': 42, 'n_seeds': 3,
    }
    print("Running Bays (2014) Figure 5 — GP Equivalent")
    results = run_experiment(config)
    plot_results(results, 'results/figure_5', show_plot=True)