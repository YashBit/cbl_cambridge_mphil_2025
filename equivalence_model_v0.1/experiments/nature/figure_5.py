"""
Bays & Brady (2024) Figure 5 — GP-Based Equivalent

Tests whether GP tuning curves + DN + Poisson noise produce the key
phenomenology from "Changing Concepts in Working Memory" (Figure 5):

    (a) Pooled error distribution across all set sizes — heavy tails
    (b) Actual SD (degrees) vs set size 1–8 — continuous rise, no plateau
    (c) Model prediction vs theoretical √l scaling from DN activity cap

=============================================================================
SCIENTIFIC QUESTION
=============================================================================

The original Figure 5 showed that:
  - Actual SD rises continuously with set size (no plateau)
  - The apparent plateau in SD_normal (from mixture fits) is an artefact
  - A variable-precision resource model accounts for the continuous rise

We ask: does the GP population coding model reproduce this?

The GP model adds heterogeneous tuning widths (location-dependent
lengthscales), which should produce MORE variable precision than
cosine tuning — potentially a better match to the continuous rise.

Panel (c) tests whether the observed SD follows the theoretical
prediction from the DN activity cap:

    Per-item gain = γN / l
    Expected spikes per item = (γN / l) × T_d
    Fisher information ∝ spikes ∝ 1/l
    Theoretical SD ∝ √l

Deviations from √l reveal effects of GP heterogeneity that go beyond
the simple gain-scaling story.

=============================================================================
PIPELINE (per set size, per trial)
=============================================================================

    1. Generate M neurons with GP tuning (lengthscale = λ_base)
    2. DN: effective gain = γ / N  (set size N divides gain)
    3. r_i(θ) = (γ/N) · g_i(θ) / (σ² + M⁻¹ Σ_j g_j(θ))   [dn_pointwise]
    4. Spike counts: n_i ~ Poisson(r_i · T_d)
    5. Full Poisson ML decode (rate-penalty retained)
    6. Circular error: ε = θ̂ − θ_true (wrapped to [-π, π))

Note: This is the single-location decoder with the γ/N shortcut,
matching the design choice in the Bays (2014) figure_1 and figure_5
experiments.  This is a different paper from Bays (2014) and does NOT
share utilities with the bays_equivalence experiment family.

=============================================================================
Usage:
    from experiments.nature.figure_5 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Tuple
import time

from core.encoder.gaussian_process import periodic_rbf_kernel, sample_gp_function
from core.encoder.poisson_spike import generate_spikes
from core.encoder.divisive_normalization import dn_pointwise
from core.decoder.ml_decoder import compute_log_likelihood, compute_circular_error



# =============================================================================
# CIRCULAR STATISTICS (self-contained — not shared with bays_equivalence)
# =============================================================================

def _circular_variance(errors: np.ndarray) -> float:
    """σ² = −2·log(ρ₁) where ρ₁ = |mean(exp(iε))|."""
    m1 = np.mean(np.exp(1j * errors))
    rho1 = np.clip(np.abs(m1), 1e-15, 1.0 - 1e-10)
    return -2.0 * np.log(rho1)


def _circular_sd_degrees(errors: np.ndarray) -> float:
    """Circular SD in degrees: √(variance) converted from radians."""
    return np.degrees(np.sqrt(_circular_variance(errors)))


# =============================================================================
# POPULATION GENERATION (self-contained)
# =============================================================================

def _generate_population(
    M: int, n_theta: int, lengthscale: float, seed: int
) -> Tuple:
    """
    Generate M neurons with GP tuning curves at a single lengthscale.

    Returns (thetas, g) where g = exp(f), shape (M, n_theta).
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    K = periodic_rbf_kernel(thetas, lengthscale)

    g = np.zeros((M, n_theta))
    for i in range(M):
        g[i] = np.exp(sample_gp_function(K, rng))

    return thetas, g


# =============================================================================
# TRIAL ENGINE — SINGLE-LOCATION, FULL POISSON ML DECODER
# =============================================================================

def _run_trials_at_gain(
    g: np.ndarray,
    thetas: np.ndarray,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    n_trials: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Run n_trials of encode → spike → decode at fixed gain γ.

    Uses full Poisson ML decoder (rate-penalty retained) and
    dn_pointwise for divisive normalisation.

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


# =============================================================================
# THEORETICAL PREDICTION
# =============================================================================

def _compute_theoretical_sd(
    sd_at_1: float,
    set_sizes: np.ndarray,
) -> np.ndarray:
    """
    Theoretical SD under DN activity cap: SD(l) = SD(1) × √l.

    Under DN, per-item gain ∝ 1/l → Fisher info ∝ 1/l → SD ∝ √l.
    """
    return sd_at_1 * np.sqrt(np.asarray(set_sizes, dtype=float))


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(config: Dict) -> Dict:
    """
    Sweep set sizes 1–8, collect error distributions and actual SD.

    Config keys
    -----------
    M            : neurons per population          (default 100)
    n_theta      : orientation bins                (default 64)
    n_trials     : trials per condition            (default 10_000)
    T_d          : decoding window (s)             (default 0.1)
    sigma_sq     : semi-saturation constant        (default 1e-6)
    lambda_base  : GP lengthscale (≡ ω in Bays)   (default 0.5)
    gamma        : gain constant (Hz)              (default 100.0)
    set_sizes    : list of N values                (default [1..8])
    seed         : master seed                     (default 42)
    n_seeds      : seeds for SE bands              (default 5)
    n_bins       : histogram bins for panel (a)    (default 60)
    """
    M          = config.get('M', 100)
    n_theta    = config.get('n_theta', 64)
    n_trials   = config.get('n_trials', 10_000)
    T_d        = config.get('T_d', 0.1)
    sigma_sq   = config.get('sigma_sq', 1e-6)
    lam        = config.get('lambda_base', 0.5)
    gamma      = config.get('gamma', 100.0)
    set_sizes  = config.get('set_sizes', [1, 2, 3, 4, 5, 6, 7, 8])
    seed       = config.get('seed', 42)
    n_seeds    = config.get('n_seeds', 5)
    n_bins     = config.get('n_bins', 60)

    t0 = time.time()
    all_seeds = []

    for s in range(n_seeds):
        current_seed = seed + s * 1000
        thetas, g = _generate_population(M, n_theta, lam, current_seed)

        seed_data = {}
        for N in set_sizes:
            effective_gamma = gamma / N  # DN: gain divided by set size
            rng = np.random.RandomState(current_seed + N)
            errors = _run_trials_at_gain(
                g, thetas, effective_gamma, T_d, sigma_sq, n_trials, rng
            )
            seed_data[N] = {
                'errors': errors,
                'sd_deg': _circular_sd_degrees(errors),
                'variance': _circular_variance(errors),
            }
            print(f"  seed={s} N={N}: SD={seed_data[N]['sd_deg']:.2f}°")

        all_seeds.append(seed_data)

    # ── Aggregate across seeds ──
    summary = {}
    for N in set_sizes:
        sds   = [sd[N]['sd_deg'] for sd in all_seeds]
        vars_ = [sd[N]['variance'] for sd in all_seeds]
        summary[N] = {
            'sd_mean': np.mean(sds),
            'sd_se':   np.std(sds, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0,
            'variance_mean': np.mean(vars_),
            'variance_se':   np.std(vars_, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0,
        }

    # ── Pooled errors across all set sizes and seeds (for panel a) ──
    all_errors = []
    for seed_data in all_seeds:
        for N in set_sizes:
            all_errors.append(seed_data[N]['errors'])
    pooled_errors = np.concatenate(all_errors)

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts, _ = np.histogram(pooled_errors, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    pooled_density = counts / (len(pooled_errors) * bin_width)

    pooled_sd_deg = _circular_sd_degrees(pooled_errors)

    # ── Theoretical √l prediction ──
    sd_at_1 = summary[set_sizes[0]]['sd_mean']
    theoretical_sd = _compute_theoretical_sd(sd_at_1, np.array(set_sizes))

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({n_seeds} seeds × {len(set_sizes)} set sizes "
          f"× {n_trials} trials)")
    print(f"  Pooled actual SD = {pooled_sd_deg:.1f}°")

    return {
        'set_sizes': set_sizes,
        'summary': summary,
        'all_seeds': all_seeds,
        'pooled_errors': pooled_errors,
        'pooled_density': pooled_density,
        'pooled_sd_deg': pooled_sd_deg,
        'bin_centers': bin_centers,
        'theoretical_sd': theoretical_sd,
        'config': config,
        'elapsed_seconds': elapsed,
    }


# =============================================================================
# PLOTTING — Three-panel figure matching Figure 5 layout
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """
    Three-panel figure:
        (a) Pooled error distribution (histogram) with actual SD annotation
        (b) Actual SD vs set size (linear axes, degrees)
        (c) Model SD vs theoretical √l prediction
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_sizes      = results['set_sizes']
    summary        = results['summary']
    bins           = results['bin_centers']
    pooled_density = results['pooled_density']
    pooled_sd      = results['pooled_sd_deg']
    theoretical_sd = results['theoretical_sd']
    config         = results['config']

    ns      = np.array(set_sizes, dtype=float)
    sd_mean = np.array([summary[N]['sd_mean'] for N in set_sizes])
    sd_se   = np.array([summary[N]['sd_se']   for N in set_sizes])

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    })

    fig = plt.figure(figsize=(14.5, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1],
                           wspace=0.35, left=0.06, right=0.97,
                           bottom=0.14, top=0.86)

    RED = '#CC2222'
    BLACK = '#222222'
    BLUE = '#2255AA'
    GRAY_HIST = '#AAAAAA'

    # ── Panel (a): Pooled error distribution ──
    ax_a = fig.add_subplot(gs[0])

    bin_width = bins[1] - bins[0]
    ax_a.bar(np.degrees(bins), pooled_density * np.pi / 180,
             width=np.degrees(bin_width) * 0.85,
             color=GRAY_HIST, edgecolor='#888888', linewidth=0.3,
             alpha=0.7, zorder=2)

    ax_a.annotate(
        f'Actual SD = {pooled_sd:.1f}\u00b0',
        xy=(0.97, 0.95), xycoords='axes fraction',
        fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9),
    )

    ax_a.set_xlim(-180, 180)
    ax_a.set_xticks([-180, -90, 0, 90, 180])
    ax_a.set_xlabel('Estimation error (degrees)')
    ax_a.set_ylabel('Probability density')
    ax_a.set_ylim(bottom=0)
    ax_a.text(-0.14, 1.06, r'$\mathbf{a}$', transform=ax_a.transAxes,
              fontsize=15, fontweight='bold', va='top')

    # ── Panel (b): Actual SD vs set size ──
    ax_b = fig.add_subplot(gs[1])

    ax_b.errorbar(ns, sd_mean, yerr=sd_se,
                  fmt='o-', color=BLACK, linewidth=1.5, markersize=5,
                  capsize=3, capthick=1.0, zorder=3,
                  label='Actual SD')

    ax_b.set_xlim(0.5, 8.5)
    ax_b.set_xticks(set_sizes)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel('Set size')
    ax_b.set_ylabel('SD (degrees)')
    ax_b.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_b.set_title('GP Model', fontsize=11)
    ax_b.text(-0.14, 1.06, r'$\mathbf{b}$', transform=ax_b.transAxes,
              fontsize=15, fontweight='bold', va='top')

    # ── Panel (c): Model vs theoretical √l prediction ──
    ax_c = fig.add_subplot(gs[2])

    ax_c.plot(ns, sd_mean, 'o-', color=RED, linewidth=1.5, markersize=5,
              zorder=3, label='GP model')
    ax_c.fill_between(ns, sd_mean - sd_se, sd_mean + sd_se,
                       color=RED, alpha=0.15)
    ax_c.plot(ns, sd_mean - sd_se, '--', color=RED, linewidth=0.6, alpha=0.5)
    ax_c.plot(ns, sd_mean + sd_se, '--', color=RED, linewidth=0.6, alpha=0.5)

    ns_smooth = np.linspace(1, 8, 100)
    sd_at_1 = summary[set_sizes[0]]['sd_mean']
    theoretical_smooth = sd_at_1 * np.sqrt(ns_smooth)

    ax_c.plot(ns_smooth, theoretical_smooth, '-', color=BLUE, linewidth=1.5,
              alpha=0.7, zorder=2, label=r'Theoretical: SD $\propto \sqrt{l}$')

    ax_c.set_xlim(0.5, 8.5)
    ax_c.set_xticks(set_sizes)
    ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel('Set size')
    ax_c.set_ylabel('SD (degrees)')
    ax_c.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_c.set_title('Model vs Theory', fontsize=11)
    ax_c.text(-0.14, 1.06, r'$\mathbf{c}$', transform=ax_c.transAxes,
              fontsize=15, fontweight='bold', va='top')

    # ── Suptitle ──
    lam = config.get('lambda_base', '?')
    gam = config.get('gamma', '?')
    M   = config.get('M', '?')
    fig.suptitle(
        f'GP Population Coding \u2014 Bays & Brady (2024) Fig 5 Equivalent '
        f'($\\lambda$={lam}, $\\gamma$={gam} Hz, M={M})',
        fontsize=12, fontweight='bold', y=0.97,
    )

    outpath = Path(output_dir) / 'figure_5_sd_vs_setsize.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot:
        plt.show()
    plt.close(fig)

    np.savez(
        Path(output_dir) / 'figure_5_data.npz',
        set_sizes=np.array(set_sizes),
        sd_mean=sd_mean, sd_se=sd_se,
        theoretical_sd=theoretical_sd,
        pooled_sd_deg=pooled_sd,
        bin_centers=bins, pooled_density=pooled_density,
        **{f'sd_N{N}': summary[N]['sd_mean'] for N in set_sizes},
    )


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == '__main__':
    config = {
        'M': 100, 'n_theta': 64, 'n_trials': 10_000,
        'T_d': 0.1, 'sigma_sq': 1e-6,
        'lambda_base': 0.5, 'gamma': 100.0,
        'set_sizes': [1, 2, 3, 4, 5, 6, 7, 8],
        'seed': 42, 'n_seeds': 5,
    }
    print("Running Bays & Brady (2024) Figure 5 \u2014 GP Model: SD vs Set Size")
    print(f"  Set sizes: {config['set_sizes']}, Trials: {config['n_trials']}, "
          f"Neurons: {config['M']}, Seeds: {config['n_seeds']}")
    results = run_experiment(config)
    plot_results(results, 'results/nature_figure_5', show_plot=True)