"""
Bays (2014) Figure 1 d,e,f — GP-Based Equivalent

Recreates panels d (variance), e (kurtosis), f (power-law exponent)
using Gaussian Process tuning curves instead of von Mises.

Pipeline per (λ_base, γ) grid point:
    1. Generate M neurons with GP tuning (lengthscale = λ_base)
    2. For each trial:
       a. Sample true orientation θ_true
       b. Compute firing rates via DN:  r_i = γ · g_i(θ) / (σ² + M⁻¹ Σ_j g_j(θ))
       c. Generate Poisson spikes:      n_i ~ Poisson(r_i · T_d)
       d. ML decode:                    θ̂ = argmax_θ Σ_i n_i · log g_i(θ)
       e. Record circular error:        ε = θ̂ − θ (wrapped to [-π, π))
    3. Compute circular variance, kurtosis from errors
    4. Exponent: α = log₂(V(γ) / V(γ/2))

Circular statistics (Fisher, 1995; Bays 2014):
    m_n = mean(exp(i·n·ε))             nth trigonometric moment
    variance = −2·log|m_1|             circular variance (= squared circ. SD)
    kurtosis = (ρ₂·cos(μ₂ − 2μ₁) − ρ₁⁴) / (1 − ρ₁)²

Usage:
    from experiments.bays_equivalence.figure_1 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Tuple
import time

# ── Core module imports (DRY: no function redefinition) ──
from core.gaussian_process import (
    periodic_rbf_kernel,
    sample_gp_function,
)
from core.poisson_spike import generate_spikes
from core.ml_decoder import compute_circular_error


# ═══════════════════════════════════════════════════════════════════════════
# CIRCULAR STATISTICS (Bays 2014 definitions, not in core modules)
# ═══════════════════════════════════════════════════════════════════════════

def _circular_moments(errors: np.ndarray) -> Tuple[complex, complex]:
    """1st and 2nd uncentered trigonometric moments."""
    return np.mean(np.exp(1j * errors)), np.mean(np.exp(2j * errors))


def circular_variance(errors: np.ndarray) -> float:
    """σ² = −2·log(ρ₁). (Fisher 1995; Bays 2014 Methods)"""
    m1, _ = _circular_moments(errors)
    rho1 = np.clip(np.abs(m1), 1e-15, 1.0 - 1e-10)
    return -2.0 * np.log(rho1)


def circular_kurtosis(errors: np.ndarray) -> float:
    """κ = (ρ₂·cos(μ₂ − 2μ₁) − ρ₁⁴) / (1 − ρ₁)². (Fisher 1995; Bays 2014)"""
    m1, m2 = _circular_moments(errors)
    rho1, rho2 = np.abs(m1), np.abs(m2)
    mu1, mu2 = np.angle(m1), np.angle(m2)
    denom = (1.0 - rho1) ** 2
    if denom < 1e-15:
        return 0.0
    return (rho2 * np.cos(mu2 - 2 * mu1) - rho1 ** 4) / denom


# ═══════════════════════════════════════════════════════════════════════════
# POPULATION GENERATION (thin wrapper around core GP functions)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_population(
    M: int, n_theta: int, lengthscale: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate M neurons with GP tuning curves at a single lengthscale.

    Returns
    -------
    thetas : (n_theta,)
    g      : (M, n_theta) — driving inputs exp(f(θ))
    log_g  : (M, n_theta) — log driving inputs (for ML decoding)
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    K = periodic_rbf_kernel(thetas, lengthscale)

    f = np.zeros((M, n_theta))
    for i in range(M):
        f[i] = sample_gp_function(K, rng)

    g = np.exp(f)
    log_g = np.log(np.maximum(g, 1e-20))
    return thetas, g, log_g


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _run_trials_at_gain(
    g: np.ndarray,
    log_g: np.ndarray,
    thetas: np.ndarray,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    n_trials: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Run n_trials of encode → spike → decode at fixed gain γ.

    DN equation:  r_i(θ) = γ · g_i(θ) / (σ² + M⁻¹ Σ_j g_j(θ))
    ML decode:    θ̂ = argmax_θ  Σ_i n_i · log g_i(θ)
                  (constant DN term drops out — see ml_decoder.py Property 1)

    Returns errors array, shape (n_trials,).
    """
    M, n_theta = g.shape
    errors = np.empty(n_trials)

    for t in range(n_trials):
        # 1. Random true stimulus index
        idx_true = rng.randint(n_theta)

        # 2. Firing rates at true orientation with DN
        g_true = g[:, idx_true]                          # (M,)
        denom = sigma_sq + np.mean(g_true)
        rates = gamma * g_true / denom                   # (M,)

        # 3. Poisson spikes (uses core.poisson_spike)
        counts = generate_spikes(rates, T_d, rng)        # (M,)

        # 4. ML decode: argmax_θ Σ_i n_i · log g_i(θ)
        ll = counts @ log_g                              # (n_theta,)
        idx_hat = np.argmax(ll)

        # 5. Circular error (uses core.ml_decoder)
        errors[t] = compute_circular_error(thetas[idx_true], thetas[idx_hat])

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: Dict) -> Dict:
    """
    Sweep (λ_base, γ) grid → collect variance, kurtosis, exponent.

    Config keys
    -----------
    M             : neurons per population          (default 100)
    n_theta       : orientation bins                (default 64)
    n_trials      : trials per grid point           (default 10_000)
    T_d           : decoding window in seconds      (default 0.1)
    sigma_sq      : semi-saturation constant        (default 1e-6)
    n_grid        : grid resolution per axis        (default 25)
    lambda_range  : (lo, hi) for lengthscale        (default (0.1, 2.5))
    gamma_range   : (lo, hi) for gain in Hz         (default (1, 256))
    seed          : master seed                     (default 42)
    """
    M         = config.get('M', 100)
    n_theta   = config.get('n_theta', 64)
    n_trials  = config.get('n_trials', 10_000)
    T_d       = config.get('T_d', 0.1)
    sigma_sq  = config.get('sigma_sq', 1e-6)
    n_grid    = config.get('n_grid', 25)
    lam_lo, lam_hi = config.get('lambda_range', (0.1, 2.5))
    gam_lo, gam_hi = config.get('gamma_range', (1.0, 256.0))
    seed      = config.get('seed', 42)

    # Log-spaced grids (matching Bays's 50×50 log grid)
    lambdas = np.logspace(np.log2(lam_lo), np.log2(lam_hi), n_grid, base=2)
    gammas  = np.logspace(np.log2(gam_lo), np.log2(gam_hi), n_grid, base=2)

    variance_grid = np.full((n_grid, n_grid), np.nan)
    kurtosis_grid = np.full((n_grid, n_grid), np.nan)
    variance_half = np.full((n_grid, n_grid), np.nan)  # for exponent

    total = n_grid * n_grid
    t0 = time.time()

    for i, lam in enumerate(lambdas):
        # Population generated ONCE per lengthscale (tuning ≠ f(γ))
        thetas, g, log_g = _generate_population(M, n_theta, lam, seed + i * 1000)

        for j, gam in enumerate(gammas):
            idx = i * n_grid + j + 1
            if idx % max(1, total // 10) == 0:
                elapsed = time.time() - t0
                eta = (total - idx) / (idx / elapsed)
                print(f"  [{idx}/{total}] λ={lam:.3f} γ={gam:.1f}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            rng = np.random.RandomState(seed + i * n_grid + j)

            # Variance & kurtosis at γ
            errs = _run_trials_at_gain(g, log_g, thetas, gam, T_d, sigma_sq, n_trials, rng)
            variance_grid[j, i] = circular_variance(errs)
            kurtosis_grid[j, i] = circular_kurtosis(errs)

            # Variance at γ/2 (for power-law exponent)
            rng2 = np.random.RandomState(seed + i * n_grid + j + total)
            errs_half = _run_trials_at_gain(g, log_g, thetas, gam / 2, T_d, sigma_sq, n_trials, rng2)
            variance_half[j, i] = circular_variance(errs_half)

    # Exponent: α = log₂(V(γ) / V(γ/2))
    # "estimated from the change of variance resulting from halving the gain"
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(variance_half > 1e-15, variance_grid / variance_half, np.nan)
        exponent_grid = np.where(ratio > 0, np.log2(ratio), np.nan)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}m, "
          f"{n_trials * total * 2:.0e} total trials)")

    return {
        'lambdas': lambdas, 'gammas': gammas,
        'variance': variance_grid,
        'kurtosis': kurtosis_grid,
        'exponent': exponent_grid,
        'config': config,
        'elapsed_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING — Publication quality, matching Bays (2014) Fig 1 d,e,f layout
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Three-panel figure: d (variance), e (kurtosis), f (exponent)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lambdas = results['lambdas']
    gammas  = results['gammas']
    V, K, E = results['variance'], results['kurtosis'], results['exponent']

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, bottom=0.15, top=0.88)

    extent = [np.log2(lambdas[0]), np.log2(lambdas[-1]),
              np.log2(gammas[0]),  np.log2(gammas[-1])]

    def _fmt_ax(ax, label):
        ax.set_xlabel(r'lengthscale, $\lambda$')
        ax.set_ylabel(r'gain, $\gamma$ (Hz)')
        xt = np.array([0.125, 0.25, 0.5, 1.0, 2.0])
        xt = xt[(xt >= lambdas[0] * 0.9) & (xt <= lambdas[-1] * 1.1)]
        ax.set_xticks(np.log2(xt))
        ax.set_xticklabels([f'{v:g}' for v in xt])
        yt = np.array([1, 4, 16, 64, 256])
        yt = yt[(yt >= gammas[0] * 0.9) & (yt <= gammas[-1] * 1.1)]
        ax.set_yticks(np.log2(yt))
        ax.set_yticklabels([f'{v:g}' for v in yt])
        ax.text(-0.12, 1.06, f'$\\mathbf{{{label}}}$',
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')

    # ── d: Variance ──
    im0 = axes[0].imshow(
        np.clip(V, 1e-3, 10), origin='lower', aspect='auto', extent=extent,
        norm=mcolors.LogNorm(vmin=0.001, vmax=10), cmap='jet', interpolation='bilinear')
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label('variance')
    cb0.set_ticks([0.001, 0.01, 0.1, 1, 10])
    cb0.set_ticklabels(['.001', '.01', '.1', '1', '10'])
    _fmt_ax(axes[0], 'd')

    # ── e: Kurtosis ──
    im1 = axes[1].imshow(
        np.clip(K, 0.01, 100), origin='lower', aspect='auto', extent=extent,
        norm=mcolors.LogNorm(vmin=0.01, vmax=100), cmap='jet', interpolation='bilinear')
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label('kurtosis')
    cb1.set_ticks([0.01, 0.1, 1, 10, 100])
    cb1.set_ticklabels(['.01', '.1', '1', '10', '100'])
    _fmt_ax(axes[1], 'e')

    # ── f: Exponent ──
    im2 = axes[2].imshow(
        np.clip(E, -3, 0), origin='lower', aspect='auto', extent=extent,
        vmin=-3, vmax=0, cmap='jet', interpolation='bilinear')
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label('exponent')
    cb2.set_ticks([0, -1, -2, -3])
    _fmt_ax(axes[2], 'f')

    fig.suptitle('GP Population Coding — Bays (2014) Fig 1 d,e,f Equivalent',
                 fontsize=12, fontweight='bold', y=0.97)

    outpath = Path(output_dir) / 'figure_1_def.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot:
        plt.show()
    plt.close(fig)

    np.savez(Path(output_dir) / 'figure_1_data.npz',
             lambdas=lambdas, gammas=gammas, variance=V, kurtosis=K, exponent=E)


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = {
        'M': 100, 'n_theta': 64, 'n_trials': 10_000,
        'T_d': 0.1, 'sigma_sq': 1e-6, 'n_grid': 25,
        'lambda_range': (0.1, 2.5), 'gamma_range': (1.0, 256.0), 'seed': 42,
    }
    print("Running Bays (2014) Figure 1 d,e,f — GP Equivalent")
    print(f"  Grid: {config['n_grid']}x{config['n_grid']}, "
          f"Trials: {config['n_trials']}, Neurons: {config['M']}")
    results = run_experiment(config)
    plot_results(results, 'results/figure_1', show_plot=True)