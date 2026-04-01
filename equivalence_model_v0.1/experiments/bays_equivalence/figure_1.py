"""
Bays (2014) Figure 1 d,e,f — GP-Based Equivalent

Recreates panels d (variance), e (kurtosis), f (power-law exponent)
using Gaussian Process tuning curves instead of von Mises.

=============================================================================
KEY DESIGN CHOICE: FULL POISSON ML DECODER (SINGLE-LOCATION)
=============================================================================

For GP tuning curves the population does NOT tile uniformly, so
Σ_i g_i(θ) fluctuates across θ.  We therefore use the FULL Poisson
log-likelihood (retaining the rate-penalty term):

    θ̂ = argmax_θ  Σ_i [ n_i · log g_i(θ) − g_i(θ) · T_d ]

This is the correct single-location ML decoder for our generative model.

Pipeline per (λ_base, γ) grid point:
    1. Generate M neurons with GP tuning (lengthscale = λ_base)
    2. For each trial:
       a. Sample true orientation θ_true
       b. DN:  r_i = γ · g_i(θ) / (σ² + M⁻¹ Σ_j g_j(θ))   [dn_pointwise]
       c. Poisson spikes: n_i ~ Poisson(r_i · T_d)
       d. Full ML decode
       e. Record circular error
    3. Compute circular variance, kurtosis
    4. Exponent: α = log₂(V(γ) / V(γ/2))

Usage:
    from experiments.bays_equivalence.figure_1 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict
import time

from core.encoder.poisson_spike import generate_spikes
from core.encoder.divisive_normalization import dn_pointwise
from core.decoder.ml_decoder import compute_log_likelihood, compute_circular_error

from experiments.bays_equivalence.bays_utils import (
    circular_variance,
    circular_kurtosis,
    generate_population,
)


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL ENGINE — SINGLE-LOCATION, FULL POISSON ML DECODER
# ═══════════════════════════════════════════════════════════════════════════

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

    DN equation (Eq. 6, single-location case):
        r_i(θ) = γ · g_i(θ) / (σ² + M⁻¹ Σ_j g_j(θ))

    Full Poisson ML decode (NOT Bays's simplified version):
        θ̂ = argmax_θ  Σ_i [ n_i · log g_i(θ)  −  g_i(θ) · T_d ]

    We decode using the DRIVING INPUTS g_i(θ), not the normalised rates.
    This is valid because the DN denominator is the same for all neurons
    at a given θ, so the γ and denominator terms factor out or cancel
    in the argmax.

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

    lambdas = np.logspace(np.log2(lam_lo), np.log2(lam_hi), n_grid, base=2)
    gammas  = np.logspace(np.log2(gam_lo), np.log2(gam_hi), n_grid, base=2)

    variance_grid = np.full((n_grid, n_grid), np.nan)
    kurtosis_grid = np.full((n_grid, n_grid), np.nan)
    variance_half = np.full((n_grid, n_grid), np.nan)

    total = n_grid * n_grid
    t0 = time.time()

    for i, lam in enumerate(lambdas):
        # Population generated ONCE per lengthscale (tuning ≠ f(γ))
        thetas, f_all = generate_population(M, n_theta, lam, n_locations=1, seed=seed + i * 1000)
        g = np.exp(f_all[0])

        for j, gam in enumerate(gammas):
            idx = i * n_grid + j + 1
            if idx % max(1, total // 10) == 0:
                elapsed = time.time() - t0
                eta = (total - idx) / (idx / elapsed)
                print(f"  [{idx}/{total}] λ={lam:.3f} γ={gam:.1f}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            rng = np.random.RandomState(seed + i * n_grid + j)

            errs = _run_trials_at_gain(g, thetas, gam, T_d, sigma_sq, n_trials, rng)
            variance_grid[j, i] = circular_variance(errs)
            kurtosis_grid[j, i] = circular_kurtosis(errs)

            rng2 = np.random.RandomState(seed + i * n_grid + j + total)
            errs_half = _run_trials_at_gain(g, thetas, gam / 2, T_d, sigma_sq, n_trials, rng2)
            variance_half[j, i] = circular_variance(errs_half)

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
# PLOTTING
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

    im0 = axes[0].imshow(
        np.clip(V, 1e-3, 10), origin='lower', aspect='auto', extent=extent,
        norm=mcolors.LogNorm(vmin=0.001, vmax=10), cmap='jet', interpolation='bilinear')
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label('variance')
    cb0.set_ticks([0.001, 0.01, 0.1, 1, 10])
    cb0.set_ticklabels(['.001', '.01', '.1', '1', '10'])
    _fmt_ax(axes[0], 'd')

    im1 = axes[1].imshow(
        np.clip(K, 0.01, 100), origin='lower', aspect='auto', extent=extent,
        norm=mcolors.LogNorm(vmin=0.01, vmax=100), cmap='jet', interpolation='bilinear')
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label('kurtosis')
    cb1.set_ticks([0.01, 0.1, 1, 10, 100])
    cb1.set_ticklabels(['.01', '.1', '1', '10', '100'])
    _fmt_ax(axes[1], 'e')

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
    print("  Decoder: FULL Poisson ML (both terms retained)")
    results = run_experiment(config)
    plot_results(results, 'results/figure_1', show_plot=True)