"""
Bays (2014) Figure 4 — GP-Based Equivalent

Effects of variations in tuning and noise correlations on decoding errors
at low gain.  Six panels:

    a — Broad GP tuning   (λ_base = 1.0)
    b — Narrow GP tuning  (λ_base = 0.3)
    c — Narrow + baseline activity  (λ_base = 0.3, baseline_frac = 0.25)
    d — Heterogeneous tuning  (random λ, amplitude, baseline per neuron)
    e — Cosine tuning  (half-wave rectified cosine, replacing GP)
    f — Correlated activity  (short-range noise correlations, c₀ = 0.25)

=============================================================================
GP EQUIVALENTS OF BAYS'S MODIFICATIONS
=============================================================================

Bays's original uses von Mises tuning with width ω.  Our GP framework
replaces ω with the kernel lengthscale λ_base.  The six panels test
robustness of the key finding (errors deviate from normality as gain
decreases) under progressive relaxation of model assumptions.

For panels a–c we use the same single-location pipeline as figure_1:
    1. Generate M neurons with GP tuning
    2. DN:  r_i = γ · g_i(θ) / (σ² + M⁻¹ Σ_j g_j(θ))
    3. Poisson spikes
    4. Full Poisson ML decode (rate-penalty retained)

Panels d–e modify the *population generation* step.
Panel f modifies the *spike generation* step (correlated Poisson).

Each panel sweeps gain γ through the same set of values and plots:
    - Top row:    example driving inputs / correlation matrix
    - Middle row: normalised error distributions (colour = gain)
    - Bottom row: deviation from matched circular normal

Two population sizes are tested: M = 100 (solid) and M = 1000 (dashed).

Usage:
    from experiments.bays_equivalence.figure_4 import run_experiment, plot_results
    results = run_experiment(config)
    plot_results(results, output_dir)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

from core.encoder.gaussian_process import periodic_rbf_kernel, sample_gp_function
from core.encoder.poisson_spike import generate_spikes
from core.encoder.divisive_normalization import dn_pointwise
from core.decoder.ml_decoder import compute_log_likelihood, compute_circular_error

from experiments.bays_equivalence.bays_utils import (
    circular_variance,
    circular_kurtosis,
    circular_moments,
    compute_deviation_from_normal,
)


# =============================================================================
# POPULATION GENERATORS (one per panel type)
# =============================================================================

def _make_gp_population(
    M: int, n_theta: int, lengthscale: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard homogeneous GP population.  Returns (thetas, g)."""
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    K = periodic_rbf_kernel(thetas, lengthscale)
    g = np.zeros((M, n_theta))
    for i in range(M):
        g[i] = np.exp(sample_gp_function(K, rng))
    return thetas, g


def _make_gp_population_with_baseline(
    M: int, n_theta: int, lengthscale: float, baseline_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """GP population + constant baseline floor.  Returns (thetas, g)."""
    thetas, g_raw = _make_gp_population(M, n_theta, lengthscale, seed)
    if baseline_frac > 1e-10:
        mean_peak = np.mean(np.max(g_raw, axis=1))
        b0 = baseline_frac * mean_peak / (1.0 - baseline_frac)
        g_raw = g_raw + b0
    return thetas, g_raw


def _make_heterogeneous_population(
    M: int,
    n_theta: int,
    lambda_mean: float,
    lambda_std: float,
    amp_mean: float,
    amp_std: float,
    baseline_mean: float,
    baseline_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heterogeneous GP population: each neuron gets its own lengthscale,
    amplitude scaling, and baseline — all drawn from truncated normals.

    GP equivalent of Bays's Eq. 17:
        f_ij(θ) = a_ij · exp(κ_ij⁻¹(cos(θ_ij − θ) − 1)) + f(0)_ij
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    g = np.zeros((M, n_theta))

    for i in range(M):
        lam_i = max(0.05, rng.normal(lambda_mean, lambda_std))
        amp_i = max(0.01, rng.normal(amp_mean, amp_std))
        bl_i  = max(0.0,  rng.normal(baseline_mean, baseline_std))

        K = periodic_rbf_kernel(thetas, lam_i)
        f_i = sample_gp_function(K, rng)
        g[i] = amp_i * np.exp(f_i) + bl_i

    return thetas, g


def _make_cosine_population(
    M: int,
    n_theta: int,
    amp_mean: float,
    amp_std: float,
    baseline_mean: float,
    baseline_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Half-wave rectified cosine tuning (GP equivalent of Bays's Eq. 18).

    g_i(θ) = a_i · [cos(θ_i − θ)]₊ + f(0)_i

    Preferred orientations θ_i are uniform on the circle.
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    prefs = rng.uniform(-np.pi, np.pi, M)
    g = np.zeros((M, n_theta))

    for i in range(M):
        amp_i = max(0.01, rng.normal(amp_mean, amp_std))
        bl_i  = max(0.0,  rng.normal(baseline_mean, baseline_std))
        cos_tuning = np.maximum(0.0, np.cos(thetas - prefs[i]))
        g[i] = amp_i * cos_tuning + bl_i

    return thetas, g


# =============================================================================
# CORRELATED SPIKE GENERATION (latent Gaussian method)
# =============================================================================

def _generate_correlated_spikes(
    rates: np.ndarray,
    T_d: float,
    correlation_matrix: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Generate correlated Poisson spike counts via the latent Gaussian method
    (Macke et al. 2008).

    1. Sample z ~ N(0, Σ) where Σ is calibrated from the desired
       spike-count correlation matrix.
    2. Transform marginals to Poisson via CDF matching:
       n_i = F_Poisson⁻¹(Φ(z_i); λ_i)

    For simplicity we use a first-order approximation where the
    Gaussian correlation ≈ the desired Poisson correlation (valid when
    rates are not too low).

    Parameters
    ----------
    rates : (M,) firing rates in Hz
    T_d : float, decoding window
    correlation_matrix : (M, M) desired pairwise correlations
    rng : RandomState
    """
    M = len(rates)
    lambdas = rates * T_d
    lambdas = np.maximum(lambdas, 1e-10)

    # Cholesky of correlation matrix
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue repair
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        C_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(C_fixed, 1.0)
        L = np.linalg.cholesky(C_fixed)

    # Sample correlated Gaussians
    z = L @ rng.randn(M)

    # Transform to Poisson via CDF matching
    from scipy.stats import norm, poisson
    u = norm.cdf(z)  # uniform marginals
    counts = poisson.ppf(u, mu=lambdas).astype(int)
    counts = np.maximum(counts, 0)

    return counts


def _build_correlation_matrix(
    M: int,
    preferred_orientations: np.ndarray,
    c0: float,
) -> np.ndarray:
    """
    Short-range pairwise correlation matrix (Bays's Eq. 19):
        c_{ij} = c₀ · exp(−|θ_i − θ_j|)

    where |·| is circular distance.
    """
    C = np.eye(M)
    for i in range(M):
        for j in range(i + 1, M):
            d = abs(preferred_orientations[i] - preferred_orientations[j])
            d = min(d, 2 * np.pi - d)
            c = c0 * np.exp(-d)
            C[i, j] = c
            C[j, i] = c
    return C


# =============================================================================
# TRIAL ENGINE (single-location, full Poisson ML)
# =============================================================================

def _run_trials(
    g: np.ndarray,
    thetas: np.ndarray,
    gamma: float,
    T_d: float,
    sigma_sq: float,
    n_trials: int,
    rng: np.random.RandomState,
    corr_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run n_trials of encode → spike → decode.

    If corr_matrix is provided, uses correlated spike generation (panel f).
    Otherwise uses independent Poisson (panels a–e).
    """
    M, n_theta = g.shape
    errors = np.empty(n_trials)

    for t in range(n_trials):
        idx_true = rng.randint(n_theta)
        rates = dn_pointwise(g[:, idx_true], gamma, sigma_sq)

        if corr_matrix is not None:
            counts = _generate_correlated_spikes(rates, T_d, corr_matrix, rng)
        else:
            counts = generate_spikes(rates, T_d, rng)

        ll = compute_log_likelihood(counts, g, T_d)
        idx_hat = np.argmax(ll)
        errors[t] = compute_circular_error(thetas[idx_true], thetas[idx_hat])

    return errors


# =============================================================================
# SINGLE PANEL RUNNER
# =============================================================================

def _run_panel(
    panel_id: str,
    gammas: np.ndarray,
    pop_sizes: List[int],
    n_trials: int,
    T_d: float,
    sigma_sq: float,
    n_bins: int,
    config: Dict,
    seed: int,
) -> Dict:
    """
    Run all (γ, M) combinations for one panel.

    Returns dict with 'errors', 'variance', 'kurtosis', 'deviation'
    keyed by (gamma, M).
    """
    thetas = None
    results = {}

    for M in pop_sizes:
        # --- Generate population ---
        if panel_id == 'a':
            thetas, g = _make_gp_population(
                M, config['n_theta'], config['lambda_broad'], seed + M)

        elif panel_id == 'b':
            thetas, g = _make_gp_population(
                M, config['n_theta'], config['lambda_narrow'], seed + M)

        elif panel_id == 'c':
            thetas, g = _make_gp_population_with_baseline(
                M, config['n_theta'], config['lambda_narrow'],
                config['baseline_frac'], seed + M)

        elif panel_id == 'd':
            thetas, g = _make_heterogeneous_population(
                M, config['n_theta'],
                lambda_mean=config['lambda_narrow'],
                lambda_std=config.get('lambda_std', 0.1),
                amp_mean=1.0, amp_std=0.5,
                baseline_mean=0.25, baseline_std=0.125,
                seed=seed + M)

        elif panel_id == 'e':
            thetas, g = _make_cosine_population(
                M, config['n_theta'],
                amp_mean=1.0, amp_std=0.5,
                baseline_mean=0.25, baseline_std=0.125,
                seed=seed + M)

        elif panel_id == 'f':
            thetas, g = _make_gp_population(
                M, config['n_theta'], config['lambda_narrow'], seed + M)
            # Build correlation matrix
            prefs = np.linspace(-np.pi, np.pi, M, endpoint=False)
            corr_matrix = _build_correlation_matrix(M, prefs, config.get('c0', 0.25))
        else:
            raise ValueError(f"Unknown panel: {panel_id}")

        # --- Sweep gains ---
        for gamma in gammas:
            rng = np.random.RandomState(seed + M + int(gamma * 100))

            corr_mat = corr_matrix if panel_id == 'f' else None
            errs = _run_trials(g, thetas, gamma, T_d, sigma_sq, n_trials, rng, corr_mat)

            dev = compute_deviation_from_normal(errs, n_bins)
            results[(gamma, M)] = {
                'errors': errs,
                'variance': circular_variance(errs),
                'kurtosis': circular_kurtosis(errs),
                'deviation': dev,
            }

        # Store example tuning curves for top-row plot
        if f'g_example_{M}' not in results:
            results[f'g_example_{M}'] = g[:min(20, M)]
            results['thetas'] = thetas
            if panel_id == 'f':
                results[f'corr_matrix_{M}'] = corr_matrix

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(config: Dict) -> Dict:
    """
    Run all six panels.

    Config keys
    -----------
    n_theta        : orientation bins                (default 64)
    n_trials       : trials per (γ, M) point         (default 10_000)
    T_d            : decoding window (s)             (default 0.1)
    sigma_sq       : semi-saturation constant        (default 1e-6)
    lambda_broad   : lengthscale for panel a         (default 1.0)
    lambda_narrow  : lengthscale for panels b–d,f    (default 0.3)
    lambda_std     : std of lengthscale (panel d)    (default 0.1)
    baseline_frac  : baseline for panel c            (default 0.25)
    c0             : peak correlation (panel f)      (default 0.25)
    gammas         : gain values to sweep            (default [1,2,4,8,16,32,64,128])
    pop_sizes      : population sizes                (default [100, 1000])
    n_bins         : histogram bins                  (default 50)
    seed           : master seed                     (default 42)
    """
    defaults = {
        'n_theta': 64,
        'n_trials': 10_000,
        'T_d': 0.1,
        'sigma_sq': 1e-6,
        'lambda_broad': 1.0,
        'lambda_narrow': 0.3,
        'lambda_std': 0.1,
        'baseline_frac': 0.25,
        'c0': 0.25,
        'gammas': [1, 2, 4, 8, 16, 32, 64, 128],
        'pop_sizes': [100, 1000],
        'n_bins': 50,
        'seed': 42,
    }
    cfg = {**defaults, **config}

    gammas = np.array(cfg['gammas'], dtype=float)
    pop_sizes = cfg['pop_sizes']
    panels = ['a', 'b', 'c', 'd', 'e', 'f']

    t0 = time.time()
    all_panels = {}

    for pid in panels:
        print(f"  Panel {pid}...", end=" ", flush=True)
        panel_t0 = time.time()

        all_panels[pid] = _run_panel(
            pid, gammas, pop_sizes,
            cfg['n_trials'], cfg['T_d'], cfg['sigma_sq'], cfg['n_bins'],
            cfg, cfg['seed'],
        )
        print(f"({time.time() - panel_t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return {
        'panels': all_panels,
        'gammas': gammas,
        'pop_sizes': pop_sizes,
        'config': cfg,
        'elapsed_seconds': elapsed,
    }


# =============================================================================
# PLOTTING — 3-row × 6-column layout matching Bays (2014) Fig 4
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """
    Three rows × six columns:
        Top:    example tuning curves / correlation matrix
        Middle: normalised error distributions (colour = gain)
        Bottom: deviation from circular normal
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    panels = results['panels']
    gammas = results['gammas']
    pop_sizes = results['pop_sizes']
    panel_ids = ['a', 'b', 'c', 'd', 'e', 'f']

    titles = ['Broad GP\ntuning', 'Narrow GP\ntuning', 'Baseline\nactivity',
              'Heterogeneous\nGP tuning', 'Cosine\ntuning', 'Correlated\nactivity']

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.6,
    })

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 6, hspace=0.45, wspace=0.30,
                           left=0.04, right=0.97, bottom=0.06, top=0.90)

    # Gain → colour map (red = low gain, blue = high gain)
    gain_cmap = plt.cm.RdYlBu
    gain_norm = mcolors.LogNorm(vmin=min(gammas), vmax=max(gammas))
    M_ref = pop_sizes[0]  # solid lines

    for col, pid in enumerate(panel_ids):
        pdata = panels[pid]
        thetas = pdata['thetas']

        # ── Top row: example tuning curves ──
        ax_top = fig.add_subplot(gs[0, col])
        if pid == 'f' and f'corr_matrix_{M_ref}' in pdata:
            # Show correlation matrix
            C = pdata[f'corr_matrix_{M_ref}']
            ax_top.imshow(C, cmap='gray_r', origin='lower', aspect='auto',
                          extent=[-np.pi, np.pi, -np.pi, np.pi])
            ax_top.set_xlabel('orientation')
            ax_top.set_ylabel('orientation')
        else:
            g_ex = pdata[f'g_example_{M_ref}']
            for i in range(min(g_ex.shape[0], 20)):
                ax_top.plot(thetas, g_ex[i], 'k-', linewidth=0.5, alpha=0.6)
            ax_top.set_xlabel('orientation')
            ax_top.set_ylabel('driving input')

        ax_top.set_title(titles[col], fontsize=10, fontweight='bold')
        ax_top.text(-0.08, 1.12, f'$\\mathbf{{{pid}}}$',
                    transform=ax_top.transAxes, fontsize=14,
                    fontweight='bold', va='top')

        # ── Middle row: normalised error distributions ──
        ax_mid = fig.add_subplot(gs[1, col])
        for M in pop_sizes:
            ls = '-' if M == pop_sizes[0] else '--'
            for gamma in gammas:
                key = (gamma, M)
                if key not in pdata:
                    continue
                dev = pdata[key]['deviation']
                emp = dev['empirical']
                if emp.max() > 0:
                    emp_norm = emp / emp.max()
                else:
                    emp_norm = emp
                c = gain_cmap(gain_norm(gamma))
                ax_mid.plot(dev['bin_centers'], emp_norm,
                            color=c, linewidth=1.0, linestyle=ls, alpha=0.85)

        ax_mid.set_xlim(-np.pi, np.pi)
        ax_mid.set_ylim(0, 1.05)
        ax_mid.set_xticks([-np.pi, 0, np.pi])
        ax_mid.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax_mid.set_xlabel('error')
        if col == 0:
            ax_mid.set_ylabel('normalised\nprobability')

        # ── Bottom row: deviation from circular normal ──
        ax_bot = fig.add_subplot(gs[2, col])
        for M in pop_sizes:
            ls = '-' if M == pop_sizes[0] else '--'
            for gamma in gammas:
                key = (gamma, M)
                if key not in pdata:
                    continue
                dev = pdata[key]['deviation']
                c = gain_cmap(gain_norm(gamma))
                ax_bot.plot(dev['bin_centers'], dev['deviation'],
                            color=c, linewidth=1.0, linestyle=ls, alpha=0.85)

        ax_bot.axhline(0, color='gray', linewidth=0.4)
        ax_bot.set_xlim(-np.pi, np.pi)
        ax_bot.set_xticks([-np.pi, 0, np.pi])
        ax_bot.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax_bot.set_xlabel('error')
        if col == 0:
            ax_bot.set_ylabel(r'$\Delta$ probability')

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    handles.append(Line2D([0], [0], color='k', linewidth=1.0, linestyle='-',
                          label=f'M = {pop_sizes[0]}'))
    if len(pop_sizes) > 1:
        handles.append(Line2D([0], [0], color='k', linewidth=1.0, linestyle='--',
                              label=f'M = {pop_sizes[1]}'))
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.04, 0.01),
               fontsize=9, frameon=False)

    # Gain colorbar
    sm = plt.cm.ScalarMappable(cmap=gain_cmap, norm=gain_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.75, 0.015, 0.20, 0.015])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'gain, $\gamma$', fontsize=9)
    cb.set_ticks(gammas)
    cb.set_ticklabels([str(int(g)) for g in gammas])

    fig.suptitle('GP Population Coding — Bays (2014) Fig 4 Equivalent\n'
                 'Robustness of error distributions under tuning and noise variations',
                 fontsize=12, fontweight='bold', y=0.97)

    outpath = Path(output_dir) / 'figure_4_robustness.png'
    fig.savefig(outpath, dpi=300)
    print(f"  Saved: {outpath}")
    if show_plot:
        plt.show()
    plt.close(fig)


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_theta': 64,
        'n_trials': 10_000,
        'T_d': 0.1,
        'sigma_sq': 1e-6,
        'lambda_broad': 1.0,
        'lambda_narrow': 0.3,
        'lambda_std': 0.1,
        'baseline_frac': 0.25,
        'c0': 0.25,
        'gammas': [1, 2, 4, 8, 16, 32, 64, 128],
        'pop_sizes': [100, 1000],
        'n_bins': 50,
        'seed': 42,
    }
    print("Running Bays (2014) Figure 4 — GP Equivalent")
    print("  Robustness: broad, narrow, baseline, heterogeneous, cosine, correlated")
    print(f"  Gains: {config['gammas']}")
    print(f"  Pop sizes: {config['pop_sizes']}")
    results = run_experiment(config)
    plot_results(results, 'results/figure_4', show_plot=True)