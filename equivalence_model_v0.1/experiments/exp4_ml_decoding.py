"""
Experiment 4: Multi-Item ML Decoding — Efficient Factorized Version

=============================================================================
SUB-EXPERIMENTS
=============================================================================

Part A — ERROR SCALING (original):
    Circular std vs set size for multi-item and single-item baseline.
    Validates sqrt(l) scaling from the causal chain.

Part B — BIAS ANALYSIS:
    Mean signed circular error across 1000 trials per set size.
    An unbiased decoder should have E[error] ≈ 0 for all l.
    Uses n_orientations=10 candidate thetas, 100 neurons.

Part C — LENGTHSCALE SWEEP:
    Decoding error vs set size for multiple base lengthscales.
    lambda_base is divided by N ∈ {1, 2, 3, 4, 5, 6}, producing
    progressively sharper tuning curves.
    Sharper tuning → more informative Fisher information → lower error.

=============================================================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

from core.gaussian_process import generate_neuron_population
from core.poisson_spike import generate_spikes
from core.ml_decoder import (
    compute_spike_weighted_log_tuning,
    compute_marginal_log_likelihood_efficient,
    decode_ml_efficient,
    decode_ml,
    compute_circular_error,
    compute_circular_std,
    compare_complexity,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Exp4Config:
    """Configuration for Experiment 4."""
    # Population parameters
    n_neurons: int = 100
    n_orientations: int = 32
    n_locations: int = 16

    # DN parameters
    gamma: float = 100.0
    sigma_sq: float = 1e-6

    # Decoding parameters
    T_d: float = 0.1
    n_trials: int = 500
    set_sizes: Tuple = (2, 4, 6, 8)

    # GP parameters
    lambda_base: float = 0.5
    sigma_lambda: float = 0.3

    # Random seed
    seed: int = 42

    @property
    def total_activity(self) -> float:
        return self.gamma * self.n_neurons


# =============================================================================
# SHARED HELPERS
# =============================================================================

def extract_f_samples_for_locations(
    population: List[Dict],
    active_locations: Tuple[int, ...]
) -> List[np.ndarray]:
    """Extract log-rate tuning functions for active locations.

    Returns list of l arrays, each shape (N, n_theta).
    """
    N = len(population)
    n_theta = population[0]['f_samples'].shape[1]

    f_samples_list = []
    for loc in active_locations:
        f_k = np.zeros((N, n_theta))
        for i, neuron in enumerate(population):
            f_k[i, :] = neuron['f_samples'][loc, :]
        f_samples_list.append(f_k)

    return f_samples_list


def run_single_trial_efficient(
    population: List[Dict],
    theta_values: np.ndarray,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    active_locations: Tuple[int, ...],
    true_orientations: np.ndarray,
    cued_index: int,
    rng: np.random.RandomState
) -> Dict:
    """Run a single trial: DN → Poisson spikes → efficient ML decode."""
    N = len(population)

    f_samples_list = extract_f_samples_for_locations(population, active_locations)

    # True orientation → nearest grid index
    theta_indices = [np.argmin(np.abs(theta_values - t)) for t in true_orientations]

    # Pre-normalised rate (product over active locations in log space)
    log_r_pre = np.zeros(N)
    for k, f_k in enumerate(f_samples_list):
        log_r_pre += f_k[:, theta_indices[k]]
    r_pre = np.exp(log_r_pre)

    # Divisive normalisation
    D = sigma_sq + np.mean(r_pre)
    rates = gamma * r_pre / D

    # Poisson spikes
    spike_counts = rng.poisson(rates * T_d)

    # Efficient ML decoding
    theta_ml, ll_max, ll_marginal = decode_ml_efficient(
        spike_counts, f_samples_list, theta_values, cued_index
    )

    theta_true = true_orientations[cued_index]
    error = compute_circular_error(theta_true, theta_ml)

    return {
        'theta_true': theta_true,
        'theta_estimate': theta_ml,
        'error': error,
        'total_spikes': int(np.sum(spike_counts)),
        'mean_rate': float(np.mean(rates)),
    }


def _run_decoding_loop(
    population, theta_values, cfg, rng, n_trials, set_sizes, desc_prefix=""
):
    """Shared trial loop used by Parts A, B, and C."""
    results = {
        l: {'errors': [], 'theta_true': [], 'theta_est': []}
        for l in set_sizes
    }

    total_trials = len(set_sizes) * n_trials
    pbar = tqdm(total=total_trials, desc=f"{desc_prefix}Decoding")

    for l in set_sizes:
        for _ in range(n_trials):
            active_locations = tuple(rng.choice(
                cfg.n_locations, size=l, replace=False
            ))
            true_orientations = rng.uniform(0, 2 * np.pi, size=l)
            cued_index = rng.randint(l)

            trial = run_single_trial_efficient(
                population, theta_values,
                cfg.gamma, cfg.sigma_sq, cfg.T_d,
                active_locations, true_orientations, cued_index, rng
            )

            results[l]['errors'].append(trial['error'])
            results[l]['theta_true'].append(trial['theta_true'])
            results[l]['theta_est'].append(trial['theta_estimate'])
            pbar.update(1)

        errors = np.array(results[l]['errors'])
        results[l]['errors'] = errors
        results[l]['theta_true'] = np.array(results[l]['theta_true'])
        results[l]['theta_est'] = np.array(results[l]['theta_est'])
        results[l]['circular_std'] = compute_circular_std(errors)
        results[l]['circular_std_deg'] = np.degrees(results[l]['circular_std'])
        results[l]['mean_error'] = float(np.mean(errors))
        results[l]['mean_error_deg'] = float(np.degrees(np.mean(errors)))
        results[l]['mean_absolute_error'] = float(np.mean(np.abs(errors)))
        results[l]['mae_deg'] = float(np.degrees(np.mean(np.abs(errors))))

    pbar.close()
    return results


# =============================================================================
# PART A — ERROR SCALING (original experiment)
# =============================================================================

def _run_part_a(population, theta_values, cfg, rng, verbose=True):
    """Multi-item efficient decoding + single-item baseline."""
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART A: ERROR SCALING  (sqrt(l) validation)")
        print(f"  {'─'*60}")
        print(f"    n_trials={cfg.n_trials}  n_theta={cfg.n_orientations}")

    multi = _run_decoding_loop(
        population, theta_values, cfg, rng,
        cfg.n_trials, cfg.set_sizes, desc_prefix="A multi  "
    )

    # Single-item baseline
    N = len(population)
    base_tuning = np.zeros((N, cfg.n_orientations))
    for i, neuron in enumerate(population):
        base_tuning[i, :] = np.exp(neuron['f_samples'][0, :])
    pop_mean = np.mean(base_tuning, axis=0, keepdims=True)
    base_tuning_dn = cfg.gamma * base_tuning / (cfg.sigma_sq + pop_mean)

    single = {l: {'errors': []} for l in cfg.set_sizes}
    for l in cfg.set_sizes:
        scaled = base_tuning_dn / l
        for _ in range(cfg.n_trials):
            theta_true = rng.uniform(0, 2 * np.pi)
            theta_idx = np.argmin(np.abs(theta_values - theta_true))
            rates = scaled[:, theta_idx]
            spikes = generate_spikes(rates, cfg.T_d, rng)
            theta_ml, _, _ = decode_ml(spikes, scaled, theta_values, cfg.T_d)
            single[l]['errors'].append(compute_circular_error(theta_true, theta_ml))
        errors = np.array(single[l]['errors'])
        single[l]['errors'] = errors
        single[l]['circular_std'] = compute_circular_std(errors)
        single[l]['circular_std_deg'] = np.degrees(single[l]['circular_std'])

    if verbose:
        print(f"\n  {'Set Size':<10} {'Multi σ (deg)':<18} {'Single σ (deg)':<18}")
        print("  " + "-" * 46)
        for l in cfg.set_sizes:
            print(f"  {l:<10} {multi[l]['circular_std_deg']:<18.2f} "
                  f"{single[l]['circular_std_deg']:<18.2f}")

    stds = [multi[l]['circular_std_deg'] for l in cfg.set_sizes]
    normalised = [stds[i] / np.sqrt(l) for i, l in enumerate(cfg.set_sizes)]
    cv = np.std(normalised) / np.mean(normalised) if len(normalised) > 1 else 0.0

    return {
        'multi_item': multi,
        'single_item': single,
        'scaling': {
            'empirical_std': stds,
            'normalised_by_sqrt_l': normalised,
            'cv_normalised': cv,
        }
    }


# =============================================================================
# PART B — BIAS ANALYSIS
# =============================================================================

def _run_part_b(cfg, rng, verbose=True):
    """Bias analysis: mean signed error over many trials.

    Uses n_orientations=10 (coarse grid), n_trials=1000, N=100.
    """
    bias_cfg = Exp4Config(
        n_neurons=100,
        n_orientations=10,
        n_locations=cfg.n_locations,
        gamma=cfg.gamma,
        sigma_sq=cfg.sigma_sq,
        T_d=cfg.T_d,
        n_trials=1000,
        set_sizes=cfg.set_sizes,
        lambda_base=cfg.lambda_base,
        sigma_lambda=cfg.sigma_lambda,
        seed=cfg.seed + 500,
    )

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART B: BIAS ANALYSIS")
        print(f"  {'─'*60}")
        print(f"    N=100  n_theta=10  n_trials=1000")

    population = generate_neuron_population(
        n_neurons=bias_cfg.n_neurons,
        n_orientations=bias_cfg.n_orientations,
        n_locations=bias_cfg.n_locations,
        base_lengthscale=bias_cfg.lambda_base,
        lengthscale_variability=bias_cfg.sigma_lambda,
        seed=bias_cfg.seed,
    )
    theta_values = population[0]['orientations']

    bias_rng = np.random.RandomState(bias_cfg.seed + 1)
    results = _run_decoding_loop(
        population, theta_values, bias_cfg, bias_rng,
        bias_cfg.n_trials, bias_cfg.set_sizes, desc_prefix="B bias   "
    )

    if verbose:
        print(f"\n  {'Set Size':<10} {'Mean Error (deg)':<20} {'σ (deg)':<15} {'|bias|/σ':<10}")
        print("  " + "-" * 55)
        for l in bias_cfg.set_sizes:
            me = results[l]['mean_error_deg']
            sd = results[l]['circular_std_deg']
            ratio = abs(me) / sd if sd > 0 else 0.0
            print(f"  {l:<10} {me:<20.3f} {sd:<15.2f} {ratio:<10.4f}")

    return {'bias_results': results, 'bias_config': bias_cfg}


# =============================================================================
# PART C — LENGTHSCALE SWEEP
# =============================================================================

# Absolute λ values chosen to stay well above the 0.1 floor in the core
# lengthscale generator, and well above the grid spacing (2π/n_theta).
# Ordered broad → sharp.  All use σ_lambda = 0 to isolate the base effect.
SWEEP_LAMBDAS = [2.5, 1.8, 1.2, 0.8, 0.5, 0.3]


def _run_part_c(cfg, rng, verbose=True):
    """Decoding error vs set size for a range of base lengthscales.

    σ_lambda is fixed at 0 so every neuron at every location gets exactly
    the stated λ — this isolates the effect of tuning width from random
    heterogeneity.  The same master seed is used for every condition so
    the *only* difference between runs is the kernel width.
    """
    lambdas = SWEEP_LAMBDAS
    n_trials_sweep = 500
    sweep_n_orientations = cfg.n_orientations

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART C: LENGTHSCALE SWEEP")
        print(f"  {'─'*60}")
        print(f"    lambdas={lambdas}  (σ_lambda=0, pure base)")
        print(f"    n_trials={n_trials_sweep}  n_theta={sweep_n_orientations}")

    sweep_results = {}

    for lam in lambdas:
        if verbose:
            print(f"\n    --- λ = {lam:.2f} ---")

        pop = generate_neuron_population(
            n_neurons=cfg.n_neurons,
            n_orientations=sweep_n_orientations,
            n_locations=cfg.n_locations,
            base_lengthscale=lam,
            lengthscale_variability=0.0,          # no jitter
            seed=cfg.seed,                         # same seed for all
        )
        theta_vals = pop[0]['orientations']

        sweep_cfg = Exp4Config(
            n_neurons=cfg.n_neurons,
            n_orientations=sweep_n_orientations,
            n_locations=cfg.n_locations,
            gamma=cfg.gamma,
            sigma_sq=cfg.sigma_sq,
            T_d=cfg.T_d,
            n_trials=n_trials_sweep,
            set_sizes=cfg.set_sizes,
            lambda_base=lam,
            sigma_lambda=0.0,
            seed=cfg.seed,
        )

        sweep_rng = np.random.RandomState(cfg.seed + 1)
        res = _run_decoding_loop(
            pop, theta_vals, sweep_cfg, sweep_rng,
            n_trials_sweep, cfg.set_sizes,
            desc_prefix=f"C λ={lam:<5.2f}"
        )

        stds_deg = [res[l]['circular_std_deg'] for l in cfg.set_sizes]
        sweep_results[lam] = {
            'lambda_effective': lam,
            'stds_deg': stds_deg,
            'per_set_size': res,
        }

        if verbose:
            tag = "  ".join(
                f"l={l}: {stds_deg[i]:.1f}°"
                for i, l in enumerate(cfg.set_sizes)
            )
            print(f"      {tag}")

    return {'sweep_results': sweep_results, 'lambdas': lambdas}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_experiment_4(config: Dict) -> Dict:
    """Run all three sub-experiments of Experiment 4."""
    cfg = Exp4Config(
        n_neurons=config.get('n_neurons', 100),
        n_orientations=config.get('n_orientations', 32),
        n_locations=config.get('n_locations', 16),
        gamma=config.get('gamma', 100.0),
        sigma_sq=config.get('sigma_sq', 1e-6),
        T_d=config.get('T_d', 0.1),
        n_trials=config.get('n_trials', 500),
        set_sizes=tuple(config.get('set_sizes', [2, 4, 6, 8])),
        seed=config.get('seed', 42),
        lambda_base=config.get('lambda_base', 0.5),
        sigma_lambda=config.get('sigma_lambda', 0.3),
    )

    print("=" * 70)
    print("EXPERIMENT 4: MULTI-ITEM ML DECODING")
    print("=" * 70)
    print(f"\n  N={cfg.n_neurons}  n_theta={cfg.n_orientations}  "
          f"L={cfg.n_locations}  T_d={cfg.T_d}  gamma={cfg.gamma}")
    print(f"  set_sizes={cfg.set_sizes}  seed={cfg.seed}")

    compare_complexity(cfg.n_neurons, cfg.n_orientations, list(cfg.set_sizes))

    rng = np.random.RandomState(cfg.seed)

    # Generate shared population for Part A
    population = generate_neuron_population(
        n_neurons=cfg.n_neurons,
        n_orientations=cfg.n_orientations,
        n_locations=cfg.n_locations,
        base_lengthscale=cfg.lambda_base,
        lengthscale_variability=cfg.sigma_lambda,
        seed=cfg.seed,
    )
    theta_values = population[0]['orientations']

    part_a = _run_part_a(population, theta_values, cfg, rng)
    part_b = _run_part_b(cfg, rng)
    part_c = _run_part_c(cfg, rng)

    print(f"\n{'='*70}")
    print("EXPERIMENT 4 COMPLETE")
    print(f"{'='*70}")

    return {
        'config': config,
        'exp_config': cfg,
        'theta_values': theta_values,
        # Part A
        'multi_item': part_a['multi_item'],
        'single_item': part_a['single_item'],
        'scaling': part_a['scaling'],
        # Part B
        'bias': part_b,
        # Part C
        'lengthscale_sweep': part_c,
        'method': 'efficient_factorized',
    }


# Backward-compatibility alias
run_experiment_4_efficient = run_experiment_4


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Generate all figures for Experiment 4 (Parts A, B, C)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("deep")

    cfg = results['exp_config']
    set_sizes = list(cfg.set_sizes)
    multi_item = results['multi_item']
    single_item = results['single_item']
    mi_std = [multi_item[l]['circular_std_deg'] for l in set_sizes]
    si_std = [single_item[l]['circular_std_deg'] for l in set_sizes]

    # ================================================================
    # PLOT 1: Part A — Error scaling + sqrt(l) check  (2 panels)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(set_sizes, mi_std, 'o-', color=palette[0], lw=2.5, ms=10,
            label='Multi-item (efficient)')
    ax.plot(set_sizes, si_std, 's--', color=palette[1], lw=2, ms=8,
            label='Single-item baseline')
    if len(set_sizes) > 1:
        ref = [mi_std[0] * np.sqrt(l / set_sizes[0]) for l in set_sizes]
        ax.plot(set_sizes, ref, ':', color='gray', lw=2, alpha=0.7,
                label=r'$\propto \sqrt{\ell}$')
    ax.set_xlabel(r'Set Size ($\ell$)')
    ax.set_ylabel('Circular Std (degrees)')
    ax.set_title('A. Decoding Error vs Set Size')
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)

    ax = axes[1]
    normalised = results['scaling']['normalised_by_sqrt_l']
    ax.plot(set_sizes, normalised, 'o-', color=palette[3], lw=2.5, ms=10)
    ax.axhline(np.mean(normalised), color='red', ls='--', lw=2,
               label=f'Mean = {np.mean(normalised):.2f} deg')
    ax.set_xlabel(r'Set Size ($\ell$)')
    ax.set_ylabel(r'Error Std / $\sqrt{\ell}$ (degrees)')
    cv = results['scaling']['cv_normalised']
    ax.set_title(f'B. $\\sqrt{{\\ell}}$ Scaling (CV = {cv:.3f})')
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)

    plt.suptitle('Experiment 4A: Error Scaling', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / 'exp4a_error_scaling.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp4a_error_scaling.png")

    # ================================================================
    # PLOT 2: Part A — Error distributions
    # ================================================================
    n_sizes = len(set_sizes)
    cols = min(n_sizes, 6)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    colors = sns.color_palette("coolwarm", n_sizes)
    if cols == 1:
        axes = [axes]
    for i, l in enumerate(set_sizes[:cols]):
        ax = axes[i]
        errors_deg = np.degrees(multi_item[l]['errors'])
        sns.histplot(errors_deg, kde=True, ax=ax, color=colors[i],
                     stat='density', alpha=0.6, bins=30)
        sd = multi_item[l]['circular_std_deg']
        ax.set_xlabel('Error (degrees)')
        ax.set_ylabel('Density' if i == 0 else '')
        ax.set_title(f'l = {l}\nσ = {sd:.1f}°')
        ax.set_xlim([-90, 90])
    plt.suptitle('Error Distributions (Multi-Item Decoder)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / 'exp4a_error_distributions.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp4a_error_distributions.png")

    # ================================================================
    # PLOT 3: Part B — Bias analysis  (2 panels)
    # ================================================================
    bias_res = results['bias']['bias_results']
    bias_cfg = results['bias']['bias_config']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: mean signed error (bias) with 95 % CI
    mean_errors_deg = [bias_res[l]['mean_error_deg'] for l in set_sizes]
    std_errors = [np.degrees(np.std(bias_res[l]['errors'])) for l in set_sizes]
    n_b = bias_cfg.n_trials
    ci95 = [1.96 * s / np.sqrt(n_b) for s in std_errors]

    ax = axes[0]
    ax.bar(set_sizes, mean_errors_deg, width=0.6, color=palette[0], alpha=0.7,
           edgecolor='black', linewidth=0.8)
    ax.errorbar(set_sizes, mean_errors_deg, yerr=ci95, fmt='none',
                color='black', capsize=6, capthick=1.5, lw=1.5)
    ax.axhline(0, color='red', ls='--', lw=1.5)
    ax.set_xlabel(r'Set Size ($\ell$)')
    ax.set_ylabel('Mean Signed Error (degrees)')
    ax.set_title('A. Decoder Bias')
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)

    # Panel 2: |bias| / sigma ratio
    ratios = [
        abs(bias_res[l]['mean_error_deg']) / bias_res[l]['circular_std_deg']
        if bias_res[l]['circular_std_deg'] > 0 else 0.0
        for l in set_sizes
    ]
    ax = axes[1]
    ax.bar(set_sizes, ratios, width=0.6, color=palette[2], alpha=0.7,
           edgecolor='black', linewidth=0.8)
    ax.axhline(0.05, color='red', ls='--', lw=1.2, label='5 % reference')
    ax.set_xlabel(r'Set Size ($\ell$)')
    ax.set_ylabel('|Bias| / σ')
    ax.set_title('B. Relative Bias')
    ax.set_xticks(set_sizes)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.suptitle(f'Experiment 4B: Bias Analysis (n_theta=10, trials={n_b})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / 'exp4b_bias.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp4b_bias.png")

    # ================================================================
    # PLOT 4: Part C — Lengthscale sweep  (dots, coloured by λ)
    # ================================================================
    sweep = results['lengthscale_sweep']
    sweep_res = sweep['sweep_results']
    lambdas = sweep['lambdas']

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min(lambdas), vmax=max(lambdas))
    jitter_w = 0.12

    rng_plot = np.random.default_rng(0)
    for lam in lambdas:
        stds = sweep_res[lam]['stds_deg']
        color = cmap(norm(lam))
        x_jit = [l + rng_plot.uniform(-jitter_w, jitter_w) for l in set_sizes]
        ax.plot(x_jit, stds, '-', color=color, lw=1.2, alpha=0.4, zorder=3)
        ax.scatter(x_jit, stds, s=90, color=color, edgecolors='black',
                   linewidths=0.6, zorder=5, label=f'λ = {lam:.2f}')

    ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
    ax.set_ylabel('Circular Std (degrees)', fontsize=13)
    ax.set_title('Experiment 4C: Lengthscale Sweep\n'
                 r'($\sigma_\lambda = 0$: pure base lengthscale, same seed)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(set_sizes)
    ax.legend(fontsize=9, title='Base Lengthscale', title_fontsize=10,
              loc='upper left', ncol=2)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out / 'exp4c_lengthscale_sweep.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp4c_lengthscale_sweep.png")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_neurons': 100,
        'n_orientations': 32,
        'n_locations': 16,
        'gamma': 100.0,
        'sigma_sq': 1e-6,
        'T_d': 0.1,
        'n_trials': 200,
        'seed': 42,
        'set_sizes': [2, 4, 6, 8],
    }
    results = run_experiment_4(config)
    plot_results(results, 'results/exp4', show_plot=True)