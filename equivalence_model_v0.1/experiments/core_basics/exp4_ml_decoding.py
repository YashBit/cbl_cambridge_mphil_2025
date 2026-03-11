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

Part D — PRECISION COMPARISON:
    Broad (λ=3) vs sharp (λ=1) at high SNR.

Part E — MULTI-SEED SUB-UNIT LAMBDA SWEEP:
    5 evenly-spaced lambdas all < 1, including l=1 baseline, averaged
    over multiple independent seeds.  Multi-seed design provides mean ±
    SEM error bars, eliminating reliance on a single stochastic
    realisation.  l=1 anchors the √l scaling reference at its true
    origin and reveals intrinsic decoder noise vs. resource-sharing cost.

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
# CIRCULAR STATISTICS HELPERS
# =============================================================================

def compute_circular_variance(errors: np.ndarray) -> float:
    """
    Circular variance: V = 1 - R_bar.

    R_bar = |mean(exp(i * errors))| is the mean resultant length.
    V ranges from 0 (no spread) to 1 (uniform on the circle).

    This is the standard definition from Fisher (1995), used in
    Bays (2014) where variance = sigma^2 = (-2 log R_bar) and
    circular variance is the complementary quantity V = 1 - R_bar.
    """
    R_bar = np.abs(np.mean(np.exp(1j * errors)))
    return 1.0 - R_bar


def compute_circular_std_from_errors(errors: np.ndarray) -> float:
    """
    Circular standard deviation: sigma = sqrt(-2 * log(R_bar)).

    This is the Fisher (1995) / Bays (2014) definition.
    Equivalent to the function in core.ml_decoder.compute_circular_std
    when period = 2*pi (the default for our orientation space).

    The relationship to circular variance V is:
        V = 1 - R_bar
        sigma^2 = -2 * log(1 - V) = -2 * log(R_bar)

    So sigma = sqrt(-2 * log(R_bar)).

    NOTE: This is NOT the same as the plain standard deviation of
    the angular errors. At low noise the two converge, but at high
    noise (when the distribution wraps around the circle) the plain
    SD underestimates spread while the circular SD correctly captures it.
    """
    R_bar = np.abs(np.mean(np.exp(1j * errors)))
    R_bar = np.clip(R_bar, 1e-10, 1.0 - 1e-10)
    return np.sqrt(-2.0 * np.log(R_bar))


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

    # Part E override: n_orientations for the sub-unit lambda sweep
    # If None, uses PART_E_N_ORIENTATIONS constant (256)
    part_e_n_theta: int = None

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

    # True orientation → nearest grid index (circular distance!)
    def _nearest_on_circle(theta_val, grid):
        """Snap a continuous angle to the nearest grid point using circular distance."""
        diff = np.abs(grid - theta_val)
        circ_diff = np.minimum(diff, 2 * np.pi - diff)
        return int(np.argmin(circ_diff))

    theta_indices = [_nearest_on_circle(t, theta_values) for t in true_orientations]

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
    """Shared trial loop used by Parts A, B, C, D, and E.

    Now computes BOTH circular variance and circular standard deviation
    for every set size, using the Fisher (1995) / Bays (2014) definitions:

        R_bar = |mean(exp(i * errors))|      (mean resultant length)
        circular_variance = 1 - R_bar
        circular_std      = sqrt(-2 * log(R_bar))
    """
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

        # ------------------------------------------------------------------
        # Circular statistics (Fisher 1995 / Bays 2014)
        # ------------------------------------------------------------------
        # Mean resultant length: R_bar = |mean(exp(i * errors))|
        R_bar = np.abs(np.mean(np.exp(1j * errors)))

        # Circular variance: V = 1 - R_bar  (range [0, 1])
        results[l]['circular_variance'] = 1.0 - R_bar

        # Circular standard deviation: sigma = sqrt(-2 * log(R_bar))
        R_bar_safe = np.clip(R_bar, 1e-10, 1.0 - 1e-10)
        results[l]['circular_std'] = np.sqrt(-2.0 * np.log(R_bar_safe))
        results[l]['circular_std_deg'] = np.degrees(results[l]['circular_std'])

        # Cross-check: also store the value from the core module
        results[l]['circular_std_core'] = compute_circular_std(errors)

        # Signed bias
        results[l]['mean_error'] = float(np.mean(errors))
        results[l]['mean_error_deg'] = float(np.degrees(np.mean(errors)))

        # Absolute error
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
            _diff = np.abs(theta_values - theta_true)
            theta_idx = np.argmin(np.minimum(_diff, 2 * np.pi - _diff))
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
# PART D — PRECISION COMPARISON: broad (λ=3) vs sharp (λ=1)
# =============================================================================

PART_D_T_D = 2.0          # 200 spk/neuron at γ=100
PART_D_LAMBDA_BROAD = 3.0  # broad tuning — gentle bumps
PART_D_LAMBDA_SHARP = 1.0  # sharp tuning — well-resolved peaks


def _run_part_d(cfg, rng, verbose=True):
    """Compare decoding error for broad vs sharp tuning at high SNR.

    λ=3.0 → tuning width ~170°, nearly flat response across orientations.
    λ=1.0 → tuning width ~57°, clear peaked selectivity.

    Both values are well above the grid spacing (2π/n_θ ≈ 0.20 rad for
    n_θ=32), ensuring the GP samples are properly resolved.

    Uses T_d=2.0 s (200 spk/neuron at γ=100) so the ML estimator
    operates efficiently and √l scaling is cleanly visible.
    """
    lambda_broad = PART_D_LAMBDA_BROAD
    lambda_sharp = PART_D_LAMBDA_SHARP
    T_d = PART_D_T_D

    conditions = {
        'broad': lambda_broad,
        'sharp': lambda_sharp,
    }

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART D: PRECISION COMPARISON  (high-SNR regime)")
        print(f"  {'─'*60}")
        print(f"    λ_broad={lambda_broad:.1f}  λ_sharp={lambda_sharp:.1f}")
        print(f"    T_d={T_d}s  (~{cfg.gamma * T_d:.0f} spk/neuron)  "
              f"n_trials={cfg.n_trials}")

    condition_results = {}

    for label, lam in conditions.items():
        if verbose:
            print(f"\n    --- {label}: λ_base = {lam:.2f} ---")

        pop = generate_neuron_population(
            n_neurons=cfg.n_neurons,
            n_orientations=cfg.n_orientations,
            n_locations=cfg.n_locations,
            base_lengthscale=lam,
            lengthscale_variability=0.0,   # pure base, no jitter
            seed=cfg.seed,
        )
        theta_vals = pop[0]['orientations']

        cond_cfg = Exp4Config(
            n_neurons=cfg.n_neurons,
            n_orientations=cfg.n_orientations,
            n_locations=cfg.n_locations,
            gamma=cfg.gamma,
            sigma_sq=cfg.sigma_sq,
            T_d=T_d,
            n_trials=cfg.n_trials,
            set_sizes=cfg.set_sizes,
            lambda_base=lam,
            sigma_lambda=0.0,
            seed=cfg.seed,
        )

        # Same RNG seed for both conditions → identical trial configs
        cond_rng = np.random.RandomState(cfg.seed + 2)
        res = _run_decoding_loop(
            pop, theta_vals, cond_cfg, cond_rng,
            cfg.n_trials, cfg.set_sizes,
            desc_prefix=f"D {label:<9s}"
        )

        stds_deg = [res[l]['circular_std_deg'] for l in cfg.set_sizes]
        condition_results[label] = {
            'lambda_base': lam,
            'stds_deg': stds_deg,
            'per_set_size': res,
        }

        if verbose:
            tag = "  ".join(
                f"l={l}: {stds_deg[i]:.1f}°"
                for i, l in enumerate(cfg.set_sizes)
            )
            print(f"      {tag}")

    if verbose:
        print(f"\n    Δ error (broad − sharp):")
        for i, l in enumerate(cfg.set_sizes):
            broad = condition_results['broad']['stds_deg'][i]
            sharp = condition_results['sharp']['stds_deg'][i]
            print(f"      l={l}: {broad:.1f}° → {sharp:.1f}°  "
                  f"(Δ = {broad - sharp:+.1f}°, "
                  f"{100*(broad - sharp)/broad:+.1f}%)")

    return {
        'condition_results': condition_results,
        'lambda_broad': lambda_broad,
        'lambda_sharp': lambda_sharp,
        'T_d': T_d,
    }


# =============================================================================
# PART E — SUB-UNIT LAMBDA SWEEP (5 lambdas < 1, single plot)
# =============================================================================

PART_E_LAMBDAS = np.linspace(0.2, 0.8, 5).tolist()  # [0.2, 0.35, 0.5, 0.65, 0.8]
PART_E_T_D = 2.0              # same high-SNR regime as Part D
PART_E_N_ORIENTATIONS = 256   # default fine grid: Δθ ≈ 1.4° ensures even λ=0.20 is well-resolved
PART_E_SET_SIZES = (1, 2, 4, 6, 8)   # include l=1 single-item baseline
PART_E_N_SEEDS = 5            # number of independent seeds for multi-seed averaging

# NOTE: Part E n_orientations can be overridden via config['part_e_n_theta']
#       to test how grid resolution interacts with tuning sharpness.
#       CLI: python scripts/run_experiments.py --exp 4 --n_theta 128


def _run_part_e(cfg, rng, verbose=True):
    """Multi-seed sub-unit lambda sweep: 5 lambdas < 1, including l=1.

    MULTI-SEED DESIGN
    -----------------
    Each (lambda, seed) pair generates an independent neural population
    and independent trial draws.  For each lambda, we collect circular
    std estimates from every seed, then report mean ± SEM across seeds.
    This eliminates reliance on a single stochastic realisation and
    provides confidence intervals for the circular SD curves.

    WHY l=1 MATTERS
    ----------------
    The single-item condition (l=1) is the irreducible baseline: no
    divisive-normalisation competition, no cross-item interference.
    Including it anchors the √l scaling reference line at its true
    origin and reveals how much of the error at l>1 is attributable
    to resource sharing vs. intrinsic decoder noise.

    All conditions use:
        - n_orientations = 256  (fine grid, Δθ ≈ 1.4°)
        - sigma_lambda = 0      (pure base lengthscale, no heterogeneity)
        - T_d = 2.0 s           (high-SNR regime, 200 spk/neuron at γ=100)
        - set_sizes = (1, 2, 4, 6, 8)
        - N_SEEDS independent population + trial seeds

    The ONLY thing that differs between conditions is the GP kernel width.
    """
    lambdas = PART_E_LAMBDAS
    T_d = PART_E_T_D
    n_ori = getattr(cfg, 'part_e_n_theta', None) or PART_E_N_ORIENTATIONS
    set_sizes_e = PART_E_SET_SIZES
    n_seeds = PART_E_N_SEEDS

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  PART E: MULTI-SEED SUB-UNIT LAMBDA SWEEP")
        print(f"  {'─'*60}")
        print(f"    lambdas={[f'{l:.2f}' for l in lambdas]}")
        print(f"    set_sizes={set_sizes_e}  (includes l=1 baseline)")
        print(f"    n_seeds={n_seeds}")
        print(f"    n_orientations={n_ori}  (Δθ = {np.degrees(2*np.pi/n_ori):.1f}°)")
        print(f"    T_d={T_d}s  (~{cfg.gamma * T_d:.0f} spk/neuron)  "
              f"n_trials={cfg.n_trials}")

    condition_results = {}

    for lam in lambdas:
        if verbose:
            print(f"\n    --- λ = {lam:.2f} ---")

        # Collect per-seed circular std arrays: shape will be (n_seeds, len(set_sizes_e))
        seed_stds = []
        seed_circ_vars = []
        # Also accumulate raw errors per set size across seeds for aggregate stats
        all_errors_by_l = {l: [] for l in set_sizes_e}

        for s_idx in range(n_seeds):
            seed_val = cfg.seed + 100 * s_idx   # well-separated seeds

            pop = generate_neuron_population(
                n_neurons=cfg.n_neurons,
                n_orientations=n_ori,
                n_locations=cfg.n_locations,
                base_lengthscale=lam,
                lengthscale_variability=0.0,
                seed=seed_val,
            )
            theta_vals = pop[0]['orientations']

            cond_cfg = Exp4Config(
                n_neurons=cfg.n_neurons,
                n_orientations=n_ori,
                n_locations=cfg.n_locations,
                gamma=cfg.gamma,
                sigma_sq=cfg.sigma_sq,
                T_d=T_d,
                n_trials=cfg.n_trials,
                set_sizes=tuple(set_sizes_e),
                lambda_base=lam,
                sigma_lambda=0.0,
                seed=seed_val,
            )

            # Independent RNG per seed → independent trial configurations
            cond_rng = np.random.RandomState(seed_val + 3)
            res = _run_decoding_loop(
                pop, theta_vals, cond_cfg, cond_rng,
                cfg.n_trials, set_sizes_e,
                desc_prefix=f"E λ={lam:<5.2f} s{s_idx} "
            )

            stds_this = [res[l]['circular_std_deg'] for l in set_sizes_e]
            cvars_this = [res[l]['circular_variance'] for l in set_sizes_e]
            seed_stds.append(stds_this)
            seed_circ_vars.append(cvars_this)

            for l in set_sizes_e:
                all_errors_by_l[l].append(res[l]['errors'])

        # --- Aggregate across seeds ---
        seed_stds = np.array(seed_stds)        # (n_seeds, n_set_sizes)
        seed_circ_vars = np.array(seed_circ_vars)

        mean_stds = seed_stds.mean(axis=0).tolist()
        sem_stds = (seed_stds.std(axis=0, ddof=1) / np.sqrt(n_seeds)).tolist()
        mean_cvars = seed_circ_vars.mean(axis=0).tolist()
        sem_cvars = (seed_circ_vars.std(axis=0, ddof=1) / np.sqrt(n_seeds)).tolist()

        condition_results[lam] = {
            'lambda_base': lam,
            'stds_deg': mean_stds,            # mean across seeds
            'stds_sem_deg': sem_stds,         # SEM across seeds
            'stds_per_seed': seed_stds.tolist(),  # full seed-level data
            'circular_variances': mean_cvars,
            'circular_variances_sem': sem_cvars,
            'circular_variances_per_seed': seed_circ_vars.tolist(),
            'n_seeds': n_seeds,
        }

        if verbose:
            tag = "  ".join(
                f"l={l}: σ={mean_stds[i]:.1f}±{sem_stds[i]:.1f}°"
                for i, l in enumerate(set_sizes_e)
            )
            print(f"      {tag}")

    # Summary table
    if verbose:
        print(f"\n    Lambda vs Circular SD mean±SEM (deg) at each set size:")
        header = f"    {'λ':<8}" + "".join(f"{'l='+str(l):<16}" for l in set_sizes_e)
        print(header)
        print("    " + "-" * (8 + 16 * len(set_sizes_e)))
        for lam in lambdas:
            row = f"    {lam:<8.2f}"
            for i, l in enumerate(set_sizes_e):
                m = condition_results[lam]['stds_deg'][i]
                s = condition_results[lam]['stds_sem_deg'][i]
                row += f"{m:.1f}±{s:.1f}{'°':<10}"
            print(row)

    return {
        'condition_results': condition_results,
        'lambdas': lambdas,
        'set_sizes': list(set_sizes_e),
        'T_d': T_d,
        'n_orientations': n_ori,
        'n_seeds': n_seeds,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_experiment_4(config: Dict) -> Dict:
    """Run all sub-experiments of Experiment 4."""
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
        part_e_n_theta=config.get('part_e_n_theta', None),
    )

    print("=" * 70)
    print("EXPERIMENT 4: MULTI-ITEM ML DECODING")
    print("=" * 70)
    print(f"\n  N={cfg.n_neurons}  n_theta={cfg.n_orientations}  "
          f"L={cfg.n_locations}  T_d={cfg.T_d}  gamma={cfg.gamma}")
    print(f"  set_sizes={cfg.set_sizes}  seed={cfg.seed}")

    compare_complexity(cfg.n_neurons, cfg.n_orientations, list(cfg.set_sizes))

    print(f"The number of orientations is: {cfg.n_orientations}")
    
    rng = np.random.RandomState(cfg.seed)

    # Optional part filtering: run only specified sub-experiments
    run_parts = config.get('run_parts', None)  # None = run all
    run_all = run_parts is None

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

    part_a = _run_part_a(population, theta_values, cfg, rng) if (run_all or 'a' in run_parts) else None
    part_b = _run_part_b(cfg, rng)  if (run_all or 'b' in run_parts) else None
    part_c = _run_part_c(cfg, rng)  if (run_all or 'c' in run_parts) else None
    part_d = _run_part_d(cfg, rng)  if (run_all or 'd' in run_parts) else None
    part_e = _run_part_e(cfg, rng)  if (run_all or 'e' in run_parts) else None

    print(f"\n{'='*70}")
    print("EXPERIMENT 4 COMPLETE")
    print(f"{'='*70}")

    results = {
        'config': config,
        'exp_config': cfg,
        'theta_values': theta_values,
        'method': 'efficient_factorized',
    }

    if part_a is not None:
        results['multi_item'] = part_a['multi_item']
        results['single_item'] = part_a['single_item']
        results['scaling'] = part_a['scaling']
    if part_b is not None:
        results['bias'] = part_b
    if part_c is not None:
        results['lengthscale_sweep'] = part_c
    if part_d is not None:
        results['precision_comparison'] = part_d
    if part_e is not None:
        results['sub_unit_sweep'] = part_e

    return results


# Backward-compatibility alias
run_experiment_4_efficient = run_experiment_4


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Generate all figures for Experiment 4 (Parts A–E)."""
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

    # ================================================================
    # PLOT 5: Part D — Precision comparison: broad vs sharp (2 panels)
    # ================================================================
    if 'precision_comparison' in results:
        pc = results['precision_comparison']
        cond_res = pc['condition_results']
        lam_broad = pc['lambda_broad']
        lam_sharp = pc['lambda_sharp']
        T_d_used = pc['T_d']
        spk = cfg.gamma * T_d_used

        broad_stds = cond_res['broad']['stds_deg']
        sharp_stds = cond_res['sharp']['stds_deg']

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

        # √l references anchored at l=2
        ref_broad = [broad_stds[0] * np.sqrt(l / set_sizes[0])
                     for l in set_sizes]
        ref_sharp = [sharp_stds[0] * np.sqrt(l / set_sizes[0])
                     for l in set_sizes]

        # --- Panel A: Broad tuning ---
        ax = axes[0]
        ax.plot(set_sizes, broad_stds, 'o-', color=palette[0],
                lw=2.5, ms=10, label='Decoder error', zorder=5)
        ax.plot(set_sizes, ref_broad, ':', color=palette[0],
                lw=1.8, alpha=0.5, label=r'$\propto \sqrt{\ell}$ ref')
        for i, l in enumerate(set_sizes):
            ax.annotate(f'{broad_stds[i]:.1f}°',
                        (l, broad_stds[i]),
                        textcoords='offset points', xytext=(10, -5),
                        fontsize=10, fontweight='bold', color=palette[0])
        ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
        ax.set_ylabel('Circular Std (degrees)', fontsize=13)
        ax.set_title(f'A.  Broad tuning  (λ = {lam_broad:.1f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xticks(set_sizes)
        sns.despine(ax=ax)

        # --- Panel B: Sharp tuning ---
        ax = axes[1]
        ax.plot(set_sizes, sharp_stds, 's-', color=palette[3],
                lw=2.5, ms=10, label='Decoder error', zorder=5)
        ax.plot(set_sizes, ref_sharp, ':', color=palette[3],
                lw=1.8, alpha=0.5, label=r'$\propto \sqrt{\ell}$ ref')
        for i, l in enumerate(set_sizes):
            ax.annotate(f'{sharp_stds[i]:.1f}°',
                        (l, sharp_stds[i]),
                        textcoords='offset points', xytext=(10, -5),
                        fontsize=10, fontweight='bold', color=palette[3])
        ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
        ax.set_title(f'B.  Sharp tuning  (λ = {lam_sharp:.1f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xticks(set_sizes)
        sns.despine(ax=ax)

        plt.suptitle(
            f'Experiment 4D: Decoding Precision  —  '
            f'T_d = {T_d_used}s  ({spk:.0f} spk/neuron)',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(out / 'exp4d_precision_comparison.png',
                    dpi=150, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
        print(f"  Saved: exp4d_precision_comparison.png")

        # ============================================================
        # PLOT 6: Part D — Bias comparison: broad vs sharp (2 panels)
        # ============================================================
        broad_bias = [cond_res['broad']['per_set_size'][l]['mean_error_deg']
                      for l in set_sizes]
        sharp_bias = [cond_res['sharp']['per_set_size'][l]['mean_error_deg']
                      for l in set_sizes]

        # 95% CI from stored errors
        broad_ci = [1.96 * np.degrees(np.std(
                        cond_res['broad']['per_set_size'][l]['errors']))
                    / np.sqrt(len(cond_res['broad']['per_set_size'][l]['errors']))
                    for l in set_sizes]
        sharp_ci = [1.96 * np.degrees(np.std(
                        cond_res['sharp']['per_set_size'][l]['errors']))
                    / np.sqrt(len(cond_res['sharp']['per_set_size'][l]['errors']))
                    for l in set_sizes]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

        # --- Panel A: Broad tuning bias ---
        ax = axes[0]
        ax.bar(set_sizes, broad_bias, width=0.6, color=palette[0],
               alpha=0.7, edgecolor='black', linewidth=0.8)
        ax.errorbar(set_sizes, broad_bias, yerr=broad_ci, fmt='none',
                    color='black', capsize=6, capthick=1.5, lw=1.5)
        ax.axhline(0, color='red', ls='--', lw=1.5)
        for i, l in enumerate(set_sizes):
            ax.annotate(f'{broad_bias[i]:+.1f}°',
                        (l, broad_bias[i]),
                        textcoords='offset points',
                        xytext=(0, 8 if broad_bias[i] >= 0 else -14),
                        ha='center', fontsize=10, fontweight='bold')
        ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
        ax.set_ylabel('Mean Signed Error (degrees)', fontsize=13)
        ax.set_title(f'A.  Broad tuning bias  (λ = {lam_broad:.1f})',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(set_sizes)
        sns.despine(ax=ax)

        # --- Panel B: Sharp tuning bias ---
        ax = axes[1]
        ax.bar(set_sizes, sharp_bias, width=0.6, color=palette[3],
               alpha=0.7, edgecolor='black', linewidth=0.8)
        ax.errorbar(set_sizes, sharp_bias, yerr=sharp_ci, fmt='none',
                    color='black', capsize=6, capthick=1.5, lw=1.5)
        ax.axhline(0, color='red', ls='--', lw=1.5)
        for i, l in enumerate(set_sizes):
            ax.annotate(f'{sharp_bias[i]:+.1f}°',
                        (l, sharp_bias[i]),
                        textcoords='offset points',
                        xytext=(0, 8 if sharp_bias[i] >= 0 else -14),
                        ha='center', fontsize=10, fontweight='bold')
        ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
        ax.set_title(f'B.  Sharp tuning bias  (λ = {lam_sharp:.1f})',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(set_sizes)
        sns.despine(ax=ax)

        plt.suptitle(
            f'Experiment 4D: Decoder Bias  —  '
            f'T_d = {T_d_used}s  ({spk:.0f} spk/neuron)',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(out / 'exp4d_bias_comparison.png',
                    dpi=150, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
        print(f"  Saved: exp4d_bias_comparison.png")

    # ================================================================
    # PLOT 7: Part E — Multi-seed sub-unit lambda sweep (with l=1)
    # ================================================================
    if 'sub_unit_sweep' in results:
        pe = results['sub_unit_sweep']
        pe_cond = pe['condition_results']
        pe_lambdas = pe['lambdas']
        pe_T_d = pe['T_d']
        pe_spk = cfg.gamma * pe_T_d
        pe_set_sizes = pe.get('set_sizes', list(cfg.set_sizes))
        pe_n_seeds = pe.get('n_seeds', 1)

        fig, ax = plt.subplots(figsize=(11, 7))

        # One colour per lambda, viridis: small λ → dark, large λ → bright
        cmap_e = plt.cm.viridis
        norm_e = plt.Normalize(vmin=min(pe_lambdas), vmax=max(pe_lambdas))
        markers = ['o', 's', 'D', '^', 'v']

        # Vertical offsets to stagger annotations and avoid overlap
        y_offset_signs = [1, -1, 1, -1, 1]
        y_offset_base = 3.0

        for idx, lam in enumerate(pe_lambdas):
            stds = pe_cond[lam]['stds_deg']
            sems = pe_cond[lam].get('stds_sem_deg', [0.0] * len(stds))
            color = cmap_e(norm_e(lam))

            # Main data line with SEM error bars
            ax.errorbar(pe_set_sizes, stds, yerr=sems,
                        marker=markers[idx % len(markers)],
                        color=color, lw=1.4, ms=7, capsize=3, capthick=1.0,
                        label=f'λ = {lam:.2f}', zorder=5)

            # √ℓ reference anchored at ℓ = 1 (the true baseline)
            l0_idx = 0  # l=1 is first
            ref = [stds[l0_idx] * np.sqrt(l / pe_set_sizes[l0_idx])
                   for l in pe_set_sizes]
            ax.plot(pe_set_sizes, ref, ':', color=color, lw=0.8, alpha=0.35)

            # Annotate each data point
            y_sign = y_offset_signs[idx % len(y_offset_signs)]
            for i, l in enumerate(pe_set_sizes):
                label_text = f'{stds[i]:.1f}°'
                if sems[i] > 0:
                    label_text = f'{stds[i]:.1f}±{sems[i]:.1f}°'
                ax.annotate(
                    label_text,
                    (l, stds[i]),
                    textcoords='offset points',
                    xytext=(0, y_sign * (y_offset_base + idx * 2)),
                    fontsize=6.5, fontweight='bold', color=color,
                    ha='center', va='bottom' if y_sign > 0 else 'top',
                    zorder=6,
                )

        ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
        ax.set_ylabel('Circular Std (degrees)', fontsize=13)
        ax.set_title(
            f'Experiment 4E: Multi-Seed Sub-Unit Lambda Sweep\n'
            f'5 evenly-spaced λ < 1  |  n_θ = {pe.get("n_orientations", 256)}  |  '
            f'T_d = {pe_T_d}s  ({pe_spk:.0f} spk/neuron)  |  '
            f'{pe_n_seeds} seeds',
            fontsize=13, fontweight='bold'
        )
        ax.set_xticks(pe_set_sizes)
        ax.legend(fontsize=10, title='Base Lengthscale λ', title_fontsize=11,
                  loc='upper left')
        sns.despine(ax=ax)

        plt.tight_layout()
        plt.savefig(out / 'exp4e_subunit_lambda_sweep.png',
                    dpi=150, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
        print(f"  Saved: exp4e_subunit_lambda_sweep.png")

        # ============================================================
        # PLOT 8: Part E — Circular VARIANCE version (V = 1 - R_bar)
        # ============================================================
        # WHY VARIANCE?
        # Circular std σ = sqrt(-2 log R_bar) is a nonlinear transform
        # of R_bar.  It compresses differences at high noise (R_bar → 0)
        # and stretches them at low noise (R_bar → 1).  Circular
        # variance V = 1 - R_bar is the *linear* measure of spread on
        # the circle.  Crucially:
        #   - V is additive under independent noise sources
        #   - The resource model predicts V ∝ ℓ (linear in set size),
        #     not √ℓ, because variance (not std) scales linearly with
        #     the inverse of per-item spike count
        #   - A linear reference line on the variance plot directly
        #     tests the 1/ℓ resource-sharing prediction
        #
        # So the std plot tests σ ∝ √ℓ; the variance plot tests V ∝ ℓ.
        # Both are the same prediction, but nonlinear compression in
        # the std transform can mask deviations that are visible in
        # the variance domain.

        fig, ax = plt.subplots(figsize=(11, 7))

        for idx, lam in enumerate(pe_lambdas):
            cvars = pe_cond[lam]['circular_variances']
            cvar_sems = pe_cond[lam].get('circular_variances_sem',
                                          [0.0] * len(cvars))
            color = cmap_e(norm_e(lam))

            # Main data line with SEM error bars
            ax.errorbar(pe_set_sizes, cvars, yerr=cvar_sems,
                        marker=markers[idx % len(markers)],
                        color=color, lw=1.4, ms=7, capsize=3, capthick=1.0,
                        label=f'λ = {lam:.2f}', zorder=5)

            # LINEAR reference: V ∝ ℓ, anchored at ℓ=1
            l0_idx = 0
            ref_var = [cvars[l0_idx] * (l / pe_set_sizes[l0_idx])
                       for l in pe_set_sizes]
            ax.plot(pe_set_sizes, ref_var, ':', color=color,
                    lw=0.8, alpha=0.35)

            # Annotate each data point
            y_sign = y_offset_signs[idx % len(y_offset_signs)]
            for i, l_val in enumerate(pe_set_sizes):
                label_text = f'{cvars[i]:.4f}'
                if cvar_sems[i] > 0:
                    label_text = f'{cvars[i]:.4f}±{cvar_sems[i]:.4f}'
                ax.annotate(
                    label_text,
                    (l_val, cvars[i]),
                    textcoords='offset points',
                    xytext=(0, y_sign * (y_offset_base + idx * 2)),
                    fontsize=6.5, fontweight='bold', color=color,
                    ha='center', va='bottom' if y_sign > 0 else 'top',
                    zorder=6,
                )

        ax.set_xlabel(r'Set Size ($\ell$)', fontsize=13)
        ax.set_ylabel(r'Circular Variance  $V = 1 - \bar{R}$', fontsize=13)
        ax.set_title(
            f'Experiment 4E: Circular Variance (Multi-Seed)\n'
            f'5 evenly-spaced λ < 1  |  n_θ = {pe.get("n_orientations", 256)}  |  '
            f'T_d = {pe_T_d}s  ({pe_spk:.0f} spk/neuron)  |  '
            f'{pe_n_seeds} seeds  |  '
            r'ref: $V \propto \ell$',
            fontsize=13, fontweight='bold'
        )
        ax.set_xticks(pe_set_sizes)
        ax.legend(fontsize=10, title='Base Lengthscale λ', title_fontsize=11,
                  loc='upper left')
        sns.despine(ax=ax)

        plt.tight_layout()
        plt.savefig(out / 'exp4e_circular_variance.png',
                    dpi=150, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
        print(f"  Saved: exp4e_circular_variance.png")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Experiment 4: Multi-Item ML Decoding',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Run all parts with defaults:
  python exp4_ml_decoding.py

  # Run only Part E with fine grid (n_theta=512):
  python exp4_ml_decoding.py --part e --n_theta 512

  # Run all parts with custom neurons/seed:
  python exp4_ml_decoding.py --n_neurons 200 --seed 99

  # Run Part E with coarse grid to test aliasing:
  python exp4_ml_decoding.py --part e --n_theta 32 --n_trials 100
        """,
    )
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_orientations', type=int, default=32,
                        help='Grid resolution for Parts A-D (default: 32)')
    parser.add_argument('--n_theta', type=int, default=None,
                        help='Grid resolution override for Part E (default: 256).\n'
                             'Controls the x-axis granularity of the orientation grid.')
    parser.add_argument('--n_locations', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--sigma_sq', type=float, default=1e-6)
    parser.add_argument('--T_d', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--set_sizes', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--output_dir', type=str, default='results/exp4')
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('--part', type=str, default=None,
                        help='Run only a specific part: a, b, c, d, or e.\n'
                             'If omitted, runs all parts.')

    args = parser.parse_args()

    config = {
        'n_neurons': args.n_neurons,
        'n_orientations': args.n_orientations,
        'n_locations': args.n_locations,
        'gamma': args.gamma,
        'sigma_sq': args.sigma_sq,
        'T_d': args.T_d,
        'n_trials': args.n_trials,
        'seed': args.seed,
        'set_sizes': args.set_sizes,
        'part_e_n_theta': args.n_theta,
    }

    if args.part:
        config['run_parts'] = [args.part.lower()]

    results = run_experiment_4(config)

    if not args.no_plot:
        plot_results(results, args.output_dir, show_plot=True)