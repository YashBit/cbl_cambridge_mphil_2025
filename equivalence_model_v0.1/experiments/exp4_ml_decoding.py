"""
Experiment 4: Multi-Item ML Decoding with Marginalisation

=============================================================================
PURPOSE
=============================================================================

Demonstrates ML decoding with marginalisation for multi-item working memory,
completing the full pipeline from GP tuning curves → DN → Poisson spikes → ML decode.

THE COMPLETE CAUSAL CHAIN:
    GP tuning (mixed selectivity) → DN caps activity → Poisson spikes
    → Joint likelihood tensor → Marginalise over non-cued items → ML decode

KEY OUTPUTS:
    1. Decoding error distributions at each set size
    2. Error std scaling with set size (√ℓ law)
    3. Comparison: single-item vs multi-item (marginalised) decoder
    4. Fisher Information and Cramér-Rao bound verification

=============================================================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass

from core.gaussian_process import generate_neuron_population
from core.poisson_spike import generate_spikes
from core.ml_decoder import (
    compute_log_likelihood,
    decode_ml,
    compute_circular_error,
    compute_circular_std,
    compute_fisher_information,
    compute_cramer_rao_bound,
    compute_tuning_curve_derivative,
    build_tuning_tensor,
    apply_dn_to_tensor,
    decode_ml_multi_item,
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
    n_locations: int = 8
    
    # DN parameters
    gamma: float = 100.0
    sigma_sq: float = 1e-6
    
    # Decoding parameters
    T_d: float = 0.1
    n_trials: int = 500
    set_sizes: Tuple = (1, 2, 4)
    
    # GP parameters
    lambda_base: float = 0.5
    sigma_lambda: float = 0.3
    
    # Random seed
    seed: int = 42
    
    @property
    def total_activity(self) -> float:
        return self.gamma * self.n_neurons


# =============================================================================
# POPULATION SETUP
# =============================================================================

def setup_population(cfg: Exp4Config) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate GP-based neural population with mixed selectivity.
    
    Returns
    -------
    population : List[Dict]
        List of neuron dictionaries with f_samples, lengthscales, etc.
    theta_values : np.ndarray
        Orientation values in radians
    """
    population = generate_neuron_population(
        n_neurons=cfg.n_neurons,
        n_orientations=cfg.n_orientations,
        n_locations=cfg.n_locations,
        base_lengthscale=cfg.lambda_base,
        lengthscale_variability=cfg.sigma_lambda,
        seed=cfg.seed
    )
    theta_values = population[0]['orientations']
    return population, theta_values


def extract_f_samples_for_locations(
    population: List[Dict],
    active_locations: Tuple[int, ...]
) -> List[np.ndarray]:
    """
    Extract log-rate tuning functions for active locations.
    
    Returns
    -------
    f_samples_per_location : List[np.ndarray]
        List of ℓ arrays, each shape (N, n_theta)
    """
    N = len(population)
    n_theta = population[0]['f_samples'].shape[1]
    ell = len(active_locations)
    
    f_samples_list = []
    for k, loc in enumerate(active_locations):
        f_k = np.zeros((N, n_theta))
        for i, neuron in enumerate(population):
            f_k[i, :] = neuron['f_samples'][loc, :]
        f_samples_list.append(f_k)
    
    return f_samples_list


# =============================================================================
# SINGLE-TRIAL DECODING
# =============================================================================

def run_single_trial_multi_item(
    population: List[Dict],
    theta_values: np.ndarray,
    cfg: Exp4Config,
    active_locations: Tuple[int, ...],
    true_orientations: np.ndarray,
    cued_index: int,
    rng: np.random.RandomState
) -> Dict:
    """
    Run a single trial of multi-item decoding.
    
    Parameters
    ----------
    population : List[Dict]
    theta_values : np.ndarray
    cfg : Exp4Config
    active_locations : Tuple[int, ...]
        Indices of active locations
    true_orientations : np.ndarray, shape (ℓ,)
        True orientation at each active location (radians)
    cued_index : int
        Which active location is cued (0-indexed within active set)
    rng : np.random.RandomState
    
    Returns
    -------
    result : Dict with theta_true, theta_estimate, error, etc.
    """
    ell = len(active_locations)
    N = len(population)
    
    # Get f_samples for active locations
    f_samples_list = extract_f_samples_for_locations(population, active_locations)
    
    # Build joint tuning tensor: shape (N, n_theta, ..., n_theta)
    tuning_tensor = build_tuning_tensor(f_samples_list)
    
    # Apply DN
    tuning_tensor = apply_dn_to_tensor(tuning_tensor, cfg.gamma, cfg.sigma_sq)
    
    # Get true orientation indices
    theta_indices = [np.argmin(np.abs(theta_values - t)) for t in true_orientations]
    
    # Get firing rates at true joint stimulus configuration
    # Index into tensor: tuning_tensor[:, θ_1, θ_2, ..., θ_ℓ]
    idx_tuple = (slice(None),) + tuple(theta_indices)
    rates_at_true = tuning_tensor[idx_tuple]  # shape (N,)
    
    # Generate Poisson spikes
    spike_counts = generate_spikes(rates_at_true, cfg.T_d, rng)
    
    # Decode with marginalisation
    theta_ml, ll_max, ll_marginal = decode_ml_multi_item(
        spike_counts, tuning_tensor, theta_values, cfg.T_d, cued_index
    )
    
    # Compute error
    theta_true = true_orientations[cued_index]
    error = compute_circular_error(theta_true, theta_ml)
    
    return {
        'theta_true': theta_true,
        'theta_estimate': theta_ml,
        'error': error,
        'spike_counts': spike_counts,
        'total_spikes': np.sum(spike_counts),
        'mean_rate': np.mean(rates_at_true),
    }


# =============================================================================
# SET SIZE EXPERIMENT
# =============================================================================

def run_set_size_experiment(
    population: List[Dict],
    theta_values: np.ndarray,
    cfg: Exp4Config,
    rng: np.random.RandomState
) -> Dict:
    """
    Run decoding experiment across multiple set sizes.
    
    For each set size ℓ:
    - Select ℓ random active locations
    - Generate random true orientations
    - Run n_trials decoding trials
    - Collect error statistics
    
    Returns
    -------
    results : Dict with errors, statistics per set size
    """
    results = {l: {'errors': [], 'theta_true': [], 'theta_est': []} 
               for l in cfg.set_sizes}
    
    for l in tqdm(cfg.set_sizes, desc="Set sizes"):
        for trial in range(cfg.n_trials):
            # Sample active locations
            active_locations = tuple(rng.choice(
                cfg.n_locations, size=l, replace=False
            ))
            
            # Sample true orientations (uniform on circle)
            true_orientations = rng.uniform(-np.pi, np.pi, size=l)
            
            # Cued location (always first in active set for simplicity)
            cued_index = 0
            
            # Run trial
            trial_result = run_single_trial_multi_item(
                population, theta_values, cfg,
                active_locations, true_orientations,
                cued_index, rng
            )
            
            results[l]['errors'].append(trial_result['error'])
            results[l]['theta_true'].append(trial_result['theta_true'])
            results[l]['theta_est'].append(trial_result['theta_estimate'])
        
        # Convert to arrays
        results[l]['errors'] = np.array(results[l]['errors'])
        results[l]['theta_true'] = np.array(results[l]['theta_true'])
        results[l]['theta_est'] = np.array(results[l]['theta_est'])
        
        # Compute statistics
        results[l]['circular_std'] = compute_circular_std(results[l]['errors'])
        results[l]['circular_std_deg'] = np.degrees(results[l]['circular_std'])
        results[l]['mean_abs_error'] = np.mean(np.abs(results[l]['errors']))
        results[l]['mean_abs_error_deg'] = np.degrees(results[l]['mean_abs_error'])
    
    return results


# =============================================================================
# SINGLE-ITEM BASELINE (for comparison)
# =============================================================================

def run_single_item_baseline(
    population: List[Dict],
    theta_values: np.ndarray,
    cfg: Exp4Config,
    rng: np.random.RandomState
) -> Dict:
    """
    Run single-item decoding baseline (no marginalisation needed).
    
    This uses the standard single-location decoder for comparison.
    """
    N = len(population)
    results = {l: {'errors': []} for l in cfg.set_sizes}
    
    for l in tqdm(cfg.set_sizes, desc="Single-item baseline"):
        # For single-item, we use location 0's tuning curves
        # and scale by 1/ℓ to simulate DN effect
        tuning_curves = np.zeros((N, cfg.n_orientations))
        for i, neuron in enumerate(population):
            f = neuron['f_samples'][0, :]
            tuning_curves[i, :] = np.exp(f)
        
        # Apply DN and scale for set size
        pop_mean = np.mean(tuning_curves, axis=0, keepdims=True)
        tuning_curves = cfg.gamma * tuning_curves / (cfg.sigma_sq + pop_mean)
        tuning_curves = tuning_curves / l  # DN scaling for ℓ items
        
        for trial in range(cfg.n_trials):
            # Sample true orientation
            theta_true = rng.uniform(-np.pi, np.pi)
            theta_idx = np.argmin(np.abs(theta_values - theta_true))
            
            # Get rates and generate spikes
            rates = tuning_curves[:, theta_idx]
            spike_counts = generate_spikes(rates, cfg.T_d, rng)
            
            # Decode
            theta_ml, _, _ = decode_ml(spike_counts, tuning_curves, theta_values, cfg.T_d)
            error = compute_circular_error(theta_true, theta_ml)
            results[l]['errors'].append(error)
        
        results[l]['errors'] = np.array(results[l]['errors'])
        results[l]['circular_std_deg'] = np.degrees(
            compute_circular_std(results[l]['errors'])
        )
    
    return results


# =============================================================================
# FISHER INFORMATION ANALYSIS
# =============================================================================

def compute_theoretical_bounds(
    population: List[Dict],
    theta_values: np.ndarray,
    cfg: Exp4Config
) -> Dict:
    """Compute Fisher Information and Cramér-Rao bounds for each set size."""
    N = len(population)
    
    # Build base tuning curves (location 0)
    base_tuning = np.zeros((N, cfg.n_orientations))
    for i, neuron in enumerate(population):
        f = neuron['f_samples'][0, :]
        base_tuning[i, :] = np.exp(f)
    
    # Apply DN
    pop_mean = np.mean(base_tuning, axis=0, keepdims=True)
    base_tuning = cfg.gamma * base_tuning / (cfg.sigma_sq + pop_mean)
    
    # Compute derivatives
    derivatives = compute_tuning_curve_derivative(base_tuning, theta_values)
    
    results = {'set_sizes': [], 'fisher_info': [], 'cr_std_deg': []}
    
    for l in cfg.set_sizes:
        scaled_tuning = base_tuning / l
        scaled_deriv = derivatives / l
        
        theta_idx = cfg.n_orientations // 2
        I_F = compute_fisher_information(scaled_tuning, scaled_deriv, theta_idx, cfg.T_d)
        cr_var = compute_cramer_rao_bound(I_F)
        
        results['set_sizes'].append(l)
        results['fisher_info'].append(I_F)
        results['cr_std_deg'].append(np.degrees(np.sqrt(cr_var)))
    
    return results


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment_4(config: Dict) -> Dict:
    """
    Run Experiment 4: Multi-Item ML Decoding.
    
    Parameters
    ----------
    config : Dict
        Configuration from run_experiments.py
        
    Returns
    -------
    results : Dict
    """
    cfg = Exp4Config(
        n_neurons=config.get('n_neurons', 100),
        n_orientations=config.get('n_orientations', 32),
        n_locations=config.get('n_locations', 8),
        gamma=config.get('gamma', 100.0),
        sigma_sq=config.get('sigma_sq', 1e-6),
        T_d=config.get('T_d', 0.1),
        n_trials=config.get('n_trials', 500),
        set_sizes=tuple(config.get('set_sizes', [1, 2, 4])),
        seed=config.get('seed', 42),
    )
    
    print("=" * 70)
    print("EXPERIMENT 4: MULTI-ITEM ML DECODING WITH MARGINALISATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N = {cfg.n_neurons} neurons")
    print(f"  n_θ = {cfg.n_orientations} orientation bins")
    print(f"  L = {cfg.n_locations} total locations")
    print(f"  γ = {cfg.gamma} Hz/neuron → Total = {cfg.total_activity} Hz")
    print(f"  T_d = {cfg.T_d} s")
    print(f"  Trials = {cfg.n_trials}")
    print(f"  Set sizes = {cfg.set_sizes}")
    print()
    
    rng = np.random.RandomState(cfg.seed)
    
    # Setup population
    print("Generating GP population with mixed selectivity...")
    population, theta_values = setup_population(cfg)
    print(f"  Generated {len(population)} neurons")
    
    # Run multi-item decoding
    print("\nRunning multi-item decoding with marginalisation...")
    multi_item_results = run_set_size_experiment(population, theta_values, cfg, rng)
    
    # Run single-item baseline
    print("\nRunning single-item baseline...")
    single_item_results = run_single_item_baseline(population, theta_values, cfg, rng)
    
    # Compute theoretical bounds
    print("\nComputing Fisher Information bounds...")
    theoretical = compute_theoretical_bounds(population, theta_values, cfg)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Set Size':<10} {'Multi-Item σ (°)':<18} {'Single-Item σ (°)':<18} {'CR Bound (°)':<15}")
    print("-" * 60)
    for i, l in enumerate(cfg.set_sizes):
        mi_std = multi_item_results[l]['circular_std_deg']
        si_std = single_item_results[l]['circular_std_deg']
        cr_std = theoretical['cr_std_deg'][i]
        print(f"{l:<10} {mi_std:<18.2f} {si_std:<18.2f} {cr_std:<15.2f}")
    
    # Check √ℓ scaling
    stds = [multi_item_results[l]['circular_std_deg'] for l in cfg.set_sizes]
    normalised = [stds[i] / np.sqrt(l) for i, l in enumerate(cfg.set_sizes)]
    cv = np.std(normalised) / np.mean(normalised)
    print(f"\n√ℓ Scaling Check:")
    print(f"  Normalised stds: {[f'{n:.2f}' for n in normalised]}")
    print(f"  CV of normalised: {cv:.3f} (< 0.1 is good)")
    
    return {
        'config': config,
        'exp_config': cfg,
        'theta_values': theta_values,
        'multi_item': multi_item_results,
        'single_item': single_item_results,
        'theoretical': theoretical,
        'scaling': {
            'empirical_std': stds,
            'normalised_by_sqrt_l': normalised,
            'cv_normalised': cv,
        }
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Generate figures for Experiment 4."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("deep")
    
    cfg = results['exp_config']
    multi_item = results['multi_item']
    single_item = results['single_item']
    theoretical = results['theoretical']
    set_sizes = list(cfg.set_sizes)
    
    mi_std = [multi_item[l]['circular_std_deg'] for l in set_sizes]
    si_std = [single_item[l]['circular_std_deg'] for l in set_sizes]
    cr_std = theoretical['cr_std_deg']
    
    # Figure 1: Error Scaling Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(set_sizes, mi_std, 'o-', color=palette[0], lw=2.5, ms=10, 
            label='Multi-item (marginalised)')
    ax.plot(set_sizes, si_std, 's--', color=palette[1], lw=2, ms=8, 
            label='Single-item baseline')
    ax.plot(set_sizes, cr_std, '^:', color=palette[2], lw=2, ms=8, 
            label='Cramér-Rao bound')
    
    ref = [mi_std[0] * np.sqrt(l / set_sizes[0]) for l in set_sizes]
    ax.plot(set_sizes, ref, ':', color='gray', lw=2, alpha=0.7, label='∝ √ℓ')
    
    ax.set_xlabel('Set Size (ℓ)')
    ax.set_ylabel('Circular Std (degrees)')
    ax.set_title('A. Decoding Error vs Set Size')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    ax = axes[1]
    normalised = results['scaling']['normalised_by_sqrt_l']
    ax.plot(set_sizes, normalised, 'o-', color=palette[3], lw=2.5, ms=10)
    ax.axhline(np.mean(normalised), color='red', ls='--', lw=2,
               label=f'Mean = {np.mean(normalised):.2f}°')
    ax.set_xlabel('Set Size (ℓ)')
    ax.set_ylabel('Error Std / √ℓ (degrees)')
    ax.set_title(f'B. √ℓ Scaling (CV = {results["scaling"]["cv_normalised"]:.3f})')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    plt.suptitle('Experiment 4: Multi-Item ML Decoding with Marginalisation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_error_scaling.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_error_scaling.png")
    
    # Figure 2: Error Distributions
    n_sizes = len(set_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(4*n_sizes, 4))
    colors = sns.color_palette("coolwarm", n_sizes)
    
    for i, l in enumerate(set_sizes):
        ax = axes[i] if n_sizes > 1 else axes
        errors = np.degrees(multi_item[l]['errors'])
        
        sns.histplot(errors, kde=True, ax=ax, color=colors[i],
                     stat='density', alpha=0.6, bins=30)
        
        std = multi_item[l]['circular_std_deg']
        ax.set_xlabel('Error (degrees)')
        ax.set_ylabel('Density' if i == 0 else '')
        ax.set_title(f'ℓ = {l}\nσ = {std:.1f}°')
        ax.set_xlim([-90, 90])
    
    plt.suptitle('Error Distributions (Multi-Item Decoder)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_error_distributions.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_error_distributions.png")


# =============================================================================
# STANDALONE ENTRY
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_neurons': 100,
        'n_orientations': 32,
        'n_locations': 8,
        'gamma': 100.0,
        'sigma_sq': 1e-6,
        'T_d': 0.1,
        'n_trials': 200,
        'seed': 42,
        'set_sizes': [1, 2, 4],
    }
    
    results = run_experiment_4(config)
    plot_results(results, 'results/exp4', show_plot=True)