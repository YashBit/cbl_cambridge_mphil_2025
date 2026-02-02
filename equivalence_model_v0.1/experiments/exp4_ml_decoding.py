"""
Experiment 4: Multi-Item ML Decoding - EFFICIENT FACTORIZED VERSION

=============================================================================
WHY THIS VERSION IS DIFFERENT
=============================================================================

The original exp4_ml_decoding.py crashes for set sizes > 4 because it builds
a joint tuning tensor of shape (N, n_theta, n_theta, ..., n_theta) requiring 
O(N * n_theta^l) memory - EXPONENTIAL in set size.

This version exploits two mathematical properties:

1. ACTIVITY CAP THEOREM: Under DN, Sum_i r_i(theta) = gamma*N (constant!)
   -> The second term in log-likelihood drops out

2. FACTORIZED LOG-RATES: log r_i^pre(theta) = Sum_k f_i,k(theta_k)
   -> The spike-weighted log-likelihood separates into 1D functions
   -> Marginalisation uses sum-of-logsumexp, not explicit summation

RESULT: O(N * l * n_theta) complexity - LINEAR in set size!

Now you can run set sizes 2, 4, 6, 8, 10, 12, 16, ... without any issues.

=============================================================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass

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
    """Configuration for Experiment 4 (Efficient Version)."""
    # Population parameters
    n_neurons: int = 100
    n_orientations: int = 32
    n_locations: int = 16  # Can be larger now!
    
    # DN parameters
    gamma: float = 100.0
    sigma_sq: float = 1e-6
    
    # Decoding parameters
    T_d: float = 0.1
    n_trials: int = 500
    set_sizes: Tuple = (2, 4, 6, 8)  # Can include larger values now!
    
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
    """Generate GP-based neural population with mixed selectivity."""
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
    """Extract log-rate tuning functions for active locations."""
    N = len(population)
    n_theta = population[0]['f_samples'].shape[1]
    
    f_samples_list = []
    for loc in active_locations:
        f_k = np.zeros((N, n_theta))
        for i, neuron in enumerate(population):
            f_k[i, :] = neuron['f_samples'][loc, :]
        f_samples_list.append(f_k)
    
    return f_samples_list


# =============================================================================
# EFFICIENT SINGLE TRIAL
# =============================================================================

def run_single_trial_efficient(
    population: List[Dict],
    theta_values: np.ndarray,
    cfg: Exp4Config,
    active_locations: Tuple[int, ...],
    true_orientations: np.ndarray,
    cued_index: int,
    rng: np.random.RandomState
) -> Dict:
    """
    Run a single trial using EFFICIENT factorized decoding.
    
    Complexity: O(N * l * n_theta) - works for ANY set size!
    """
    ell = len(active_locations)
    N = len(population)
    n_theta = len(theta_values)
    
    # Extract f_samples for active locations
    f_samples_list = extract_f_samples_for_locations(population, active_locations)
    
    # Get true orientation indices
    theta_indices = [np.argmin(np.abs(theta_values - t)) for t in true_orientations]
    
    # Compute firing rates at true configuration WITH DN
    # r_i = gamma * r_pre_i / D  where D = sigma^2 + mean(r_pre)
    log_r_pre = np.zeros(N)
    for k, f_k in enumerate(f_samples_list):
        log_r_pre += f_k[:, theta_indices[k]]
    r_pre = np.exp(log_r_pre)
    
    D = cfg.sigma_sq + np.mean(r_pre)
    rates = cfg.gamma * r_pre / D
    
    # Generate Poisson spikes
    spike_counts = rng.poisson(rates * cfg.T_d)
    
    # =========================================================================
    # EFFICIENT DECODING (the key innovation!)
    # =========================================================================
    theta_ml, ll_max, ll_marginal = decode_ml_efficient(
        spike_counts, f_samples_list, theta_values, cued_index
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
        'mean_rate': np.mean(rates),
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
    
    NOW WORKS FOR ANY SET SIZE thanks to efficient factorization!
    """
    results = {l: {'errors': [], 'theta_true': [], 'theta_est': []} 
               for l in cfg.set_sizes}
    
    for l in tqdm(cfg.set_sizes, desc="Set sizes"):
        for trial in range(cfg.n_trials):
            # Sample active locations
            active_locations = tuple(rng.choice(
                cfg.n_locations, size=l, replace=False
            ))
            
            # Sample true orientations
            true_orientations = rng.uniform(0, 2*np.pi, size=l)
            
            # Cue random location
            cued_index = rng.randint(l)
            
            # Run trial (efficient!)
            trial_result = run_single_trial_efficient(
                population, theta_values, cfg,
                active_locations, true_orientations, cued_index, rng
            )
            
            results[l]['errors'].append(trial_result['error'])
            results[l]['theta_true'].append(trial_result['theta_true'])
            results[l]['theta_est'].append(trial_result['theta_estimate'])
        
        # Compute statistics
        errors = np.array(results[l]['errors'])
        results[l]['errors'] = errors
        results[l]['theta_true'] = np.array(results[l]['theta_true'])
        results[l]['theta_est'] = np.array(results[l]['theta_est'])
        results[l]['circular_std'] = compute_circular_std(errors)
        results[l]['circular_std_deg'] = np.degrees(results[l]['circular_std'])
        results[l]['mean_absolute_error'] = np.mean(np.abs(errors))
        results[l]['mae_deg'] = np.degrees(results[l]['mean_absolute_error'])
        
        print(f"  l={l}: sigma = {results[l]['circular_std_deg']:.2f} deg")
    
    return results


def run_single_item_baseline(
    population: List[Dict],
    theta_values: np.ndarray,
    cfg: Exp4Config,
    rng: np.random.RandomState
) -> Dict:
    """Run single-item decoding as baseline."""
    N = len(population)
    
    # Build base tuning curves (location 0)
    base_tuning = np.zeros((N, cfg.n_orientations))
    for i, neuron in enumerate(population):
        f = neuron['f_samples'][0, :]
        base_tuning[i, :] = np.exp(f)
    
    # Apply DN
    pop_mean = np.mean(base_tuning, axis=0, keepdims=True)
    base_tuning = cfg.gamma * base_tuning / (cfg.sigma_sq + pop_mean)
    
    results = {l: {'errors': [], 'theta_true': [], 'theta_est': []} 
               for l in cfg.set_sizes}
    
    for l in cfg.set_sizes:
        scaled_tuning = base_tuning / l
        
        for trial in range(cfg.n_trials):
            theta_true = rng.uniform(0, 2*np.pi)
            theta_idx = np.argmin(np.abs(theta_values - theta_true))
            
            rates = scaled_tuning[:, theta_idx]
            spikes = generate_spikes(rates, cfg.T_d, rng)
            
            theta_ml, _, _ = decode_ml(spikes, scaled_tuning, theta_values, cfg.T_d)
            error = compute_circular_error(theta_true, theta_ml)
            
            results[l]['errors'].append(error)
        
        errors = np.array(results[l]['errors'])
        results[l]['errors'] = errors
        results[l]['circular_std'] = compute_circular_std(errors)
        results[l]['circular_std_deg'] = np.degrees(results[l]['circular_std'])
    
    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment_4_efficient(config: Dict) -> Dict:
    """
    Run Experiment 4 using EFFICIENT factorized decoding.
    
    Key difference from original: ANY set size is now feasible!
    """
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
    )
    
    print("=" * 70)
    print("EXPERIMENT 4: EFFICIENT MULTI-ITEM ML DECODING")
    print("=" * 70)
    print(f"\nUsing FACTORIZED method: O(N * l * n_theta) complexity")
    print(f"All set sizes are now feasible!\n")
    
    # Show complexity comparison
    compare_complexity(cfg.n_neurons, cfg.n_orientations, list(cfg.set_sizes))
    
    print(f"Configuration:")
    print(f"  N = {cfg.n_neurons} neurons")
    print(f"  n_theta = {cfg.n_orientations} orientation bins")
    print(f"  L = {cfg.n_locations} total locations")
    print(f"  gamma = {cfg.gamma} Hz/neuron -> Total = {cfg.total_activity} Hz")
    print(f"  T_d = {cfg.T_d} s")
    print(f"  Trials = {cfg.n_trials}")
    print(f"  Set sizes = {cfg.set_sizes}")
    print()
    
    rng = np.random.RandomState(cfg.seed)
    
    # Setup population
    print("Generating GP population with mixed selectivity...")
    population, theta_values = setup_population(cfg)
    print(f"  Generated {len(population)} neurons")
    
    # Run multi-item decoding (EFFICIENT!)
    print("\nRunning EFFICIENT multi-item decoding...")
    multi_item_results = run_set_size_experiment(population, theta_values, cfg, rng)
    
    # Run single-item baseline
    print("\nRunning single-item baseline...")
    single_item_results = run_single_item_baseline(population, theta_values, cfg, rng)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Set Size':<10} {'Multi-Item sigma':<18} {'Single-Item sigma':<18}")
    print("-" * 50)
    for l in cfg.set_sizes:
        mi_std = multi_item_results[l]['circular_std_deg']
        si_std = single_item_results[l]['circular_std_deg']
        print(f"{l:<10} {mi_std:<18.2f} {si_std:<18.2f}")
    
    # Check sqrt(l) scaling
    stds = [multi_item_results[l]['circular_std_deg'] for l in cfg.set_sizes]
    normalised = [stds[i] / np.sqrt(l) for i, l in enumerate(cfg.set_sizes)]
    cv = np.std(normalised) / np.mean(normalised) if len(normalised) > 1 else 0.0
    
    print(f"\nsqrt(l) Scaling Check:")
    print(f"  Normalised stds: {[f'{n:.2f}' for n in normalised]}")
    print(f"  CV of normalised: {cv:.3f} (< 0.1 is good)")
    
    return {
        'config': config,
        'exp_config': cfg,
        'theta_values': theta_values,
        'multi_item': multi_item_results,
        'single_item': single_item_results,
        'scaling': {
            'empirical_std': stds,
            'normalised_by_sqrt_l': normalised,
            'cv_normalised': cv,
        },
        'method': 'efficient_factorized',
    }


# =============================================================================
# ALIAS FOR BACKWARD COMPATIBILITY WITH run_experiments.py
# =============================================================================

# This is the key fix - provide the expected function name
run_experiment_4 = run_experiment_4_efficient


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
    set_sizes = list(cfg.set_sizes)
    
    mi_std = [multi_item[l]['circular_std_deg'] for l in set_sizes]
    si_std = [single_item[l]['circular_std_deg'] for l in set_sizes]
    
    # Figure 1: Error Scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(set_sizes, mi_std, 'o-', color=palette[0], lw=2.5, ms=10, 
            label='Multi-item (efficient)')
    ax.plot(set_sizes, si_std, 's--', color=palette[1], lw=2, ms=8, 
            label='Single-item baseline')
    
    if len(set_sizes) > 1:
        ref = [mi_std[0] * np.sqrt(l / set_sizes[0]) for l in set_sizes]
        ax.plot(set_sizes, ref, ':', color='gray', lw=2, alpha=0.7, label=r'$\propto \sqrt{\ell}$')
    
    ax.set_xlabel(r'Set Size ($\ell$)')
    ax.set_ylabel('Circular Std (degrees)')
    ax.set_title('A. Decoding Error vs Set Size')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    ax = axes[1]
    normalised = results['scaling']['normalised_by_sqrt_l']
    ax.plot(set_sizes, normalised, 'o-', color=palette[3], lw=2.5, ms=10)
    ax.axhline(np.mean(normalised), color='red', ls='--', lw=2,
               label=f'Mean = {np.mean(normalised):.2f} deg')
    ax.set_xlabel(r'Set Size ($\ell$)')
    ax.set_ylabel(r'Error Std / $\sqrt{\ell}$ (degrees)')
    ax.set_title(f'B. $\\sqrt{{\\ell}}$ Scaling (CV = {results["scaling"]["cv_normalised"]:.3f})')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    plt.suptitle('Experiment 4: Efficient Multi-Item ML Decoding',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_efficient_scaling.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp4_efficient_scaling.png")
    
    # Figure 2: Error Distributions
    n_sizes = len(set_sizes)
    fig, axes = plt.subplots(1, min(n_sizes, 6), figsize=(4*min(n_sizes, 6), 4))
    colors = sns.color_palette("coolwarm", n_sizes)
    
    if n_sizes == 1:
        axes = [axes]
    
    for i, l in enumerate(set_sizes[:6]):  # Show up to 6
        ax = axes[i]
        errors = np.degrees(multi_item[l]['errors'])
        
        sns.histplot(errors, kde=True, ax=ax, color=colors[i],
                     stat='density', alpha=0.6, bins=30)
        
        std = multi_item[l]['circular_std_deg']
        ax.set_xlabel('Error (degrees)')
        ax.set_ylabel('Density' if i == 0 else '')
        ax.set_title(f'l = {l}\nsigma = {std:.1f} deg')
        ax.set_xlim([-90, 90])
    
    plt.suptitle('Error Distributions (Efficient Multi-Item Decoder)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_efficient_distributions.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  Saved: exp4_efficient_distributions.png")


# =============================================================================
# STANDALONE ENTRY
# =============================================================================

if __name__ == '__main__':
    # Now we can test LARGE set sizes!
    config = {
        'n_neurons': 100,
        'n_orientations': 32,
        'n_locations': 16,
        'gamma': 100.0,
        'sigma_sq': 1e-6,
        'T_d': 0.1,
        'n_trials': 200,
        'seed': 42,
        'set_sizes': [2, 4, 6, 8, 10, 12],  # Previously impossible!
    }
    
    results = run_experiment_4_efficient(config)
    plot_results(results, 'results/exp4_efficient', show_plot=True)