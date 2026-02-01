"""
Experiment 3: Poisson Noise Analysis (v3.1 - DRY)

=============================================================================
THEORETICAL FOUNDATION
=============================================================================

This experiment demonstrates how Poisson spiking noise creates the capacity 
limit in working memory through the 1/√l degradation of per-item precision.

THE KEY INSIGHT:
----------------
- Total population activity: Σᵢ rᵢ = γN (CONSTANT due to DN)
- Total spikes: λ_total = γN × T_d (CONSTANT)
- SNR of total spikes: √(λ_total) (CONSTANT) ← NOT the capacity limit!

What DOES degrade with set size l:
- Per-item firing rate: r_item ≈ γN/l (decreases with l)
- Per-item spike count: λ_item = (γN/l) × T_d (decreases with l)  
- Per-item SNR: √(λ_item) ∝ 1/√l ← THIS is the capacity limit!

SUB-EXPERIMENTS:
----------------
3A: Per-Item SNR Scaling — demonstrates SNR ∝ 1/√l
3B: Time-Accuracy Trade-off — SNR_item = √(γN × T_d / l)
3C: Per-Item Spike Distributions — visualize resource allocation

Author: Mixed Selectivity Project
Date: January 2026
Version: 3.1 (DRY - uses core module functions)
=============================================================================
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from itertools import combinations

# Import from core modules (no re-implementation)
from core.gaussian_process import generate_neuron_population
from core.divisive_normalization import (
    compute_total_post_dn_analytical,
    compute_per_item_activity_efficient,
)
from core.poisson_spike import (
    generate_spikes_multi_trial,
    compute_theoretical_snr,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExpConfig:
    """Configuration for Experiment 3."""
    n_neurons: int = 100
    n_orientations: int = 10
    n_locations: int = 8
    gamma: float = 100.0
    sigma_sq: float = 1e-6
    T_d: float = 0.1
    n_trials: int = 1000
    set_sizes: tuple = (1, 2, 4, 6, 8)
    T_d_values: tuple = (0.05, 0.1, 0.2, 0.4)
    lambda_base: float = 0.3
    sigma_lambda: float = 0.5
    seed: int = 42
    
    @property
    def total_activity(self) -> float:
        """Total population activity budget (γN)."""
        return self.gamma * self.n_neurons


# =============================================================================
# HELPER: COMPUTE POST-DN RATES FOR A STIMULUS CONFIGURATION
# =============================================================================

def compute_post_dn_rates(
    G_stacked: np.ndarray,
    subset: Tuple[int, ...],
    fixed_thetas: np.ndarray,
    gamma: float,
    sigma_sq: float
) -> np.ndarray:
    """
    Compute post-DN firing rates for each neuron given a stimulus configuration.
    
    This implements the DN equation:
        r^{post}_i = γ · r^{pre}_i / [σ² + (1/N) Σ_j r^{pre}_j]
    
    where r^{pre}_i = ∏_{k ∈ S} g_i(θ_k, k) is the pre-DN response.
    
    Parameters
    ----------
    G_stacked : np.ndarray
        Pre-computed g(θ) = exp(f(θ)), shape (N, n_locations, n_orientations)
    subset : Tuple[int, ...]
        Active location indices
    fixed_thetas : np.ndarray
        Orientation index at each location, shape (n_locations,)
    gamma : float
        Gain constant
    sigma_sq : float
        Semi-saturation constant
        
    Returns
    -------
    R_post : np.ndarray
        Post-DN firing rates, shape (N,)
    """
    N = G_stacked.shape[0]
    
    # Step 1: Pre-DN response (product over active locations)
    R_pre = np.ones(N)
    for loc in subset:
        theta_idx = fixed_thetas[loc]
        R_pre *= G_stacked[:, loc, theta_idx]
    
    # Step 2: Population divisive normalization
    pop_mean = np.mean(R_pre)
    R_post = gamma * R_pre / (sigma_sq + pop_mean)
    
    return R_post


def generate_population_and_setup(
    config: ExpConfig,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GP population and sample fixed orientations.
    
    Returns
    -------
    G_stacked : np.ndarray
        Shape (N, n_locations, n_orientations)
    fixed_thetas : np.ndarray
        Fixed orientation indices, shape (n_locations,)
    """
    # Generate neuron population using core module
    population = generate_neuron_population(
        n_neurons=config.n_neurons,
        n_orientations=config.n_orientations,
        n_locations=config.n_locations,
        base_lengthscale=config.lambda_base,
        lengthscale_variability=config.sigma_lambda,
        seed=config.seed
    )
    
    # Pre-compute G = exp(f)
    f_samples_population = [neuron['f_samples'] for neuron in population]
    G_stacked = np.stack([np.exp(f) for f in f_samples_population], axis=0)
    
    # Sample fixed orientations
    fixed_thetas = rng.integers(0, config.n_orientations, size=config.n_locations)
    
    return G_stacked, fixed_thetas


# =============================================================================
# EXPERIMENT 3A: PER-ITEM SNR SCALING
# =============================================================================

def run_exp3a_snr_scaling(config: ExpConfig) -> Dict:
    """
    Experiment 3A: Demonstrate per-item SNR ∝ 1/√l scaling.
    
    Measures:
    1. Total population SNR (constant due to DN)
    2. Per-item SNR (degrades as 1/√l)
    """
    rng = np.random.default_rng(config.seed)
    rng_poisson = np.random.RandomState(config.seed + 1000)
    
    G_stacked, fixed_thetas = generate_population_and_setup(config, rng)
    
    # Use core module for theoretical predictions
    total_activity = compute_total_post_dn_analytical(config.gamma, config.n_neurons)
    
    results = {
        'set_sizes': list(config.set_sizes),
        'theoretical': {
            'lambda_total': [],
            'lambda_per_item': [],
            'snr_total': [],
            'snr_per_item': [],
        },
        'empirical': {
            'snr_total': [],
            'snr_per_item': [],
            'fano_mean': [],
            'rate_total': [],
            'rate_per_item': [],
        },
    }
    
    for l in config.set_sizes:
        # === THEORETICAL PREDICTIONS (using core module) ===
        lambda_total = total_activity * config.T_d
        per_item_activity = compute_per_item_activity_efficient(
            config.gamma, config.n_neurons, l
        )
        lambda_per_item = per_item_activity * config.T_d
        
        snr_total_theory = compute_theoretical_snr(lambda_total)
        snr_per_item_theory = compute_theoretical_snr(lambda_per_item)
        
        results['theoretical']['lambda_total'].append(lambda_total)
        results['theoretical']['lambda_per_item'].append(lambda_per_item)
        results['theoretical']['snr_total'].append(snr_total_theory)
        results['theoretical']['snr_per_item'].append(snr_per_item_theory)
        
        # === EMPIRICAL MEASUREMENT ===
        all_subsets = list(combinations(range(config.n_locations), l))
        n_subsets = len(all_subsets)
        
        total_spikes_all = []
        per_item_spikes_all = []
        all_rates = []
        
        for subset in all_subsets:
            # Compute post-DN rates
            R_post = compute_post_dn_rates(
                G_stacked, subset, fixed_thetas,
                config.gamma, config.sigma_sq
            )
            all_rates.append(R_post)
            
            # Generate Poisson spikes using core module
            spike_counts = generate_spikes_multi_trial(
                rates=R_post,
                T_d=config.T_d,
                n_trials=config.n_trials // n_subsets + 1,
                rng=rng_poisson
            )
            
            total_spikes = np.sum(spike_counts, axis=1)
            total_spikes_all.extend(total_spikes)
            per_item_spikes_all.extend(total_spikes / l)
        
        total_arr = np.array(total_spikes_all)
        per_item_arr = np.array(per_item_spikes_all)
        all_rates_arr = np.array(all_rates)
        
        # Empirical SNR
        snr_total_emp = np.mean(total_arr) / np.std(total_arr) if np.std(total_arr) > 0 else 0
        snr_per_item_emp = np.mean(per_item_arr) / np.std(per_item_arr) if np.std(per_item_arr) > 0 else 0
        fano = np.var(total_arr) / np.mean(total_arr) if np.mean(total_arr) > 0 else 1.0
        
        results['empirical']['snr_total'].append(snr_total_emp)
        results['empirical']['snr_per_item'].append(snr_per_item_emp)
        results['empirical']['fano_mean'].append(fano)
        results['empirical']['rate_total'].append(np.mean(np.sum(all_rates_arr, axis=1)))
        results['empirical']['rate_per_item'].append(np.mean(np.sum(all_rates_arr, axis=1)) / l)
    
    return results


# =============================================================================
# EXPERIMENT 3B: TIME-ACCURACY TRADE-OFF
# =============================================================================

def run_exp3b_time_tradeoff(config: ExpConfig) -> Dict:
    """
    Experiment 3B: Time-accuracy trade-off for per-item precision.
    
    Per-item SNR = √(γN × T_d / l)
    
    Doubling T_d compensates for doubling l.
    """
    total_activity = compute_total_post_dn_analytical(config.gamma, config.n_neurons)
    
    results = {
        'set_sizes': list(config.set_sizes),
        'T_d_values': list(config.T_d_values),
        'snr_total_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
        'snr_per_item_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
        'lambda_per_item_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
    }
    
    for i, l in enumerate(config.set_sizes):
        per_item_activity = compute_per_item_activity_efficient(
            config.gamma, config.n_neurons, l
        )
        
        for j, T_d in enumerate(config.T_d_values):
            lambda_total = total_activity * T_d
            lambda_per_item = per_item_activity * T_d
            
            results['snr_total_grid'][i, j] = compute_theoretical_snr(lambda_total)
            results['snr_per_item_grid'][i, j] = compute_theoretical_snr(lambda_per_item)
            results['lambda_per_item_grid'][i, j] = lambda_per_item
    
    return results


# =============================================================================
# EXPERIMENT 3C: SPIKE DISTRIBUTIONS
# =============================================================================

def run_exp3c_spike_distributions(config: ExpConfig) -> Dict:
    """
    Experiment 3C: Visualize per-item spike count distributions.
    """
    rng = np.random.default_rng(config.seed)
    rng_poisson = np.random.RandomState(config.seed + 3000)
    
    G_stacked, fixed_thetas = generate_population_and_setup(config, rng)
    
    selected_sizes = [1, 4, 8]
    total_activity = compute_total_post_dn_analytical(config.gamma, config.n_neurons)
    
    results = {
        'set_sizes': selected_sizes,
        'total_distributions': {},
        'per_item_distributions': {},
        'stats': {},
    }
    
    for l in selected_sizes:
        all_subsets = list(combinations(range(config.n_locations), l))
        n_subsets = len(all_subsets)
        
        total_spikes_all = []
        per_item_spikes_all = []
        
        for subset in all_subsets:
            R_post = compute_post_dn_rates(
                G_stacked, subset, fixed_thetas,
                config.gamma, config.sigma_sq
            )
            
            spike_counts = generate_spikes_multi_trial(
                rates=R_post,
                T_d=config.T_d,
                n_trials=config.n_trials // n_subsets + 1,
                rng=rng_poisson
            )
            
            total_spikes = np.sum(spike_counts, axis=1)
            total_spikes_all.extend(total_spikes)
            per_item_spikes_all.extend(total_spikes / l)
        
        total_arr = np.array(total_spikes_all)
        per_item_arr = np.array(per_item_spikes_all)
        
        lambda_total = total_activity * config.T_d
        lambda_per_item = compute_per_item_activity_efficient(
            config.gamma, config.n_neurons, l
        ) * config.T_d
        
        results['total_distributions'][l] = total_arr
        results['per_item_distributions'][l] = per_item_arr
        results['stats'][l] = {
            'lambda_total': lambda_total,
            'lambda_per_item': lambda_per_item,
            'mean_total': np.mean(total_arr),
            'std_total': np.std(total_arr),
            'snr_total': np.mean(total_arr) / np.std(total_arr),
            'mean_per_item': np.mean(per_item_arr),
            'std_per_item': np.std(per_item_arr),
            'snr_per_item': np.mean(per_item_arr) / np.std(per_item_arr),
        }
    
    return results


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment_3(config: Dict) -> Dict:
    """
    Run all sub-experiments for Experiment 3.
    
    Parameters
    ----------
    config : Dict
        Configuration from run_experiments.py
        
    Returns
    -------
    results : Dict
        Combined results from all sub-experiments
    """
    exp_config = ExpConfig(
        n_neurons=config.get('n_neurons', 100),
        n_orientations=config.get('n_orientations', 10),
        n_locations=config.get('n_locations', 8),
        gamma=config.get('gamma', 100.0),
        sigma_sq=config.get('sigma_sq', 1e-6),
        T_d=config.get('T_d', 0.1),
        n_trials=config.get('n_trials', 1000),
        set_sizes=tuple(config.get('set_sizes', [1, 2, 4, 6, 8])),
        lambda_base=config.get('lambda_base', 0.3),
        sigma_lambda=config.get('sigma_lambda', 0.5),
        seed=config.get('seed', 42),
    )
    
    print("=" * 70)
    print("EXPERIMENT 3: POISSON NOISE ANALYSIS (v3.1 - DRY)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N = {exp_config.n_neurons} neurons")
    print(f"  γ = {exp_config.gamma} Hz/neuron")
    print(f"  T_d = {exp_config.T_d} s")
    print(f"  Total activity: γN = {exp_config.total_activity:.0f} Hz (CONSTANT)")
    print(f"  Trials: {exp_config.n_trials}")
    print()
    print("KEY INSIGHT:")
    print("  Total SNR ∝ √(γN×T_d) is CONSTANT")
    print("  Per-item SNR ∝ √(γN×T_d/l) DEGRADES as 1/√l")
    print()
    
    print("Running Experiment 3A: Per-Item SNR Scaling...")
    results_3a = run_exp3a_snr_scaling(exp_config)
    
    print("Running Experiment 3B: Time-Accuracy Trade-off...")
    results_3b = run_exp3b_time_tradeoff(exp_config)
    
    print("Running Experiment 3C: Per-Item Spike Distributions...")
    results_3c = run_exp3c_spike_distributions(exp_config)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'l':<5} {'λ_total':<10} {'λ_item':<10} {'SNR_total':<12} {'SNR_item':<12}")
    print("-" * 55)
    
    for i, l in enumerate(results_3a['set_sizes']):
        print(f"{l:<5} "
              f"{results_3a['theoretical']['lambda_total'][i]:<10.0f} "
              f"{results_3a['theoretical']['lambda_per_item'][i]:<10.0f} "
              f"{results_3a['theoretical']['snr_total'][i]:<12.1f} "
              f"{results_3a['theoretical']['snr_per_item'][i]:<12.1f}")
    
    print("\nNote: λ_total constant, λ_item ∝ 1/l → SNR_item ∝ 1/√l")
    print("=" * 70)
    
    return {
        'config': config,
        'exp_config': exp_config,
        'exp3a': results_3a,
        'exp3b': results_3b,
        'exp3c': results_3c,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Generate all figures for Experiment 3."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("deep")
    
    exp_config = results['exp_config']
    r3a = results['exp3a']
    r3b = results['exp3b']
    r3c = results['exp3c']
    set_sizes = r3a['set_sizes']
    
    # =========================================================================
    # Figure 1: SNR Scaling (Experiment 3A)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Total vs Per-Item SNR
    ax = axes[0]
    ax.plot(set_sizes, r3a['theoretical']['snr_total'], 'o-', 
            color=palette[0], linewidth=2.5, markersize=10, label='Total SNR (constant)')
    ax.plot(set_sizes, r3a['theoretical']['snr_per_item'], 's-', 
            color=palette[3], linewidth=2.5, markersize=10, label='Per-item SNR ∝ 1/√l')
    ax.plot(set_sizes, r3a['empirical']['snr_per_item'], 'x', 
            color=palette[3], markersize=12, markeredgewidth=2, label='Per-item (empirical)')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('A. The Capacity Limit: Per-Item SNR Degrades')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    # Panel B: Lambda comparison
    ax = axes[1]
    x = np.arange(len(set_sizes))
    width = 0.35
    ax.bar(x - width/2, r3a['theoretical']['lambda_total'], width, 
           color=palette[0], alpha=0.7, label='λ_total', edgecolor='black')
    ax.bar(x + width/2, r3a['theoretical']['lambda_per_item'], width,
           color=palette[3], alpha=0.7, label='λ_per_item', edgecolor='black')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Expected Spike Count (λ)')
    ax.set_title('B. Resource Allocation')
    ax.set_xticks(x)
    ax.set_xticklabels(set_sizes)
    ax.legend()
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path / 'exp3a_snr_scaling.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3a_snr_scaling.png")
    
    # =========================================================================
    # Figure 2: Time-Accuracy Trade-off (Experiment 3B)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = sns.color_palette("viridis", len(r3b['T_d_values']))
    for j, T_d in enumerate(r3b['T_d_values']):
        ax.plot(r3b['set_sizes'], r3b['snr_per_item_grid'][:, j], 'o-', 
                color=colors[j], linewidth=2.5, markersize=8, label=f'T_d = {T_d:.2f}s')
    
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Per-Item SNR')
    ax.set_title('Time-Accuracy Trade-off: Per-Item SNR = √(γN × T_d / l)')
    ax.legend(title='Integration Time')
    ax.set_xticks(r3b['set_sizes'])
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path / 'exp3b_time_tradeoff.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3b_time_tradeoff.png")
    
    # =========================================================================
    # Figure 3: Spike Distributions (Experiment 3C)
    # =========================================================================
    n_sizes = len(r3c['set_sizes'])
    fig, axes = plt.subplots(2, n_sizes, figsize=(4*n_sizes, 7))
    colors = sns.color_palette("coolwarm", n_sizes)
    
    for i, l in enumerate(r3c['set_sizes']):
        stats = r3c['stats'][l]
        
        # Top: Total spikes
        ax = axes[0, i]
        sns.histplot(r3c['total_distributions'][l], kde=True, ax=ax, 
                     color='gray', stat='density', alpha=0.6)
        ax.axvline(stats['mean_total'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Total Spike Count')
        ax.set_title(f'l={l}: Total (λ={stats["lambda_total"]:.0f})')
        
        # Bottom: Per-item spikes
        ax = axes[1, i]
        sns.histplot(r3c['per_item_distributions'][l], kde=True, ax=ax, 
                     color=colors[i], stat='density', alpha=0.6)
        ax.axvline(stats['mean_per_item'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Per-Item Spike Count')
        ax.set_title(f'l={l}: Per-Item (λ={stats["lambda_per_item"]:.0f})')
    
    plt.suptitle('Exp 3C: Total (constant) vs Per-Item (decreasing) Spike Distributions',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp3c_distributions.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3c_distributions.png")
    
    # =========================================================================
    # Figure 4: Summary
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A: Core result
    ax = axes[0, 0]
    ax.plot(set_sizes, r3a['theoretical']['snr_total'], 'o-', 
            color=palette[0], linewidth=3, markersize=10, label='Total SNR')
    ax.plot(set_sizes, r3a['theoretical']['snr_per_item'], 's-', 
            color=palette[3], linewidth=3, markersize=10, label='Per-item SNR')
    ax.fill_between(set_sizes, r3a['theoretical']['snr_per_item'],
                    r3a['theoretical']['snr_total'], alpha=0.2, color='red')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('SNR')
    ax.set_title('A. THE CAPACITY LIMIT', fontweight='bold')
    ax.legend()
    ax.set_xticks(set_sizes)
    
    # B: Resource allocation
    ax = axes[0, 1]
    x = np.arange(len(set_sizes))
    ax.bar(x - 0.2, r3a['theoretical']['lambda_total'], 0.4, 
           color=palette[0], alpha=0.7, label='λ_total')
    ax.bar(x + 0.2, r3a['theoretical']['lambda_per_item'], 0.4,
           color=palette[3], alpha=0.7, label='λ_per_item')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Expected Spikes')
    ax.set_title('B. RESOURCE ALLOCATION', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(set_sizes)
    ax.legend()
    
    # C: Time trade-off
    ax = axes[1, 0]
    for j, T_d in enumerate(r3b['T_d_values']):
        ax.plot(r3b['set_sizes'], r3b['snr_per_item_grid'][:, j], 'o-', 
                linewidth=2, markersize=7, label=f'T_d={T_d:.2f}s')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Per-Item SNR')
    ax.set_title('C. TIME-ACCURACY TRADE-OFF', fontweight='bold')
    ax.legend()
    ax.set_xticks(r3b['set_sizes'])
    
    # D: Distribution comparison
    ax = axes[1, 1]
    for i, l in enumerate(r3c['set_sizes']):
        data = r3c['per_item_distributions'][l]
        data_norm = (data - np.mean(data)) / np.std(data)
        sns.kdeplot(data_norm, ax=ax, label=f'l={l}', linewidth=2)
    ax.set_xlabel('Normalized Per-Item Spikes')
    ax.set_ylabel('Density')
    ax.set_title('D. PER-ITEM DISTRIBUTIONS', fontweight='bold')
    ax.legend()
    
    plt.suptitle(f'Experiment 3: The 1/√l Capacity Limit\n'
                 f'(N={exp_config.n_neurons}, γ={exp_config.gamma} Hz, T_d={exp_config.T_d}s)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp3_summary.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3_summary.png")


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_neurons': 100,
        'n_orientations': 10,
        'n_locations': 8,
        'gamma': 100.0,
        'sigma_sq': 1e-6,
        'T_d': 0.1,
        'n_trials': 1000,
        'seed': 42,
        'set_sizes': [1, 2, 4, 6, 8],
        'lambda_base': 0.3,
        'sigma_lambda': 0.5,
    }
    
    results = run_experiment_3(config)
    plot_results(results, 'results/exp3', show_plot=True)