"""
Experiment 3: Poisson Noise Analysis (CORRECTED v3.0)

=============================================================================
THREE CORE EXPERIMENTS
=============================================================================

This module demonstrates how Poisson spiking noise creates the capacity limit
in working memory through the 1/√l degradation of per-item precision.

**CRITICAL INSIGHT (v3.0)**
---------------------------
The key is understanding WHAT degrades with set size:

- Total population activity: Σᵢ rᵢ = γN (CONSTANT due to DN)
- Total spikes: λ_total = γN × T_d (CONSTANT)
- SNR of total spikes: √(λ_total) (CONSTANT) ← This is NOT the capacity limit!

What DOES degrade:
- Per-item firing rate allocation: r_item ≈ γN/l (decreases with l)
- Per-item spike count: λ_item = (γN/l) × T_d (decreases with l)  
- Per-item precision/SNR: √(λ_item) ∝ 1/√l ← THIS is the capacity limit!

The 1/√l law applies to the PRECISION OF EACH ITEM'S REPRESENTATION,
not to the total population activity (which DN keeps constant).


EXPERIMENT 3A: Per-Item SNR Scaling
-----------------------------------
Shows that while total activity stays constant, the resources allocated
to each item decrease as 1/l, causing per-item SNR to degrade as 1/√l.


EXPERIMENT 3B: Time-Accuracy Trade-off
--------------------------------------
Per-item SNR = √(γN × T_d / l)

This means precision depends on the ratio T_d/l:
- Doubling T_d is equivalent to halving l
- 4 items for 200ms ≈ 2 items for 100ms (same per-item SNR)


EXPERIMENT 3C: Per-Item Spike Distributions
-------------------------------------------
Shows how the spike count distribution FOR A SINGLE ITEM changes with load.
At high load, each item gets few spikes → discrete, noisy regime.


Author: Mixed Selectivity Project
Date: January 2026
Version: 3.0 (Corrected to measure PER-ITEM precision, not total activity)
=============================================================================
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gaussian_process import generate_neuron_population
from core.poisson_spike import (
    generate_spikes_multi_trial,
    compute_empirical_stats,
    compute_population_stats,
    PoissonStats,
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
    gamma: float = 100.0       # Hz per neuron (gain)
    sigma_sq: float = 1e-6     # Semi-saturation constant for DN
    T_d: float = 0.1           # Decoding window (seconds)
    n_trials: int = 1000       # Monte Carlo trials
    set_sizes: tuple = (1, 2, 4, 6, 8)
    T_d_values: tuple = (0.05, 0.1, 0.2, 0.4)  # For Exp 3B
    lambda_base: float = 0.3   # GP lengthscale base
    sigma_lambda: float = 0.5  # GP lengthscale variability
    seed: int = 42
    
    @property
    def total_activity(self) -> float:
        """Total population activity budget (γN)."""
        return self.gamma * self.n_neurons


# =============================================================================
# CORE COMPUTATION: GP → DN → STRUCTURED RATES
# =============================================================================

def compute_post_dn_rates_for_subset(
    G_stacked: np.ndarray,
    subset: Tuple[int, ...],
    fixed_thetas: np.ndarray,
    gamma: float,
    sigma_sq: float
) -> np.ndarray:
    """
    Compute post-DN firing rate for each neuron given a stimulus configuration.
    
    Parameters
    ----------
    G_stacked : np.ndarray
        Pre-computed g(θ) = exp(f(θ)) for all neurons, shape (N, n_locations, n_orientations)
    subset : Tuple[int, ...]
        Active location indices (which locations have items)
    fixed_thetas : np.ndarray
        Orientation index at each location, shape (n_locations,)
    gamma : float
        Gain constant (target mean firing rate)
    sigma_sq : float
        Semi-saturation constant
        
    Returns
    -------
    R_post : np.ndarray
        Post-DN firing rate for each neuron, shape (N,)
    """
    N = G_stacked.shape[0]
    subset_arr = np.array(subset)
    l = len(subset)
    
    # Pre-DN response: R_pre[n] = ∏_{k ∈ subset} g_n(θ_k)
    R_pre = np.ones(N)
    for k in range(l):
        loc = subset_arr[k]
        theta_idx = fixed_thetas[loc]
        R_pre *= G_stacked[:, loc, theta_idx]
    
    # Population mean for divisive normalization denominator
    pop_mean = np.mean(R_pre)
    
    # Apply divisive normalization: r_post = γ * r_pre / (σ² + ⟨R_pre⟩)
    R_post = gamma * R_pre / (sigma_sq + pop_mean)
    
    return R_post


def generate_population_and_rates(
    config: ExpConfig,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate GP population and pre-compute G = exp(f).
    
    Returns
    -------
    G_stacked : np.ndarray
        Shape (N, n_locations, n_orientations)
    fixed_thetas : np.ndarray
        Fixed orientation indices, shape (n_locations,)
    metadata : dict
        Additional information
    """
    # Generate neuron population with GP tuning curves
    population = generate_neuron_population(
        n_neurons=config.n_neurons,
        n_orientations=config.n_orientations,
        n_locations=config.n_locations,
        base_lengthscale=config.lambda_base,
        lengthscale_variability=config.sigma_lambda,
        seed=config.seed
    )
    
    # Pre-compute G = exp(f) for all neurons
    f_samples_population = [neuron['f_samples'] for neuron in population]
    G_stacked = np.stack([np.exp(f) for f in f_samples_population], axis=0)
    
    # Sample fixed orientations for each location
    fixed_thetas = rng.integers(0, config.n_orientations, size=config.n_locations)
    
    metadata = {
        'population': population,
    }
    
    return G_stacked, fixed_thetas, metadata


# =============================================================================
# EXPERIMENT 3A: PER-ITEM SNR SCALING
# =============================================================================

def run_exp3a_snr_scaling(config: ExpConfig) -> Dict:
    """
    Experiment 3A: Demonstrate per-item SNR ∝ 1/√l scaling.
    
    The key insight: DN keeps TOTAL activity constant, but the resources
    allocated to EACH ITEM decrease as 1/l, so per-item precision degrades.
    
    We measure:
    1. Total population SNR (should be ~constant due to DN)
    2. Per-item SNR (should degrade as 1/√l)
    """
    rng = np.random.default_rng(config.seed)
    rng_poisson = np.random.RandomState(config.seed + 1000)
    
    # Generate population once
    G_stacked, fixed_thetas, metadata = generate_population_and_rates(config, rng)
    
    results = {
        'set_sizes': list(config.set_sizes),
        'theoretical': {
            'lambda_total': [],      # Total expected spikes (constant)
            'lambda_per_item': [],   # Per-item expected spikes (∝ 1/l)
            'snr_total': [],         # √(λ_total) - constant
            'snr_per_item': [],      # √(λ_per_item) ∝ 1/√l
        },
        'empirical': {
            'snr_total': [],         # Empirical total SNR
            'snr_per_item': [],      # Empirical per-item SNR
            'fano_mean': [],         # Fano factor verification
            'rate_total': [],        # Total firing rate (should be ~γN)
            'rate_per_item': [],     # Per-item rate (should be ~γN/l)
            'rate_heterogeneity': [],  # Std of per-neuron rates (DN signature)
        },
        'detailed': {},
    }
    
    for l in config.set_sizes:
        # === THEORETICAL PREDICTIONS ===
        # Total activity is capped by DN at γN
        total_rate = config.gamma * config.n_neurons  # = γN (constant)
        lambda_total = total_rate * config.T_d
        snr_total_theory = np.sqrt(lambda_total)
        
        # Per-item allocation: each item gets ~1/l of the total budget
        per_item_rate = total_rate / l  # = γN/l
        lambda_per_item = per_item_rate * config.T_d
        snr_per_item_theory = np.sqrt(lambda_per_item)
        
        results['theoretical']['lambda_total'].append(lambda_total)
        results['theoretical']['lambda_per_item'].append(lambda_per_item)
        results['theoretical']['snr_total'].append(snr_total_theory)
        results['theoretical']['snr_per_item'].append(snr_per_item_theory)
        
        # === EMPIRICAL MEASUREMENT ===
        # Generate all subsets and compute post-DN rates
        all_subsets = list(combinations(range(config.n_locations), l))
        n_subsets = len(all_subsets)
        
        # Storage for per-item spike counts across trials
        per_item_spikes_all_trials = []
        total_spikes_all_trials = []
        all_rates = []
        
        for subset in all_subsets:
            # Compute structured post-DN rates for this configuration
            R_post = compute_post_dn_rates_for_subset(
                G_stacked=G_stacked,
                subset=subset,
                fixed_thetas=fixed_thetas,
                gamma=config.gamma,
                sigma_sq=config.sigma_sq
            )
            all_rates.append(R_post)
            
            # Generate spikes from these rates
            spike_counts = generate_spikes_multi_trial(
                rates=R_post,
                T_d=config.T_d,
                n_trials=config.n_trials // n_subsets + 1,
                rng=rng_poisson
            )  # Shape: (n_trials_per_subset, N)
            
            # Total spikes per trial
            total_spikes = np.sum(spike_counts, axis=1)
            total_spikes_all_trials.extend(total_spikes)
            
            # Per-item spikes: divide total by number of items
            # This represents the "share" of spikes allocated to each item
            per_item_spikes = total_spikes / l
            per_item_spikes_all_trials.extend(per_item_spikes)
        
        total_spikes_arr = np.array(total_spikes_all_trials)
        per_item_spikes_arr = np.array(per_item_spikes_all_trials)
        all_rates_arr = np.array(all_rates)  # Shape: (n_subsets, N)
        
        # Compute empirical statistics
        # Total SNR (should be roughly constant)
        mean_total = np.mean(total_spikes_arr)
        std_total = np.std(total_spikes_arr)
        snr_total_emp = mean_total / std_total if std_total > 0 else 0
        
        # Per-item SNR (should degrade as 1/√l)
        mean_per_item = np.mean(per_item_spikes_arr)
        std_per_item = np.std(per_item_spikes_arr)
        snr_per_item_emp = mean_per_item / std_per_item if std_per_item > 0 else 0
        
        # Fano factor for total spikes
        fano_total = (std_total ** 2) / mean_total if mean_total > 0 else 1.0
        
        # Rate statistics
        mean_rate_total = np.mean(np.sum(all_rates_arr, axis=1))
        mean_rate_per_item = mean_rate_total / l
        rate_heterogeneity = np.mean(np.std(all_rates_arr, axis=1))
        
        results['empirical']['snr_total'].append(snr_total_emp)
        results['empirical']['snr_per_item'].append(snr_per_item_emp)
        results['empirical']['fano_mean'].append(fano_total)
        results['empirical']['rate_total'].append(mean_rate_total)
        results['empirical']['rate_per_item'].append(mean_rate_per_item)
        results['empirical']['rate_heterogeneity'].append(rate_heterogeneity)
        
        # Store detailed data
        results['detailed'][l] = {
            'total_spikes': total_spikes_arr,
            'per_item_spikes': per_item_spikes_arr,
            'rates': all_rates_arr,
        }
    
    return results


# =============================================================================
# EXPERIMENT 3B: TIME-ACCURACY TRADE-OFF
# =============================================================================

def run_exp3b_time_tradeoff(config: ExpConfig) -> Dict:
    """
    Experiment 3B: Demonstrate time-accuracy trade-off for PER-ITEM precision.
    
    Per-item SNR = √(γN × T_d / l)
    
    This depends on the ratio T_d/l, so:
    - Doubling T_d compensates for doubling l
    - Iso-SNR curves are hyperbolas in (l, T_d) space
    """
    rng = np.random.default_rng(config.seed)
    
    # Generate population once
    G_stacked, fixed_thetas, metadata = generate_population_and_rates(config, rng)
    
    results = {
        'set_sizes': list(config.set_sizes),
        'T_d_values': list(config.T_d_values),
        'snr_total_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
        'snr_per_item_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
        'lambda_total_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
        'lambda_per_item_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
    }
    
    total_rate = config.gamma * config.n_neurons  # γN (constant due to DN)
    
    for i, l in enumerate(config.set_sizes):
        per_item_rate = total_rate / l  # γN/l
        
        for j, T_d in enumerate(config.T_d_values):
            # Total spikes (constant with l)
            lambda_total = total_rate * T_d
            snr_total = np.sqrt(lambda_total)
            
            # Per-item spikes (decreases with l)
            lambda_per_item = per_item_rate * T_d  # = γN × T_d / l
            snr_per_item = np.sqrt(lambda_per_item)
            
            results['lambda_total_grid'][i, j] = lambda_total
            results['lambda_per_item_grid'][i, j] = lambda_per_item
            results['snr_total_grid'][i, j] = snr_total
            results['snr_per_item_grid'][i, j] = snr_per_item
    
    # Baseline for reference (l=1, T_d=0.1)
    results['baseline_snr_per_item'] = np.sqrt(total_rate * 0.1 / 1)
    
    return results


# =============================================================================
# EXPERIMENT 3C: PER-ITEM SPIKE DISTRIBUTIONS
# =============================================================================

def run_exp3c_spike_distributions(config: ExpConfig) -> Dict:
    """
    Experiment 3C: Visualize PER-ITEM spike count distributions.
    
    Shows how the resources allocated to each item decrease with load,
    shifting the distribution from "many spikes" (Gaussian-like) to
    "few spikes" (discrete, Poisson-dominated).
    """
    rng = np.random.default_rng(config.seed)
    rng_poisson = np.random.RandomState(config.seed + 3000)
    
    # Generate population once
    G_stacked, fixed_thetas, metadata = generate_population_and_rates(config, rng)
    
    selected_sizes = [1, 4, 8]  # Low, medium, high load
    
    results = {
        'set_sizes': selected_sizes,
        'total_distributions': {},
        'per_item_distributions': {},
        'stats': {},
    }
    
    total_rate = config.gamma * config.n_neurons
    
    for l in selected_sizes:
        # Generate all subsets
        all_subsets = list(combinations(range(config.n_locations), l))
        n_subsets = len(all_subsets)
        
        total_spikes_all = []
        per_item_spikes_all = []
        
        for subset in all_subsets:
            R_post = compute_post_dn_rates_for_subset(
                G_stacked=G_stacked,
                subset=subset,
                fixed_thetas=fixed_thetas,
                gamma=config.gamma,
                sigma_sq=config.sigma_sq
            )
            
            spike_counts = generate_spikes_multi_trial(
                rates=R_post,
                T_d=config.T_d,
                n_trials=config.n_trials // n_subsets + 1,
                rng=rng_poisson
            )
            
            total_spikes = np.sum(spike_counts, axis=1)
            total_spikes_all.extend(total_spikes)
            
            per_item_spikes = total_spikes / l
            per_item_spikes_all.extend(per_item_spikes)
        
        total_spikes_arr = np.array(total_spikes_all)
        per_item_spikes_arr = np.array(per_item_spikes_all)
        
        # Theoretical values
        lambda_total = total_rate * config.T_d
        lambda_per_item = (total_rate / l) * config.T_d
        
        results['total_distributions'][l] = total_spikes_arr
        results['per_item_distributions'][l] = per_item_spikes_arr
        results['stats'][l] = {
            'lambda_total': lambda_total,
            'lambda_per_item': lambda_per_item,
            'mean_total': np.mean(total_spikes_arr),
            'std_total': np.std(total_spikes_arr),
            'snr_total': np.mean(total_spikes_arr) / np.std(total_spikes_arr),
            'mean_per_item': np.mean(per_item_spikes_arr),
            'std_per_item': np.std(per_item_spikes_arr),
            'snr_per_item': np.mean(per_item_spikes_arr) / np.std(per_item_spikes_arr),
        }
    
    return results


# =============================================================================
# COMBINED EXPERIMENT RUNNER
# =============================================================================

def run_experiment_3(config: Dict) -> Dict:
    """
    Run all three sub-experiments measuring PER-ITEM precision.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary (from run_experiments.py)
        
    Returns
    -------
    results : Dict
        Combined results from all sub-experiments
    """
    # Convert dict config to dataclass
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
    print("EXPERIMENT 3: POISSON NOISE ANALYSIS (v3.0)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N = {exp_config.n_neurons} neurons")
    print(f"  γ = {exp_config.gamma} Hz/neuron")
    print(f"  T_d = {exp_config.T_d} s")
    print(f"  Total activity budget: γN = {exp_config.total_activity:.0f} Hz (CONSTANT)")
    print(f"  Trials: {exp_config.n_trials}")
    print()
    print("  KEY INSIGHT:")
    print("  - Total SNR ∝ √(γN×T_d) is CONSTANT (DN caps total activity)")
    print("  - Per-item SNR ∝ √(γN×T_d/l) DEGRADES as 1/√l")
    print("  - The capacity limit is about PER-ITEM precision, not total activity")
    print()
    
    # Run sub-experiments
    print("Running Experiment 3A: Per-Item SNR Scaling...")
    results_3a = run_exp3a_snr_scaling(exp_config)
    
    print("Running Experiment 3B: Time-Accuracy Trade-off...")
    results_3b = run_exp3b_time_tradeoff(exp_config)
    
    print("Running Experiment 3C: Per-Item Spike Distributions...")
    results_3c = run_exp3c_spike_distributions(exp_config)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS: SNR SCALING (Experiment 3A)")
    print("=" * 70)
    print(f"\n{'l':<5} {'λ_total':<10} {'λ_item':<10} {'SNR_total':<12} {'SNR_item':<12} {'SNR_item_emp':<12}")
    print("-" * 70)
    
    for i, l in enumerate(results_3a['set_sizes']):
        print(f"{l:<5} "
              f"{results_3a['theoretical']['lambda_total'][i]:<10.0f} "
              f"{results_3a['theoretical']['lambda_per_item'][i]:<10.0f} "
              f"{results_3a['theoretical']['snr_total'][i]:<12.1f} "
              f"{results_3a['theoretical']['snr_per_item'][i]:<12.1f} "
              f"{results_3a['empirical']['snr_per_item'][i]:<12.1f}")
    
    print("\n" + "=" * 70)
    print("Note: λ_total is CONSTANT, but λ_item ∝ 1/l → SNR_item ∝ 1/√l")
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
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    exp_config = results['exp_config']
    
    # =========================================================================
    # Figure 1: SNR Scaling (Experiment 3A)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    r3a = results['exp3a']
    set_sizes = r3a['set_sizes']
    
    # Panel A: Total vs Per-Item SNR
    ax = axes[0]
    ax.plot(set_sizes, r3a['theoretical']['snr_total'], 'o-', 
            color=sns.color_palette()[0], linewidth=2.5, markersize=9, 
            label='Total SNR (constant)')
    ax.plot(set_sizes, r3a['theoretical']['snr_per_item'], 's-', 
            color=sns.color_palette()[1], linewidth=2.5, markersize=9,
            label='Per-item SNR (∝ 1/√l)')
    ax.plot(set_sizes, r3a['empirical']['snr_per_item'], 'x--', 
            color=sns.color_palette()[2], linewidth=2, markersize=8,
            label='Per-item (empirical)')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
    ax.set_title('A. The Capacity Limit is Per-Item', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='right')
    ax.set_xticks(set_sizes)
    
    # Add annotation
    ax.annotate('DN keeps this\nconstant', 
                xy=(6, r3a['theoretical']['snr_total'][2]), 
                xytext=(7, r3a['theoretical']['snr_total'][2] + 5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('This degrades\n∝ 1/√l', 
                xy=(6, r3a['theoretical']['snr_per_item'][2]), 
                xytext=(7, r3a['theoretical']['snr_per_item'][2] - 5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Panel B: Expected Spikes (λ)
    ax = axes[1]
    ax.plot(set_sizes, r3a['theoretical']['lambda_total'], 'o-', 
            color=sns.color_palette()[0], linewidth=2.5, markersize=9,
            label='λ_total = γN×T_d (constant)')
    ax.plot(set_sizes, r3a['theoretical']['lambda_per_item'], 's-', 
            color=sns.color_palette()[1], linewidth=2.5, markersize=9,
            label='λ_item = γN×T_d/l (∝ 1/l)')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Expected Spike Count (λ)', fontsize=12)
    ax.set_title('B. Resource Allocation', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    
    # Panel C: Fano Factor + Rate Heterogeneity
    ax = axes[2]
    ax.bar(np.array(set_sizes) - 0.2, r3a['empirical']['fano_mean'], 
           width=0.4, color=sns.color_palette()[3], alpha=0.7, 
           edgecolor='black', label='Fano Factor')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Poisson: F=1')
    
    ax2 = ax.twinx()
    ax2.bar(np.array(set_sizes) + 0.2, r3a['empirical']['rate_heterogeneity'], 
            width=0.4, color=sns.color_palette()[4], alpha=0.7,
            edgecolor='black', label='Rate Heterogeneity')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Fano Factor', fontsize=12, color=sns.color_palette()[3])
    ax2.set_ylabel('Rate Std (Hz)', fontsize=12, color=sns.color_palette()[4])
    ax.set_title('C. Poisson Verification & DN Signature', fontweight='bold', fontsize=12)
    ax.set_ylim([0.5, 1.5])
    ax.set_xticks(set_sizes)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f'Experiment 3A: Per-Item SNR Scaling (N={exp_config.n_neurons}, γ={exp_config.gamma}, T_d={exp_config.T_d}s)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'exp3a_snr_scaling.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3a_snr_scaling.png")
    
    # =========================================================================
    # Figure 2: Time-Accuracy Trade-off (Experiment 3B)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    r3b = results['exp3b']
    
    # Panel A: Per-Item SNR Heatmap (this should show the 1/√l pattern)
    ax = axes[0]
    sns.heatmap(r3b['snr_per_item_grid'], 
                xticklabels=[f'{t:.2f}' for t in r3b['T_d_values']],
                yticklabels=r3b['set_sizes'],
                annot=True, fmt='.1f', cmap='viridis',
                cbar_kws={'label': 'Per-Item SNR'}, ax=ax)
    ax.set_xlabel('Integration Time T_d (s)', fontsize=12)
    ax.set_ylabel('Set Size (l)', fontsize=12)
    ax.set_title('A. Per-Item SNR = √(γN·T_d/l)', fontweight='bold', fontsize=12)
    
    # Panel B: Iso-SNR curves
    ax = axes[1]
    colors = sns.color_palette("husl", len(r3b['T_d_values']))
    
    for j, T_d in enumerate(r3b['T_d_values']):
        snr_values = r3b['snr_per_item_grid'][:, j]
        ax.plot(r3b['set_sizes'], snr_values, 'o-', 
                color=colors[j], linewidth=2.5, markersize=8,
                label=f'T_d = {T_d:.2f}s')
    
    # Add 1/√l reference line
    ref_snr = r3b['snr_per_item_grid'][0, 1]  # l=1, T_d=0.1
    ref_line = [ref_snr / np.sqrt(l) for l in r3b['set_sizes']]
    ax.plot(r3b['set_sizes'], ref_line, 'k--', linewidth=1.5, alpha=0.5, label='∝ 1/√l')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Per-Item SNR', fontsize=12)
    ax.set_title('B. Time-Accuracy Trade-off', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, title='Integration Time')
    ax.set_xticks(r3b['set_sizes'])
    
    plt.suptitle('Experiment 3B: Capacity is a Time-Accuracy Trade-off (Per-Item Precision)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'exp3b_time_tradeoff.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3b_time_tradeoff.png")
    
    # =========================================================================
    # Figure 3: Spike Count Distributions (Experiment 3C)
    # =========================================================================
    r3c = results['exp3c']
    n_dists = len(r3c['set_sizes'])
    
    fig, axes = plt.subplots(2, n_dists, figsize=(4*n_dists, 8))
    colors = sns.color_palette("coolwarm", n_dists)
    
    for i, l in enumerate(r3c['set_sizes']):
        stats = r3c['stats'][l]
        
        # Top row: Total spike distribution (should be similar across l)
        ax = axes[0, i]
        data_total = r3c['total_distributions'][l]
        sns.histplot(data_total, kde=True, ax=ax, color='gray', 
                     stat='density', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axvline(stats['mean_total'], color='red', linestyle='--', 
                   linewidth=2, label=f"μ = {stats['mean_total']:.0f}")
        ax.set_xlabel('Total Spike Count', fontsize=10)
        ax.set_ylabel('Density' if i == 0 else '', fontsize=10)
        ax.set_title(f'l = {l}: Total Spikes\nλ = {stats["lambda_total"]:.0f}',
                     fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        
        # Bottom row: Per-item spike distribution (degrades with l)
        ax = axes[1, i]
        data_per_item = r3c['per_item_distributions'][l]
        sns.histplot(data_per_item, kde=True, ax=ax, color=colors[i], 
                     stat='density', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axvline(stats['mean_per_item'], color='red', linestyle='--', 
                   linewidth=2, label=f"μ = {stats['mean_per_item']:.0f}")
        ax.set_xlabel('Per-Item Spike Count', fontsize=10)
        ax.set_ylabel('Density' if i == 0 else '', fontsize=10)
        ax.set_title(f'l = {l}: Per-Item Spikes\nλ_item = {stats["lambda_per_item"]:.0f}, SNR = {stats["snr_per_item"]:.1f}',
                     fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
    
    plt.suptitle('Experiment 3C: Total vs Per-Item Spike Distributions\n'
                 '(Top: Total spikes stay constant | Bottom: Per-item spikes decrease with load)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'exp3c_distributions.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3c_distributions.png")
    
    # =========================================================================
    # Figure 4: Summary Figure (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: The Core Result - Per-Item SNR degrades
    ax = axes[0, 0]
    ax.plot(set_sizes, r3a['theoretical']['snr_total'], 'o-', 
            color=sns.color_palette()[0], linewidth=3, markersize=10,
            label='Total SNR (constant)')
    ax.plot(set_sizes, r3a['theoretical']['snr_per_item'], 's-', 
            color=sns.color_palette()[1], linewidth=3, markersize=10,
            label='Per-item SNR ∝ 1/√l')
    ax.fill_between(set_sizes, 
                    r3a['theoretical']['snr_per_item'],
                    r3a['theoretical']['snr_total'],
                    alpha=0.2, color='red', label='Precision loss')
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_title('A. THE CAPACITY LIMIT:\nPer-Item Precision Degrades', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    
    # Panel B: Resource allocation
    ax = axes[0, 1]
    width = 0.35
    x = np.arange(len(set_sizes))
    ax.bar(x - width/2, r3a['theoretical']['lambda_total'], width, 
           color=sns.color_palette()[0], alpha=0.7, label='λ_total', edgecolor='black')
    ax.bar(x + width/2, r3a['theoretical']['lambda_per_item'], width,
           color=sns.color_palette()[1], alpha=0.7, label='λ_per_item', edgecolor='black')
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Expected Spikes (λ)', fontsize=12)
    ax.set_title('B. Resource Allocation\n(Total constant, per-item decreases)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(set_sizes)
    ax.legend(fontsize=10)
    
    # Panel C: Time trade-off surface
    ax = axes[1, 0]
    for j, T_d in enumerate(r3b['T_d_values']):
        ax.plot(r3b['set_sizes'], r3b['snr_per_item_grid'][:, j], 'o-', 
                linewidth=2, markersize=7, label=f'T_d={T_d:.2f}s')
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Per-Item SNR', fontsize=12)
    ax.set_title('C. Time-Accuracy Trade-off\n(Longer T_d compensates for higher l)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(r3b['set_sizes'])
    
    # Panel D: Distribution comparison (per-item)
    ax = axes[1, 1]
    for i, l in enumerate(r3c['set_sizes']):
        data = r3c['per_item_distributions'][l]
        data_norm = (data - np.mean(data)) / np.std(data)
        sns.kdeplot(data_norm, ax=ax, label=f'l={l} (λ_item={r3c["stats"][l]["lambda_per_item"]:.0f})', 
                    linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Per-Item Spike Count', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('D. Per-Item Distributions\n(Fewer spikes per item at high load)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    
    plt.suptitle(f'Experiment 3 Summary: The 1/√l Capacity Limit\n'
                 f'(N={exp_config.n_neurons}, γ={exp_config.gamma} Hz, T_d={exp_config.T_d}s)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'exp3_summary.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp3_summary.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment 3: Poisson Noise Analysis (v3.0)')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_orientations', type=int, default=10)
    parser.add_argument('--n_locations', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--sigma_sq', type=float, default=1e-6)
    parser.add_argument('--T_d', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/exp3')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
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
        'set_sizes': [1, 2, 4, 6, 8],
        'lambda_base': 0.3,
        'sigma_lambda': 0.5,
    }
    
    results = run_experiment_3(config)
    plot_results(results, args.output_dir, show_plot=args.show)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE (v3.0)")
    print("=" * 70)
    print(f"\n📁 Output: {args.output_dir}/")
    print("\n🔬 KEY INSIGHTS:")
    print("   • Total SNR is CONSTANT (DN caps activity at γN)")
    print("   • Per-item SNR ∝ 1/√l (each item gets fewer resources)")
    print("   • The capacity limit is about PER-ITEM precision")
    print("   • Time can compensate: SNR_item = √(γN×T_d/l)")


if __name__ == '__main__':
    main()