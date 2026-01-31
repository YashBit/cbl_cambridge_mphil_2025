"""
Experiment 3: Poisson Noise Analysis

=============================================================================
THREE CORE EXPERIMENTS
=============================================================================

This module contains three experiments that demonstrate how Poisson spiking
noise creates the capacity limit in working memory.

EXPERIMENT 3A: SNR Scaling with Set Size
----------------------------------------
THE QUESTION: How does signal-to-noise ratio change as we remember more items?

THE MECHANISM:
    1. DN caps total activity: Σᵢ rᵢ = γN (constant)
    2. Per-item rate: r_item = γN/l (decreases with load)
    3. Expected spikes: λ = r_item × T_d ∝ 1/l
    4. SNR = √λ ∝ 1/√l

THE INSIGHT: The √l degradation is mathematically inevitable given Poisson
statistics. It's not a design choice—it's a noise floor.


EXPERIMENT 3B: Time-Accuracy Trade-off
--------------------------------------
THE QUESTION: Can we trade integration time for capacity?

THE MECHANISM:
    SNR = √(γN × T_d / l)
    
    This means SNR depends on the PRODUCT (T_d / l). Therefore:
    - Doubling T_d is equivalent to halving l
    - 4 items for 200ms ≈ 2 items for 100ms (same SNR)

THE INSIGHT: "Capacity" is not fixed—it's one dimension of a three-way
trade-off between items, precision, and time.


EXPERIMENT 3C: Spike Count Distributions
----------------------------------------
THE QUESTION: What do the actual spike distributions look like?

THE MECHANISM:
    Poisson(λ) has qualitatively different shapes:
    - λ < 5:  Discrete, asymmetric, far from Gaussian
    - λ ~ 20: Bell-shaped but still discrete
    - λ > 50: Approximately Gaussian (CLT)

THE INSIGHT: At high load (low λ), we're in the "few spikes" regime where
the discrete, stochastic nature of neural communication dominates.


Author: Mixed Selectivity Project
Date: January 2026
=============================================================================
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.poisson_spike import (
    generate_spikes_multi_trial,
    compute_expected_lambda,
    compute_theoretical_snr,
    compute_theoretical_cv,
    compute_empirical_stats,
    compute_population_stats,
    create_heterogeneous_rates,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExpConfig:
    """Configuration for Experiment 3."""
    n_neurons: int = 100
    gamma: float = 100.0       # Hz per neuron (gain)
    T_d: float = 0.1           # Decoding window (seconds)
    n_trials: int = 1000       # Monte Carlo trials
    set_sizes: tuple = (1, 2, 4, 6, 8)
    T_d_values: tuple = (0.05, 0.1, 0.2, 0.4)  # For Exp 3B
    seed: int = 42
    
    @property
    def total_activity(self) -> float:
        """Total population activity (γN)."""
        return self.gamma * self.n_neurons


# =============================================================================
# EXPERIMENT 3A: SNR SCALING WITH SET SIZE
# =============================================================================

def run_exp3a_snr_scaling(config: ExpConfig) -> Dict:
    """
    Experiment 3A: Verify SNR ∝ 1/√l scaling.
    
    For each set size:
    1. Compute theoretical SNR from λ = γN×T_d/l
    2. Generate spike counts and compute empirical SNR
    3. Verify Fano factor ≈ 1 (Poisson signature)
    """
    rng = np.random.RandomState(config.seed)
    
    results = {
        'set_sizes': list(config.set_sizes),
        'theoretical': {'snr': [], 'lambda': [], 'cv': []},
        'empirical': {'snr': [], 'fano': [], 'cv': []},
    }
    
    for l in config.set_sizes:
        # Theoretical predictions
        lambda_exp = compute_expected_lambda(
            config.gamma, config.n_neurons, l, config.T_d
        )
        snr_theory = compute_theoretical_snr(lambda_exp)
        cv_theory = compute_theoretical_cv(lambda_exp)
        
        results['theoretical']['lambda'].append(lambda_exp)
        results['theoretical']['snr'].append(snr_theory)
        results['theoretical']['cv'].append(cv_theory)
        
        # Empirical verification
        per_item_rate = config.total_activity / l
        neuron_rates = create_heterogeneous_rates(
            per_item_rate, config.n_neurons, heterogeneity=2.0, rng=rng
        )
        
        spike_counts = generate_spikes_multi_trial(
            neuron_rates, config.T_d, config.n_trials, rng
        )
        
        # Sum across neurons to get total spikes per trial
        total_spikes = np.sum(spike_counts, axis=1)
        stats = compute_empirical_stats(total_spikes)
        
        results['empirical']['snr'].append(stats.snr)
        results['empirical']['fano'].append(stats.fano_factor)
        results['empirical']['cv'].append(stats.cv)
    
    return results


# =============================================================================
# EXPERIMENT 3B: TIME-ACCURACY TRADE-OFF
# =============================================================================

def run_exp3b_time_tradeoff(config: ExpConfig) -> Dict:
    """
    Experiment 3B: Demonstrate time-accuracy trade-off.
    
    SNR = √(γN × T_d / l) depends on ratio T_d/l.
    Show iso-SNR curves where different (l, T_d) pairs give same SNR.
    """
    rng = np.random.RandomState(config.seed)
    
    # Create grid of (l, T_d) combinations
    results = {
        'set_sizes': list(config.set_sizes),
        'T_d_values': list(config.T_d_values),
        'snr_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
        'lambda_grid': np.zeros((len(config.set_sizes), len(config.T_d_values))),
    }
    
    for i, l in enumerate(config.set_sizes):
        for j, T_d in enumerate(config.T_d_values):
            lambda_exp = compute_expected_lambda(
                config.gamma, config.n_neurons, l, T_d
            )
            snr = compute_theoretical_snr(lambda_exp)
            
            results['snr_grid'][i, j] = snr
            results['lambda_grid'][i, j] = lambda_exp
    
    # Find iso-SNR pairs (where l × T_d = constant)
    # Reference: l=1, T_d=0.1 gives baseline SNR
    baseline_lambda = compute_expected_lambda(
        config.gamma, config.n_neurons, 1, 0.1
    )
    results['baseline_snr'] = compute_theoretical_snr(baseline_lambda)
    results['baseline_lambda'] = baseline_lambda
    
    return results


# =============================================================================
# EXPERIMENT 3C: SPIKE COUNT DISTRIBUTIONS
# =============================================================================

def run_exp3c_spike_distributions(config: ExpConfig) -> Dict:
    """
    Experiment 3C: Visualize spike count distributions.
    
    Show how Poisson distribution shape changes with λ:
    - High λ (low load): Approximately Gaussian
    - Low λ (high load): Discrete, asymmetric
    """
    rng = np.random.RandomState(config.seed)
    
    # Select representative set sizes
    selected_sizes = [1, 4, 8]  # Low, medium, high load
    
    results = {
        'set_sizes': selected_sizes,
        'distributions': {},
        'stats': {},
    }
    
    for l in selected_sizes:
        lambda_exp = compute_expected_lambda(
            config.gamma, config.n_neurons, l, config.T_d
        )
        
        # Generate many samples
        per_item_rate = config.total_activity / l
        neuron_rates = create_heterogeneous_rates(
            per_item_rate, config.n_neurons, heterogeneity=2.0, rng=rng
        )
        
        spike_counts = generate_spikes_multi_trial(
            neuron_rates, config.T_d, config.n_trials, rng
        )
        total_spikes = np.sum(spike_counts, axis=1)
        
        results['distributions'][l] = total_spikes
        results['stats'][l] = {
            'lambda': lambda_exp,
            'mean': np.mean(total_spikes),
            'std': np.std(total_spikes),
            'snr': np.mean(total_spikes) / np.std(total_spikes),
        }
    
    return results


# =============================================================================
# COMBINED EXPERIMENT RUNNER
# =============================================================================

def run_experiment_3(config: Dict) -> Dict:
    """
    Run all three sub-experiments.
    
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
        gamma=config.get('gamma', 100.0),
        T_d=config.get('T_d', 0.1),
        n_trials=config.get('n_trials', 1000),
        set_sizes=tuple(config.get('set_sizes', [1, 2, 4, 6, 8])),
        seed=config.get('seed', 42),
    )
    
    print("=" * 70)
    print("EXPERIMENT 3: POISSON NOISE ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N = {exp_config.n_neurons} neurons")
    print(f"  γ = {exp_config.gamma} Hz/neuron")
    print(f"  T_d = {exp_config.T_d} s")
    print(f"  Total activity budget: γN = {exp_config.total_activity:.0f} Hz")
    print(f"  Trials: {exp_config.n_trials}")
    print()
    
    # Run sub-experiments
    print("Running Experiment 3A: SNR Scaling...")
    results_3a = run_exp3a_snr_scaling(exp_config)
    
    print("Running Experiment 3B: Time-Accuracy Trade-off...")
    results_3b = run_exp3b_time_tradeoff(exp_config)
    
    print("Running Experiment 3C: Spike Distributions...")
    results_3c = run_exp3c_spike_distributions(exp_config)
    
    # Print summary table for 3A
    print("\n" + "=" * 70)
    print("RESULTS: SNR SCALING (Experiment 3A)")
    print("=" * 70)
    print(f"\n{'Set Size':<10} {'λ (expected)':<15} {'SNR (theory)':<15} {'SNR (empirical)':<15} {'Fano':<10}")
    print("-" * 70)
    
    for i, l in enumerate(results_3a['set_sizes']):
        print(f"{l:<10} "
              f"{results_3a['theoretical']['lambda'][i]:<15.1f} "
              f"{results_3a['theoretical']['snr'][i]:<15.2f} "
              f"{results_3a['empirical']['snr'][i]:<15.2f} "
              f"{results_3a['empirical']['fano'][i]:<10.3f}")
    
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
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="muted")
    
    exp_config = results['exp_config']
    
    # =========================================================================
    # Figure 1: SNR Scaling (Experiment 3A) - Main Result
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    r3a = results['exp3a']
    set_sizes = r3a['set_sizes']
    
    # Panel A: SNR vs Set Size
    ax = axes[0]
    ax.plot(set_sizes, r3a['theoretical']['snr'], 'o-', 
            color=sns.color_palette()[0], linewidth=2, markersize=8, 
            label='Theory: √(γNT_d/l)')
    ax.plot(set_sizes, r3a['empirical']['snr'], 's--', 
            color=sns.color_palette()[1], linewidth=2, markersize=7,
            label='Empirical')
    
    ax.set_xlabel('Set Size (l)', fontsize=11)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
    ax.set_title('A. SNR Scaling', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    
    # Panel B: Expected Spikes (λ)
    ax = axes[1]
    bars = ax.bar(set_sizes, r3a['theoretical']['lambda'], 
                  color=sns.color_palette()[2], alpha=0.7, edgecolor='black')
    
    # Add 1/l reference line
    ref = r3a['theoretical']['lambda'][0]
    ref_line = [ref / l for l in set_sizes]
    ax.plot(set_sizes, ref_line, 'o--', color=sns.color_palette()[3], 
            linewidth=2, markersize=6, label='∝ 1/l')
    
    ax.set_xlabel('Set Size (l)', fontsize=11)
    ax.set_ylabel('Expected Spikes (λ)', fontsize=11)
    ax.set_title('B. Spike Budget per Item', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    
    # Panel C: Fano Factor Verification
    ax = axes[2]
    ax.bar(set_sizes, r3a['empirical']['fano'], 
           color=sns.color_palette()[4], alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Poisson: F=1')
    
    ax.set_xlabel('Set Size (l)', fontsize=11)
    ax.set_ylabel('Fano Factor (Var/Mean)', fontsize=11)
    ax.set_title('C. Poisson Verification', fontweight='bold', fontsize=12)
    ax.set_ylim([0.8, 1.2])
    ax.legend(fontsize=9)
    ax.set_xticks(set_sizes)
    
    plt.suptitle(f'Experiment 3A: SNR Scaling with Set Size (N={exp_config.n_neurons}, T_d={exp_config.T_d}s)',
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    r3b = results['exp3b']
    
    # Panel A: SNR Heatmap
    ax = axes[0]
    sns.heatmap(r3b['snr_grid'], 
                xticklabels=[f'{t:.2f}' for t in r3b['T_d_values']],
                yticklabels=r3b['set_sizes'],
                annot=True, fmt='.1f', cmap='viridis',
                cbar_kws={'label': 'SNR'}, ax=ax)
    ax.set_xlabel('Integration Time T_d (s)', fontsize=11)
    ax.set_ylabel('Set Size (l)', fontsize=11)
    ax.set_title('A. SNR = √(γN·T_d/l)', fontweight='bold', fontsize=12)
    
    # Panel B: Iso-SNR curves (SNR vs l for different T_d)
    ax = axes[1]
    colors = sns.color_palette("husl", len(r3b['T_d_values']))
    
    for j, T_d in enumerate(r3b['T_d_values']):
        snr_values = r3b['snr_grid'][:, j]
        ax.plot(r3b['set_sizes'], snr_values, 'o-', 
                color=colors[j], linewidth=2, markersize=7,
                label=f'T_d = {T_d:.2f}s')
    
    ax.set_xlabel('Set Size (l)', fontsize=11)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
    ax.set_title('B. Time-Accuracy Trade-off', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, title='Integration Time')
    ax.set_xticks(r3b['set_sizes'])
    
    plt.suptitle('Experiment 3B: Capacity is a Time-Accuracy Trade-off',
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
    
    fig, axes = plt.subplots(1, n_dists, figsize=(4*n_dists, 4))
    colors = sns.color_palette("coolwarm", n_dists)
    
    for i, l in enumerate(r3c['set_sizes']):
        ax = axes[i]
        data = r3c['distributions'][l]
        stats = r3c['stats'][l]
        
        # Histogram with KDE
        sns.histplot(data, kde=True, ax=ax, color=colors[i], 
                     stat='density', alpha=0.6, edgecolor='black', linewidth=0.5)
        
        # Add vertical line at mean
        ax.axvline(stats['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"μ = {stats['mean']:.0f}")
        
        ax.set_xlabel('Total Spike Count', fontsize=11)
        ax.set_ylabel('Density' if i == 0 else '', fontsize=11)
        ax.set_title(f'l = {l} items\nλ = {stats["lambda"]:.0f}, SNR = {stats["snr"]:.1f}',
                     fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
    
    plt.suptitle('Experiment 3C: Spike Count Distributions Across Load',
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
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Panel A: The Causal Chain (SNR scaling)
    ax = axes[0, 0]
    ax.plot(set_sizes, r3a['theoretical']['snr'], 'o-', 
            color=sns.color_palette()[0], linewidth=2.5, markersize=9)
    ax.plot(set_sizes, r3a['empirical']['snr'], 's--', 
            color=sns.color_palette()[1], linewidth=2, markersize=7)
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('SNR')
    ax.set_title('A. SNR ∝ 1/√l', fontweight='bold')
    ax.legend(['Theory', 'Empirical'], fontsize=9)
    ax.set_xticks(set_sizes)
    
    # Panel B: Lambda scaling
    ax = axes[0, 1]
    ax.plot(set_sizes, r3a['theoretical']['lambda'], 'o-', 
            color=sns.color_palette()[2], linewidth=2.5, markersize=9)
    ref = r3a['theoretical']['lambda'][0]
    ax.plot(set_sizes, [ref/l for l in set_sizes], '--', 
            color='gray', linewidth=2, alpha=0.7)
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Expected Spikes (λ)')
    ax.set_title('B. λ ∝ 1/l', fontweight='bold')
    ax.legend(['Observed', '1/l reference'], fontsize=9)
    ax.set_xticks(set_sizes)
    
    # Panel C: Time trade-off
    ax = axes[1, 0]
    for j, T_d in enumerate(r3b['T_d_values'][:3]):  # Show 3 curves
        ax.plot(r3b['set_sizes'], r3b['snr_grid'][:, j], 'o-', 
                linewidth=2, markersize=7, label=f'T_d={T_d:.2f}s')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('SNR')
    ax.set_title('C. Time-Accuracy Trade-off', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(r3b['set_sizes'])
    
    # Panel D: Distribution comparison
    ax = axes[1, 1]
    for i, l in enumerate(r3c['set_sizes']):
        data = r3c['distributions'][l]
        # Normalize for comparison
        data_norm = (data - np.mean(data)) / np.std(data)
        sns.kdeplot(data_norm, ax=ax, label=f'l={l}', linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Spike Count')
    ax.set_ylabel('Density')
    ax.set_title('D. Distribution Shape vs Load', fontweight='bold')
    ax.legend(fontsize=9)
    
    plt.suptitle(f'Experiment 3 Summary: Poisson Noise Creates the Capacity Limit\n'
                 f'(N={exp_config.n_neurons}, γ={exp_config.gamma} Hz, T_d={exp_config.T_d}s)',
                 fontsize=13, fontweight='bold', y=1.02)
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
    
    parser = argparse.ArgumentParser(description='Experiment 3: Poisson Noise Analysis')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--T_d', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/exp3')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    config = {
        'n_neurons': args.n_neurons,
        'gamma': args.gamma,
        'T_d': args.T_d,
        'n_trials': args.n_trials,
        'seed': args.seed,
        'set_sizes': [1, 2, 4, 6, 8],
    }
    
    results = run_experiment_3(config)
    plot_results(results, args.output_dir, show_plot=args.show)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output: {args.output_dir}/")
    print("\n🔬 KEY INSIGHTS:")
    print("   3A: SNR ∝ 1/√l — The fundamental capacity limit")
    print("   3B: SNR = √(γN·T_d/l) — Capacity is a time-accuracy trade-off")
    print("   3C: High load → Few spikes → Discrete, noisy distributions")
    print("\n✓ The capacity limit is a noise floor, not a design choice.")


if __name__ == '__main__':
    main()