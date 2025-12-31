"""
Experiment 1: Pre-Normalized Response Analysis

This experiment analyzes the RAW (pre-normalized) neural responses
from the GP-based mixed selectivity framework.

KEY OUTPUT:
- R.mean vs Set Size (shows exponential growth)
- Distribution of R.mean across neurons
- Scaling analysis (fold-change between set sizes)

The pre-normalized response is: R = exp(Σ_k f_k(θ_k))
where f_k are GP samples for each active location.

This grows EXPONENTIALLY with set size because:
- More locations → more terms in sum → larger exponent → exponential growth

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# GP GENERATION (shared with exp2)
# ============================================================================

def generate_neuron_gp_samples(
    n_orientations: int,
    total_locations: int,
    theta_lengthscale: float,
    lengthscale_variability: float,
    random_state: np.random.RandomState
) -> Dict:
    """
    Generate GP samples for a single neuron.
    
    Returns dictionary with:
        - lengthscale_vector: (total_locations,)
        - f_samples: (total_locations, n_theta)
        - orientations: (n_theta,)
    """
    n_theta = n_orientations
    orientations = np.linspace(-np.pi, np.pi, n_theta)
    
    # Location-dependent lengthscales (source of mixed selectivity)
    random_factors = 1.0 + lengthscale_variability * random_state.randn(total_locations)
    random_factors = np.abs(random_factors)
    lengthscale_vector = theta_lengthscale * random_factors
    
    # Sample GP functions
    f_samples = np.zeros((total_locations, n_theta))
    
    for loc in range(total_locations):
        lengthscale = lengthscale_vector[loc]
        
        # Build covariance matrix (periodic kernel)
        K = np.zeros((n_theta, n_theta))
        for i in range(n_theta):
            for j in range(n_theta):
                dist = np.abs(orientations[i] - orientations[j])
                dist = np.minimum(dist, 2*np.pi - dist)
                K[i, j] = np.exp(-dist**2 / (2 * lengthscale**2))
        
        K += 1e-6 * np.eye(n_theta)
        L = np.linalg.cholesky(K)
        
        z = random_state.randn(n_theta)
        f_loc = L @ z
        
        gain = 1.0 + 0.2 * random_state.randn()
        f_samples[loc, :] = f_loc * np.abs(gain)
    
    return {
        'lengthscale_vector': lengthscale_vector,
        'f_samples': f_samples,
        'orientations': orientations,
        'n_theta': n_theta
    }


def compute_pre_normalized_responses(
    f_samples: np.ndarray,
    subset_sizes: List[int],
    verbose: bool = False
) -> Dict:
    """
    Compute PRE-NORMALIZED response R = exp(G) for all subsets.
    
    G = Σ_k f_k(θ_k) is the log-rate tensor
    R = exp(G) is the pre-normalized response
    
    Returns R.mean for each set size l.
    """
    total_locations, n_theta = f_samples.shape
    results = {}
    
    for l in subset_sizes:
        subsets = list(combinations(range(total_locations), l))
        subset_means = []
        
        iterator = tqdm(subsets, desc=f"    l={l}", leave=False) if verbose else subsets
        
        for subset in iterator:
            f_subset = [f_samples[loc, :] for loc in subset]
            
            # Build log-rate tensor G
            G = np.zeros([n_theta] * l)
            for dim_idx, f_loc in enumerate(f_subset):
                shape = [1] * l
                shape[dim_idx] = n_theta
                G = G + f_loc.reshape(shape)
            
            # Pre-normalized response: R = exp(G)
            R = np.exp(G)
            subset_means.append(np.mean(R))
        
        results[l] = {
            'all_means': np.array(subset_means),
            'R_mean': np.mean(subset_means),
            'R_std': np.std(subset_means),
            'n_subsets': len(subsets)
        }
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment1(
    n_neurons: int = 1,
    n_orientations: int = 10,
    total_locations: int = 8,
    subset_sizes: List[int] = [2, 4, 6, 8],
    theta_lengthscale: float = 0.3,
    lengthscale_variability: float = 0.5,
    seed: int = 22,
    verbose: bool = True
) -> Dict:
    """
    Run Experiment 1: Pre-Normalized Response Analysis.
    
    For n_neurons=1: Single neuron analysis
    For n_neurons>1: Population average
    """
    master_rng = np.random.RandomState(seed)
    
    if verbose:
        print("\n" + "="*70)
        print("  EXPERIMENT 1: PRE-NORMALIZED RESPONSE ANALYSIS")
        print("="*70)
        print(f"\n  📊 Configuration:")
        print(f"     n_neurons:       {n_neurons}")
        print(f"     n_orientations:  {n_orientations}")
        print(f"     total_locations: {total_locations}")
        print(f"     subset_sizes:    {subset_sizes}")
        print(f"     λ_base:          {theta_lengthscale}")
        print(f"     σ_λ:             {lengthscale_variability}")
        print(f"     seed:            {seed}")
    
    # Storage
    all_neuron_data = []
    population_R_means = {l: [] for l in subset_sizes}
    
    start_time = time.time()
    
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  GENERATING {n_neurons} NEURON(S)")
        print(f"  {'─'*60}")
    
    # Progress bar for multiple neurons
    if n_neurons > 1 and verbose:
        neuron_iter = tqdm(range(n_neurons), desc="  Neurons", unit="neuron")
    else:
        neuron_iter = range(n_neurons)
    
    for neuron_idx in neuron_iter:
        neuron_seed = master_rng.randint(0, 2**31)
        neuron_rng = np.random.RandomState(neuron_seed)
        
        # Generate GP samples
        neuron_data = generate_neuron_gp_samples(
            n_orientations=n_orientations,
            total_locations=total_locations,
            theta_lengthscale=theta_lengthscale,
            lengthscale_variability=lengthscale_variability,
            random_state=neuron_rng
        )
        
        # Print single neuron details
        if n_neurons == 1 and verbose:
            print(f"\n  STAGE 1: Location-Dependent Lengthscales")
            print(f"  λ_vector: [{', '.join([f'{v:.3f}' for v in neuron_data['lengthscale_vector']])}]")
            lv = neuron_data['lengthscale_vector']
            print(f"  Range: [{lv.min():.3f}, {lv.max():.3f}], Ratio: {lv.max()/lv.min():.2f}×")
            
            print(f"\n  STAGE 2: GP Samples")
            fs = neuron_data['f_samples']
            print(f"  Shape: {fs.shape}")
            print(f"  Range: [{fs.min():.3f}, {fs.max():.3f}]")
            
            print(f"\n  STAGE 3: Pre-Normalized Responses")
        
        # Compute pre-normalized responses
        pre_norm = compute_pre_normalized_responses(
            f_samples=neuron_data['f_samples'],
            subset_sizes=subset_sizes,
            verbose=(n_neurons == 1 and verbose)
        )
        
        # Store
        all_neuron_data.append({
            'neuron_idx': neuron_idx,
            'lengthscale_vector': neuron_data['lengthscale_vector'],
            'f_samples': neuron_data['f_samples'],
            'pre_norm': pre_norm
        })
        
        for l in subset_sizes:
            population_R_means[l].append(pre_norm[l]['R_mean'])
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    results = {
        'experiment': 'pre_normalized',
        'n_neurons': n_neurons,
        'config': {
            'n_orientations': n_orientations,
            'total_locations': total_locations,
            'subset_sizes': subset_sizes,
            'theta_lengthscale': theta_lengthscale,
            'lengthscale_variability': lengthscale_variability,
            'seed': seed
        },
        'neuron_data': all_neuron_data,
        'population_summary': {},
        'timing': {'total_seconds': elapsed, 'per_neuron': elapsed / n_neurons}
    }
    
    # Population summary
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  RESULTS: PRE-NORMALIZED R.mean")
        print(f"  {'─'*60}")
        
        header = "  " + f"{'l':<5} {'R.mean':<18}"
        if n_neurons > 1:
            header += f" {'Std':<18}"
        header += f" {'# Subsets':<12}"
        print(header)
        print("  " + "-"*55)
    
    for l in subset_sizes:
        values = np.array(population_R_means[l])
        
        if n_neurons == 1:
            summary_value = values[0]
        else:
            summary_value = np.mean(values)
        
        results['population_summary'][l] = {
            'R_mean': summary_value,
            'all_values': values,
            'std': np.std(values) if n_neurons > 1 else 0.0,
            'n_subsets': len(list(combinations(range(total_locations), l)))
        }
        
        if verbose:
            line = f"  {l:<5} {summary_value:<18.4e}"
            if n_neurons > 1:
                line += f" {np.std(values):<18.4e}"
            line += f" {results['population_summary'][l]['n_subsets']:<12}"
            print(line)
    
    # Scaling analysis
    if verbose:
        print(f"\n  SCALING ANALYSIS:")
        for i in range(len(subset_sizes) - 1):
            l1, l2 = subset_sizes[i], subset_sizes[i+1]
            r1 = results['population_summary'][l1]['R_mean']
            r2 = results['population_summary'][l2]['R_mean']
            fold = r2 / (r1 + 1e-10)
            print(f"    l={l1} → l={l2}: {fold:.2f}× increase")
        
        # Overall
        r_first = results['population_summary'][subset_sizes[0]]['R_mean']
        r_last = results['population_summary'][subset_sizes[-1]]['R_mean']
        print(f"\n    Overall (l={subset_sizes[0]} → l={subset_sizes[-1]}): {r_last/r_first:.2f}× increase")
        
        print(f"\n  ⏱️  Time: {elapsed:.2f}s")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_experiment1(
    results: Dict,
    save_dir: str = 'figures/exp1_pre_norm',
    show_plot: bool = True
) -> Dict[str, plt.Figure]:
    """
    Create plots for Experiment 1 (Pre-Normalized).
    
    Plots:
    1. Set size vs R.mean (log scale) - shows exponential growth
    2. Set size vs R.mean (linear scale)
    3. Distribution across neurons (if n_neurons > 1)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_neurons = results['n_neurons']
    population_summary = results['population_summary']
    config = results['config']
    subset_sizes = config['subset_sizes']
    
    R_means = [population_summary[l]['R_mean'] for l in subset_sizes]
    
    figures = {}
    
    print(f"\n  {'='*60}")
    print(f"  CREATING PLOTS")
    print(f"  {'='*60}")
    
    sns.set_style("whitegrid")
    
    # ========================================
    # PLOT 1: Log Scale
    # ========================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    ax1.set_yscale('log')
    
    ax1.plot(subset_sizes, R_means, 'o-', linewidth=2.5, markersize=12,
             color='#E74C3C', label='Pre-Normalized R.mean')
    ax1.scatter(subset_sizes, R_means, s=200, c='#C0392B',
                alpha=0.7, edgecolors='white', linewidths=2, zorder=5)
    
    for l, val in zip(subset_sizes, R_means):
        ax1.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 15),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='gray', alpha=0.9))
    
    ax1.set_xlabel('Set Size (l) - Number of Active Locations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pre-Normalized R.mean', fontsize=14, fontweight='bold')
    
    neuron_str = "1 neuron" if n_neurons == 1 else f"{n_neurons} neurons (avg)"
    ax1.set_title(f'Pre-Normalized Response vs Set Size\n({neuron_str}, log scale)',
                  fontsize=16, fontweight='bold', pad=20)
    
    ax1.set_xticks(subset_sizes)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp1_pre_norm_{n_neurons}neurons_log.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['log_scale'] = fig1
    
    # ========================================
    # PLOT 2: Linear Scale
    # ========================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    ax2.plot(subset_sizes, R_means, 'o-', linewidth=2.5, markersize=12,
             color='#E74C3C', label='Pre-Normalized R.mean')
    ax2.scatter(subset_sizes, R_means, s=200, c='#C0392B',
                alpha=0.7, edgecolors='white', linewidths=2, zorder=5)
    
    for l, val in zip(subset_sizes, R_means):
        ax2.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 15),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='gray', alpha=0.9))
    
    ax2.set_xlabel('Set Size (l) - Number of Active Locations', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Pre-Normalized R.mean', fontsize=14, fontweight='bold')
    ax2.set_title(f'Pre-Normalized Response vs Set Size\n({neuron_str}, linear scale)',
                  fontsize=16, fontweight='bold', pad=20)
    
    ax2.set_xticks(subset_sizes)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp1_pre_norm_{n_neurons}neurons_linear.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['linear_scale'] = fig2
    
    # ========================================
    # PLOT 3: Distribution (population only)
    # ========================================
    if n_neurons > 1:
        fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, l in enumerate(subset_sizes):
            ax = axes[idx]
            all_values = population_summary[l]['all_values']
            
            sns.histplot(all_values, kde=True, ax=ax, color='#E74C3C',
                        edgecolor='white', linewidth=1.5, alpha=0.7)
            
            mean_val = np.mean(all_values)
            ax.axvline(mean_val, color='#C0392B', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2e}')
            
            ax.set_xlabel('R.mean per neuron', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            n_subsets = len(list(combinations(range(8), l)))
            ax.set_title(f'l = {l} (C(8,{l}) = {n_subsets} subsets)',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
        
        fig3.suptitle(f'Distribution of Pre-Normalized R.mean\n({n_neurons} neurons)',
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        filepath = Path(save_dir) / f'exp1_pre_norm_{n_neurons}neurons_dist.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        
        figures['distributions'] = fig3
    
    if show_plot:
        plt.show()
    
    print(f"\n  ✅ All plots saved to: {save_dir}/")
    
    return figures


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment 1: Pre-Normalized Response')
    parser.add_argument('--n_neurons', type=int, default=1)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--save_dir', type=str, default='figures/exp1_pre_norm')
    parser.add_argument('--no_plot', action='store_true')
    
    args = parser.parse_args()
    
    results = run_experiment1(
        n_neurons=args.n_neurons,
        seed=args.seed,
        verbose=True
    )
    
    if not args.no_plot:
        figures = plot_experiment1(results, save_dir=args.save_dir)
    
    # Save results
    save_path = Path(args.save_dir) / f'exp1_results_{args.n_neurons}neurons.npy'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, results, allow_pickle=True)
    print(f"\n  💾 Results saved to: {save_path}")