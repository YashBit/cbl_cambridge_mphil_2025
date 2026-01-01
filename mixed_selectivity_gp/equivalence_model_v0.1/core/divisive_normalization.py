"""
Experiment 2: Post-Normalized Response Analysis (Divisive Normalization)

This experiment analyzes the POST-NORMALIZED neural responses
after applying Divisive Normalization (DN) with GLOBAL normalization.

KEY OUTPUT:
- R.mean vs Set Size (with Global DN, same shape as Pre-DN but scaled)
- Comparison with pre-normalized
- Compression ratio by set size

GLOBAL DN EQUATION:
    r_S = γ · R_pre / (Σ_{j=1}^{L} ḡ_j + σ²)
    
    The denominator is computed over ALL locations, not just active ones.
    This ensures the scaling factor is constant across all set sizes.

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
# GP GENERATION (same as exp1)
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
    """
    n_theta = n_orientations
    orientations = np.linspace(-np.pi, np.pi, n_theta)
    
    random_factors = 1.0 + lengthscale_variability * random_state.randn(total_locations)
    random_factors = np.abs(random_factors)
    lengthscale_vector = theta_lengthscale * random_factors
    
    f_samples = np.zeros((total_locations, n_theta))
    
    for loc in range(total_locations):
        lengthscale = lengthscale_vector[loc]
        
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


def compute_global_dn_denominator(f_samples: np.ndarray, sigma_sq: float = 1e-6) -> float:
    """
    Compute the GLOBAL DN denominator over ALL locations.
    
    denominator = Σ_{j=1}^{L} ḡ_j + σ²
    
    where ḡ_j = mean over θ of exp(f_j(θ))
    """
    g_all = np.exp(f_samples)  # (total_locations, n_theta)
    g_bar = np.mean(g_all, axis=1)  # (total_locations,)
    return np.sum(g_bar) + sigma_sq


def compute_normalized_responses_global_dn(
    f_samples: np.ndarray,
    subset_sizes: List[int],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6,
    verbose: bool = False
) -> Dict:
    """
    Compute pre-normalized AND post-normalized responses with GLOBAL DN.
    
    KEY DIFFERENCE FROM PER-SUBSET DN:
        The denominator is computed over ALL locations once,
        then applied to all subsets. This ensures Post-DN has
        the same shape as Pre-DN (just scaled by constant).
    """
    total_locations, n_theta = f_samples.shape
    results = {'pre_norm': {}, 'post_norm': {}}
    
    # ================================================================
    # COMPUTE GLOBAL DENOMINATOR (over ALL locations)
    # ================================================================
    denominator_global = compute_global_dn_denominator(f_samples, sigma_sq)
    
    if verbose:
        g_all = np.exp(f_samples)
        g_bar = np.mean(g_all, axis=1)
        print(f"\n  GLOBAL DN DENOMINATOR:")
        print(f"    ḡ per location: [{', '.join([f'{v:.3f}' for v in g_bar])}]")
        print(f"    Σ ḡ_j = {np.sum(g_bar):.4f}")
        print(f"    Denominator (global) = {denominator_global:.4f}")
    
    # ================================================================
    # COMPUTE RESPONSES FOR EACH SUBSET
    # ================================================================
    for l in subset_sizes:
        subsets = list(combinations(range(total_locations), l))
        
        pre_means = []
        post_means = []
        
        iterator = tqdm(subsets, desc=f"    l={l}", leave=False) if verbose else subsets
        
        for subset in iterator:
            f_subset = [f_samples[loc, :] for loc in subset]
            
            # Build log-rate tensor G
            G = np.zeros([n_theta] * l)
            for dim_idx, f_loc in enumerate(f_subset):
                shape = [1] * l
                shape[dim_idx] = n_theta
                G = G + f_loc.reshape(shape)
            
            # Pre-normalized: R_pre = exp(G)
            R_pre = np.exp(G)
            pre_means.append(np.mean(R_pre))
            
            # ========================================================
            # GLOBAL DN: Use the SAME denominator for ALL subsets
            # ========================================================
            R_post = gamma * R_pre / denominator_global
            post_means.append(np.mean(R_post))
        
        results['pre_norm'][l] = {
            'all_means': np.array(pre_means),
            'R_mean': np.mean(pre_means),
            'R_std': np.std(pre_means),
            'n_subsets': len(subsets)
        }
        
        results['post_norm'][l] = {
            'all_means': np.array(post_means),
            'R_mean': np.mean(post_means),
            'R_std': np.std(post_means),
            'n_subsets': len(subsets)
        }
    
    results['denominator_global'] = denominator_global
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment2(
    n_neurons: int = 1,
    n_orientations: int = 10,
    total_locations: int = 8,
    subset_sizes: List[int] = [2, 4, 6, 8],
    theta_lengthscale: float = 0.3,
    lengthscale_variability: float = 0.5,
    gamma: float = 100.0,
    sigma_sq: float = 1e-6,
    seed: int = 22,
    verbose: bool = True
) -> Dict:
    """
    Run Experiment 2: Post-Normalized Response Analysis with GLOBAL DN.
    """
    master_rng = np.random.RandomState(seed)
    
    if verbose:
        print("\n" + "="*70)
        print("  EXPERIMENT 2: POST-NORMALIZED RESPONSE (GLOBAL DN)")
        print("="*70)
        print(f"\n  📊 Configuration:")
        print(f"     n_neurons:       {n_neurons}")
        print(f"     n_orientations:  {n_orientations}")
        print(f"     total_locations: {total_locations}")
        print(f"     subset_sizes:    {subset_sizes}")
        print(f"     λ_base:          {theta_lengthscale}")
        print(f"     σ_λ:             {lengthscale_variability}")
        print(f"     γ (gain):        {gamma} Hz")
        print(f"     σ²:              {sigma_sq}")
        print(f"     seed:            {seed}")
        print(f"\n  ⚠️  Using GLOBAL DN (denominator over ALL locations)")
    
    # Storage
    all_neuron_data = []
    population_pre = {l: [] for l in subset_sizes}
    population_post = {l: [] for l in subset_sizes}
    
    start_time = time.time()
    
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  GENERATING {n_neurons} NEURON(S) WITH GLOBAL DN")
        print(f"  {'─'*60}")
    
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
        
        if n_neurons == 1 and verbose:
            print(f"\n  STAGE 1: Location-Dependent Lengthscales")
            lv = neuron_data['lengthscale_vector']
            print(f"  λ_vector: [{', '.join([f'{v:.3f}' for v in lv])}]")
            print(f"  Range: [{lv.min():.3f}, {lv.max():.3f}], Ratio: {lv.max()/lv.min():.2f}×")
            
            print(f"\n  STAGE 2: GP Samples")
            fs = neuron_data['f_samples']
            print(f"  Shape: {fs.shape}, Range: [{fs.min():.3f}, {fs.max():.3f}]")
            
            print(f"\n  STAGE 3: Computing Pre & Post DN Responses (GLOBAL DN)")
        
        # Compute BOTH pre and post normalized with GLOBAL DN
        responses = compute_normalized_responses_global_dn(
            f_samples=neuron_data['f_samples'],
            subset_sizes=subset_sizes,
            gamma=gamma,
            sigma_sq=sigma_sq,
            verbose=(n_neurons == 1 and verbose)
        )
        
        all_neuron_data.append({
            'neuron_idx': neuron_idx,
            'lengthscale_vector': neuron_data['lengthscale_vector'],
            'f_samples': neuron_data['f_samples'],
            'responses': responses,
            'denominator_global': responses['denominator_global']
        })
        
        for l in subset_sizes:
            population_pre[l].append(responses['pre_norm'][l]['R_mean'])
            population_post[l].append(responses['post_norm'][l]['R_mean'])
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    results = {
        'experiment': 'post_normalized_global_dn',
        'n_neurons': n_neurons,
        'gamma': gamma,
        'sigma_sq': sigma_sq,
        'config': {
            'n_orientations': n_orientations,
            'total_locations': total_locations,
            'subset_sizes': subset_sizes,
            'theta_lengthscale': theta_lengthscale,
            'lengthscale_variability': lengthscale_variability,
            'seed': seed
        },
        'neuron_data': all_neuron_data,
        'population_summary': {'pre_norm': {}, 'post_norm': {}},
        'timing': {'total_seconds': elapsed, 'per_neuron': elapsed / n_neurons}
    }
    
    # Population summary
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  RESULTS: PRE vs POST NORMALIZED R.mean (GLOBAL DN)")
        print(f"  {'─'*60}")
        print(f"\n  {'l':<5} {'Pre-DN':<18} {'Post-DN':<18} {'Ratio':<12}")
        print("  " + "-"*55)
    
    for l in subset_sizes:
        pre_values = np.array(population_pre[l])
        post_values = np.array(population_post[l])
        
        if n_neurons == 1:
            pre_summary = pre_values[0]
            post_summary = post_values[0]
        else:
            pre_summary = np.mean(pre_values)
            post_summary = np.mean(post_values)
        
        ratio = post_summary / (pre_summary + 1e-10)
        
        results['population_summary']['pre_norm'][l] = {
            'R_mean': pre_summary,
            'all_values': pre_values,
            'std': np.std(pre_values) if n_neurons > 1 else 0.0
        }
        
        results['population_summary']['post_norm'][l] = {
            'R_mean': post_summary,
            'all_values': post_values,
            'std': np.std(post_values) if n_neurons > 1 else 0.0
        }
        
        if verbose:
            print(f"  {l:<5} {pre_summary:<18.4e} {post_summary:<18.4e} {ratio:<12.4f}")
    
    # Scaling analysis
    if verbose:
        print(f"\n  SCALING ANALYSIS:")
        print(f"  {'Transition':<15} {'Pre-DN':<15} {'Post-DN':<15}")
        print("  " + "-"*45)
        
        for i in range(len(subset_sizes) - 1):
            l1, l2 = subset_sizes[i], subset_sizes[i+1]
            
            pre1 = results['population_summary']['pre_norm'][l1]['R_mean']
            pre2 = results['population_summary']['pre_norm'][l2]['R_mean']
            post1 = results['population_summary']['post_norm'][l1]['R_mean']
            post2 = results['population_summary']['post_norm'][l2]['R_mean']
            
            pre_fold = pre2 / (pre1 + 1e-10)
            post_fold = post2 / (post1 + 1e-10)
            
            print(f"  l={l1}→{l2:<8} {pre_fold:<15.2f}× {post_fold:<15.2f}×")
        
        # KEY INSIGHT: With GLOBAL DN, pre_fold == post_fold
        print(f"\n  📊 KEY INSIGHT (GLOBAL DN):")
        print(f"     Pre-DN and Post-DN scaling should be IDENTICAL")
        print(f"     because denominator is constant across all set sizes!")
        
        print(f"\n  ⏱️  Time: {elapsed:.2f}s")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_experiment2(
    results: Dict,
    save_dir: str = 'figures/exp2_post_norm',
    show_plot: bool = True
) -> Dict[str, plt.Figure]:
    """
    Create plots for Experiment 2 (Post-Normalized with Global DN).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_neurons = results['n_neurons']
    population_summary = results['population_summary']
    config = results['config']
    gamma = results['gamma']
    subset_sizes = config['subset_sizes']
    
    pre_means = [population_summary['pre_norm'][l]['R_mean'] for l in subset_sizes]
    post_means = [population_summary['post_norm'][l]['R_mean'] for l in subset_sizes]
    
    figures = {}
    
    print(f"\n  {'='*60}")
    print(f"  CREATING PLOTS (GLOBAL DN)")
    print(f"  {'='*60}")
    
    sns.set_style("whitegrid")
    neuron_str = "1 neuron" if n_neurons == 1 else f"{n_neurons} neurons (avg)"
    
    # ========================================
    # PLOT 1: Post-DN (linear scale)
    # ========================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    ax1.plot(subset_sizes, post_means, 's-', linewidth=2.5, markersize=12,
             color='#2E86AB', label=f'Post-DN (γ={gamma}, Global)')
    ax1.scatter(subset_sizes, post_means, s=200, c='#1A5276',
                alpha=0.7, edgecolors='white', linewidths=2, zorder=5)
    
    for l, val in zip(subset_sizes, post_means):
        ax1.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 15),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='gray', alpha=0.9))
    
    ax1.set_xlabel('Set Size (l) - Number of Active Locations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Post-DN R.mean', fontsize=14, fontweight='bold')
    ax1.set_title(f'Post-Normalized Response vs Set Size (GLOBAL DN)\n({neuron_str}, γ={gamma} Hz)',
                  fontsize=16, fontweight='bold', pad=20)
    
    ax1.set_xticks(subset_sizes)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp2_post_norm_global_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['post_norm'] = fig1
    
    # ========================================
    # PLOT 2: Pre vs Post comparison (log scale)
    # ========================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    ax2.set_yscale('log')
    
    ax2.plot(subset_sizes, pre_means, 'o-', linewidth=2.5, markersize=10,
             color='#E74C3C', label='Pre-DN (raw)')
    ax2.plot(subset_sizes, post_means, 's-', linewidth=2.5, markersize=10,
             color='#2E86AB', label=f'Post-DN (γ={gamma}, Global)')
    
    ax2.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R.mean (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Pre-DN vs Post-DN (GLOBAL DN)\n({neuron_str})',
                  fontsize=16, fontweight='bold', pad=20)
    
    ax2.set_xticks(subset_sizes)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add annotation about parallel curves
    ax2.annotate('Curves are parallel\n(same scaling, constant offset)',
                xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp2_comparison_global_{n_neurons}neurons_log.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['comparison_log'] = fig2
    
    # ========================================
    # PLOT 3: Scaling comparison (bar chart)
    # ========================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    pre_folds = [pre_means[i+1] / (pre_means[i] + 1e-10) for i in range(len(subset_sizes)-1)]
    post_folds = [post_means[i+1] / (post_means[i] + 1e-10) for i in range(len(subset_sizes)-1)]
    
    x = np.arange(len(pre_folds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, pre_folds, width, label='Pre-DN', color='#E74C3C', alpha=0.8)
    bars2 = ax3.bar(x + width/2, post_folds, width, label='Post-DN (Global)', color='#2E86AB', alpha=0.8)
    
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Transition', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Fold Change', fontsize=14, fontweight='bold')
    ax3.set_title(f'Scaling: Pre-DN vs Post-DN (GLOBAL DN)\n({neuron_str})\nBars should be IDENTICAL',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'l={subset_sizes[i]}→{subset_sizes[i+1]}'
                        for i in range(len(subset_sizes)-1)])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp2_scaling_global_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['scaling'] = fig3
    
    # ========================================
    # PLOT 4: Distribution (population only)
    # ========================================
    if n_neurons > 1:
        fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, l in enumerate(subset_sizes):
            ax = axes[idx]
            
            post_values = population_summary['post_norm'][l]['all_values']
            
            sns.histplot(post_values, kde=True, ax=ax, color='#2E86AB',
                        edgecolor='white', linewidth=1.5, alpha=0.7)
            
            mean_val = np.mean(post_values)
            ax.axvline(mean_val, color='#1A5276', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2e}')
            
            ax.set_xlabel('Post-DN R.mean per neuron', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'l = {l}', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
        
        fig4.suptitle(f'Distribution of Post-DN R.mean\n({n_neurons} neurons, γ={gamma})',
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        filepath = Path(save_dir) / f'exp2_post_norm_{n_neurons}neurons_dist.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
        
        figures['distributions'] = fig4
    
    if show_plot:
        plt.show()
    
    print(f"\n  ✅ All plots saved to: {save_dir}/")
    
    return figures


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment 2: Post-Normalized Response (Global DN)')
    parser.add_argument('--n_neurons', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=100.0, help='DN gain constant (Hz)')
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--save_dir', type=str, default='figures/exp2_post_norm')
    parser.add_argument('--no_plot', action='store_true')
    
    args = parser.parse_args()
    
    results = run_experiment2(
        n_neurons=args.n_neurons,
        gamma=args.gamma,
        seed=args.seed,
        verbose=True
    )
    
    if not args.no_plot:
        figures = plot_experiment2(results, save_dir=args.save_dir)
    
    # Save results
    save_path = Path(args.save_dir) / f'exp2_results_global_{args.n_neurons}neurons.npy'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, results, allow_pickle=True)
    print(f"\n  💾 Results saved to: {save_path}")