"""
Experiment 2: Post-Normalized Response Analysis (Divisive Normalization)

This experiment analyzes the POST-NORMALIZED neural responses
after applying Divisive Normalization (DN) with GLOBAL normalization.

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from itertools import combinations
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# GP GENERATION
# ============================================================================

def generate_neuron_gp_samples(
    n_orientations: int,
    total_locations: int,
    theta_lengthscale: float,
    lengthscale_variability: float,
    random_state: np.random.RandomState
) -> Dict:
    """Generate GP samples for a single neuron."""
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


# ============================================================================
# DIVISIVE NORMALIZATION (SELF-CONTAINED)
# ============================================================================

def compute_global_dn_denominator(f_samples: np.ndarray, sigma_sq: float = 1e-6) -> float:
    """
    Compute GLOBAL DN denominator: Σ_{j=1}^{L} ḡ_j + σ²
    where ḡ_j = mean over θ of exp(f_j(θ))
    """
    g_all = np.exp(f_samples)
    g_bar = np.mean(g_all, axis=1)
    return np.sum(g_bar) + sigma_sq


def compute_normalized_responses_global_dn(
    f_samples: np.ndarray,
    subset_sizes: List[int],
    gamma: float = 100.0,
    sigma_sq: float = 1e-6,
    verbose: bool = False
) -> Dict:
    """Compute pre and post-normalized responses with GLOBAL DN."""
    total_locations, n_theta = f_samples.shape
    results = {'pre_norm': {}, 'post_norm': {}}
    
    # GLOBAL denominator (computed once)
    denominator_global = compute_global_dn_denominator(f_samples, sigma_sq)
    
    if verbose:
        g_all = np.exp(f_samples)
        g_bar = np.mean(g_all, axis=1)
        print(f"\n  GLOBAL DN DENOMINATOR:")
        print(f"    ḡ per location: [{', '.join([f'{v:.3f}' for v in g_bar])}]")
        print(f"    Denominator = {denominator_global:.4f}")
    
    for l in subset_sizes:
        subsets = list(combinations(range(total_locations), l))
        pre_means, post_means = [], []
        
        iterator = tqdm(subsets, desc=f"    l={l}", leave=False) if verbose else subsets
        
        for subset in iterator:
            f_subset = [f_samples[loc, :] for loc in subset]
            
            G = np.zeros([n_theta] * l)
            for dim_idx, f_loc in enumerate(f_subset):
                shape = [1] * l
                shape[dim_idx] = n_theta
                G = G + f_loc.reshape(shape)
            
            R_pre = np.exp(G)
            pre_means.append(np.mean(R_pre))
            
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
    """Run Experiment 2: Post-Normalized Response with GLOBAL DN."""
    master_rng = np.random.RandomState(seed)
    
    if verbose:
        print("\n" + "="*70)
        print("  EXPERIMENT 2: POST-NORMALIZED RESPONSE (GLOBAL DN)")
        print("="*70)
        print(f"\n  📊 Configuration:")
        print(f"     n_neurons:       {n_neurons}")
        print(f"     γ (gain):        {gamma} Hz")
        print(f"     seed:            {seed}")
    
    all_neuron_data = []
    population_pre = {l: [] for l in subset_sizes}
    population_post = {l: [] for l in subset_sizes}
    
    start_time = time.time()
    
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  GENERATING {n_neurons} NEURON(S)")
        print(f"  {'─'*60}")
    
    neuron_iter = tqdm(range(n_neurons), desc="  Neurons") if (n_neurons > 1 and verbose) else range(n_neurons)
    
    for neuron_idx in neuron_iter:
        neuron_rng = np.random.RandomState(master_rng.randint(0, 2**31))
        
        neuron_data = generate_neuron_gp_samples(
            n_orientations, total_locations, theta_lengthscale,
            lengthscale_variability, neuron_rng
        )
        
        responses = compute_normalized_responses_global_dn(
            neuron_data['f_samples'], subset_sizes, gamma, sigma_sq,
            verbose=(n_neurons == 1 and verbose)
        )
        
        all_neuron_data.append({
            'neuron_idx': neuron_idx,
            'lengthscale_vector': neuron_data['lengthscale_vector'],
            'f_samples': neuron_data['f_samples'],
            'responses': responses
        })
        
        for l in subset_sizes:
            population_pre[l].append(responses['pre_norm'][l]['R_mean'])
            population_post[l].append(responses['post_norm'][l]['R_mean'])
    
    elapsed = time.time() - start_time
    
    results = {
        'experiment': 'post_normalized_global_dn',
        'n_neurons': n_neurons,
        'gamma': gamma,
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
        'timing': {'total_seconds': elapsed}
    }
    
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  RESULTS")
        print(f"  {'─'*60}")
        print(f"\n  {'l':<5} {'Pre-DN':<18} {'Post-DN':<18}")
        print("  " + "-"*45)
    
    for l in subset_sizes:
        pre_vals = np.array(population_pre[l])
        post_vals = np.array(population_post[l])
        
        pre_summary = pre_vals[0] if n_neurons == 1 else np.mean(pre_vals)
        post_summary = post_vals[0] if n_neurons == 1 else np.mean(post_vals)
        
        results['population_summary']['pre_norm'][l] = {'R_mean': pre_summary, 'all_values': pre_vals}
        results['population_summary']['post_norm'][l] = {'R_mean': post_summary, 'all_values': post_vals}
        
        if verbose:
            print(f"  {l:<5} {pre_summary:<18.4e} {post_summary:<18.4e}")
    
    if verbose:
        print(f"\n  ⏱️  Time: {elapsed:.2f}s")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_experiment2(results: Dict, save_dir: str = 'figures/exp2_post_norm', show_plot: bool = True):
    """Create plots for Experiment 2."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_neurons = results['n_neurons']
    pop = results['population_summary']
    subset_sizes = results['config']['subset_sizes']
    gamma = results['gamma']
    
    pre_means = [pop['pre_norm'][l]['R_mean'] for l in subset_sizes]
    post_means = [pop['post_norm'][l]['R_mean'] for l in subset_sizes]
    
    print(f"\n  {'='*60}")
    print(f"  CREATING PLOTS")
    print(f"  {'='*60}")
    
    sns.set_style("whitegrid")
    neuron_str = "1 neuron" if n_neurons == 1 else f"{n_neurons} neurons (avg)"
    
    # Plot 1: Comparison (log scale)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale('log')
    ax.plot(subset_sizes, pre_means, 'o-', lw=2.5, ms=10, color='#E74C3C', label='Pre-DN')
    ax.plot(subset_sizes, post_means, 's-', lw=2.5, ms=10, color='#2E86AB', label=f'Post-DN (γ={gamma})')
    ax.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R.mean (log scale)', fontsize=14, fontweight='bold')
    ax.set_title(f'Pre-DN vs Post-DN (GLOBAL DN)\n({neuron_str})', fontsize=16, fontweight='bold')
    ax.set_xticks(subset_sizes)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp2_comparison_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    if show_plot:
        plt.show()
    
    return {'comparison': fig}


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neurons', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--save_dir', type=str, default='figures/exp2_post_norm')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()
    
    results = run_experiment2(n_neurons=args.n_neurons, gamma=args.gamma, seed=args.seed)
    
    if not args.no_plot:
        plot_experiment2(results, save_dir=args.save_dir)
    
    np.save(Path(args.save_dir) / f'exp2_results_{args.n_neurons}neurons.npy', results, allow_pickle=True)