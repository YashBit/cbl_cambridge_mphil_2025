"""
Experiment 3: Precision vs Set Size (Behavioral Predictions)

This experiment tests the core behavioral prediction of the model:

    PRECISION DECLINES WITH SET SIZE

The mechanism:
    1. More items → DN spreads activity across more locations
    2. Less activity per item → fewer spikes per item  
    3. Fewer spikes → more Poisson noise relative to signal
    4. More noise → larger decoding errors → lower precision

This is the key link between neural mechanism and behavioral limitation.

Author: Mixed Selectivity Project
Date: December 2025
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import from core modules, fall back to self-contained if not available
try:
    from core.poisson_spiking import generate_spikes, compute_dn_firing_rates
    from core.ml_decoder import decode_ml_single_location, compute_circular_error, compute_all_metrics
    USING_CORE_MODULES = True
except ImportError:
    USING_CORE_MODULES = False


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
    orientations = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    
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
# SELF-CONTAINED FUNCTIONS (fallback if core modules not available)
# ============================================================================

def _generate_spikes(firing_rates: np.ndarray, T_d: float, rng: np.random.RandomState) -> np.ndarray:
    """Generate Poisson spike counts."""
    lambda_param = np.maximum(firing_rates * T_d, 0)
    return rng.poisson(lambda_param)


def _compute_dn_firing_rates(
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    theta_indices: Tuple[int, ...],
    gamma: float,
    sigma_sq: float
) -> Tuple[np.ndarray, float]:
    """Compute post-DN firing rates for given stimulus."""
    f_at_theta = np.array([f_samples[loc, idx] for loc, idx in zip(active_locations, theta_indices)])
    g_at_theta = np.exp(f_at_theta)
    
    g_all = np.exp(f_samples[list(active_locations), :])
    g_bar = np.mean(g_all, axis=1)
    denominator = np.sum(g_bar) + sigma_sq
    
    firing_rates = gamma * g_at_theta / denominator
    return firing_rates, denominator


def _decode_ml(
    spike_counts: np.ndarray,
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    denominator: float,
    gamma: float
) -> np.ndarray:
    """ML decode each location independently."""
    n_active = len(active_locations)
    decoded = np.zeros(n_active, dtype=int)
    
    for i, loc in enumerate(active_locations):
        f_loc = f_samples[loc, :]
        r_theta = gamma * np.exp(f_loc) / denominator
        r_theta = np.maximum(r_theta, 1e-10)
        log_lik = spike_counts[i] * np.log(r_theta)
        decoded[i] = np.argmax(log_lik)
    
    return decoded


def _circular_error(true_idx: int, decoded_idx: int, n_theta: int) -> float:
    """Compute circular error in radians."""
    theta_vals = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    error = theta_vals[decoded_idx] - theta_vals[true_idx]
    return np.arctan2(np.sin(error), np.cos(error))


# ============================================================================
# TRIAL SIMULATION
# ============================================================================

def simulate_trials(
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    n_trials: int,
    gamma: float,
    sigma_sq: float,
    T_d: float,
    rng: np.random.RandomState
) -> Dict:
    """Simulate multiple trials for a given set size."""
    n_theta = f_samples.shape[1]
    n_active = len(active_locations)
    
    all_errors = []
    all_spikes = []
    all_rates = []
    
    for _ in range(n_trials):
        # Random stimulus
        true_indices = tuple(rng.randint(0, n_theta, size=n_active))
        
        # Get firing rates (use core or fallback)
        if USING_CORE_MODULES:
            rates, denom = compute_dn_firing_rates(
                f_samples, active_locations, true_indices, gamma, sigma_sq
            )
            spikes = generate_spikes(rates, T_d, rng)
        else:
            rates, denom = _compute_dn_firing_rates(
                f_samples, active_locations, true_indices, gamma, sigma_sq
            )
            spikes = _generate_spikes(rates, T_d, rng)
        
        # Decode
        decoded = _decode_ml(spikes, f_samples, active_locations, denom, gamma)
        
        # Errors
        for true_idx, dec_idx in zip(true_indices, decoded):
            all_errors.append(_circular_error(true_idx, dec_idx, n_theta))
        
        all_spikes.extend(spikes)
        all_rates.extend(rates)
    
    errors = np.array(all_errors)
    
    return {
        'set_size': n_active,
        'n_trials': n_trials,
        'errors': errors,
        'spike_counts': np.array(all_spikes),
        'firing_rates': np.array(all_rates),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean_abs_error': np.mean(np.abs(errors)),
        'precision': 1 / (np.var(errors) + 1e-10),
        'circular_variance': 1 - np.abs(np.mean(np.exp(1j * errors))),
        'mean_spikes': np.mean(all_spikes),
        'mean_rate': np.mean(all_rates)
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment3(
    n_neurons: int = 10,
    n_orientations: int = 36,
    total_locations: int = 8,
    subset_sizes: List[int] = [2, 4, 6, 8],  # Fixed to standard set sizes
    n_trials_per_size: int = 200,
    theta_lengthscale: float = 0.3,
    lengthscale_variability: float = 0.5,
    gamma: float = 1000.0,  # Increased from 100 to get more spikes
    sigma_sq: float = 1e-6,
    T_d: float = 0.25,  # Increased from 0.1 to 250ms for more spikes
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run Experiment 3: Precision vs Set Size.
    
    Tests the key behavioral prediction that precision declines with set size.
    """
    master_rng = np.random.RandomState(seed)
    
    if verbose:
        print("\n" + "="*70)
        print("  EXPERIMENT 3: PRECISION VS SET SIZE")
        print("="*70)
        print(f"\n  📊 Configuration:")
        print(f"     n_neurons:        {n_neurons}")
        print(f"     n_orientations:   {n_orientations}")
        print(f"     subset_sizes:     {subset_sizes}")
        print(f"     n_trials/size:    {n_trials_per_size}")
        print(f"     γ (gain):         {gamma} Hz")
        print(f"     T_d (window):     {T_d*1000:.0f} ms")
        print(f"     seed:             {seed}")
    
    start_time = time.time()
    
    # Storage
    all_results = {l: [] for l in subset_sizes}
    
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  SIMULATING {n_neurons} NEURON(S)")
        print(f"  {'─'*60}")
    
    neuron_iter = tqdm(range(n_neurons), desc="  Neurons") if verbose else range(n_neurons)
    
    for neuron_idx in neuron_iter:
        neuron_rng = np.random.RandomState(master_rng.randint(0, 2**31))
        
        # Generate GP
        neuron_data = generate_neuron_gp_samples(
            n_orientations, total_locations, theta_lengthscale,
            lengthscale_variability, neuron_rng
        )
        f_samples = neuron_data['f_samples']
        
        # Test each set size
        for l in subset_sizes:
            active_locations = tuple(range(min(l, total_locations)))
            
            trial_rng = np.random.RandomState(neuron_rng.randint(0, 2**31))
            
            result = simulate_trials(
                f_samples, active_locations, n_trials_per_size,
                gamma, sigma_sq, T_d, trial_rng
            )
            
            all_results[l].append(result)
    
    elapsed = time.time() - start_time
    
    # Aggregate across neurons
    results = {
        'experiment': 'precision_vs_set_size',
        'n_neurons': n_neurons,
        'config': {
            'n_orientations': n_orientations,
            'total_locations': total_locations,
            'subset_sizes': subset_sizes,
            'n_trials_per_size': n_trials_per_size,
            'gamma': gamma,
            'T_d': T_d,
            'seed': seed
        },
        'by_set_size': {},
        'timing': {'total_seconds': elapsed}
    }
    
    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  RESULTS: PRECISION VS SET SIZE")
        print(f"  {'─'*60}")
        print(f"\n  {'l':<5} {'RMSE (rad)':<12} {'Precision':<12} {'Mean Spikes':<12} {'Mean Rate':<12}")
        print(f"  {'-'*55}")
    
    for l in subset_sizes:
        neuron_results = all_results[l]
        
        # Aggregate errors across all neurons
        all_errors = np.concatenate([r['errors'] for r in neuron_results])
        all_spikes = np.concatenate([r['spike_counts'] for r in neuron_results])
        all_rates = np.concatenate([r['firing_rates'] for r in neuron_results])
        
        rmse = np.sqrt(np.mean(all_errors**2))
        precision = 1 / (np.var(all_errors) + 1e-10)
        mean_spikes = np.mean(all_spikes)
        mean_rate = np.mean(all_rates)
        
        results['by_set_size'][l] = {
            'rmse': rmse,
            'precision': precision,
            'mean_spikes': mean_spikes,
            'mean_rate': mean_rate,
            'all_errors': all_errors,
            'circular_variance': 1 - np.abs(np.mean(np.exp(1j * all_errors)))
        }
        
        if verbose:
            print(f"  {l:<5} {rmse:<12.4f} {precision:<12.2f} {mean_spikes:<12.2f} {mean_rate:<12.2f}")
    
    # Fit power law
    log_l = np.log(subset_sizes)
    precision_vals = [results['by_set_size'][l]['precision'] for l in subset_sizes]
    log_prec = np.log(np.array(precision_vals) + 1e-10)
    
    slope, intercept = np.polyfit(log_l, log_prec, 1)
    
    results['scaling'] = {
        'precision_exponent': slope,
        'precision_values': precision_vals,
        'rmse_values': [results['by_set_size'][l]['rmse'] for l in subset_sizes],
        'spike_values': [results['by_set_size'][l]['mean_spikes'] for l in subset_sizes]
    }
    
    if verbose:
        print(f"\n  📊 SCALING ANALYSIS:")
        print(f"     Precision ∝ l^{slope:.2f}")
        if slope < -0.5:
            print(f"     ✓ Strong precision decline with set size")
        elif slope < 0:
            print(f"     ✓ Precision declines with set size")
        else:
            print(f"     ⚠ Unexpected: precision doesn't decline")
        
        print(f"\n  ⏱️  Time: {elapsed:.2f}s")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_experiment3(
    results: Dict,
    save_dir: str = 'figures/exp3_precision',
    show_plot: bool = True
) -> Dict[str, plt.Figure]:
    """Create plots for Experiment 3."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    subset_sizes = results['config']['subset_sizes']
    n_neurons = results['n_neurons']
    gamma = results['config']['gamma']
    T_d = results['config']['T_d']
    
    precision_vals = results['scaling']['precision_values']
    rmse_vals = results['scaling']['rmse_values']
    spike_vals = results['scaling']['spike_values']
    exponent = results['scaling']['precision_exponent']
    
    figures = {}
    
    print(f"\n  {'='*60}")
    print(f"  CREATING PLOTS")
    print(f"  {'='*60}")
    
    sns.set_style("whitegrid")
    
    # ========================================
    # PLOT 1: Precision vs Set Size (log-log)
    # ========================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax1.plot(subset_sizes, precision_vals, 'o-', lw=2.5, ms=12,
             color='#27AE60', label='Observed')
    
    # Fit line
    fit_x = np.array(subset_sizes)
    fit_y = np.exp(results['scaling']['precision_exponent'] * np.log(fit_x) + 
                   np.log(precision_vals[0]) - results['scaling']['precision_exponent'] * np.log(subset_sizes[0]))
    ax1.plot(fit_x, fit_y, '--', lw=2, color='gray', 
             label=f'Fit: l^{exponent:.2f}')
    
    ax1.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precision (1/variance)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Precision vs Set Size\n({n_neurons} neurons, γ={gamma} Hz, T={T_d*1000:.0f}ms)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3_precision_vs_setsize_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['precision'] = fig1
    
    # ========================================
    # PLOT 2: RMSE vs Set Size
    # ========================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    ax2.plot(subset_sizes, rmse_vals, 's-', lw=2.5, ms=12,
             color='#E74C3C', label='RMSE')
    
    # Convert to degrees for interpretation
    rmse_deg = [r * 180 / np.pi for r in rmse_vals]
    
    ax2_deg = ax2.twinx()
    ax2_deg.set_ylabel('RMSE (degrees)', fontsize=12, color='gray')
    ax2_deg.tick_params(axis='y', labelcolor='gray')
    ax2_deg.set_ylim([min(rmse_deg) * 0.9, max(rmse_deg) * 1.1])
    
    for l, rmse, rmse_d in zip(subset_sizes, rmse_vals, rmse_deg):
        ax2.annotate(f'{rmse_d:.1f}°', xy=(l, rmse), xytext=(5, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE (radians)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Recall Error vs Set Size\n({n_neurons} neurons)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(subset_sizes)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3_rmse_vs_setsize_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['rmse'] = fig2
    
    # ========================================
    # PLOT 3: Mean Spikes vs Set Size
    # ========================================
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    
    ax3.plot(subset_sizes, spike_vals, 'D-', lw=2.5, ms=12,
             color='#3498DB', label='Mean spike count')
    
    ax3.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Spike Count per Item', fontsize=14, fontweight='bold')
    ax3.set_title(f'Spikes per Item vs Set Size\n({n_neurons} neurons, T={T_d*1000:.0f}ms)',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(subset_sizes)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3_spikes_vs_setsize_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['spikes'] = fig3
    
    # ========================================
    # PLOT 4: Error Distribution by Set Size
    # ========================================
    n_sizes = len(subset_sizes)
    n_cols = 2
    n_rows = (n_sizes + 1) // 2
    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    
    for idx, l in enumerate(subset_sizes):
        ax = axes[idx]
        errors = results['by_set_size'][l]['all_errors']
        errors_deg = errors * 180 / np.pi
        
        sns.histplot(errors_deg, kde=True, ax=ax, color='#9B59B6',
                    edgecolor='white', alpha=0.7, bins=36)
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        rmse_deg = results['by_set_size'][l]['rmse'] * 180 / np.pi
        ax.set_title(f'l = {l} (RMSE = {rmse_deg:.1f}°)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Error (degrees)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim([-180, 180])
    
    # Hide unused subplots
    for idx in range(len(subset_sizes), len(axes)):
        axes[idx].set_visible(False)
    
    fig4.suptitle(f'Error Distributions by Set Size\n({n_neurons} neurons)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3_error_distributions_{n_neurons}neurons.png'
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
    
    parser = argparse.ArgumentParser(description='Experiment 3: Precision vs Set Size')
    parser.add_argument('--n_neurons', type=int, default=10)
    parser.add_argument('--n_orientations', type=int, default=36)
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--T_d', type=float, default=0.1, help='Decoding window in seconds')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='figures/exp3_precision')
    parser.add_argument('--no_plot', action='store_true')
    
    args = parser.parse_args()
    
    results = run_experiment3(
        n_neurons=args.n_neurons,
        n_orientations=args.n_orientations,
        n_trials_per_size=args.n_trials,
        gamma=args.gamma,
        T_d=args.T_d,
        seed=args.seed,
        verbose=True
    )
    
    if not args.no_plot:
        figures = plot_experiment3(results, save_dir=args.save_dir)
    
    # Save results
    save_path = Path(args.save_dir) / f'exp3_results_{args.n_neurons}neurons.npy'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, results, allow_pickle=True)
    print(f"\n  💾 Results saved to: {save_path}")