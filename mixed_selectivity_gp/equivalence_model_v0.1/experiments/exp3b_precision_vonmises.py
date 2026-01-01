"""
Experiment 3b: Precision vs Set Size with Proper Tuning Curves

This version uses von Mises tuning curves (like Bays 2014) instead of
arbitrary GP samples. Each location has a preferred orientation, and
the firing rate peaks at that orientation.

The key equation for the tuning curve is:
    f(θ) = A · exp(κ · cos(θ - θ_pref))

Where:
    - θ_pref is the preferred orientation for this location
    - κ is the concentration parameter (higher = sharper tuning)
    - A is the amplitude

This ensures the ML decoder has meaningful information to decode from.

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


# ============================================================================
# TUNING CURVE GENERATION (von Mises style)
# ============================================================================

def generate_tuning_curves(
    n_orientations: int,
    total_locations: int,
    kappa: float = 2.0,  # Concentration parameter (higher = sharper)
    amplitude_mean: float = 1.0,
    amplitude_std: float = 0.2,
    random_state: np.random.RandomState = None
) -> Dict:
    """
    Generate von Mises-style tuning curves for each location.
    
    Each location gets a preferred orientation, and the tuning curve
    peaks at that orientation.
    
    Parameters:
        n_orientations: Number of orientation bins
        total_locations: Number of spatial locations
        kappa: Concentration parameter (2-4 is typical, higher = sharper)
        amplitude_mean: Mean amplitude of tuning curves
        amplitude_std: Std of amplitude variation
        random_state: For reproducibility
    
    Returns:
        Dictionary with tuning curve parameters and f_samples
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    n_theta = n_orientations
    orientations = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    
    # Each location has a random preferred orientation
    preferred_orientations = random_state.uniform(-np.pi, np.pi, total_locations)
    
    # Each location has slightly different amplitude and sharpness
    amplitudes = amplitude_mean + amplitude_std * random_state.randn(total_locations)
    amplitudes = np.abs(amplitudes)  # Ensure positive
    
    # Allow some variation in kappa across locations
    kappas = kappa * (1 + 0.2 * random_state.randn(total_locations))
    kappas = np.maximum(kappas, 0.5)  # Ensure positive and not too flat
    
    # Generate tuning curves: f(θ) = A * exp(κ * cos(θ - θ_pref)) / exp(κ)
    # We normalize by exp(κ) so the max is approximately A
    f_samples = np.zeros((total_locations, n_theta))
    
    for loc in range(total_locations):
        theta_pref = preferred_orientations[loc]
        k = kappas[loc]
        A = amplitudes[loc]
        
        # von Mises-like tuning curve (log-rate)
        # f(θ) = A * (cos(θ - θ_pref) + 1) / 2  # Simple version, range [0, A]
        
        # Better version using actual von Mises shape:
        # This gives log-firing rate, so exp(f) will be the firing rate modulation
        cos_diff = np.cos(orientations - theta_pref)
        f_samples[loc, :] = A * k * (cos_diff - 1) / k  # Ranges from -2A to 0
        # Shift so it's centered around 0 with peak at A
        f_samples[loc, :] = A * cos_diff  # Simple: ranges from -A to A
    
    return {
        'preferred_orientations': preferred_orientations,
        'amplitudes': amplitudes,
        'kappas': kappas,
        'f_samples': f_samples,
        'orientations': orientations,
        'n_theta': n_theta
    }


def generate_population_tuning_curves(
    n_orientations: int,
    total_locations: int,
    n_neurons: int,
    kappa: float = 2.0,
    random_state: np.random.RandomState = None
) -> Dict:
    """
    Generate a population of neurons, each with tuning curves for all locations.
    
    In this version, different neurons have DIFFERENT preferred orientations
    at each location, creating a population code.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    n_theta = n_orientations
    orientations = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    
    # Each neuron × location combination has a preferred orientation
    # Shape: (n_neurons, total_locations)
    preferred_orientations = random_state.uniform(-np.pi, np.pi, (n_neurons, total_locations))
    
    # Amplitudes vary slightly
    amplitudes = 1.0 + 0.2 * random_state.randn(n_neurons, total_locations)
    amplitudes = np.abs(amplitudes)
    
    # Generate all tuning curves
    # Shape: (n_neurons, total_locations, n_theta)
    f_all = np.zeros((n_neurons, total_locations, n_theta))
    
    for neuron in range(n_neurons):
        for loc in range(total_locations):
            theta_pref = preferred_orientations[neuron, loc]
            A = amplitudes[neuron, loc]
            cos_diff = np.cos(orientations - theta_pref)
            f_all[neuron, loc, :] = A * cos_diff
    
    return {
        'preferred_orientations': preferred_orientations,
        'amplitudes': amplitudes,
        'f_all': f_all,
        'orientations': orientations,
        'n_theta': n_theta,
        'n_neurons': n_neurons,
        'kappa': kappa
    }


# ============================================================================
# SPIKING AND DECODING (same as before but clearer)
# ============================================================================

def compute_dn_firing_rates(
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    theta_indices: Tuple[int, ...],
    gamma: float,
    sigma_sq: float
) -> Tuple[np.ndarray, float]:
    """Compute post-DN firing rates for given stimulus."""
    # f values at the true stimulus orientations
    f_at_theta = np.array([f_samples[loc, idx] for loc, idx in zip(active_locations, theta_indices)])
    
    # Pre-normalized: g = exp(f)
    g_at_theta = np.exp(f_at_theta)
    
    # Global DN denominator over active locations
    g_all = np.exp(f_samples[list(active_locations), :])
    g_bar = np.mean(g_all, axis=1)
    denominator = np.sum(g_bar) + sigma_sq
    
    # Post-DN firing rates
    firing_rates = gamma * g_at_theta / denominator
    
    return firing_rates, denominator


def generate_spikes(firing_rates: np.ndarray, T_d: float, rng: np.random.RandomState) -> np.ndarray:
    """Generate Poisson spike counts."""
    lambda_param = np.maximum(firing_rates * T_d, 0)
    return rng.poisson(lambda_param)


def decode_ml(
    spike_counts: np.ndarray,
    f_samples: np.ndarray,
    active_locations: Tuple[int, ...],
    denominator: float,
    gamma: float
) -> np.ndarray:
    """
    ML decode each location independently.
    
    For each location, find θ that maximizes: n · log(r(θ))
    Since r(θ) = γ · exp(f(θ)) / D, this simplifies to: n · f(θ)
    """
    n_active = len(active_locations)
    decoded = np.zeros(n_active, dtype=int)
    
    for i, loc in enumerate(active_locations):
        f_loc = f_samples[loc, :]
        
        # Log-likelihood ∝ n · log(r(θ)) = n · (log(γ) + f(θ) - log(D))
        # For argmax, we just need n · f(θ)
        log_lik = spike_counts[i] * f_loc
        
        decoded[i] = np.argmax(log_lik)
    
    return decoded


def circular_error(true_idx: int, decoded_idx: int, n_theta: int) -> float:
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
        # Random stimulus orientations
        true_indices = tuple(rng.randint(0, n_theta, size=n_active))
        
        # Get firing rates with DN
        rates, denom = compute_dn_firing_rates(
            f_samples, active_locations, true_indices, gamma, sigma_sq
        )
        
        # Generate spikes
        spikes = generate_spikes(rates, T_d, rng)
        
        # Decode
        decoded = decode_ml(spikes, f_samples, active_locations, denom, gamma)
        
        # Compute errors
        for true_idx, dec_idx in zip(true_indices, decoded):
            all_errors.append(circular_error(true_idx, dec_idx, n_theta))
        
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

def run_experiment3b(
    n_neurons: int = 50,
    n_orientations: int = 72,  # 5° bins for finer resolution
    total_locations: int = 8,
    subset_sizes: List[int] = [2, 4, 6, 8],
    n_trials_per_size: int = 500,
    kappa: float = 2.0,  # Tuning curve sharpness
    gamma: float = 100.0,  # Back to original, since tuning curves now have structure
    sigma_sq: float = 1e-6,
    T_d: float = 0.1,  # 100ms window
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run Experiment 3b: Precision vs Set Size with proper tuning curves.
    """
    master_rng = np.random.RandomState(seed)
    
    if verbose:
        print("\n" + "="*70)
        print("  EXPERIMENT 3b: PRECISION VS SET SIZE (von Mises tuning)")
        print("="*70)
        print(f"\n  📊 Configuration:")
        print(f"     n_neurons:        {n_neurons}")
        print(f"     n_orientations:   {n_orientations} ({360/n_orientations:.1f}° bins)")
        print(f"     subset_sizes:     {subset_sizes}")
        print(f"     n_trials/size:    {n_trials_per_size}")
        print(f"     κ (sharpness):    {kappa}")
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
        
        # Generate von Mises tuning curves (NOT GP samples!)
        neuron_data = generate_tuning_curves(
            n_orientations=n_orientations,
            total_locations=total_locations,
            kappa=kappa,
            random_state=neuron_rng
        )
        f_samples = neuron_data['f_samples']
        
        # Test each set size
        for l in subset_sizes:
            active_locations = tuple(range(l))
            
            trial_rng = np.random.RandomState(neuron_rng.randint(0, 2**31))
            
            result = simulate_trials(
                f_samples, active_locations, n_trials_per_size,
                gamma, sigma_sq, T_d, trial_rng
            )
            
            all_results[l].append(result)
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    results = {
        'experiment': 'precision_vs_set_size_vonmises',
        'n_neurons': n_neurons,
        'config': {
            'n_orientations': n_orientations,
            'total_locations': total_locations,
            'subset_sizes': subset_sizes,
            'n_trials_per_size': n_trials_per_size,
            'kappa': kappa,
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
        print(f"\n  {'l':<5} {'RMSE (°)':<12} {'Precision':<12} {'Mean Spikes':<12} {'Mean Rate':<12}")
        print(f"  {'-'*55}")
    
    for l in subset_sizes:
        neuron_results = all_results[l]
        
        all_errors = np.concatenate([r['errors'] for r in neuron_results])
        all_spikes = np.concatenate([r['spike_counts'] for r in neuron_results])
        all_rates = np.concatenate([r['firing_rates'] for r in neuron_results])
        
        rmse = np.sqrt(np.mean(all_errors**2))
        rmse_deg = rmse * 180 / np.pi
        precision = 1 / (np.var(all_errors) + 1e-10)
        mean_spikes = np.mean(all_spikes)
        mean_rate = np.mean(all_rates)
        
        results['by_set_size'][l] = {
            'rmse': rmse,
            'rmse_deg': rmse_deg,
            'precision': precision,
            'mean_spikes': mean_spikes,
            'mean_rate': mean_rate,
            'all_errors': all_errors,
            'circular_variance': 1 - np.abs(np.mean(np.exp(1j * all_errors)))
        }
        
        if verbose:
            print(f"  {l:<5} {rmse_deg:<12.1f} {precision:<12.2f} {mean_spikes:<12.2f} {mean_rate:<12.2f}")
    
    # Fit power law
    log_l = np.log(subset_sizes)
    precision_vals = [results['by_set_size'][l]['precision'] for l in subset_sizes]
    log_prec = np.log(np.array(precision_vals) + 1e-10)
    
    slope, intercept = np.polyfit(log_l, log_prec, 1)
    
    results['scaling'] = {
        'precision_exponent': slope,
        'precision_values': precision_vals,
        'rmse_values': [results['by_set_size'][l]['rmse'] for l in subset_sizes],
        'rmse_deg_values': [results['by_set_size'][l]['rmse_deg'] for l in subset_sizes],
        'spike_values': [results['by_set_size'][l]['mean_spikes'] for l in subset_sizes]
    }
    
    if verbose:
        print(f"\n  📊 SCALING ANALYSIS:")
        print(f"     Precision ∝ l^{slope:.2f}")
        if slope < -0.5:
            print(f"     ✓ Strong precision decline (like Bays ~l^-1)")
        elif slope < 0:
            print(f"     ✓ Precision declines with set size")
        else:
            print(f"     ⚠ Unexpected: precision doesn't decline")
        
        print(f"\n  ⏱️  Time: {elapsed:.2f}s")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_experiment3b(
    results: Dict,
    save_dir: str = 'figures/exp3b_precision',
    show_plot: bool = True
) -> Dict[str, plt.Figure]:
    """Create plots for Experiment 3b."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    subset_sizes = results['config']['subset_sizes']
    n_neurons = results['n_neurons']
    gamma = results['config']['gamma']
    T_d = results['config']['T_d']
    kappa = results['config']['kappa']
    
    precision_vals = results['scaling']['precision_values']
    rmse_deg_vals = results['scaling']['rmse_deg_values']
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
    fit_y = precision_vals[0] * (fit_x / subset_sizes[0]) ** exponent
    ax1.plot(fit_x, fit_y, '--', lw=2, color='gray', 
             label=f'Fit: l^{exponent:.2f}')
    
    # Reference: Bays prediction (l^-1)
    bays_y = precision_vals[0] * (fit_x / subset_sizes[0]) ** (-1)
    ax1.plot(fit_x, bays_y, ':', lw=2, color='orange', alpha=0.7,
             label='Bays (l^-1)')
    
    ax1.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precision (1/variance)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Precision vs Set Size (von Mises tuning)\n({n_neurons} neurons, κ={kappa}, γ={gamma} Hz)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3b_precision_vs_setsize_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['precision'] = fig1
    
    # ========================================
    # PLOT 2: RMSE vs Set Size
    # ========================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    ax2.plot(subset_sizes, rmse_deg_vals, 's-', lw=2.5, ms=12, color='#E74C3C')
    
    for l, rmse_d in zip(subset_sizes, rmse_deg_vals):
        ax2.annotate(f'{rmse_d:.1f}°', xy=(l, rmse_d), xytext=(5, 10),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE (degrees)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Recall Error vs Set Size\n({n_neurons} neurons, von Mises tuning)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(subset_sizes)
    ax2.grid(True, alpha=0.3)
    
    # Add reference line for random guessing
    ax2.axhline(y=104, color='gray', linestyle=':', alpha=0.5, label='Random (104°)')
    ax2.legend()
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3b_rmse_vs_setsize_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['rmse'] = fig2
    
    # ========================================
    # PLOT 3: Mean Spikes vs Set Size
    # ========================================
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    
    ax3.plot(subset_sizes, spike_vals, 'D-', lw=2.5, ms=12, color='#3498DB')
    
    ax3.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Spike Count per Item', fontsize=14, fontweight='bold')
    ax3.set_title(f'Spikes per Item vs Set Size\n({n_neurons} neurons, T={T_d*1000:.0f}ms)',
                  fontsize=14, fontweight='bold', pad=20)
    ax3.set_xticks(subset_sizes)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3b_spikes_vs_setsize_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    figures['spikes'] = fig3
    
    # ========================================
    # PLOT 4: Error Distribution by Set Size
    # ========================================
    n_sizes = len(subset_sizes)
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, l in enumerate(subset_sizes):
        ax = axes[idx]
        errors = results['by_set_size'][l]['all_errors']
        errors_deg = errors * 180 / np.pi
        
        sns.histplot(errors_deg, kde=True, ax=ax, color='#9B59B6',
                    edgecolor='white', alpha=0.7, bins=36, stat='density')
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        rmse_deg = results['by_set_size'][l]['rmse_deg']
        ax.set_title(f'l = {l} (RMSE = {rmse_deg:.1f}°)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Error (degrees)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlim([-180, 180])
    
    fig4.suptitle(f'Error Distributions by Set Size\n({n_neurons} neurons, von Mises tuning)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    filepath = Path(save_dir) / f'exp3b_error_distributions_{n_neurons}neurons.png'
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
    
    parser = argparse.ArgumentParser(description='Experiment 3b: Precision with von Mises tuning')
    parser.add_argument('--n_neurons', type=int, default=50)
    parser.add_argument('--n_orientations', type=int, default=72)
    parser.add_argument('--n_trials', type=int, default=500)
    parser.add_argument('--kappa', type=float, default=2.0, help='Tuning curve sharpness')
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--T_d', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='figures/exp3b_precision')
    parser.add_argument('--no_plot', action='store_true')
    
    args = parser.parse_args()
    
    results = run_experiment3b(
        n_neurons=args.n_neurons,
        n_orientations=args.n_orientations,
        n_trials_per_size=args.n_trials,
        kappa=args.kappa,
        gamma=args.gamma,
        T_d=args.T_d,
        seed=args.seed,
        verbose=True
    )
    
    if not args.no_plot:
        figures = plot_experiment3b(results, save_dir=args.save_dir)
    
    # Save results
    save_path = Path(args.save_dir) / f'exp3b_results_{args.n_neurons}neurons.npy'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, results, allow_pickle=True)
    print(f"\n  💾 Results saved to: {save_path}")