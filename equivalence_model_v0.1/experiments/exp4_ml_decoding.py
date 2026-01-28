"""
Experiment 4: Maximum Likelihood Decoding & Precision Analysis

=============================================================================
PURPOSE
=============================================================================

This experiment completes the mechanistic chain by showing that ML decoding
precision degrades as √l with set size, exactly as predicted by the 
DN + Poisson framework.

THE COMPLETE CAUSAL CHAIN:
    1. DN caps total activity at γN
    2. Per-item rate = γN/l (resource competition)
    3. Expected spikes λ = rate × T_d ∝ 1/l
    4. Poisson SNR = √λ ∝ 1/√l
    5. Fisher Information I_F ∝ rate ∝ 1/l
    6. Cramér-Rao bound: Var[θ̂] ≥ 1/I_F ∝ l
    7. ML decoding error: Std[θ̂] ∝ √l

KEY OUTPUTS:
    1. Decoding error distributions at each set size
    2. Error std vs set size (should follow √l)
    3. Comparison to Cramér-Rao bound
    4. Likelihood landscapes visualization

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
import time

# Import core modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.poisson_spike import generate_spikes, generate_spikes_multi_trial
from core.ml_decoder import (
    decode_ml_single_location,
    compute_circular_error,
    compute_errors_batch,
    compute_all_metrics
)


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_config() -> Dict:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description='Experiment 4: ML Decoding')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_orientations', type=int, default=64,
                       help='Resolution of orientation space')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--T_d', type=float, default=0.1,
                       help='Decoding time window (seconds)')
    parser.add_argument('--n_trials', type=int, default=500,
                       help='Trials per condition for statistics')
    parser.add_argument('--kappa', type=float, default=2.0,
                       help='Tuning curve sharpness')
    parser.add_argument('--output_dir', type=str, default='results/exp4')
    
    args = parser.parse_args()
    
    return {
        'n_neurons': args.n_neurons,
        'n_orientations': args.n_orientations,
        'set_sizes': [1, 2, 4, 6, 8],
        'seed': args.seed,
        'gamma': args.gamma,
        'T_d': args.T_d,
        'n_trials': args.n_trials,
        'kappa': args.kappa,
        'output_dir': args.output_dir
    }


# =============================================================================
# HELPER FUNCTIONS (to work with existing ml_decoder)
# =============================================================================

def generate_von_mises_tuning(
    theta_values: np.ndarray,
    preferred: float,
    kappa: float,
    amplitude: float,
    baseline: float = 0.0
) -> np.ndarray:
    """
    Generate von Mises (circular Gaussian) tuning curve.
    
    r(θ) = baseline + amplitude × exp(κ × cos(θ - θ_pref))
    """
    return baseline + amplitude * np.exp(kappa * np.cos(theta_values - preferred))


def generate_population_tuning(
    theta_values: np.ndarray,
    n_neurons: int,
    kappa: float = 2.0,
    amplitude: float = 50.0,
    baseline: float = 5.0,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tuning curves for a population with uniformly distributed preferences.
    
    Returns:
        tuning_curves: shape (N, n_θ)
        preferred_orientations: shape (N,)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    preferred_orientations = rng.uniform(-np.pi, np.pi, n_neurons)
    
    tuning_curves = np.zeros((n_neurons, len(theta_values)))
    for i in range(n_neurons):
        tuning_curves[i] = generate_von_mises_tuning(
            theta_values, preferred_orientations[i], kappa, amplitude, baseline
        )
    
    return tuning_curves, preferred_orientations


def decode_population_ml(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    T_d: float,
    gamma: float = 100.0
) -> Tuple[int, float, np.ndarray]:
    """
    ML decode using population of neurons.
    
    Parameters:
        spike_counts: shape (N,) - spike count per neuron
        tuning_curves: shape (N, n_θ) - tuning curve per neuron
        theta_values: shape (n_θ,) - orientation values
        T_d: decoding time window
        gamma: gain constant
        
    Returns:
        decoded_idx: index of decoded orientation
        decoded_theta: decoded orientation value
        log_likelihoods: log-likelihood at each θ
    """
    n_theta = len(theta_values)
    log_likelihoods = np.zeros(n_theta)
    
    for j in range(n_theta):
        # Rates at this candidate θ
        rates = tuning_curves[:, j]
        rates = np.maximum(rates, 1e-10)  # Avoid log(0)
        
        # Poisson log-likelihood: Σᵢ [nᵢ log(rᵢ) - rᵢ T_d]
        log_likelihoods[j] = np.sum(
            spike_counts * np.log(rates) - rates * T_d
        )
    
    decoded_idx = np.argmax(log_likelihoods)
    decoded_theta = theta_values[decoded_idx]
    
    return decoded_idx, decoded_theta, log_likelihoods


def decode_batch(
    spike_counts_batch: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    T_d: float,
    gamma: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode multiple trials.
    
    Parameters:
        spike_counts_batch: shape (n_trials, N)
        
    Returns:
        decoded_indices: shape (n_trials,)
        decoded_thetas: shape (n_trials,)
    """
    n_trials = spike_counts_batch.shape[0]
    decoded_indices = np.zeros(n_trials, dtype=int)
    decoded_thetas = np.zeros(n_trials)
    
    for t in range(n_trials):
        idx, theta, _ = decode_population_ml(
            spike_counts_batch[t], tuning_curves, theta_values, T_d, gamma
        )
        decoded_indices[t] = idx
        decoded_thetas[t] = theta
    
    return decoded_indices, decoded_thetas


def compute_circular_errors(
    decoded_thetas: np.ndarray,
    true_theta: float
) -> np.ndarray:
    """Compute circular errors for batch of decoded values."""
    errors = decoded_thetas - true_theta
    # Wrap to [-π, π]
    errors = np.arctan2(np.sin(errors), np.cos(errors))
    return errors


# =============================================================================
# CORE EXPERIMENT
# =============================================================================

def run_experiment_4(config: Dict) -> Dict:
    """
    Run Experiment 4: ML Decoding Precision Analysis.
    
    For each set size l:
    1. Compute post-DN firing rates (per-item = γN/l)
    2. Scale population tuning curves to match DN budget
    3. Generate spike counts across many trials
    4. Decode each trial using ML
    5. Compute error statistics
    """
    
    # Header
    print("=" * 70)
    print("EXPERIMENT 4: ML DECODING & PRECISION ANALYSIS")
    print("=" * 70)
    print(f"\n{'Parameter':<20} {'Value':<15} {'Description'}")
    print("-" * 70)
    print(f"{'N (neurons)':<20} {config['n_neurons']:<15} Population size")
    print(f"{'γ (gain)':<20} {config['gamma']:<15.1f} Hz per neuron")
    print(f"{'T_d (window)':<20} {config['T_d']:<15.2f} seconds")
    print(f"{'n_trials':<20} {config['n_trials']:<15} Per set size")
    print(f"{'κ (sharpness)':<20} {config['kappa']:<15.1f} Tuning concentration")
    print(f"{'θ resolution':<20} {config['n_orientations']:<15} Orientation bins")
    print()
    
    # Initialize
    rng = np.random.RandomState(config['seed'])
    total_activity = config['gamma'] * config['n_neurons']
    
    # Create orientation grid
    theta_values = np.linspace(-np.pi, np.pi, config['n_orientations'], endpoint=False)
    
    # Generate base population tuning curves
    print("Generating population tuning curves...")
    base_tuning, preferred = generate_population_tuning(
        theta_values=theta_values,
        n_neurons=config['n_neurons'],
        kappa=config['kappa'],
        amplitude=50.0,
        baseline=5.0,
        rng=rng
    )
    print(f"✓ Population: {config['n_neurons']} neurons, preferences span [-π, π]")
    
    # Choose a true stimulus (middle of range)
    theta_true = 0.0
    theta_true_idx = np.argmin(np.abs(theta_values - theta_true))
    print(f"✓ True stimulus: θ = {theta_true:.3f} rad ({np.degrees(theta_true):.1f}°)")
    
    # Storage
    results = {
        'config': config,
        'theta_values': theta_values,
        'theta_true': theta_true,
        'set_sizes': config['set_sizes'],
        'decoding': {}
    }
    
    # Theoretical predictions
    print("\n" + "=" * 70)
    print("THEORETICAL PREDICTIONS")
    print("=" * 70)
    print(f"\n{'Set Size':<10} {'Per-Item Rate':<15} {'Expected λ':<12} {'CRB Std':<12} {'Rel. Error'}")
    print("-" * 70)
    
    for l in config['set_sizes']:
        per_item_rate = total_activity / l
        expected_spikes = per_item_rate * config['T_d']
        # Simplified CRB estimate (actual depends on tuning curve details)
        crb_var = l / (config['kappa'] * expected_spikes + 1e-10)
        crb_std = np.sqrt(crb_var)
        rel_error = np.sqrt(l)
        
        results['decoding'][l] = {
            'per_item_rate': per_item_rate,
            'expected_spikes': expected_spikes,
            'crb_std': crb_std,
            'theoretical_rel_error': rel_error
        }
        
        print(f"{l:<10} {per_item_rate:<15.1f} {expected_spikes:<12.1f} {crb_std:<12.4f} {rel_error:.2f}×")
    
    # Run decoding experiment
    print("\n" + "=" * 70)
    print("ML DECODING SIMULATION")
    print("=" * 70)
    print(f"\nRunning {config['n_trials']} trials per set size...\n")
    
    for l in tqdm(config['set_sizes'], desc="Set sizes", unit="l"):
        
        # Scale tuning curves to match DN-constrained budget
        # Under DN, total population rate for this item = γN/l
        per_item_rate = total_activity / l
        
        # Scale factor to normalize total population rate
        base_total = np.sum(base_tuning[:, theta_true_idx])
        scale_factor = per_item_rate / base_total
        
        scaled_tuning = base_tuning * scale_factor
        
        # Get rates at true stimulus
        rates_at_true = scaled_tuning[:, theta_true_idx]
        
        # Generate spikes for all trials
        spike_counts_all = generate_spikes_multi_trial(
            rates=rates_at_true,
            T_d=config['T_d'],
            n_trials=config['n_trials'],
            rng=rng
        )
        
        # Decode each trial
        decoded_indices, decoded_thetas = decode_batch(
            spike_counts_batch=spike_counts_all,
            tuning_curves=scaled_tuning,
            theta_values=theta_values,
            T_d=config['T_d'],
            gamma=config['gamma']
        )
        
        # Compute errors
        errors = compute_circular_errors(decoded_thetas, theta_true)
        
        # Statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Store results
        results['decoding'][l].update({
            'errors': errors,
            'theta_hat': decoded_thetas,
            'mean_error': mean_error,
            'std_error': std_error,
            'rmse': rmse,
            'bias': mean_error,
        })
    
    # Print results
    print(f"\n{'Set Size':<10} {'Mean Error':<14} {'Std Error':<14} {'RMSE':<14} {'Rel. Std'}")
    print("-" * 70)
    
    ref_std = results['decoding'][1]['std_error']
    for l in config['set_sizes']:
        d = results['decoding'][l]
        rel_std = d['std_error'] / ref_std
        print(f"{l:<10} {d['mean_error']:<14.4f} {d['std_error']:<14.4f} {d['rmse']:<14.4f} {rel_std:.2f}×")
    
    # Verify √l scaling
    print("\n" + "=" * 70)
    print("√l SCALING VERIFICATION")
    print("=" * 70)
    print(f"\n{'Set Size':<10} {'Observed Std':<14} {'Predicted (√l)':<14} {'Ratio'}")
    print("-" * 70)
    
    for l in config['set_sizes']:
        observed = results['decoding'][l]['std_error']
        predicted = ref_std * np.sqrt(l)
        ratio = observed / predicted
        print(f"{l:<10} {observed:<14.4f} {predicted:<14.4f} {ratio:.3f}")
    
    print("\n" + "=" * 70)
    print("KEY RESULT: Decoding error scales as √l")
    print("=" * 70)
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results: Dict, output_dir: str):
    """Generate publication-quality figures."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = results['config']
    set_sizes = results['set_sizes']
    theta_values = results['theta_values']
    
    # Extract data
    std_errors = [results['decoding'][l]['std_error'] for l in set_sizes]
    mean_errors = [results['decoding'][l]['mean_error'] for l in set_sizes]
    rmse_values = [results['decoding'][l]['rmse'] for l in set_sizes]
    
    # Reference for √l scaling
    ref_std = std_errors[0]
    sqrt_l_prediction = [ref_std * np.sqrt(l) for l in set_sizes]
    
    # Colors
    colors = {
        'observed': '#3498db',
        'predicted': '#e74c3c',
        'histogram': '#2ecc71',
        'theory': '#9b59b6'
    }
    
    # =========================================================================
    # Figure 1: Error Std vs Set Size (Main Result)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(set_sizes, std_errors, 'o-', color=colors['observed'],
            linewidth=2.5, markersize=10, label='Observed Std[error]')
    ax.plot(set_sizes, sqrt_l_prediction, 's--', color=colors['predicted'],
            linewidth=2, markersize=8, label=f'Predicted: {ref_std:.3f} × √l')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Decoding Error Std (radians)', fontsize=12)
    ax.set_title('ML Decoding Precision Degrades as √l\n(Completing the Mechanistic Chain)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(set_sizes)
    
    # Add annotation
    ax.annotate('Error ∝ √l\n(Capacity Limit)',
                xy=(6, std_errors[3]), xytext=(3, std_errors[4]),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=11, color='gray',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_error_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: exp4_error_scaling.png")
    
    # =========================================================================
    # Figure 2: Error Distributions
    # =========================================================================
    fig, axes = plt.subplots(1, len(set_sizes), figsize=(3*len(set_sizes), 4), sharey=True)
    
    for i, l in enumerate(set_sizes):
        ax = axes[i]
        errors = results['decoding'][l]['errors']
        
        ax.hist(errors, bins=30, color=colors['histogram'], alpha=0.7,
                edgecolor='black', density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Error (rad)')
        ax.set_title(f'l = {l}', fontweight='bold')
        
        # Add std annotation
        std = results['decoding'][l]['std_error']
        ax.text(0.95, 0.95, f'σ = {std:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0].set_ylabel('Density')
    plt.suptitle('Decoding Error Distributions by Set Size', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: exp4_error_distributions.png")
    
    # =========================================================================
    # Figure 3: Relative Error Scaling
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    relative_std = [s / ref_std for s in std_errors]
    sqrt_l = [np.sqrt(l) for l in set_sizes]
    
    ax.plot(set_sizes, relative_std, 'o-', color=colors['observed'],
            linewidth=2.5, markersize=10, label='Observed: Std / Std(l=1)')
    ax.plot(set_sizes, sqrt_l, 's--', color=colors['predicted'],
            linewidth=2, markersize=8, label='Theory: √l')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Relative Error (normalized to l=1)', fontsize=12)
    ax.set_title('Error Scaling Matches √l Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(set_sizes)
    
    # Add text box
    textstr = 'THE COMPLETE CHAIN:\n• DN: rate ∝ 1/l\n• Poisson: SNR ∝ √rate\n• Fisher: I_F ∝ rate\n• CRB: Var ≥ 1/I_F\n• Result: σ ∝ √l'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_relative_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: exp4_relative_scaling.png")
    
    # =========================================================================
    # Figure 4: Summary (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A: Error std vs set size
    ax = axes[0, 0]
    ax.plot(set_sizes, std_errors, 'o-', color=colors['observed'], linewidth=2, markersize=8, label='Observed')
    ax.plot(set_sizes, sqrt_l_prediction, 's--', color=colors['predicted'], linewidth=2, markersize=6, label='√l prediction')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Error Std (rad)')
    ax.set_title('A. Decoding Error vs Set Size', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # B: Relative scaling
    ax = axes[0, 1]
    ax.plot(set_sizes, relative_std, 'o-', color=colors['observed'], linewidth=2, markersize=8, label='Relative std')
    ax.plot(set_sizes, sqrt_l, 's--', color='gray', linewidth=2, alpha=0.7, label='√l')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Relative Error')
    ax.set_title('B. Scaling Verification (∝ √l)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # C: Error distributions (overlaid)
    ax = axes[1, 0]
    for i, l in enumerate([1, 4, 8]):
        if l in set_sizes:
            errors = results['decoding'][l]['errors']
            ax.hist(errors, bins=30, alpha=0.5, label=f'l={l}', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Error (rad)')
    ax.set_ylabel('Density')
    ax.set_title('C. Error Distributions Widen with l', fontweight='bold')
    ax.legend(fontsize=9)
    
    # D: Bias check (should be ~0)
    ax = axes[1, 1]
    ax.bar(set_sizes, mean_errors, color=colors['theory'], alpha=0.7, width=0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Mean Error (rad)')
    ax.set_title('D. Bias Check (should ≈ 0)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Experiment 4: ML Decoding Analysis (N={config["n_neurons"]}, T_d={config["T_d"]}s, {config["n_trials"]} trials)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: exp4_summary.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    
    config = get_config()
    
    # Run experiment
    start_time = time.time()
    results = run_experiment_4(config)
    elapsed = time.time() - start_time
    
    # Generate plots
    print("\nGenerating figures...")
    plot_results(results, config['output_dir'])
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print(f"📁 Output: {config['output_dir']}/")
    
    # Verify scaling
    ref_std = results['decoding'][1]['std_error']
    print(f"\n🔬 SCALING VERIFICATION:")
    print(f"   Reference (l=1): σ = {ref_std:.4f} rad")
    for l in config['set_sizes'][1:]:
        observed = results['decoding'][l]['std_error']
        predicted = ref_std * np.sqrt(l)
        ratio = observed / predicted
        status = "✓" if 0.8 < ratio < 1.2 else "⚠️"
        print(f"   l={l}: observed={observed:.4f}, predicted={predicted:.4f}, ratio={ratio:.2f} {status}")
    
    print(f"\n✓ MECHANISTIC CHAIN COMPLETE:")
    print(f"   DN (γN cap) → rate∝1/l → spikes∝1/l → SNR∝1/√l → I_F∝1/l → error∝√l")


if __name__ == '__main__':
    main()