"""
Experiment 4: Maximum Likelihood Decoding (v2.0 - DRY)

=============================================================================
PURPOSE
=============================================================================

Demonstrates how ML decoding error scales with set size under DN,
completing the causal chain from neural activity to behavioral errors.

THE COMPLETE CHAIN:
    DN caps activity → Per-item rate ∝ 1/l → Spikes ∝ 1/l → SNR ∝ 1/√l
    → Fisher Information ∝ 1/l → Cramér-Rao bound ∝ l → Error std ∝ √l

KEY OUTPUTS:
    1. Decoded error distributions at each set size
    2. Error std scaling (should follow √l)
    3. Comparison to Cramér-Rao bound (theoretical minimum)
    4. Fisher Information scaling verification

=============================================================================
VERSION 2.0: DRY IMPLEMENTATION
=============================================================================

Uses core modules properly:
- core.gaussian_process: GP tuning curve generation
- core.poisson_spike: Spike generation
- core.ml_decoder: ML decoding and Fisher information
- core.divisive_normalization: DN application

Author: Mixed Selectivity Project
Date: January 2026
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

# Import from core modules (NO re-implementation)
from core.gaussian_process import generate_neuron_population
from core.poisson_spike import generate_spikes
from core.ml_decoder import (
    compute_log_likelihood,
    decode_ml,
    compute_circular_error,
    compute_circular_std,
    compute_fisher_information,
    compute_cramer_rao_bound,
    compute_tuning_curve_derivative,
)

# Optional progress bar
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
    """Configuration for Experiment 4."""
    n_neurons: int = 100
    n_orientations: int = 64
    n_locations: int = 8
    gamma: float = 100.0
    sigma_sq: float = 1e-6
    T_d: float = 0.1
    n_trials: int = 500
    set_sizes: Tuple = (1, 2, 4, 8)
    seed: int = 42
    
    # GP parameters (matching other experiments)
    lambda_base: float = 0.5
    sigma_lambda: float = 0.3
    
    @property
    def total_activity(self) -> float:
        return self.gamma * self.n_neurons


# =============================================================================
# TUNING CURVE GENERATION (using core.gaussian_process)
# =============================================================================

def generate_orientation_tuning_curves(
    cfg: Exp4Config,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GP-based tuning curves for orientation decoding.
    
    Uses core.gaussian_process.generate_neuron_population for a single location,
    then applies DN to normalize total activity to γN.
    
    Returns
    -------
    tuning_curves : np.ndarray, shape (N, n_orientations)
        Post-DN firing rates f(θ) for each neuron
    theta_values : np.ndarray, shape (n_orientations,)
        Orientation values in radians
    """
    # Generate population using core module
    # We use location 0 only (single-location decoding task)
    population = generate_neuron_population(
        n_neurons=cfg.n_neurons,
        n_orientations=cfg.n_orientations,
        n_locations=1,  # Single location for orientation decoding
        base_lengthscale=cfg.lambda_base,
        lengthscale_variability=cfg.sigma_lambda,
        seed=cfg.seed
    )
    
    # Extract tuning curves for location 0
    # f_samples shape: (n_locations, n_orientations) per neuron
    tuning_curves = np.zeros((cfg.n_neurons, cfg.n_orientations))
    for i, neuron in enumerate(population):
        f = neuron['f_samples'][0, :]  # Location 0
        tuning_curves[i, :] = np.exp(f)  # Convert log-rate to rate
    
    # Apply divisive normalization: r_post = γ * r_pre / (σ² + mean(r_pre))
    pop_mean = np.mean(tuning_curves, axis=0, keepdims=True)
    tuning_curves = cfg.gamma * tuning_curves / (cfg.sigma_sq + pop_mean)
    
    # Orientation values (must match what generate_neuron_population uses)
    theta_values = population[0]['orientations']
    
    return tuning_curves, theta_values


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment_4(config: Dict) -> Dict:
    """
    Run Experiment 4: ML Decoding.
    
    Parameters
    ----------
    config : Dict
        Configuration from run_experiments.py
        
    Returns
    -------
    results : Dict
        Complete experimental results
    """
    cfg = Exp4Config(
        n_neurons=config.get('n_neurons', 100),
        n_orientations=config.get('n_orientations', 64),
        gamma=config.get('gamma', 100.0),
        sigma_sq=config.get('sigma_sq', 1e-6),
        T_d=config.get('T_d', 0.1),
        n_trials=config.get('n_trials', 500),
        set_sizes=tuple(config.get('set_sizes', [1, 2, 4, 8])),
        seed=config.get('seed', 42),
    )
    
    print("=" * 70)
    print("EXPERIMENT 4: ML DECODING (v2.0 - DRY)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N = {cfg.n_neurons} neurons")
    print(f"  γ = {cfg.gamma} Hz/neuron → Total = {cfg.total_activity} Hz")
    print(f"  T_d = {cfg.T_d} s")
    print(f"  Trials per set size = {cfg.n_trials}")
    print(f"  Set sizes = {cfg.set_sizes}")
    print()
    
    rng = np.random.RandomState(cfg.seed)
    
    # Generate GP tuning curves (using core module)
    print("Generating GP tuning curves...")
    base_tuning, theta_values = generate_orientation_tuning_curves(cfg, rng)
    
    # Verify DN normalization
    total_act = np.sum(np.mean(base_tuning, axis=1))
    print(f"  Total activity = {total_act:.1f} Hz (expected: {cfg.total_activity})")
    
    # Compute derivatives for Fisher Information (using core module)
    tuning_derivatives = compute_tuning_curve_derivative(base_tuning, theta_values)
    
    # Storage
    decoding_results = {}
    theoretical_results = {
        'set_sizes': list(cfg.set_sizes),
        'fisher_info': [],
        'cramer_rao_var': [],
        'cramer_rao_std_deg': [],
    }
    
    print(f"\nRunning ML decoding...")
    
    for l in tqdm(cfg.set_sizes, desc="Set sizes"):
        # Scale tuning curves for set size: under DN, per-item rate = γN/l
        # This is equivalent to scaling the tuning curves by 1/l
        scaled_tuning = base_tuning / l
        scaled_derivatives = tuning_derivatives / l
        
        # Compute Fisher Information at middle of stimulus range (using core module)
        theta_idx = cfg.n_orientations // 2
        I_F = compute_fisher_information(
            scaled_tuning, scaled_derivatives, theta_idx, cfg.T_d
        )
        cr_var = compute_cramer_rao_bound(I_F)
        cr_std_deg = np.degrees(np.sqrt(cr_var))
        
        theoretical_results['fisher_info'].append(I_F)
        theoretical_results['cramer_rao_var'].append(cr_var)
        theoretical_results['cramer_rao_std_deg'].append(cr_std_deg)
        
        # Run decoding trials
        errors = np.zeros(cfg.n_trials)
        
        for trial in range(cfg.n_trials):
            # Random true stimulus
            theta_idx_true = rng.randint(0, cfg.n_orientations)
            theta_true = theta_values[theta_idx_true]
            
            # Get rates at true stimulus
            rates = scaled_tuning[:, theta_idx_true]
            
            # Generate Poisson spikes (using core module)
            spikes = generate_spikes(rates, cfg.T_d, rng)
            
            # ML decode (using core module)
            theta_est, _, _ = decode_ml(spikes, scaled_tuning, theta_values, cfg.T_d)
            
            # Circular error (using core module)
            errors[trial] = compute_circular_error(theta_true, theta_est)
        
        # Circular std (using core module)
        circ_std_rad = compute_circular_std(errors)
        circ_std_deg = np.degrees(circ_std_rad)
        
        decoding_results[l] = {
            'set_size': l,
            'errors_rad': errors,
            'errors_deg': np.degrees(errors),
            'circular_std_rad': circ_std_rad,
            'circular_std_deg': circ_std_deg,
            'mean_abs_error_deg': np.degrees(np.mean(np.abs(errors))),
            'lambda_item': cfg.gamma * cfg.n_neurons * cfg.T_d / l,
        }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: DECODING ERROR vs SET SIZE")
    print("=" * 70)
    print(f"\n{'l':<6} {'λ_item':<10} {'Empirical σ':<14} {'CR Bound σ':<14} {'Ratio':<10}")
    print("-" * 54)
    
    for i, l in enumerate(cfg.set_sizes):
        empirical = decoding_results[l]['circular_std_deg']
        theoretical = theoretical_results['cramer_rao_std_deg'][i]
        ratio = empirical / theoretical if theoretical > 0 else np.inf
        lambda_item = decoding_results[l]['lambda_item']
        print(f"{l:<6} {lambda_item:<10.0f} {empirical:<14.2f} {theoretical:<14.2f} {ratio:<10.2f}")
    
    # Verify √l scaling
    stds = [decoding_results[l]['circular_std_deg'] for l in cfg.set_sizes]
    normalized = [s / np.sqrt(l) for s, l in zip(stds, cfg.set_sizes)]
    cv = np.std(normalized) / np.mean(normalized)
    
    print(f"\n√l Scaling Check:")
    print(f"  Normalized stds: {[f'{n:.2f}' for n in normalized]}")
    print(f"  CV of normalized: {cv:.3f} (< 0.1 is good)")
    
    return {
        'config': config,
        'exp_config': cfg,
        'theta_values': theta_values,
        'base_tuning': base_tuning,
        'theoretical': theoretical_results,
        'decoding': decoding_results,
        'scaling': {
            'empirical_std': stds,
            'normalized_by_sqrt_l': normalized,
            'cv_normalized': cv,
        }
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Generate all figures for Experiment 4."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("deep")
    
    cfg = results['exp_config']
    decoding = results['decoding']
    theoretical = results['theoretical']
    set_sizes = list(cfg.set_sizes)
    
    empirical_std = [decoding[l]['circular_std_deg'] for l in set_sizes]
    theoretical_std = theoretical['cramer_rao_std_deg']
    normalized = results['scaling']['normalized_by_sqrt_l']
    fisher_info = theoretical['fisher_info']
    
    # =========================================================================
    # Figure 1: Error Scaling
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Error std vs set size
    ax = axes[0]
    ax.plot(set_sizes, empirical_std, 'o-', color=palette[0],
            lw=2.5, ms=10, label='Empirical (ML)')
    ax.plot(set_sizes, theoretical_std, 's--', color=palette[1],
            lw=2, ms=8, label='Cramér-Rao bound')
    
    # √l reference
    ref = [empirical_std[0] * np.sqrt(l / set_sizes[0]) for l in set_sizes]
    ax.plot(set_sizes, ref, ':', color='gray', lw=2, alpha=0.7, label='∝ √l')
    
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Circular Std (degrees)')
    ax.set_title('A. Decoding Error Scales as √l')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    # Panel B: Normalized
    ax = axes[1]
    ax.plot(set_sizes, normalized, 'o-', color=palette[2], lw=2.5, ms=10)
    ax.axhline(np.mean(normalized), color='red', ls='--', lw=2,
               label=f'Mean = {np.mean(normalized):.2f}°')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Error Std / √l (degrees)')
    ax.set_title(f'B. √l Scaling Verified (CV = {results["scaling"]["cv_normalized"]:.3f})')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    plt.suptitle(f'Experiment 4: ML Decoding (N={cfg.n_neurons}, γ={cfg.gamma} Hz)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_error_scaling.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_error_scaling.png")
    
    # =========================================================================
    # Figure 2: Error Distributions
    # =========================================================================
    n_sizes = len(set_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(3*n_sizes, 4))
    colors = sns.color_palette("coolwarm", n_sizes)
    
    for i, l in enumerate(set_sizes):
        ax = axes[i] if n_sizes > 1 else axes
        errors = decoding[l]['errors_deg']
        
        sns.histplot(errors, kde=True, ax=ax, color=colors[i],
                     stat='density', alpha=0.6, bins=30)
        
        std = decoding[l]['circular_std_deg']
        ax.set_xlabel('Error (degrees)')
        ax.set_ylabel('Density' if i == 0 else '')
        ax.set_title(f'l = {l}\nσ = {std:.1f}°')
        ax.set_xlim([-60, 60])
    
    plt.suptitle('Experiment 4: Error Distributions Across Set Size',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_error_distributions.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_error_distributions.png")
    
    # =========================================================================
    # Figure 3: Fisher Information
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(set_sizes, fisher_info, 'o-', color=palette[3], lw=2.5, ms=10)
    ref_I = [fisher_info[0] * set_sizes[0] / l for l in set_sizes]
    ax.plot(set_sizes, ref_I, '--', color='gray', lw=2, alpha=0.7, label='∝ 1/l')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Fisher Information')
    ax.set_title('A. Fisher Information ∝ 1/l')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    ax = axes[1]
    cr_var = theoretical['cramer_rao_var']
    ax.plot(set_sizes, cr_var, 'o-', color=palette[4], lw=2.5, ms=10)
    ref_var = [cr_var[0] * l / set_sizes[0] for l in set_sizes]
    ax.plot(set_sizes, ref_var, '--', color='gray', lw=2, alpha=0.7, label='∝ l')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Cramér-Rao Variance')
    ax.set_title('B. Minimum Variance ∝ l')
    ax.legend()
    ax.set_xticks(set_sizes)
    sns.despine(ax=ax)
    
    plt.suptitle('Experiment 4: Theoretical Bounds', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_fisher_info.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_fisher_info.png")
    
    # =========================================================================
    # Figure 4: Summary
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    ax = axes[0, 0]
    ax.plot(set_sizes, empirical_std, 'o-', lw=2.5, ms=10, label='Empirical')
    ax.plot(set_sizes, theoretical_std, 's--', lw=2, ms=7, label='CR bound')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Error Std (degrees)')
    ax.set_title('A. Error ∝ √l')
    ax.legend()
    ax.set_xticks(set_sizes)
    
    ax = axes[0, 1]
    l_low, l_high = set_sizes[0], set_sizes[-1]
    sns.kdeplot(decoding[l_low]['errors_deg'], ax=ax, lw=2, label=f'l={l_low}')
    sns.kdeplot(decoding[l_high]['errors_deg'], ax=ax, lw=2, label=f'l={l_high}')
    ax.set_xlabel('Error (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('B. Distributions Widen with Load')
    ax.legend()
    ax.set_xlim([-60, 60])
    
    ax = axes[1, 0]
    ax.plot(set_sizes, fisher_info, 'o-', color=palette[3], lw=2.5, ms=10)
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Fisher Information')
    ax.set_title('C. Information ∝ 1/l')
    ax.set_xticks(set_sizes)
    
    ax = axes[1, 1]
    efficiency = [theoretical_std[i] / empirical_std[i] for i in range(len(set_sizes))]
    ax.bar(set_sizes, efficiency, color=palette[5], alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='red', ls='--', lw=2, label='Optimal')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Efficiency (CR / Empirical)')
    ax.set_title('D. Decoder Efficiency')
    ax.set_ylim([0, 1.2])
    ax.legend()
    ax.set_xticks(set_sizes)
    
    plt.suptitle(f'Experiment 4 Summary (N={cfg.n_neurons}, T_d={cfg.T_d}s)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_summary.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_summary.png")


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_neurons': 100,
        'n_orientations': 64,
        'gamma': 100.0,
        'sigma_sq': 1e-6,
        'T_d': 0.1,
        'n_trials': 500,
        'seed': 42,
        'set_sizes': [1, 2, 4, 8],
    }
    
    results = run_experiment_4(config)
    plot_results(results, 'results/exp4', show_plot=True)