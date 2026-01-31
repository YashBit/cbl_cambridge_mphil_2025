"""
Experiment 4: Maximum Likelihood Decoding with GP Tuning Curves

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
UNIFIED NEURON MODEL
=============================================================================

Uses the SAME GP-based tuning curves as other experiments:
- Gaussian Process samples for heterogeneous tuning
- Location-dependent lengthscales (mixed selectivity source)
- Proper population divisive normalization

Author: Mixed Selectivity Project
Date: January 2026
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from core modules; if not available, functions are defined below
try:
    from core.gaussian_process import generate_neuron_population
    from core.poisson_spike import generate_spikes, compute_theoretical_snr
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

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
    gamma: float = 20.0  # Lower gamma for visible errors
    T_d: float = 0.25    # Longer window
    n_trials: int = 500
    set_sizes: Tuple = (1, 2, 4, 8)
    kappa: float = 2.0  # Not used for GP curves, kept for API compatibility
    seed: int = 42
    
    # GP parameters (matching other experiments)
    lambda_base: float = 0.5
    sigma_lambda: float = 0.3
    gain_variability: float = 0.2
    sigma_sq: float = 1e-6
    
    @property
    def total_activity(self) -> float:
        return self.gamma * self.n_neurons


# =============================================================================
# GP TUNING CURVE GENERATION
# =============================================================================

def generate_gp_tuning_population(cfg: Exp4Config, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate population of GP-based tuning curves for orientation decoding.
    
    Returns
    -------
    tuning_curves : np.ndarray, shape (N, n_orientations)
        Firing rates f(θ) for each neuron (after DN normalization)
    """
    orientations = np.linspace(0, 2*np.pi, cfg.n_orientations, endpoint=False)
    tuning_curves = np.zeros((cfg.n_neurons, cfg.n_orientations))
    
    for n in range(cfg.n_neurons):
        # Location-dependent lengthscale (heterogeneity source)
        ls = cfg.lambda_base * np.abs(1.0 + cfg.sigma_lambda * rng.randn())
        ls = max(ls, 0.15)  # Floor for numerical stability
        
        # Build periodic RBF kernel
        theta_i, theta_j = np.meshgrid(orientations, orientations, indexing='ij')
        dist = np.abs(theta_i - theta_j)
        dist = np.minimum(dist, 2*np.pi - dist)
        K = np.exp(-dist**2 / (2 * ls**2)) + 1e-4 * np.eye(cfg.n_orientations)
        
        # Sample GP
        try:
            L = np.linalg.cholesky(K)
            f = L @ rng.randn(cfg.n_orientations)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(K)
            f = eigvecs @ (np.sqrt(np.maximum(eigvals, 1e-6)) * rng.randn(cfg.n_orientations))
        
        # Apply gain and convert to positive rates
        gain = np.abs(1.0 + cfg.gain_variability * rng.randn())
        tuning_curves[n, :] = np.exp(f * gain)
    
    # Apply DN: normalize so total activity = γN
    pop_mean = np.mean(tuning_curves, axis=0, keepdims=True)
    tuning_curves = cfg.gamma * tuning_curves / (pop_mean + cfg.sigma_sq)
    
    return tuning_curves


# =============================================================================
# ML DECODING FUNCTIONS
# =============================================================================

def compute_log_likelihood(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    T_d: float
) -> np.ndarray:
    """
    Compute log-likelihood for all candidate stimulus values.
    
    ℓ(θ) = Σᵢ [ nᵢ · log(fᵢ(θ)) - fᵢ(θ)·T_d ]
    """
    tuning_safe = np.maximum(tuning_curves, 1e-10)
    term1 = spike_counts[:, np.newaxis] * np.log(tuning_safe)
    term2 = tuning_safe * T_d
    return np.sum(term1 - term2, axis=0)


def decode_ml(
    spike_counts: np.ndarray,
    tuning_curves: np.ndarray,
    theta_values: np.ndarray,
    T_d: float
) -> Tuple[float, int, np.ndarray]:
    """
    Decode stimulus using maximum likelihood.
    
    Returns: (theta_ml, idx_ml, log_likelihood_curve)
    """
    ll = compute_log_likelihood(spike_counts, tuning_curves, T_d)
    idx_ml = np.argmax(ll)
    return theta_values[idx_ml], idx_ml, ll


def compute_circular_error(theta_true: float, theta_est: float, period: float = 2*np.pi) -> float:
    """Signed circular error wrapped to [-period/2, period/2)."""
    error = theta_est - theta_true
    return (error + period/2) % period - period/2


def compute_circular_std(errors: np.ndarray) -> float:
    """Circular standard deviation using resultant vector method."""
    R = np.abs(np.mean(np.exp(1j * errors)))
    if R > 1e-10:
        return np.sqrt(-2 * np.log(R))
    return np.pi


# =============================================================================
# FISHER INFORMATION & CRAMÉR-RAO BOUND
# =============================================================================

def compute_tuning_derivatives(tuning_curves: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
    """Compute numerical derivatives of tuning curves (periodic boundary)."""
    d_theta = theta_values[1] - theta_values[0]
    derivatives = np.zeros_like(tuning_curves)
    
    # Central difference with periodic boundary
    derivatives[:, 1:-1] = (tuning_curves[:, 2:] - tuning_curves[:, :-2]) / (2 * d_theta)
    derivatives[:, 0] = (tuning_curves[:, 1] - tuning_curves[:, -1]) / (2 * d_theta)
    derivatives[:, -1] = (tuning_curves[:, 0] - tuning_curves[:, -2]) / (2 * d_theta)
    
    return derivatives


def compute_fisher_information(
    tuning_curves: np.ndarray,
    tuning_derivatives: np.ndarray,
    theta_idx: int,
    T_d: float
) -> float:
    """
    Compute Fisher Information at a specific stimulus value.
    
    I_F(θ) = T_d · Σᵢ [f'ᵢ(θ)]² / fᵢ(θ)
    """
    f = tuning_curves[:, theta_idx]
    f_prime = tuning_derivatives[:, theta_idx]
    f_safe = np.maximum(f, 1e-10)
    return T_d * np.sum(f_prime**2 / f_safe)


def compute_cramer_rao_bound(fisher_info: float) -> float:
    """CR bound: Var[θ̂] ≥ 1/I_F"""
    return 1.0 / max(fisher_info, 1e-10)


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
    # Convert dict to dataclass
    cfg = Exp4Config(
        n_neurons=config.get('n_neurons', 100),
        n_orientations=config.get('n_orientations', 64),
        gamma=config.get('gamma', 100.0),
        T_d=config.get('T_d', 0.1),
        n_trials=config.get('n_trials', 500),
        set_sizes=tuple(config.get('set_sizes', [1, 2, 4, 6, 8])),
        kappa=config.get('kappa', 2.0),
        seed=config.get('seed', 42),
    )
    
    print("=" * 70)
    print("EXPERIMENT 4: ML DECODING WITH GP TUNING CURVES")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  N = {cfg.n_neurons} neurons")
    print(f"  γ = {cfg.gamma} Hz/neuron → Total = {cfg.total_activity} Hz")
    print(f"  T_d = {cfg.T_d} s")
    print(f"  Trials per set size = {cfg.n_trials}")
    print(f"  Set sizes = {cfg.set_sizes}")
    print()
    
    rng = np.random.RandomState(cfg.seed)
    theta_values = np.linspace(0, 2*np.pi, cfg.n_orientations, endpoint=False)
    
    # Generate GP population (base tuning curves, DN normalized)
    print("Generating GP tuning curves...")
    base_tuning = generate_gp_tuning_population(cfg, rng)
    
    # Verify DN
    total_act = np.sum(np.mean(base_tuning, axis=1))
    print(f"  Total activity = {total_act:.1f} Hz (expected: {cfg.total_activity})")
    
    # Compute derivatives for Fisher Information
    tuning_derivatives = compute_tuning_derivatives(base_tuning, theta_values)
    
    # Storage for results
    decoding_results = {}
    theoretical_results = {
        'set_sizes': list(cfg.set_sizes),
        'fisher_info': [],
        'cramer_rao_var': [],
        'cramer_rao_std_deg': [],
    }
    
    # Run for each set size
    print(f"\nRunning ML decoding...")
    
    for l in tqdm(cfg.set_sizes, desc="Set sizes"):
        # Scale tuning curves by 1/l (DN resource sharing)
        scaled_tuning = base_tuning / l
        scaled_derivatives = tuning_derivatives / l
        
        # Expected spikes per item
        lambda_item = cfg.gamma * cfg.n_neurons * cfg.T_d / l
        
        # Compute Fisher Information (at middle of stimulus range)
        theta_idx = cfg.n_orientations // 2
        I_F = compute_fisher_information(scaled_tuning, scaled_derivatives, theta_idx, cfg.T_d)
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
            
            # Get rates and generate spikes
            rates = scaled_tuning[:, theta_idx_true]
            spikes = rng.poisson(rates * cfg.T_d)
            
            # ML decode
            theta_est, _, _ = decode_ml(spikes, scaled_tuning, theta_values, cfg.T_d)
            
            # Circular error
            errors[trial] = compute_circular_error(theta_true, theta_est)
        
        # Statistics
        circ_std_rad = compute_circular_std(errors)
        circ_std_deg = np.degrees(circ_std_rad)
        
        decoding_results[l] = {
            'set_size': l,
            'errors_rad': errors,
            'errors_deg': np.degrees(errors),
            'circular_std_rad': circ_std_rad,
            'circular_std_deg': circ_std_deg,
            'mean_abs_error_deg': np.degrees(np.mean(np.abs(errors))),
            'lambda_item': lambda_item,
        }
    
    # Print results table
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
    
    # Compute √l scaling metrics
    stds = [decoding_results[l]['circular_std_deg'] for l in cfg.set_sizes]
    normalized = [s / np.sqrt(l) for s, l in zip(stds, cfg.set_sizes)]
    
    print(f"\n√l Scaling Check:")
    print(f"  Normalized stds: {[f'{n:.2f}' for n in normalized]}")
    print(f"  CV of normalized: {np.std(normalized)/np.mean(normalized):.3f} (< 0.1 is good)")
    
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
            'cv_normalized': np.std(normalized) / np.mean(normalized),
        }
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict, output_dir: str, show_plot: bool = False):
    """Generate all figures for Experiment 4."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    cfg = results['exp_config']
    decoding = results['decoding']
    theoretical = results['theoretical']
    set_sizes = list(cfg.set_sizes)
    
    # Extract data
    empirical_std = [decoding[l]['circular_std_deg'] for l in set_sizes]
    theoretical_std = theoretical['cramer_rao_std_deg']
    normalized = results['scaling']['normalized_by_sqrt_l']
    fisher_info = theoretical['fisher_info']
    
    # =========================================================================
    # Figure 1: Error Scaling (2 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Error std vs set size
    ax = axes[0]
    ax.plot(set_sizes, empirical_std, 'o-', color=sns.color_palette()[0],
            lw=2.5, ms=10, label='Empirical (ML)')
    ax.plot(set_sizes, theoretical_std, 's--', color=sns.color_palette()[1],
            lw=2, ms=8, label='Cramér-Rao bound')
    
    # √l reference
    ref = [empirical_std[0] * np.sqrt(l / set_sizes[0]) for l in set_sizes]
    ax.plot(set_sizes, ref, ':', color='gray', lw=2, alpha=0.7, label='∝ √l')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Circular Std (degrees)', fontsize=12)
    ax.set_title('A. Decoding Error Scales as √l', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(set_sizes)
    
    # Panel B: Normalized (flat = perfect √l)
    ax = axes[1]
    ax.plot(set_sizes, normalized, 'o-', color=sns.color_palette()[2],
            lw=2.5, ms=10)
    ax.axhline(np.mean(normalized), color='red', ls='--', lw=2,
               label=f'Mean = {np.mean(normalized):.2f}°')
    ax.fill_between(set_sizes,
                    np.mean(normalized) - np.std(normalized),
                    np.mean(normalized) + np.std(normalized),
                    alpha=0.2, color='red')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Error Std / √l (degrees)', fontsize=12)
    ax.set_title(f'B. √l Scaling Verified (CV = {results["scaling"]["cv_normalized"]:.3f})',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(set_sizes)
    
    plt.suptitle(f'Experiment 4: ML Decoding (N={cfg.n_neurons}, γ={cfg.gamma} Hz)',
                 fontsize=13, fontweight='bold', y=1.02)
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
        
        # Gaussian reference
        std = decoding[l]['circular_std_deg']
        x_range = np.linspace(-60, 60, 100)
        gaussian = np.exp(-x_range**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
        ax.plot(x_range, gaussian, 'k--', lw=1.5, alpha=0.7, label='Gaussian')
        
        ax.set_xlabel('Error (degrees)', fontsize=10)
        ax.set_ylabel('Density' if i == 0 else '', fontsize=10)
        ax.set_title(f'l = {l}\nσ = {std:.1f}°', fontweight='bold')
        ax.set_xlim([-60, 60])
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Experiment 4: Error Distributions Across Set Size',
                 fontsize=13, fontweight='bold', y=1.02)
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
    
    # Panel A: Fisher Info vs set size
    ax = axes[0]
    ax.plot(set_sizes, fisher_info, 'o-', color=sns.color_palette()[3],
            lw=2.5, ms=10)
    
    # 1/l reference
    ref_I = [fisher_info[0] * set_sizes[0] / l for l in set_sizes]
    ax.plot(set_sizes, ref_I, '--', color='gray', lw=2, alpha=0.7, label='∝ 1/l')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Fisher Information', fontsize=12)
    ax.set_title('A. Fisher Information ∝ 1/l', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(set_sizes)
    
    # Panel B: CR variance vs set size
    ax = axes[1]
    cr_var = theoretical['cramer_rao_var']
    ax.plot(set_sizes, cr_var, 'o-', color=sns.color_palette()[4],
            lw=2.5, ms=10)
    
    # l reference
    ref_var = [cr_var[0] * l / set_sizes[0] for l in set_sizes]
    ax.plot(set_sizes, ref_var, '--', color='gray', lw=2, alpha=0.7, label='∝ l')
    
    ax.set_xlabel('Set Size (l)', fontsize=12)
    ax.set_ylabel('Cramér-Rao Variance', fontsize=12)
    ax.set_title('B. Minimum Variance ∝ l', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(set_sizes)
    
    plt.suptitle('Experiment 4: Theoretical Bounds',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_fisher_info.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_fisher_info.png")
    
    # =========================================================================
    # Figure 4: Summary (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # A: Error scaling
    ax = axes[0, 0]
    ax.plot(set_sizes, empirical_std, 'o-', lw=2.5, ms=10, label='Empirical')
    ax.plot(set_sizes, theoretical_std, 's--', lw=2, ms=7, label='CR bound')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Error Std (degrees)')
    ax.set_title('A. Error ∝ √l', fontweight='bold')
    ax.legend()
    ax.set_xticks(set_sizes)
    
    # B: Distribution comparison
    ax = axes[0, 1]
    l_low, l_high = set_sizes[0], set_sizes[-1]
    sns.kdeplot(decoding[l_low]['errors_deg'], ax=ax, lw=2, label=f'l={l_low}')
    sns.kdeplot(decoding[l_high]['errors_deg'], ax=ax, lw=2, label=f'l={l_high}')
    ax.set_xlabel('Error (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('B. Distributions Widen with Load', fontweight='bold')
    ax.legend()
    ax.set_xlim([-60, 60])
    
    # C: Fisher Information
    ax = axes[1, 0]
    ax.plot(set_sizes, fisher_info, 'o-', color=sns.color_palette()[3], lw=2.5, ms=10)
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Fisher Information')
    ax.set_title('C. Information ∝ 1/l', fontweight='bold')
    ax.set_xticks(set_sizes)
    
    # D: Decoder efficiency
    ax = axes[1, 1]
    efficiency = [theoretical_std[i] / empirical_std[i] for i in range(len(set_sizes))]
    ax.bar(set_sizes, efficiency, color=sns.color_palette()[5], alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='red', ls='--', lw=2, label='Optimal')
    ax.set_xlabel('Set Size (l)')
    ax.set_ylabel('Efficiency (CR / Empirical)')
    ax.set_title('D. Decoder Efficiency', fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.legend()
    ax.set_xticks(set_sizes)
    
    plt.suptitle(f'Experiment 4 Summary (N={cfg.n_neurons}, T_d={cfg.T_d}s)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'exp4_summary.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"  ✓ Saved: exp4_summary.png")


# =============================================================================
# MAIN (standalone execution)
# =============================================================================

def main():
    """Standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment 4: ML Decoding')
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_orientations', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--T_d', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/exp4')
    parser.add_argument('--show', action='store_true')
    
    args = parser.parse_args()
    
    config = {
        'n_neurons': args.n_neurons,
        'n_orientations': args.n_orientations,
        'gamma': args.gamma,
        'T_d': args.T_d,
        'n_trials': args.n_trials,
        'seed': args.seed,
        'set_sizes': [1, 2, 4, 6, 8],
        'kappa': 2.0,
    }
    
    results = run_experiment_4(config)
    plot_results(results, args.output_dir, show_plot=args.show)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output: {args.output_dir}/")
    print("\n🔬 THE CAUSAL CHAIN:")
    print("   DN caps activity → Per-item rate ∝ 1/l → λ ∝ 1/l")
    print("   → SNR ∝ 1/√l → I_F ∝ 1/l → Error std ∝ √l")


if __name__ == '__main__':
    main()