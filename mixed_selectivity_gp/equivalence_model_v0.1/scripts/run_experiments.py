#!/usr/bin/env python
"""
Unified Experiment Runner for Mixed Selectivity Analysis

This script runs either:
  --exp1  Pre-Normalized Response (exponential growth with set size)
  --exp2  Post-Normalized Response with DN (should be FLAT)
  --both  Run both experiments with same seed (for direct comparison)

USAGE:
    # Run pre-normalized (Experiment 1)
    python run_experiments.py --exp1 --n_neurons 1
    
    # Run post-normalized with DN (Experiment 2)
    python run_experiments.py --exp2 --n_neurons 1 --gamma 100
    
    # Run both for comparison
    python run_experiments.py --both --n_neurons 10 --gamma 100
    
    # Population analysis
    python run_experiments.py --both --n_neurons 100 --gamma 100

KEY INSIGHT:
    - Exp1 (Pre-DN): R.mean grows EXPONENTIALLY with set size
    - Exp2 (Post-DN): R.mean is nearly FLAT (constant) - this is the 
      metabolic budget constraint that explains WM capacity limits

Author: Mixed Selectivity Project
Date: December 2025
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from experiments.exp1_pre_normalized import (
    run_experiment1, plot_experiment1
)
from experiments.exp2_post_normalized import (
    run_experiment2, plot_experiment2
)

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Mixed Selectivity Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single neuron, pre-normalized only
    python run_experiments.py --exp1 --n_neurons 1
    
    # Single neuron, post-normalized with DN
    python run_experiments.py --exp2 --n_neurons 1 --gamma 100
    
    # Both experiments for comparison
    python run_experiments.py --both --n_neurons 1
    
    # Population analysis (100 neurons)
    python run_experiments.py --both --n_neurons 100 --gamma 100

Key Insight:
    Pre-DN (Exp1):  R.mean grows EXPONENTIALLY with set size
    Post-DN (Exp2): R.mean is nearly FLAT (metabolic budget)
        """
    )
    
    # Experiment selection
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument('--exp1', action='store_true',
                          help='Run Experiment 1: Pre-Normalized Response')
    exp_group.add_argument('--exp2', action='store_true',
                          help='Run Experiment 2: Post-Normalized Response (DN)')
    exp_group.add_argument('--both', action='store_true',
                          help='Run both experiments for comparison')
    
    # Common parameters
    parser.add_argument('--n_neurons', type=int, default=1,
                        help='Number of neurons (default: 1)')
    parser.add_argument('--n_orientations', type=int, default=10,
                        help='Number of orientation bins (default: 10)')
    parser.add_argument('--theta_lengthscale', type=float, default=0.3,
                        help='Base GP lengthscale (default: 0.3)')
    parser.add_argument('--lengthscale_variability', type=float, default=0.5,
                        help='Lengthscale variability σ_λ (default: 0.5)')
    parser.add_argument('--seed', type=int, default=22,
                        help='Random seed (default: 22)')
    
    # DN parameters (for exp2)
    parser.add_argument('--gamma', type=float, default=100.0,
                        help='DN gain constant in Hz (default: 100)')
    parser.add_argument('--sigma_sq', type=float, default=1e-6,
                        help='DN semi-saturation constant (default: 1e-6)')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='figures',
                        help='Base directory for saving figures (default: figures)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')
    
    return parser.parse_args()


def run_both_experiments(args):
    """Run both experiments with same seed for direct comparison."""
    
    print("\n" + "🧠 "*20)
    print("  MIXED SELECTIVITY EXPERIMENT SUITE")
    print("🧠 "*20)
    
    print(f"\n  Running BOTH experiments with seed={args.seed}")
    print(f"  This allows direct comparison of Pre-DN vs Post-DN")
    
    # Run Experiment 1
    print("\n" + "─"*70)
    results1 = run_experiment1(
        n_neurons=args.n_neurons,
        n_orientations=args.n_orientations,
        theta_lengthscale=args.theta_lengthscale,
        lengthscale_variability=args.lengthscale_variability,
        seed=args.seed,
        verbose=True
    )
    
    # Run Experiment 2 with SAME seed
    print("\n" + "─"*70)
    results2 = run_experiment2(
        n_neurons=args.n_neurons,
        n_orientations=args.n_orientations,
        theta_lengthscale=args.theta_lengthscale,
        lengthscale_variability=args.lengthscale_variability,
        gamma=args.gamma,
        sigma_sq=args.sigma_sq,
        seed=args.seed,  # SAME SEED for comparison
        verbose=True
    )
    
    # Create comparison summary
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    
    subset_sizes = [2, 4, 6, 8]
    
    print(f"\n  {'l':<5} {'Pre-DN':<18} {'Post-DN':<18} {'Fold Reduction':<15}")
    print("  " + "-"*60)
    
    for l in subset_sizes:
        pre = results1['population_summary'][l]['R_mean']
        post = results2['population_summary']['post_norm'][l]['R_mean']
        fold = pre / (post + 1e-10)
        print(f"  {l:<5} {pre:<18.4e} {post:<18.4e} {fold:<15.2f}×")
    
    # Key insight check
    pre_values = [results1['population_summary'][l]['R_mean'] for l in subset_sizes]
    post_values = [results2['population_summary']['post_norm'][l]['R_mean'] for l in subset_sizes]
    
    pre_fold_total = pre_values[-1] / pre_values[0]
    post_fold_total = post_values[-1] / post_values[0]
    
    print(f"\n  KEY INSIGHT:")
    print(f"    Pre-DN:  l=2→l=8 shows {pre_fold_total:.1f}× increase (EXPONENTIAL)")
    print(f"    Post-DN: l=2→l=8 shows {post_fold_total:.1f}× increase (FLAT)")
    
    post_cv = np.std(post_values) / np.mean(post_values)
    print(f"\n    Post-DN Coefficient of Variation: {post_cv:.2%}")
    
    if post_cv < 0.3:
        print(f"    ✓ DN successfully CAPS total activity across set sizes")
    else:
        print(f"    ⚠ Post-DN shows more variation than expected")
    
    # Plot if requested
    if not args.no_plot:
        save_dir1 = Path(args.save_dir) / 'exp1_pre_norm'
        save_dir2 = Path(args.save_dir) / 'exp2_post_norm'
        
        plot_experiment1(results1, save_dir=str(save_dir1), show_plot=False)
        plot_experiment2(results2, save_dir=str(save_dir2), show_plot=False)
        
        # Create combined comparison plot
        create_comparison_plot(results1, results2, args)
        
        plt.show()
    
    return results1, results2


def create_comparison_plot(results1, results2, args):
    """Create a combined comparison plot."""
    import seaborn as sns
    
    save_dir = Path(args.save_dir) / 'comparison'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    subset_sizes = [2, 4, 6, 8]
    
    pre_means = [results1['population_summary'][l]['R_mean'] for l in subset_sizes]
    post_means = [results2['population_summary']['post_norm'][l]['R_mean'] for l in subset_sizes]
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_neurons = args.n_neurons
    neuron_str = "1 neuron" if n_neurons == 1 else f"{n_neurons} neurons (avg)"
    
    # Left: Log scale comparison
    ax1 = axes[0]
    ax1.set_yscale('log')
    ax1.plot(subset_sizes, pre_means, 'o-', linewidth=2.5, markersize=10,
             color='#E74C3C', label='Pre-DN (raw)')
    ax1.plot(subset_sizes, post_means, 's-', linewidth=2.5, markersize=10,
             color='#2E86AB', label=f'Post-DN (γ={args.gamma})')
    
    ax1.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R.mean (log scale)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Pre-DN vs Post-DN\n({neuron_str})', fontsize=16, fontweight='bold')
    ax1.set_xticks(subset_sizes)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Right: Post-DN only (linear) with horizontal line
    ax2 = axes[1]
    ax2.plot(subset_sizes, post_means, 's-', linewidth=2.5, markersize=12,
             color='#2E86AB', label=f'Post-DN (γ={args.gamma})')
    ax2.scatter(subset_sizes, post_means, s=200, c='#1A5276',
                alpha=0.7, edgecolors='white', linewidths=2, zorder=5)
    
    mean_post = np.mean(post_means)
    ax2.axhline(y=mean_post, color='#27AE60', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Mean: {mean_post:.2e}')
    
    for l, val in zip(subset_sizes, post_means):
        ax2.annotate(f'{val:.2e}', xy=(l, val), xytext=(0, 12),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.9))
    
    ax2.set_xlabel('Set Size (l)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Post-DN R.mean', fontsize=14, fontweight='bold')
    ax2.set_title(f'Post-DN Response (should be ~FLAT)\n({neuron_str})',
                  fontsize=16, fontweight='bold')
    ax2.set_xticks(subset_sizes)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = save_dir / f'comparison_{n_neurons}neurons.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved comparison plot: {filepath}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.both:
        results1, results2 = run_both_experiments(args)
        
    elif args.exp1:
        print("\n" + "🧠 "*20)
        print("  EXPERIMENT 1: PRE-NORMALIZED RESPONSE")
        print("🧠 "*20)
        
        results = run_experiment1(
            n_neurons=args.n_neurons,
            n_orientations=args.n_orientations,
            theta_lengthscale=args.theta_lengthscale,
            lengthscale_variability=args.lengthscale_variability,
            seed=args.seed,
            verbose=True
        )
        
        if not args.no_plot:
            save_dir = Path(args.save_dir) / 'exp1_pre_norm'
            plot_experiment1(results, save_dir=str(save_dir))
        
        # Save
        save_path = Path(args.save_dir) / 'exp1_pre_norm' / f'exp1_results_{args.n_neurons}neurons.npy'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, results, allow_pickle=True)
        print(f"\n  💾 Results saved to: {save_path}")
        
    elif args.exp2:
        print("\n" + "🧠 "*20)
        print("  EXPERIMENT 2: POST-NORMALIZED RESPONSE (DN)")
        print("🧠 "*20)
        
        results = run_experiment2(
            n_neurons=args.n_neurons,
            n_orientations=args.n_orientations,
            theta_lengthscale=args.theta_lengthscale,
            lengthscale_variability=args.lengthscale_variability,
            gamma=args.gamma,
            sigma_sq=args.sigma_sq,
            seed=args.seed,
            verbose=True
        )
        
        if not args.no_plot:
            save_dir = Path(args.save_dir) / 'exp2_post_norm'
            plot_experiment2(results, save_dir=str(save_dir))
        
        # Save
        save_path = Path(args.save_dir) / 'exp2_post_norm' / f'exp2_results_{args.n_neurons}neurons.npy'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, results, allow_pickle=True)
        print(f"\n  💾 Results saved to: {save_path}")
    
    print("\n" + "✅ "*20)
    print("  EXPERIMENT COMPLETE")
    print("✅ "*20)


if __name__ == '__main__':
    main()