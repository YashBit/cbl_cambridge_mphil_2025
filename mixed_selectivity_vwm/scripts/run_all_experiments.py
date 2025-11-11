#!/usr/bin/env python
"""
Run Mixed Selectivity Experiments - All Three Methods

This script runs experiments with all three generation methods:
1. Direct (Engineered) - Guaranteed mixed selectivity
2. GP Interaction - Biologically plausible
3. Simple Conjunctive - Pedagogical demonstration

Usage from project root:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --quick      # Fast test
    python scripts/run_all_experiments.py --full       # Full analysis
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

# Import from project structure
from mixed_selectivity.experiments.exp1_validation import run_experiment1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run mixed selectivity experiments with all methods'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'direct', 'gp_interaction', 'simple_conjunctive', 'compare'],
        help='Which method(s) to run (default: all)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with minimal neurons (5 neurons, fast)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full analysis with many neurons (100 neurons, slow)'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    return parser.parse_args()


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def run_single_method(method, config, method_name):
    """Run experiment with a single method."""
    print_header(f"RUNNING: {method_name.upper()}")
    
    # Update config with method
    method_config = config.copy()
    method_config['method'] = method
    method_config['save_dir'] = f'figures/exp1/{method}'
    
    # Run experiment
    results = run_experiment1(**method_config)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {method_name.upper()}")
    print(f"{'='*70}")
    print(f"Mean Separability: {results['separability_stats']['mean']:.3f}")
    print(f"Std Separability:  {results['separability_stats']['std']:.3f}")
    print(f"Mixed Selectivity: {results['separability_stats']['percent_mixed']:.1f}%")
    print(f"Success:           {'✓ YES' if results['success'] else '✗ NO'}")
    print(f"{'='*70}\n")
    
    return results


def compare_all_methods(results_dict):
    """Print comparison of all methods."""
    print_header("COMPARISON OF ALL METHODS")
    
    # Sort by mean separability (lower is better)
    sorted_methods = sorted(results_dict.items(), 
                          key=lambda x: x[1]['separability_stats']['mean'])
    
    print(f"{'Rank':<6} {'Method':<20} {'Mean Sep':<12} {'Mixed %':<12} {'Status'}")
    print("-" * 70)
    
    for rank, (method, results) in enumerate(sorted_methods, 1):
        mean_sep = results['separability_stats']['mean']
        mixed_pct = results['separability_stats']['percent_mixed']
        status = "✓ PASS" if results['success'] else "✗ FAIL"
        
        print(f"{rank:<6} {method:<20} {mean_sep:<12.3f} {mixed_pct:<12.1f} {status}")
    
    print("-" * 70)
    print(f"\n🏆 WINNER: {sorted_methods[0][0]}")
    print(f"   Mean Separability: {sorted_methods[0][1]['separability_stats']['mean']:.3f}")
    print(f"   Mixed Selectivity: {sorted_methods[0][1]['separability_stats']['percent_mixed']:.1f}%")


def main():
    args = parse_args()
    
    print_header("MIXED SELECTIVITY EXPERIMENT SUITE")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    
    # Configure based on flags
    if args.quick:
        print("⚡ QUICK MODE: Fast test with minimal neurons")
        n_neurons = 5
        n_orientations = 10
        n_locations = 3
    elif args.full:
        print("🔬 FULL MODE: Comprehensive analysis")
        n_neurons = 100
        n_orientations = 30
        n_locations = 6
    else:
        print("📊 STANDARD MODE: Balanced analysis")
        n_neurons = 20
        n_orientations = 20
        n_locations = 4
    
    # Base configuration
    config = {
        'n_neurons': n_neurons,
        'n_orientations': n_orientations,
        'n_locations': n_locations,
        'theta_lengthscale': 0.3,
        'spatial_lengthscale': 1.5,
        'plot': not args.no_plot,
        'seed': args.seed,
        'verbose': True
    }
    
    print(f"\nConfiguration:")
    print(f"  Neurons: {n_neurons}")
    print(f"  Orientations: {n_orientations}")
    print(f"  Locations: {n_locations}")
    print(f"  Plotting: {not args.no_plot}")
    
    # Run experiments based on mode
    results = {}
    
    if args.mode == 'all':
        # Run all three methods
        methods = [
            ('direct', 'Direct (Engineered)'),
            ('gp_interaction', 'GP Interaction'),
            ('simple_conjunctive', 'Simple Conjunctive')
        ]
        
        for method, name in methods:
            results[method] = run_single_method(method, config, name)
        
        # Compare results
        compare_all_methods(results)
        
    elif args.mode == 'compare':
        # Use built-in comparison mode
        print_header("RUNNING COMPARISON MODE")
        config['method'] = 'compare'
        results = run_experiment1(**config)
        
    else:
        # Run single method
        method_names = {
            'direct': 'Direct (Engineered)',
            'gp_interaction': 'GP Interaction',
            'simple_conjunctive': 'Simple Conjunctive'
        }
        results[args.mode] = run_single_method(
            args.mode, 
            config, 
            method_names[args.mode]
        )
    
    # Final summary
    print_header("EXPERIMENT COMPLETE")
    
    if not args.no_plot:
        print("📊 Figures saved to: figures/exp1/")
        print("👁️  Close plot windows to exit...")
        plt.show()
    
    print("\n✓ All experiments finished successfully!")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)