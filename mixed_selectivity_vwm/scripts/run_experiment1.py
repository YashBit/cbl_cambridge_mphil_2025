#!/usr/bin/env python
"""
Run Experiment 1: Validate Mixed Selectivity in Neural Populations

This script runs the first experiment to validate that GP-based
neural populations exhibit mixed selectivity.

Enhanced version with multiple generation methods.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path to import mixed_selectivity package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mixed_selectivity.experiments.exp1_validation import run_experiment1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run mixed selectivity validation experiment'
    )
    parser.add_argument(
        '--method', 
        type=str, 
        default='direct',
        choices=['direct', 'gp_interaction', 'simple_conjunctive', 'compare'],
        help='Generation method to use (default: direct)'
    )
    parser.add_argument(
        '--n_neurons',
        type=int,
        default=20,
        help='Number of neurons to generate (default: 20)'
    )
    parser.add_argument(
        '--n_orientations',
        type=int,
        default=20,
        help='Number of orientations (default: 20)'
    )
    parser.add_argument(
        '--n_locations',
        type=int,
        default=4,
        help='Number of locations (default: 4)'
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


def main(method='direct', n_neurons=20, n_orientations=20, n_locations=4, 
         plot=True, seed=42):
    """
    Run Experiment 1 with optimized parameters for MacBook.
    
    Parameters:
        method: Generation method ('direct', 'gp_interaction', 'simple_conjunctive', 'compare')
        n_neurons: Number of neurons to generate
        n_orientations: Number of orientation values
        n_locations: Number of spatial locations
        plot: Whether to generate plots
        seed: Random seed for reproducibility
    """
    
    print("\n" + "="*60)
    print(" MIXED SELECTIVITY VALIDATION EXPERIMENT")
    print("="*60)
    print("\nStarting Experiment 1...")
    print("This will generate neural populations and test for mixed selectivity.\n")
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration - REMOVED 'verbose' parameter
    config = {
        'n_neurons': n_neurons,
        'n_orientations': n_orientations,
        'n_locations': n_locations,
        'theta_lengthscale': 0.3,      # Smaller for more local tuning
        'spatial_lengthscale': 1.5,    # Smaller for more specificity
        'plot': plot,
        'seed': seed,
        'method': method  # Specify generation method
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Special message for method selection
    if method == 'direct':
        print("📌 Using DIRECT method: Guaranteed mixed selectivity through")
        print("   explicit non-separable pattern construction\n")
    elif method == 'gp_interaction':
        print("📌 Using GP INTERACTION method: GP-based with strong")
        print("   interaction terms between orientation and location\n")
    elif method == 'simple_conjunctive':
        print("📌 Using SIMPLE CONJUNCTIVE method: Location-dependent")
        print("   orientation preferences for clear mixed selectivity\n")
    elif method == 'compare':
        print("📌 COMPARISON mode: Testing all methods to find the best\n")
    
    # Run the experiment
    try:
        results = run_experiment1(**config)
        
        # Print final summary
        print("\n" + "="*60)
        print(" EXPERIMENT COMPLETE")
        print("="*60)
        
        if results['success']:
            print("✅ MAIN FINDING: Successfully generated mixed selectivity!")
            print(f"   Method used: {results.get('method', 'unknown')}")
            print(f"\n📊 Key Statistics:")
            print(f"   - Mean separability: {results['separability_stats']['mean']:.3f}")
            print(f"   - Std separability: {results['separability_stats']['std']:.3f}")
            print(f"   - Neurons with mixed selectivity: {results['separability_stats']['percent_mixed']:.1f}%")
            
            # Save results to file
            save_path = os.path.join(parent_dir, 'data', 'results', f'exp1_results_{method}.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, results)
            print(f"\n💾 Results saved to: {save_path}")
            
        else:
            print("⚠️  Experiment did not achieve strong mixed selectivity.")
            print(f"   Mean separability: {results['separability_stats']['mean']:.3f} (target: < 0.8)")
            
            if method in ['gp_interaction', 'simple_conjunctive']:
                print("\n💡 SUGGESTIONS:")
                print("   1. Try --method direct for guaranteed mixed selectivity")
                print("   2. Try --method compare to see all methods")
            else:
                print("\n   Consider adjusting parameters or trying a different method.")
        
        # If comparison mode, show recommendations
        if method == 'compare' and 'all_results' in results:
            print("\n" + "="*60)
            print(" RECOMMENDATIONS")
            print("="*60)
            
            all_results = results['all_results']
            
            # Sort methods by performance
            sorted_methods = sorted(all_results.items(), 
                                  key=lambda x: x[1]['mean_sep'])
            
            print("\nMethods ranked by mixed selectivity (lower is better):")
            for i, (m, r) in enumerate(sorted_methods, 1):
                status = "✓" if r['mean_sep'] < 0.8 else "✗"
                print(f"  {i}. {m:20s} - Sep: {r['mean_sep']:.3f} {status}")
            
            best_method = sorted_methods[0][0]
            print(f"\n🏆 Recommended method: {best_method}")
            print(f"   Use: python run_experiment1.py --method {best_method}")
        
        return results
        
    except ImportError as e:
        print(f"\n❌ Import error: {str(e)}")
        print("\nMake sure you've updated the following files:")
        print("1. mixed_selectivity/core/gaussian_process.py")
        print("2. mixed_selectivity/experiments/exp1_validation.py")
        raise
        
    except Exception as e:
        print(f"\n❌ Error running experiment: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure all required packages are installed:")
        print("   pip install numpy torch matplotlib tqdm")
        print("2. Ensure the mixed_selectivity package is properly set up:")
        print("   pip install -e .")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run the experiment
    results = main(
        method=args.method,
        n_neurons=args.n_neurons,
        n_orientations=args.n_orientations,
        n_locations=args.n_locations,
        plot=not args.no_plot,
        seed=args.seed
    )
    
    # Keep plots open for viewing
    if results is not None and not args.no_plot:
        print("\n👁️  Viewing results. Close plot windows when done...")
        plt.show()