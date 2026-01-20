"""
Master script to run all experiments.

Usage:
    python scripts/run_experiments.py --exp 1          # Run Experiment 1 only
    python scripts/run_experiments.py --exp 2          # Run Experiment 2 only
    python scripts/run_experiments.py --exp 3          # Run Experiment 3 only
    python scripts/run_experiments.py --exp 4          # Run Experiment 4 only
    python scripts/run_experiments.py --exp all        # Run all experiments
    
    # With custom parameters:
    python scripts/run_experiments.py --exp 4 --n_neurons 100 --seed 42 --n_trials 500
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import experiment modules
from experiments import exp1_pre_normalized
from experiments import exp2_post_normalized
from experiments import exp3_poisson_noise_snr
from experiments import exp4_ml_decoding


def main():
    parser = argparse.ArgumentParser(description='Run Mixed Selectivity Experiments')
    parser.add_argument('--exp', type=str, choices=['1', '2', '3', '4', 'all'], default='all',
                       help='Which experiment to run (1, 2, 3, 4, or all)')
    parser.add_argument('--n_neurons', type=int, default=100,
                       help='Number of neurons')
    parser.add_argument('--n_orientations', type=int, default=10,
                       help='Number of orientation bins (64 for exp4)')
    parser.add_argument('--n_locations', type=int, default=8,
                       help='Number of locations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--gamma', type=float, default=100.0,
                       help='Gain constant (Hz)')
    parser.add_argument('--sigma_sq', type=float, default=1e-6,
                       help='Semi-saturation constant')
    parser.add_argument('--T_d', type=float, default=0.1,
                       help='Decoding time window (seconds)')
    parser.add_argument('--n_trials', type=int, default=500,
                       help='Number of trials for empirical stats')
    parser.add_argument('--kappa', type=float, default=2.0,
                       help='Tuning curve sharpness (Exp 4)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MIXED SELECTIVITY & DIVISIVE NORMALIZATION FRAMEWORK")
    print("=" * 70)
    print(f"\nExperiment: {args.exp}")
    print(f"\nConfiguration:")
    print(f"  N neurons:     {args.n_neurons}")
    print(f"  n_θ:           {args.n_orientations}")
    print(f"  L:             {args.n_locations}")
    print(f"  γ:             {args.gamma} Hz")
    print(f"  σ²:            {args.sigma_sq}")
    print(f"  Seed:          {args.seed}")
    if args.exp in ['3', '4', 'all']:
        print(f"  T_d:           {args.T_d}s")
        print(f"  n_trials:      {args.n_trials}")
    if args.exp in ['4', 'all']:
        print(f"  κ (kappa):     {args.kappa}")
    print()
    
    # Build argument lists for experiment modules
    base_args = [
        '--n_neurons', str(args.n_neurons),
        '--n_orientations', str(args.n_orientations),
        '--n_locations', str(args.n_locations),
        '--seed', str(args.seed),
        '--gamma', str(args.gamma),
        '--sigma_sq', str(args.sigma_sq)
    ]
    
    exp3_args = base_args + [
        '--T_d', str(args.T_d),
        '--n_trials', str(args.n_trials)
    ]
    
    exp4_args = [
        '--n_neurons', str(args.n_neurons),
        '--n_orientations', '64',  # Higher resolution for decoding
        '--seed', str(args.seed),
        '--gamma', str(args.gamma),
        '--T_d', str(args.T_d),
        '--n_trials', str(args.n_trials),
        '--kappa', str(args.kappa)
    ]
    
    # Run experiments
    if args.exp in ['1', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: PRE-NORMALIZED ANALYSIS")
        print("=" * 70 + "\n")
        sys.argv = ['exp1_pre_normalized.py'] + base_args
        exp1_pre_normalized.main()
    
    if args.exp in ['2', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: POST-NORMALIZED ANALYSIS (POPULATION DN)")
        print("=" * 70 + "\n")
        sys.argv = ['exp2_post_normalized.py'] + base_args
        exp2_post_normalized.main()
    
    if args.exp in ['3', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: POISSON NOISE & SNR ANALYSIS")
        print("=" * 70 + "\n")
        sys.argv = ['exp3_poisson_noise_snr.py'] + exp3_args
        exp3_poisson_noise_snr.main()
    
    if args.exp in ['4', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: ML DECODING & PRECISION ANALYSIS")
        print("=" * 70 + "\n")
        sys.argv = ['exp4_ml_decoding.py'] + exp4_args
        exp4_ml_decoding.main()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()