"""
Master script to run all experiments with multiple seeds and neuron counts.

=============================================================================
EXPERIMENT INDEX
=============================================================================

Core framework experiments (exp 1-4):
    1 — Pre-normalised response analysis
    2 — Post-normalised (DN) response analysis
    3 — Poisson noise and SNR degradation
    4 — Efficient ML decoding

Bays (2014) equivalence experiments (exp 5-9):
    5  — Figure 1 d,e,f: variance, kurtosis, exponent vs (λ, γ)
    6  — Figure 2: error distributions and scaling vs set size
    7  — Figure 3: cued recall with optimal weighting
    8  — Figure 4: robustness (broad, narrow, baseline, hetero, cosine, corr)
    9  — Figure 5: effects of baseline activity on ML parameters

Bays & Brady (2024) equivalence experiments (exp 10):
    10 — Figure 5: SD vs set size, continuous rise, √l prediction

=============================================================================
USAGE
=============================================================================

Single experiment:
    python scripts/run_experiments.py --exp 1
    python scripts/run_experiments.py --exp 6 --n_neurons 50 --seed 22

Experiment groups:
    python scripts/run_experiments.py --core          # experiments 1-4 only
    python scripts/run_experiments.py --bays          # experiments 5-10 only
    python scripts/run_experiments.py --all           # all experiments 1-10

Multiple experiments:
    python scripts/run_experiments.py --exp 5 6 7 8 9 10

Multiple seeds (for statistical robustness):
    python scripts/run_experiments.py --exp 3 --seeds 42 43 44 45 46

Multiple neuron counts (scaling analysis):
    python scripts/run_experiments.py --exp all --neurons 100 1000 10000

Full batch run (all experiments, default seeds [42-46], neurons [100, 10000]):
    python scripts/run_experiments.py --batch

Custom batch:
    python scripts/run_experiments.py --exp 3 4 --seeds 1 2 3 --neurons 100 10000

Override experiment parameters:
    python scripts/run_experiments.py --exp 6 --gamma 200.0 --T_d 0.2
    python scripts/run_experiments.py --exp 8 --n_trials 50000

=============================================================================
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from itertools import product

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))


# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_EXPERIMENTS = list(range(1, 11))  # 1 through 10

EXPERIMENT_NAMES = {
    1: 'Pre-normalised response analysis',
    2: 'Post-normalised (DN) response analysis',
    3: 'Poisson noise and SNR degradation',
    4: 'Efficient ML decoding',
    5: 'Bays (2014) Fig 1 d,e,f — variance/kurtosis/exponent grid',
    6: 'Bays (2014) Fig 2 — error distributions vs set size',
    7: 'Bays (2014) Fig 3 — cued recall with optimal weighting',
    8: 'Bays (2014) Fig 4 — robustness of error distributions',
    9: 'Bays (2014) Fig 5 — baseline activity effects',
    10: 'Bays & Brady (2024) Fig 5 — SD vs set size',
}

DEFAULT_CONFIG = {
    'n_orientations': 100,
    'n_locations': 8,
    'set_sizes': [2, 4, 6, 8],
    'gamma': 100.0,
    'sigma_sq': 1e-6,
    'lambda_base': 0.3,
    'sigma_lambda': 0.5,
    'T_d': 0.1,
    'n_trials': 500,
    'kappa': 2.0,
}

BATCH_SEEDS = [42, 43, 44, 45, 46]
BATCH_NEURONS = [100, 10000]


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_single_experiment(
    exp_num: int,
    n_neurons: int,
    seed: int,
    config: dict,
    output_base: str = 'results'
) -> dict:
    """
    Run a single experiment with specified parameters.
    
    Returns
    -------
    result : dict
        Contains status, timing, and output path
    """
    
    # Create output directory with descriptive name including date
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_base}/exp{exp_num}_seed{seed}_{timestamp}/N{n_neurons}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # =================================================================
        # CORE FRAMEWORK EXPERIMENTS (1-4)
        # =================================================================

        if exp_num == 1:
            from experiments.core_basics.exp1_pre_normalized import run_experiment_1, plot_results
            
            exp_config = {
                'n_neurons': n_neurons,
                'n_orientations': config['n_orientations'],
                'n_locations': config['n_locations'],
                'set_sizes': config['set_sizes'],
                'seed': seed,
                'lambda_base': config['lambda_base'],
                'sigma_lambda': config['sigma_lambda'],
            }
            results = run_experiment_1(exp_config)
            plot_results(results, output_dir, show_plot=False)
            
        elif exp_num == 2:
            from experiments.core_basics.exp2_post_normalized import run_experiment_2, plot_results
            
            exp_config = {
                'n_neurons': n_neurons,
                'n_orientations': config['n_orientations'],
                'n_locations': config['n_locations'],
                'set_sizes': config['set_sizes'],
                'seed': seed,
                'gamma': config['gamma'],
                'sigma_sq': config['sigma_sq'],
                'lambda_base': config['lambda_base'],
                'sigma_lambda': config['sigma_lambda'],
            }
            results = run_experiment_2(exp_config)
            plot_results(results, output_dir, show_plot=False)
            
        elif exp_num == 3:
            from experiments.core_basics.exp3_poisson_noise_snr import run_experiment_3, plot_results
            
            exp_config = {
                'n_neurons': n_neurons,
                'n_orientations': config['n_orientations'],
                'n_locations': config['n_locations'],
                'set_sizes': config['set_sizes'],
                'seed': seed,
                'gamma': config['gamma'],
                'sigma_sq': config['sigma_sq'],
                'lambda_base': config['lambda_base'],
                'sigma_lambda': config['sigma_lambda'],
                'T_d': config['T_d'],
                'n_trials': config['n_trials'],
            }
            results = run_experiment_3(exp_config)
            plot_results(results, output_dir, show_plot=False)
            
        elif exp_num == 4:
            from experiments.core_basics.exp4_ml_decoding import run_experiment_4, plot_results
            
            exp_config = {
                'n_neurons': n_neurons,
                'n_orientations': 200,  # Finer resolution for decoding
                'seed': seed,
                'gamma': config['gamma'],
                'T_d': config['T_d'],
                'n_trials': config['n_trials'],
                'kappa': config['kappa'],
            }
            results = run_experiment_4(exp_config)
            plot_results(results, output_dir, show_plot=False)

        # =================================================================
        # BAYS (2014) EQUIVALENCE EXPERIMENTS (5-9)
        # =================================================================
            
        elif exp_num == 5:
            from experiments.bays_equivalence.figure_1 import run_experiment, plot_results
            
            exp_config = {
                'M': n_neurons,
                'n_theta': 64,
                'n_trials': config['n_trials'],
                'T_d': config['T_d'],
                'sigma_sq': config['sigma_sq'],
                'n_grid': 25,
                'lambda_range': (0.1, 2.5),
                'gamma_range': (1.0, 256.0),
                'seed': seed,
            }
            results = run_experiment(exp_config)
            plot_results(results, output_dir, show_plot=False)
            
        elif exp_num == 6:
            from experiments.bays_equivalence.figure_2 import run_experiment, plot_results

            exp_config = {
                'M': n_neurons,
                'n_theta': 64,
                'n_trials': config['n_trials'],
                'T_d': config['T_d'],
                'sigma_sq': config['sigma_sq'],
                'lambda_base': config['lambda_base'],
                'gamma': config['gamma'],
                'set_sizes': config['set_sizes'],
                'seed': seed,
                'n_seeds': 5,
            }
            results = run_experiment(exp_config)
            plot_results(results, output_dir, show_plot=False)
            
        elif exp_num == 7:
            from experiments.bays_equivalence.figure_3 import run_experiment, plot_results

            exp_config = {
                'M': n_neurons,
                'n_theta': 64,
                'n_trials': config['n_trials'],
                'n_trials_sweep': min(config['n_trials'], 2000),
                'T_d': config['T_d'],
                'sigma_sq': config['sigma_sq'],
                'lambda_base': config['lambda_base'],
                'gamma': config['gamma'],
                'set_sizes': [2, 4, 8],
                'cue_ratio': 3.0,
                'alpha_range': (1.0, 8.0),
                'n_alpha': 15,
                'seed': seed,
                'n_seeds': 3,
                'n_bins': 50,
            }
            results = run_experiment(exp_config)
            plot_results(results, output_dir, show_plot=False)

        elif exp_num == 8:
            from experiments.bays_equivalence.figure_4 import run_experiment, plot_results

            exp_config = {
                'n_theta': 64,
                'n_trials': config['n_trials'],
                'T_d': config['T_d'],
                'sigma_sq': config['sigma_sq'],
                'lambda_broad': 1.0,
                'lambda_narrow': 0.3,
                'lambda_std': 0.1,
                'baseline_frac': 0.25,
                'c0': 0.25,
                'gammas': [1, 2, 4, 8, 16, 32, 64, 128],
                'pop_sizes': [100, 1000],
                'n_bins': 50,
                'seed': seed,
            }
            results = run_experiment(exp_config)
            plot_results(results, output_dir, show_plot=False)

        elif exp_num == 9:
            from experiments.bays_equivalence.figure_5 import run_experiment, plot_results

            exp_config = {
                'M': n_neurons,
                'n_theta': 64,
                'T_d': config['T_d'],
                'sigma_sq': config['sigma_sq'],
                'lambda_ref': 0.5,
                'gamma_ref': config['gamma'],
                'set_sizes': [1, 2, 4, 8],
                'baseline_fracs': [0.0, 0.05, 0.20, 0.50, 0.80, 0.95],
                'n_trials_fit': min(config['n_trials'], 3000),
                'n_trials_final': config['n_trials'],
                'n_gamma_grid': 20,
                'n_lambda_grid': 8,
                'gamma_range': (10.0, 1e6),
                'lambda_range': (0.3, 1.0),
                'seed': seed,
                'n_seeds': 3,
            }
            results = run_experiment(exp_config)
            plot_results(results, output_dir, show_plot=False)

        # =================================================================
        # BAYS & BRADY (2024) EQUIVALENCE EXPERIMENTS (10)
        # =================================================================

        elif exp_num == 10:
            from experiments.nature.figure_5 import run_experiment, plot_results

            exp_config = {
                'M': n_neurons,
                'n_theta': 64,
                'n_trials': config['n_trials'],
                'T_d': config['T_d'],
                'sigma_sq': config['sigma_sq'],
                'lambda_base': config['lambda_base'],
                'gamma': config['gamma'],
                'set_sizes': [1, 2, 3, 4, 5, 6, 7, 8],
                'seed': seed,
                'n_seeds': 5,
            }
            results = run_experiment(exp_config)
            plot_results(results, output_dir, show_plot=False)
            
        else:
            raise ValueError(f"Unknown experiment: {exp_num}")
        
        elapsed = time.time() - start_time
        return {
            'status': 'success',
            'experiment': exp_num,
            'name': EXPERIMENT_NAMES.get(exp_num, ''),
            'n_neurons': n_neurons,
            'seed': seed,
            'elapsed_seconds': elapsed,
            'output_dir': output_dir
        }
        
    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        return {
            'status': 'error',
            'experiment': exp_num,
            'name': EXPERIMENT_NAMES.get(exp_num, ''),
            'n_neurons': n_neurons,
            'seed': seed,
            'elapsed_seconds': elapsed,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_batch(
    experiments: list,
    neurons_list: list,
    seeds_list: list,
    config: dict,
    output_base: str = 'results'
) -> list:
    """
    Run multiple experiments across different seeds and neuron counts.
    """
    
    total_runs = len(experiments) * len(neurons_list) * len(seeds_list)
    
    print("=" * 70)
    print("BATCH EXPERIMENT RUN")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Experiments:   {experiments}")
    for e in experiments:
        print(f"                 {e}: {EXPERIMENT_NAMES.get(e, '?')}")
    print(f"  Neuron counts: {neurons_list}")
    print(f"  Seeds:         {seeds_list}")
    print(f"  Total runs:    {total_runs}")
    print(f"  Output base:   {output_base}")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path(output_base) / f"batch_{timestamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    run_count = 0
    batch_start = time.time()
    
    for exp_num in experiments:
        for n_neurons in neurons_list:
            for seed in seeds_list:
                run_count += 1
                
                print("\n" + "=" * 70)
                print(f"RUN {run_count}/{total_runs}: "
                      f"Exp {exp_num} ({EXPERIMENT_NAMES.get(exp_num, '')}), "
                      f"N={n_neurons:,}, seed={seed}")
                print("=" * 70)
                
                result = run_single_experiment(
                    exp_num=exp_num,
                    n_neurons=n_neurons,
                    seed=seed,
                    config=config,
                    output_base=str(summary_dir)
                )
                
                results.append(result)
                
                status_icon = "OK" if result['status'] == 'success' else "FAIL"
                print(f"\n[{status_icon}] Completed in {result['elapsed_seconds']:.1f}s")
    
    batch_elapsed = time.time() - batch_start
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_runs': total_runs,
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'error'),
        'total_time_seconds': batch_elapsed,
        'config': config,
        'experiments': experiments,
        'neurons_list': neurons_list,
        'seeds_list': seeds_list,
        'results': results
    }
    
    summary_file = summary_dir / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"   Total runs:    {total_runs}")
    print(f"   Successful:    {summary['successful']}")
    print(f"   Failed:        {summary['failed']}")
    print(f"   Total time:    {batch_elapsed/60:.1f} minutes")
    print(f"\n   Output: {summary_dir}")
    print(f"   Summary: {summary_file}")
    
    print(f"\nTiming by experiment:")
    for exp_num in experiments:
        exp_results = [r for r in results if r['experiment'] == exp_num and r['status'] == 'success']
        if exp_results:
            avg_time = sum(r['elapsed_seconds'] for r in exp_results) / len(exp_results)
            print(f"   Exp {exp_num} ({EXPERIMENT_NAMES.get(exp_num, '')}): {avg_time:.1f}s average")
    
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"   Exp {e['experiment']}, N={e['n_neurons']}, seed={e['seed']}: {e['error']}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Mixed Selectivity Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  Core framework (1-4):
    1  Pre-normalised response analysis
    2  Post-normalised (DN) response analysis
    3  Poisson noise and SNR degradation
    4  Efficient ML decoding

  Bays (2014) equivalence (5-9):
    5  Figure 1 d,e,f — variance/kurtosis/exponent vs (lambda, gamma)
    6  Figure 2 — error distributions vs set size
    7  Figure 3 — cued recall with optimal weighting
    8  Figure 4 — robustness (broad, narrow, baseline, hetero, cosine, corr)
    9  Figure 5 — baseline activity effects on ML parameters

  Bays & Brady (2024) equivalence (10):
    10 Figure 5 — SD vs set size, continuous rise, sqrt(l) prediction

Examples:
  python run_experiments.py --exp 2
  python run_experiments.py --exp 2 --n_neurons 50 --seed 22
  python run_experiments.py --exp 5 6 7 8 9
  python run_experiments.py --exp 2 --seeds 42 43 44
  python run_experiments.py --exp 2 --neurons 100 10000
  python run_experiments.py --batch
  python run_experiments.py --exp 2 3 --seeds 1 2 3 --neurons 100 1000
        """
    )
    
    # Experiment selection
    parser.add_argument('--exp', type=int, nargs='+', default=None,
                       help='Experiment number(s): 1-4 (core), 5-9 (Bays 2014), 10 (Bays & Brady 2024)')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments (1-9)')
    parser.add_argument('--batch', action='store_true',
                       help='Run full batch with default seeds [42-46] and neurons [100, 10000]')
    parser.add_argument('--core', action='store_true',
                       help='Run core experiments only (1-4)')
    parser.add_argument('--bays', action='store_true',
                       help='Run Bays equivalence experiments only (5-9)')
    
    # Multi-run parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seed(s) to use')
    parser.add_argument('--neurons', type=int, nargs='+', default=[100],
                       help='Neuron count(s) to use')
    
    # Single-run parameters (for backward compatibility)
    parser.add_argument('--n_neurons', type=int, default=None,
                       help='Number of neurons (single run)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (single run)')
    
    # Experiment parameters
    parser.add_argument('--n_orientations', type=int, default=100)
    parser.add_argument('--n_locations', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=100.0)
    parser.add_argument('--sigma_sq', type=float, default=1e-6)
    parser.add_argument('--T_d', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=500)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'n_orientations': args.n_orientations,
        'n_locations': args.n_locations,
        'set_sizes': [2, 4, 6, 8],
        'gamma': args.gamma,
        'sigma_sq': args.sigma_sq,
        'lambda_base': 0.3,
        'sigma_lambda': 0.5,
        'T_d': args.T_d,
        'n_trials': args.n_trials,
        'kappa': args.kappa,
    }
    
    # Determine experiments to run
    if args.batch:
        experiments = ALL_EXPERIMENTS
        seeds = BATCH_SEEDS
        neurons = BATCH_NEURONS
    elif args.all:
        experiments = ALL_EXPERIMENTS
        seeds = args.seeds
        neurons = args.neurons
    elif args.core:
        experiments = [1, 2, 3, 4]
        seeds = args.seeds
        neurons = args.neurons
    elif args.bays:
        experiments = [5, 6, 7, 8, 9, 10]
        seeds = args.seeds
        neurons = args.neurons
    elif args.exp:
        experiments = args.exp
        seeds = args.seeds
        neurons = args.neurons
    else:
        print("Error: Specify --exp, --all, --core, --bays, or --batch")
        parser.print_help()
        sys.exit(1)
    
    # Handle single-run backward compatibility
    if args.n_neurons is not None:
        neurons = [args.n_neurons]
    if args.seed is not None:
        seeds = [args.seed]
    
    # Validate experiments
    for e in experiments:
        if e not in ALL_EXPERIMENTS:
            print(f"Error: Invalid experiment number: {e}. Valid: {ALL_EXPERIMENTS}")
            sys.exit(1)
    
    # Run
    print("=" * 70)
    print("MIXED SELECTIVITY & DIVISIVE NORMALIZATION FRAMEWORK")
    print("=" * 70)
    
    if len(experiments) == 1 and len(seeds) == 1 and len(neurons) == 1:
        # Single run
        exp_num = experiments[0]
        print(f"\nExperiment {exp_num}: {EXPERIMENT_NAMES.get(exp_num, '')}")
        print(f"  N={neurons[0]}, seed={seeds[0]}")
        
        result = run_single_experiment(
            exp_num=exp_num,
            n_neurons=neurons[0],
            seed=seeds[0],
            config=config,
            output_base=args.output_dir
        )
        
        if result['status'] == 'success':
            print(f"\n[OK] Experiment {exp_num} complete ({result['elapsed_seconds']:.1f}s)")
            print(f"  Output: {result['output_dir']}")
        else:
            print(f"\n[FAIL] Experiment {exp_num} failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(result['traceback'])
    else:
        # Batch run
        run_batch(
            experiments=experiments,
            neurons_list=neurons,
            seeds_list=seeds,
            config=config,
            output_base=args.output_dir
        )


if __name__ == '__main__':
    main()