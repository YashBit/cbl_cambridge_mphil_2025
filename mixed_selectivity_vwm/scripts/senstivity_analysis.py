#!/usr/bin/env python
"""
Rigorous Parameter Sensitivity Analysis for Mixed Selectivity

REAL SCIENCE: Comprehensive testing with results stored in JSON for later analysis.
No visualization - pure data collection for reproducible research.

Outputs:
    - results/sensitivity_analysis_TIMESTAMP.json (main results)
    - results/sensitivity_summary.json (latest summary)
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mixed_selectivity.core.gaussian_process import NeuralGaussianProcess
from mixed_selectivity.experiments.exp1_validation import analyze_population_separability


def safe_sample_neurons(gp: NeuralGaussianProcess, n_neurons: int, max_retries: int = 3) -> np.ndarray:
    """
    Safely sample neurons with fallback for numerical issues.
    
    Some lengthscale combinations can cause ill-conditioned covariance matrices.
    This wrapper catches those cases and skips problematic configurations.
    """
    # Temporarily suppress the internal GP progress bars
    import os
    old_tqdm_disable = os.environ.get('TQDM_DISABLE', None)
    os.environ['TQDM_DISABLE'] = '1'
    
    try:
        for attempt in range(max_retries):
            try:
                result = gp.sample_neurons(n_neurons)
                return result
            except Exception as e:
                if 'positive-definite' in str(e) or 'Cholesky' in str(e):
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None
                else:
                    raise
        return None
    finally:
        # Restore original TQDM setting
        if old_tqdm_disable is None:
            os.environ.pop('TQDM_DISABLE', None)
        else:
            os.environ['TQDM_DISABLE'] = old_tqdm_disable


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of values."""
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'sem': float(np.std(values) / np.sqrt(len(values))),  # Standard error
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
        'n': len(values)
    }


def test_lengthscale_sensitivity(verbose: bool = True) -> Dict[str, Any]:
    """
    Test how GP lengthscales affect separability.
    
    Comprehensive grid search over parameter space.
    """
    
    if verbose:
        print("="*70)
        print("TEST 1: GP LENGTHSCALE SENSITIVITY")
        print("="*70)
        print("Testing 7 × 6 = 42 parameter combinations...")
    
    theta_scales = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    spatial_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    results = {
        'test_name': 'lengthscale_sensitivity',
        'description': 'Grid search over GP lengthscale parameters',
        'parameters': {
            'theta_lengthscales': theta_scales,
            'spatial_lengthscales': spatial_scales,
            'n_neurons': 20,
            'n_orientations': 20,
            'n_locations': 4,
            'method': 'gp_interaction',
            'seed': 42
        },
        'grid_results': [],
        'summary': {}
    }
    
    start_time = time.time()
    
    all_separabilities = []
    
    # Create progress bar for grid search
    total_combinations = len(theta_scales) * len(spatial_scales)
    pbar = tqdm(total=total_combinations, desc="Lengthscale grid", 
                unit=" configs", disable=not verbose, ncols=100)
    
    failed_configs = []
    
    for i, theta_ls in enumerate(theta_scales):
        for j, spatial_ls in enumerate(spatial_scales):
            pbar.set_postfix_str(f'θ={theta_ls:.1f} s={spatial_ls:.1f}')
            
            gp = NeuralGaussianProcess(
                n_orientations=20,
                n_locations=4,
                theta_lengthscale=theta_ls,
                spatial_lengthscale=spatial_ls,
                method='gp_interaction',
                seed=42
            )
            
            # Use safe sampling
            tuning = safe_sample_neurons(gp, 20)
            
            if tuning is None:
                # Configuration failed - record and skip
                failed_configs.append({
                    'theta_lengthscale': float(theta_ls),
                    'spatial_lengthscale': float(spatial_ls),
                    'reason': 'numerical_instability'
                })
                pbar.update(1)
                continue
            
            sep_stats = analyze_population_separability(tuning, show_progress=False)
            
            result_entry = {
                'theta_lengthscale': float(theta_ls),
                'spatial_lengthscale': float(spatial_ls),
                'mean_separability': sep_stats['mean'],
                'std_separability': sep_stats['std'],
                'median_separability': sep_stats['median'],
                'percent_mixed': sep_stats['percent_mixed'],
                'passes_threshold': bool(sep_stats['mean'] < 0.8),
                'all_values': sep_stats['all_values'].tolist()
            }
            
            results['grid_results'].append(result_entry)
            all_separabilities.append(sep_stats['mean'])
            
            pbar.update(1)
    
    pbar.close()
    
    # Report failed configurations
    if failed_configs and verbose:
        print(f"\n⚠️  {len(failed_configs)} configurations failed due to numerical issues:")
        for fc in failed_configs[:3]:  # Show first 3
            print(f"    theta={fc['theta_lengthscale']:.2f}, spatial={fc['spatial_lengthscale']:.2f}")
        if len(failed_configs) > 3:
            print(f"    ... and {len(failed_configs)-3} more")
    
    elapsed_time = time.time() - start_time
    
    # Find optimal parameters (only from successful runs)
    if len(all_separabilities) == 0:
        if verbose:
            print("\n❌ All configurations failed - cannot determine optimal parameters")
        results['summary'] = {
            'error': 'all_configurations_failed',
            'failed_configs': failed_configs
        }
        return results
    
    min_sep_idx = np.argmin(all_separabilities)
    optimal_params = results['grid_results'][min_sep_idx]
    
    # Find parameters closest to baseline (0.3, 1.5)
    baseline_idx = None
    for idx, entry in enumerate(results['grid_results']):
        if entry['theta_lengthscale'] == 0.3 and entry['spatial_lengthscale'] == 1.5:
            baseline_idx = idx
            break
    
    results['summary'] = {
        'total_combinations': len(theta_scales) * len(spatial_scales),
        'successful_combinations': len(results['grid_results']),
        'failed_combinations': len(failed_configs),
        'failed_configs': failed_configs,
        'elapsed_time_seconds': float(elapsed_time),
        'optimal_parameters': {
            'theta_lengthscale': optimal_params['theta_lengthscale'],
            'spatial_lengthscale': optimal_params['spatial_lengthscale'],
            'mean_separability': optimal_params['mean_separability']
        },
        'baseline_parameters': results['grid_results'][baseline_idx] if baseline_idx else None,
        'separability_statistics': compute_statistics(all_separabilities),
        'passing_combinations': sum(1 for r in results['grid_results'] if r['passes_threshold']),
        'passing_percentage': 100 * sum(1 for r in results['grid_results'] if r['passes_threshold']) / len(results['grid_results']) if len(results['grid_results']) > 0 else 0
    }
    
    if verbose:
        print(f"\n✓ Complete in {elapsed_time:.1f}s")
        print(f"  Successful: {len(results['grid_results'])}/{total_combinations} configurations")
        if len(failed_configs) > 0:
            print(f"  Failed: {len(failed_configs)} (numerical instability)")
        print(f"  Optimal: theta={optimal_params['theta_lengthscale']:.2f}, spatial={optimal_params['spatial_lengthscale']:.2f} → sep={optimal_params['mean_separability']:.3f}")
        print(f"  Passing threshold: {results['summary']['passing_combinations']}/{len(results['grid_results'])} ({results['summary']['passing_percentage']:.1f}%)")
    
    return results


def test_population_size_effect(verbose: bool = True) -> Dict[str, Any]:
    """
    Test stability of results across different population sizes.
    
    Multiple repeats at each size to estimate variance.
    """
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 2: POPULATION SIZE STABILITY")
        print("="*70)
        print("Testing 3 methods × 6 sizes × 5 repeats = 90 runs...")
    
    pop_sizes = [5, 10, 20, 50, 100, 200]
    n_repeats = 5
    methods = ['direct', 'gp_interaction', 'simple_conjunctive']
    
    results = {
        'test_name': 'population_size_stability',
        'description': 'Test if separability converges with population size',
        'parameters': {
            'population_sizes': pop_sizes,
            'n_repeats': n_repeats,
            'methods': methods,
            'n_orientations': 20,
            'n_locations': 4
        },
        'method_results': {},
        'summary': {}
    }
    
    start_time = time.time()
    
    # Create progress bar for all runs
    total_runs = len(methods) * len(pop_sizes) * n_repeats
    pbar = tqdm(total=total_runs, desc="Population stability", 
                unit=" runs", disable=not verbose, ncols=100)
    
    for method in methods:
        method_data = {
            'method': method,
            'size_results': []
        }
        
        for n in pop_sizes:
            sep_values = []
            percent_mixed_values = []
            
            for rep in range(n_repeats):
                pbar.set_postfix_str(f'{method[:6]} n={n} r={rep+1}')
                
                gp = NeuralGaussianProcess(
                    n_orientations=20,
                    n_locations=4,
                    method=method,
                    seed=42 + rep
                )
                
                tuning = gp.sample_neurons(n)
                sep_stats = analyze_population_separability(tuning, show_progress=False)
                
                sep_values.append(sep_stats['mean'])
                percent_mixed_values.append(sep_stats['percent_mixed'])
                
                pbar.update(1)
            
            size_result = {
                'population_size': n,
                'separability': compute_statistics(sep_values),
                'percent_mixed': compute_statistics(percent_mixed_values),
                'raw_separability_values': sep_values,
                'raw_percent_mixed_values': percent_mixed_values
            }
            
            method_data['size_results'].append(size_result)
        
        results['method_results'][method] = method_data
    
    pbar.close()
    
    elapsed_time = time.time() - start_time
    
    # Summary statistics
    results['summary'] = {
        'elapsed_time_seconds': float(elapsed_time),
        'total_runs': len(methods) * len(pop_sizes) * n_repeats,
        'convergence_analysis': {}
    }
    
    # Check for convergence (variance reduction with size)
    for method in methods:
        size_results = results['method_results'][method]['size_results']
        variances = [r['separability']['std'] for r in size_results]
        
        results['summary']['convergence_analysis'][method] = {
            'variance_at_n5': variances[0],
            'variance_at_n200': variances[-1],
            'variance_reduction': variances[0] - variances[-1],
            'converged': bool(variances[-1] < 0.02)  # Threshold for convergence
        }
    
    if verbose:
        print(f"\n✓ Complete in {elapsed_time:.1f}s")
        print(f"  Convergence check:")
        for method, conv in results['summary']['convergence_analysis'].items():
            status = "✓" if conv['converged'] else "✗"
            print(f"    {method}: variance {conv['variance_at_n5']:.3f} → {conv['variance_at_n200']:.3f} {status}")
    
    return results


def test_stimulus_space_effect(verbose: bool = True) -> Dict[str, Any]:
    """
    Test how stimulus space dimensionality affects separability.
    
    Focuses on GP interaction method.
    """
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 3: STIMULUS DIMENSIONALITY EFFECT")
        print("="*70)
        print("Testing 4 configurations with varying dimensionality...")
    
    configs = [
        (10, 2),   # 20 total conditions
        (20, 4),   # 80 total conditions (baseline)
        (30, 6),   # 180 total conditions
        (40, 8),   # 320 total conditions
    ]
    
    results = {
        'test_name': 'stimulus_dimensionality',
        'description': 'Effect of stimulus space size on mixed selectivity',
        'parameters': {
            'configurations': [{'n_orientations': c[0], 'n_locations': c[1]} for c in configs],
            'method': 'gp_interaction',
            'n_neurons': 20,
            'seed': 42
        },
        'configuration_results': [],
        'summary': {}
    }
    
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(configs, desc="Dimensionality test", unit=" configs", 
                disable=not verbose, ncols=100)
    
    for n_ori, n_loc in pbar:
        pbar.set_postfix_str(f'{n_ori}×{n_loc}')
        
        gp = NeuralGaussianProcess(
            n_orientations=n_ori,
            n_locations=n_loc,
            method='gp_interaction',
            seed=42
        )
        
        tuning = gp.sample_neurons(20)
        sep_stats = analyze_population_separability(tuning, show_progress=False)
        
        result_entry = {
            'n_orientations': n_ori,
            'n_locations': n_loc,
            'total_conditions': n_ori * n_loc,
            'mean_separability': sep_stats['mean'],
            'std_separability': sep_stats['std'],
            'median_separability': sep_stats['median'],
            'percent_mixed': sep_stats['percent_mixed'],
            'passes_threshold': bool(sep_stats['mean'] < 0.8),
            'all_separability_values': sep_stats['all_values'].tolist()
        }
        
        results['configuration_results'].append(result_entry)
    
    elapsed_time = time.time() - start_time
    
    # Analyze trend
    separabilities = [r['mean_separability'] for r in results['configuration_results']]
    dimensions = [r['total_conditions'] for r in results['configuration_results']]
    
    # Linear regression to detect trend
    coeffs = np.polyfit(dimensions, separabilities, 1)
    trend_slope = float(coeffs[0])
    
    results['summary'] = {
        'elapsed_time_seconds': float(elapsed_time),
        'trend_analysis': {
            'slope': trend_slope,
            'interpretation': 'increasing' if trend_slope > 0.0001 else ('decreasing' if trend_slope < -0.0001 else 'flat'),
            'correlation': float(np.corrcoef(dimensions, separabilities)[0, 1])
        },
        'best_configuration': min(results['configuration_results'], key=lambda x: x['mean_separability']),
        'worst_configuration': max(results['configuration_results'], key=lambda x: x['mean_separability'])
    }
    
    if verbose:
        print(f"\n✓ Complete in {elapsed_time:.1f}s")
        print(f"  Trend: {results['summary']['trend_analysis']['interpretation']} (slope={trend_slope:.6f})")
        best = results['summary']['best_configuration']
        print(f"  Best: {best['n_orientations']}×{best['n_locations']} → sep={best['mean_separability']:.3f}")
    
    return results


def test_random_seed_stability(verbose: bool = True) -> Dict[str, Any]:
    """
    Test reproducibility across random seeds.
    
    Critical for understanding if results are robust or seed-dependent.
    """
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 4: RANDOM SEED REPRODUCIBILITY")
        print("="*70)
        print("Testing 3 methods × 20 seeds = 60 runs...")
    
    n_seeds = 20
    methods = ['direct', 'gp_interaction', 'simple_conjunctive']
    
    results = {
        'test_name': 'random_seed_stability',
        'description': 'Test reproducibility across different random seeds',
        'parameters': {
            'n_seeds': n_seeds,
            'seed_range': [0, n_seeds-1],
            'methods': methods,
            'n_neurons': 20,
            'n_orientations': 20,
            'n_locations': 4
        },
        'method_results': {},
        'summary': {}
    }
    
    start_time = time.time()
    
    # Create progress bar for all runs
    total_runs = len(methods) * n_seeds
    pbar = tqdm(total=total_runs, desc="Seed reproducibility", 
                unit=" runs", disable=not verbose, ncols=100)
    
    for method in methods:
        sep_values = []
        percent_mixed_values = []
        passes_threshold_count = 0
        
        seed_results = []
        
        for seed in range(n_seeds):
            pbar.set_postfix_str(f'{method[:6]} s={seed}')
            
            gp = NeuralGaussianProcess(
                n_orientations=20,
                n_locations=4,
                method=method,
                seed=seed
            )
            
            tuning = gp.sample_neurons(20)
            sep_stats = analyze_population_separability(tuning, show_progress=False)
            
            sep_values.append(sep_stats['mean'])
            percent_mixed_values.append(sep_stats['percent_mixed'])
            
            passes = sep_stats['mean'] < 0.8
            if passes:
                passes_threshold_count += 1
            
            seed_results.append({
                'seed': seed,
                'mean_separability': sep_stats['mean'],
                'percent_mixed': sep_stats['percent_mixed'],
                'passes_threshold': bool(passes)
            })
            
            pbar.update(1)
        
        results['method_results'][method] = {
            'method': method,
            'seed_results': seed_results,
            'separability_statistics': compute_statistics(sep_values),
            'percent_mixed_statistics': compute_statistics(percent_mixed_values),
            'passes_threshold_count': passes_threshold_count,
            'passes_threshold_percentage': 100 * passes_threshold_count / n_seeds,
            'raw_separability_values': sep_values,
            'raw_percent_mixed_values': percent_mixed_values
        }
    
    pbar.close()
    
    elapsed_time = time.time() - start_time
    
    results['summary'] = {
        'elapsed_time_seconds': float(elapsed_time),
        'total_runs': len(methods) * n_seeds,
        'robustness_ranking': []
    }
    
    # Rank methods by robustness (low variance + high pass rate)
    for method in methods:
        mr = results['method_results'][method]
        robustness_score = mr['passes_threshold_percentage'] / (1 + mr['separability_statistics']['std'])
        
        results['summary']['robustness_ranking'].append({
            'method': method,
            'robustness_score': float(robustness_score),
            'pass_rate': mr['passes_threshold_percentage'],
            'variance': mr['separability_statistics']['std']
        })
    
    results['summary']['robustness_ranking'].sort(key=lambda x: x['robustness_score'], reverse=True)
    
    if verbose:
        print(f"\n✓ Complete in {elapsed_time:.1f}s")
        print(f"  Pass rates:")
        for method in methods:
            mr = results['method_results'][method]
            print(f"    {method}: {mr['passes_threshold_count']}/{n_seeds} ({mr['passes_threshold_percentage']:.1f}%)")
    
    return results


def run_comprehensive_analysis(verbose: bool = True, run_all: bool = True) -> Dict[str, Any]:
    """
    Run all sensitivity analyses and compile comprehensive report.
    
    Args:
        verbose: Print progress information
        run_all: If False, run tests one at a time with user prompts
    """
    
    if verbose:
        print("\n" + "="*70)
        print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        print("\nRigorous testing protocol:")
        print("  1. Lengthscale sensitivity (42 configurations)")
        print("  2. Population size stability (90 runs)")
        print("  3. Stimulus dimensionality (4 configurations)")
        print("  4. Random seed reproducibility (60 runs)")
        print("\nTotal: 196 experiments")
        print("="*70 + "\n")
        
        if not run_all:
            input("Press Enter to start Test 1 (Lengthscale Sensitivity)...")
    
    analysis_start = time.time()
    
    # Run test 1
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST 1 / 4: LENGTHSCALE SENSITIVITY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    test1_results = test_lengthscale_sensitivity(verbose=verbose)
    
    if not run_all and verbose:
        input("\nPress Enter to continue to Test 2 (Population Stability)...")
    
    # Run test 2
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST 2 / 4: POPULATION SIZE STABILITY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    test2_results = test_population_size_effect(verbose=verbose)
    
    if not run_all and verbose:
        input("\nPress Enter to continue to Test 3 (Dimensionality)...")
    
    # Run test 3
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST 3 / 4: STIMULUS DIMENSIONALITY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    test3_results = test_stimulus_space_effect(verbose=verbose)
    
    if not run_all and verbose:
        input("\nPress Enter to continue to Test 4 (Seed Reproducibility)...")
    
    # Run test 4
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST 4 / 4: RANDOM SEED REPRODUCIBILITY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    test4_results = test_random_seed_stability(verbose=verbose)
    
    total_elapsed = time.time() - analysis_start
    
    # Compile comprehensive report
    comprehensive_results = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': 196,
            'total_elapsed_time_seconds': float(total_elapsed),
            'total_elapsed_time_human': f"{int(total_elapsed//60)}m {int(total_elapsed%60)}s"
        },
        'test_1_lengthscale_sensitivity': test1_results,
        'test_2_population_stability': test2_results,
        'test_3_dimensionality': test3_results,
        'test_4_seed_reproducibility': test4_results,
        'executive_summary': {}
    }
    
    # Generate executive summary
    exec_summary = {
        'key_findings': [],
        'recommendations': [],
        'critical_parameters': {}
    }
    
    # Finding 1: Optimal lengthscales
    optimal = test1_results['summary']['optimal_parameters']
    baseline = test1_results['summary']['baseline_parameters']
    exec_summary['key_findings'].append({
        'finding': 'optimal_lengthscales',
        'description': f"Optimal parameters: theta={optimal['theta_lengthscale']:.2f}, spatial={optimal['spatial_lengthscale']:.2f} → sep={optimal['mean_separability']:.3f}",
        'improvement_over_baseline': float(baseline['mean_separability'] - optimal['mean_separability'])
    })
    
    # Finding 2: Population convergence
    gp_convergence = test2_results['summary']['convergence_analysis']['gp_interaction']
    exec_summary['key_findings'].append({
        'finding': 'population_convergence',
        'description': f"Results converge at n=200 (variance: {gp_convergence['variance_at_n200']:.3f})",
        'converged': gp_convergence['converged']
    })
    
    # Finding 3: Dimensionality effect
    dim_trend = test3_results['summary']['trend_analysis']
    exec_summary['key_findings'].append({
        'finding': 'dimensionality_trend',
        'description': f"Separability trend is {dim_trend['interpretation']} with dimension",
        'correlation': dim_trend['correlation']
    })
    
    # Finding 4: Seed robustness
    gp_robustness = test4_results['method_results']['gp_interaction']
    exec_summary['key_findings'].append({
        'finding': 'seed_robustness',
        'description': f"GP interaction passes threshold in {gp_robustness['passes_threshold_percentage']:.1f}% of seeds",
        'robust': bool(gp_robustness['passes_threshold_percentage'] > 80)
    })
    
    # Recommendations
    if optimal['mean_separability'] < baseline['mean_separability']:
        exec_summary['recommendations'].append({
            'priority': 'HIGH',
            'recommendation': f"Use optimized lengthscales: theta={optimal['theta_lengthscale']:.2f}, spatial={optimal['spatial_lengthscale']:.2f}",
            'expected_improvement': f"{100*(baseline['mean_separability']-optimal['mean_separability'])/baseline['mean_separability']:.1f}% reduction in separability"
        })
    
    if dim_trend['interpretation'] == 'decreasing':
        best_dim = test3_results['summary']['best_configuration']
        exec_summary['recommendations'].append({
            'priority': 'MEDIUM',
            'recommendation': f"Use higher dimensionality: {best_dim['n_orientations']}×{best_dim['n_locations']} conditions",
            'rationale': 'Larger stimulus spaces produce stronger mixed selectivity'
        })
    
    if gp_robustness['passes_threshold_percentage'] < 80:
        exec_summary['recommendations'].append({
            'priority': 'HIGH',
            'recommendation': 'Current parameters are not robust across seeds',
            'rationale': f'Only {gp_robustness["passes_threshold_percentage"]:.1f}% pass rate - need parameter tuning'
        })
    
    comprehensive_results['executive_summary'] = exec_summary
    
    if verbose:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nTotal time: {int(total_elapsed//60)}m {int(total_elapsed%60)}s")
        print(f"Total experiments: 196")
        print("\nKey Findings:")
        for i, finding in enumerate(exec_summary['key_findings'], 1):
            print(f"  {i}. {finding['description']}")
        print("\nRecommendations:")
        for i, rec in enumerate(exec_summary['recommendations'], 1):
            print(f"  {i}. [{rec['priority']}] {rec['recommendation']}")
    
    return comprehensive_results


def save_results(results: Dict[str, Any], output_dir: str = 'data/results') -> None:
    """Save results to JSON files."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save timestamped full results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_path = Path(output_dir) / f'sensitivity_analysis_{timestamp}.json'
    
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n💾 Full results saved: {full_path}")
    
    # Save summary (overwrite latest)
    summary_path = Path(output_dir) / 'sensitivity_summary.json'
    
    summary = {
        'timestamp': results['analysis_metadata']['timestamp'],
        'executive_summary': results['executive_summary'],
        'test_summaries': {
            'lengthscale': results['test_1_lengthscale_sensitivity']['summary'],
            'population': results['test_2_population_stability']['summary'],
            'dimensionality': results['test_3_dimensionality']['summary'],
            'seed_stability': results['test_4_seed_reproducibility']['summary']
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"💾 Summary saved: {summary_path}")


if __name__ == '__main__':
    import sys
    
    # Check for interactive mode
    interactive = '--interactive' in sys.argv or '-i' in sys.argv
    
    print("\n" + "="*70)
    print("RIGOROUS SENSITIVITY ANALYSIS - DATA COLLECTION MODE")
    print("="*70)
    print("\nNo visualization - pure data collection")
    print("Results will be saved to JSON for later analysis")
    
    if interactive:
        print("\n[INTERACTIVE MODE] - Press Enter between tests")
    
    print("="*70 + "\n")
    
    # Suppress nested progress bars from GP
    import os
    os.environ['TQDM_DISABLE'] = '0'  # Keep main progress bars
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis(verbose=True, run_all=not interactive)
    
    # Save results
    save_results(results)
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ALL TESTS COMPLETE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print("\nResults saved and ready for analysis.")
    print("Use the JSON files for:")
    print("  - Statistical analysis")
    print("  - Visualization in separate scripts")
    print("  - Comparison with future runs")
    print("  - Publication-ready tables")