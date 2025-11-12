"""
Experiment 1: Validation of mixed selectivity in synthetic neural populations.

This experiment tests whether the generated neural tuning curves exhibit genuine
mixed selectivity - i.e., non-separable conjunctive responses to orientation and
spatial location.

Scientific rationale:
    Mixed selectivity is a hallmark of flexible neural computation in prefrontal
    cortex and other brain regions. Neurons with mixed selectivity respond to
    conjunctions of features rather than single features in isolation, enabling
    high-dimensional representations that support complex cognitive operations.

Validation approach:
    We use Singular Value Decomposition (SVD) to quantify separability:
        Separability = σ₁² / Σᵢ σᵢ²
    
    - Separability ≈ 1.0: Response is separable (r(θ,L) ≈ f(θ)·g(L))
    - Separability < 0.8: True mixed selectivity (non-separable)
    
    Target: Mean population separability < 0.8

References:
    Rigotti et al. (2013) Nature
    Fusi et al. (2016) Neuron
"""

import numpy as np
from typing import Dict, Literal, Optional
import os
from pathlib import Path

# Use Plotly for elegant, publication-quality visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import from the actual location in your project
# Adjust this import path based on your project structure
try:
    from mixed_selectivity.core.gaussian_process import NeuralGaussianProcess
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.gaussian_process import NeuralGaussianProcess


def run_experiment1(
    n_neurons: int = 20,
    n_orientations: int = 20,
    n_locations: int = 4,
    theta_lengthscale: float = 0.3,
    spatial_lengthscale: float = 1.5,
    plot: bool = True,
    seed: int = 42,
    save_dir: str = 'figures/exp1',
    method: Literal['direct', 'gp_interaction', 'simple_conjunctive', 'compare'] = 'direct'
) -> Dict:
    """
    Run Experiment 1: Generate neural population and validate mixed selectivity.
    
    Experimental pipeline:
        1. Generate population with specified method
        2. Compute separability index for each neuron via SVD
        3. Test hypothesis: mean separability < 0.8
        4. Visualize results and save figures
    
    Args:
        n_neurons: Population size
        n_orientations: Number of orientation values in [-π, π]
        n_locations: Number of spatial locations
        theta_lengthscale: Orientation kernel width (GP method only)
        spatial_lengthscale: Spatial kernel width (GP method only)
        plot: Whether to generate and save figures
        seed: Random seed for reproducibility
        save_dir: Directory to save results
        method: Generation method or 'compare' to test all
    
    Returns:
        Dictionary containing:
            - tuning_curves: Neural responses (n_neurons, n_orientations, n_locations)
            - separability_stats: Statistical summary of separability indices
            - success: Boolean indicating if hypothesis test passed
            - method: Method used for generation
    """
    print("=" * 70)
    print("EXPERIMENT 1: VALIDATION OF MIXED SELECTIVITY")
    print("=" * 70)
    print(f"\nExperimental parameters:")
    print(f"  Population size: {n_neurons} neurons")
    print(f"  Stimulus space: {n_orientations} orientations × {n_locations} locations")
    print(f"  Generation method: {method}")
    print(f"  Random seed: {seed}")
    print(f"  Hypothesis: Mean separability < 0.8")
    
    if method == 'compare':
        # Compare all methods
        return _compare_methods(
            n_neurons, n_orientations, n_locations,
            theta_lengthscale, spatial_lengthscale,
            plot, seed, save_dir
        )
    
    # Standard single-method experiment
    return _run_single_method(
        n_neurons, n_orientations, n_locations,
        theta_lengthscale, spatial_lengthscale,
        method, plot, seed, save_dir
    )


def _run_single_method(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    theta_lengthscale: float,
    spatial_lengthscale: float,
    method: str,
    plot: bool,
    seed: int,
    save_dir: str
) -> Dict:
    """Execute experiment with a single generation method."""
    
    # ========================================
    # PHASE 1: GENERATE NEURAL POPULATION
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: NEURAL POPULATION GENERATION")
    print("=" * 70)
    
    gp = NeuralGaussianProcess(
        n_orientations=n_orientations,
        n_locations=n_locations,
        theta_lengthscale=theta_lengthscale,
        spatial_lengthscale=spatial_lengthscale,
        seed=seed,
        method=method
    )
    
    tuning_curves = gp.sample_neurons(n_neurons)
    
    print(f"\n✓ Generated population:")
    print(f"  Shape: {tuning_curves.shape}")
    print(f"  Mean activity: {tuning_curves.mean():.3f}")
    print(f"  Activity range: [{tuning_curves.min():.3f}, {tuning_curves.max():.3f}]")
    
    # ========================================
    # PHASE 2: SEPARABILITY ANALYSIS
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: SEPARABILITY ANALYSIS")
    print("=" * 70)
    print("\nComputing SVD-based separability for each neuron...")
    
    sep_results = analyze_population_separability(tuning_curves, show_progress=True)
    
    # Report statistics
    print(f"\nPopulation statistics:")
    print(f"  Mean separability:   {sep_results['mean']:.3f} ± {sep_results['std']:.3f}")
    print(f"  Median separability: {sep_results['median']:.3f}")
    print(f"  Range: [{sep_results['min']:.3f}, {sep_results['max']:.3f}]")
    print(f"  Neurons with mixed selectivity (<0.8): {sep_results['percent_mixed']:.1f}%")
    
    # ========================================
    # PHASE 3: HYPOTHESIS TEST
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 3: HYPOTHESIS TEST")
    print("=" * 70)
    
    threshold = 0.8
    success = sep_results['mean'] < threshold
    
    print(f"\nH₀: Mean separability < {threshold} (mixed selectivity)")
    print(f"Result: Mean separability = {sep_results['mean']:.3f}")
    
    # Add method-specific interpretation
    if method == 'gp_interaction':
        print(f"\nGP Interaction Method Notes:")
        print(f"  - Non-separability from parameter coupling (Phases 1-2)")
        print(f"  - No artificial conjunction injection")
        print(f"  - Expected range: 0.55-0.65")
        if sep_results['mean'] < 0.55:
            print(f"  ✓ Exceptionally strong non-separability achieved")
        elif sep_results['mean'] < 0.65:
            print(f"  ✓ Within expected range for pure parameter coupling")
        elif sep_results['mean'] < 0.8:
            print(f"  ✓ Moderate mixed selectivity (still passes threshold)")
        else:
            print(f"  ⚠️  Above expected range - consider adjusting lengthscales")
    
    if success:
        print(f"\n✓✓✓ HYPOTHESIS CONFIRMED")
        print(f"    Population exhibits mixed selectivity!")
        print(f"    {sep_results['percent_mixed']:.1f}% of neurons are non-separable")
    else:
        print(f"\n✗✗✗ HYPOTHESIS REJECTED")
        print(f"    Population shows predominantly separable tuning")
        print(f"    Only {sep_results['percent_mixed']:.1f}% of neurons are non-separable")
        
        # Provide diagnostic feedback
        if method in ['gp_interaction', 'simple_conjunctive']:
            print(f"\n    💡 Suggestion: Try method='direct' for guaranteed mixed selectivity")
    
    # ========================================
    # PHASE 4: VISUALIZATION
    # ========================================
    if plot:
        print("\n" + "=" * 70)
        print("PHASE 4: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _create_result_figures(tuning_curves, sep_results, save_dir, method)
        print(f"✓ Figures saved to: {save_dir}/")
    
    # ========================================
    # RETURN RESULTS
    # ========================================
    return {
        'tuning_curves': tuning_curves,
        'separability_stats': sep_results,
        'success': success,
        'method': method,
        'threshold': threshold
    }


def _compare_methods(
    n_neurons: int,
    n_orientations: int,
    n_locations: int,
    theta_lengthscale: float,
    spatial_lengthscale: float,
    plot: bool,
    seed: int,
    save_dir: str
) -> Dict:
    """Compare all available generation methods."""
    
    print("\n" + "=" * 70)
    print("COMPARISON MODE: TESTING ALL METHODS")
    print("=" * 70)
    
    methods = ['direct', 'gp_interaction', 'simple_conjunctive']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"  TESTING METHOD: {method.upper()}")
        print(f"{'='*70}")
        
        # Run with this method
        result = _run_single_method(
            n_neurons, n_orientations, n_locations,
            theta_lengthscale, spatial_lengthscale,
            method, plot=False, seed=seed + len(all_results),
            save_dir=f"{save_dir}/{method}"
        )
        
        all_results[method] = {
            'tuning_curves': result['tuning_curves'],
            'separability_stats': result['separability_stats'],
            'success': result['success'],
            'mean_sep': result['separability_stats']['mean'],
            'percent_mixed': result['separability_stats']['percent_mixed']
        }
    
    # Create comparison visualization
    if plot:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _create_comparison_figures(all_results, save_dir)
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    sorted_methods = sorted(all_results.items(), key=lambda x: x[1]['mean_sep'])
    
    print(f"\n{'Rank':<6} {'Method':<20} {'Mean Sep':<12} {'Mixed %':<12} {'Status'}")
    print("-" * 70)
    
    for rank, (method, r) in enumerate(sorted_methods, 1):
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{rank:<6} {method:<20} {r['mean_sep']:<12.3f} {r['percent_mixed']:<12.1f} {status}")
    
    best_method = sorted_methods[0][0]
    print(f"\n🏆 BEST METHOD: {best_method}")
    print(f"   Mean separability: {sorted_methods[0][1]['mean_sep']:.3f}")
    
    return {
        'all_results': all_results,
        'best_method': best_method,
        'success': sorted_methods[0][1]['success']
    }


def _create_result_figures(
    tuning_curves: np.ndarray,
    sep_results: Dict,
    save_dir: str,
    method: str
) -> None:
    """Create elegant, publication-quality result figures using Plotly."""
    
    # Color scheme
    COLORS = {
        'primary': '#2E86AB',
        'success': '#06A77D',
        'danger': '#D90429',
        'neutral': '#6C757D',
    }
    
    sep_values = sep_results['all_values']
    best_idx = np.argmin(sep_values)
    worst_idx = np.argmax(sep_values)
    n_neurons = len(sep_values)
    
    # =================================================================
    # FIGURE 1: Tuning Curves Comparison (Best vs Worst)
    # =================================================================
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'<b>Highest Mixed Selectivity</b><br><sub>Neuron {best_idx} | Separability = {sep_values[best_idx]:.3f}</sub>',
            f'<b>Lowest Mixed Selectivity</b><br><sub>Neuron {worst_idx} | Separability = {sep_values[worst_idx]:.3f}</sub>'
        ),
        horizontal_spacing=0.15
    )
    
    # Best neuron heatmap
    fig1.add_trace(
        go.Heatmap(
            z=tuning_curves[best_idx],
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title='<b>Firing<br>Rate</b>', x=0.45, len=0.85),
            hovertemplate='Location: %{x}<br>Orientation: %{y}<br>Rate: %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Worst neuron heatmap
    fig1.add_trace(
        go.Heatmap(
            z=tuning_curves[worst_idx],
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title='<b>Firing<br>Rate</b>', x=1.02, len=0.85),
            hovertemplate='Location: %{x}<br>Orientation: %{y}<br>Rate: %{z:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig1.update_xaxes(title_text='<b>Spatial Location</b>', row=1, col=1)
    fig1.update_xaxes(title_text='<b>Spatial Location</b>', row=1, col=2)
    fig1.update_yaxes(title_text='<b>Orientation Index</b>', row=1, col=1)
    fig1.update_yaxes(title_text='<b>Orientation Index</b>', row=1, col=2)
    
    fig1.update_layout(
        title=dict(
            text=f'<b>Neural Tuning Curves: Mixed vs. Separable Selectivity ({method})</b><br>' +
                 '<sub>Lower separability → stronger conjunctive coding of orientation × location</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
        width=1100,
        height=500,
        margin=dict(l=70, r=70, t=120, b=70)
    )
    
    filepath1 = Path(save_dir) / f'fig1_tuning_curves_{method}.html'
    fig1.write_html(str(filepath1))
    print(f"  ✓ Saved: {filepath1.name}")
    
    # =================================================================
    # FIGURE 2: Separability Distribution Histogram
    # =================================================================
    fig2 = go.Figure()
    
    fig2.add_trace(go.Histogram(
        x=sep_values,
        nbinsx=20,
        marker=dict(color=COLORS['primary'], line=dict(color='white', width=1)),
        opacity=0.8,
        hovertemplate='Separability: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Threshold line
    fig2.add_vline(
        x=0.8,
        line=dict(color=COLORS['danger'], width=3, dash='dash'),
        annotation=dict(
            text='Threshold (0.8)',
            textangle=-90,
            yshift=10,
            font=dict(size=11, color=COLORS['danger'])
        )
    )
    
    # Mean line
    fig2.add_vline(
        x=sep_results['mean'],
        line=dict(color=COLORS['success'], width=3),
        annotation=dict(
            text=f'Mean ({sep_results["mean"]:.3f})',
            textangle=-90,
            yshift=-10,
            font=dict(size=11, color=COLORS['success'])
        )
    )
    
    fig2.update_layout(
        title=dict(
            text=f'<b>Distribution of Separability Indices ({method})</b><br>' +
                 f'<sub>{sep_results["percent_mixed"]:.1f}% of neurons exhibit mixed selectivity (separability < 0.8)</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='<b>Separability Index</b>',
            gridcolor='lightgray',
            range=[max(0, sep_results['min']-0.05), min(1, sep_results['max']+0.05)]
        ),
        yaxis=dict(title='<b>Number of Neurons</b>', gridcolor='lightgray'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
        showlegend=False,
        width=900,
        height=550,
        margin=dict(l=70, r=70, t=110, b=70)
    )
    
    filepath2 = Path(save_dir) / f'fig2_separability_distribution_{method}.html'
    fig2.write_html(str(filepath2))
    print(f"  ✓ Saved: {filepath2.name}")
    
    # =================================================================
    # FIGURE 3: Population Scatter Plot
    # =================================================================
    mixed = sep_values < 0.8
    colors_scatter = [COLORS['success'] if m else COLORS['neutral'] for m in mixed]
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=np.arange(n_neurons),
        y=sep_values,
        mode='markers',
        marker=dict(
            size=11,
            color=colors_scatter,
            line=dict(width=1, color='white'),
            opacity=0.85
        ),
        text=[f'Neuron {i}<br>Sep: {s:.3f}<br>{"Mixed" if m else "Separable"}' 
              for i, (s, m) in enumerate(zip(sep_values, mixed))],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Threshold line
    fig3.add_hline(
        y=0.8,
        line=dict(color=COLORS['danger'], width=2, dash='dash'),
        annotation=dict(text='Threshold (0.8)', xanchor='left', x=0.02, 
                       font=dict(size=10, color=COLORS['danger']))
    )
    
    # Mean line
    fig3.add_hline(
        y=sep_results['mean'],
        line=dict(color=COLORS['primary'], width=2),
        annotation=dict(text=f'Mean ({sep_results["mean"]:.3f})', xanchor='left', x=0.02,
                       font=dict(size=10, color=COLORS['primary']))
    )
    
    fig3.update_layout(
        title=dict(
            text=f'<b>Separability Index Across Individual Neurons ({method})</b><br>' +
                 f'<sub>Population: {n_neurons} neurons | Mixed selectivity: {sep_results["percent_mixed"]:.1f}%</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(title='<b>Neuron Index</b>', gridcolor='lightgray'),
        yaxis=dict(
            title='<b>Separability Index</b>',
            gridcolor='lightgray',
            range=[max(0, sep_results['min']-0.05), min(1, sep_results['max']+0.05)]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
        showlegend=False,
        width=950,
        height=550,
        margin=dict(l=70, r=70, t=110, b=70)
    )
    
    filepath3 = Path(save_dir) / f'fig3_population_scatter_{method}.html'
    fig3.write_html(str(filepath3))
    print(f"  ✓ Saved: {filepath3.name}")
    
    # =================================================================
    # FIGURE 4: Summary Statistics Card
    # =================================================================
    fig4 = go.Figure()
    
    metrics = [
        ('Mean Separability', f"{sep_results['mean']:.3f} ± {sep_results['std']:.3f}"),
        ('Median Separability', f"{sep_results['median']:.3f}"),
        ('Range', f"[{sep_results['min']:.3f}, {sep_results['max']:.3f}]"),
        ('Mixed Neurons (<0.8)', f"{sep_results['percent_mixed']:.1f}%"),
        ('Separable Neurons (≥0.8)', f"{100-sep_results['percent_mixed']:.1f}%"),
    ]
    
    y_positions = np.linspace(0.8, 0.25, len(metrics))
    
    for (label, value), y_pos in zip(metrics, y_positions):
        fig4.add_annotation(
            text=f'<b>{label}:</b>',
            x=0.28, y=y_pos,
            xref='paper', yref='paper',
            xanchor='right',
            showarrow=False,
            font=dict(size=14, color=COLORS['neutral'])
        )
        
        color = COLORS['success'] if 'Mixed' in label else COLORS['primary']
        fig4.add_annotation(
            text=f'<b>{value}</b>',
            x=0.32, y=y_pos,
            xref='paper', yref='paper',
            xanchor='left',
            showarrow=False,
            font=dict(size=15, color=color)
        )
    
    # Interpretation
    interpretation = _get_interpretation(sep_results['mean'], sep_results['percent_mixed'], method)
    fig4.add_annotation(
        text=f'<i>{interpretation}</i>',
        x=0.5, y=0.08,
        xref='paper', yref='paper',
        xanchor='center',
        showarrow=False,
        font=dict(size=11, color=COLORS['neutral']),
        bgcolor='rgba(248, 249, 250, 0.9)',
        bordercolor=COLORS['primary'],
        borderwidth=1.5,
        borderpad=12
    )
    
    fig4.update_layout(
        title=dict(
            text=f'<b>Population Statistics Summary</b><br><sub>Method: {method}</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor='white',
        paper_bgcolor='#F8F9FA',
        font=dict(family='Arial'),
        width=750,
        height=550,
        margin=dict(l=50, r=50, t=90, b=50)
    )
    
    filepath4 = Path(save_dir) / f'fig4_summary_statistics_{method}.html'
    fig4.write_html(str(filepath4))
    print(f"  ✓ Saved: {filepath4.name}")


def _create_comparison_figures(results: Dict, save_dir: str) -> None:
    """Create elegant side-by-side comparison of all methods using Plotly."""
    
    COLORS = {
        'direct': '#06A77D',
        'gp_interaction': '#2E86AB',
        'simple_conjunctive': '#A23B72',
        'threshold': '#D90429',
    }
    
    methods = list(results.keys())
    
    # =================================================================
    # FIGURE: Method Comparison
    # =================================================================
    fig = make_subplots(
        rows=2, cols=len(methods),
        subplot_titles=[f'<b>{m}</b><br><sub>Sep={results[m]["mean_sep"]:.3f}</sub>' 
                       for m in methods],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
        specs=[[{'type': 'xy'}]*len(methods), [{'type': 'xy'}]*len(methods)]
    )
    
    # Row 1: Histograms for each method
    for i, method in enumerate(methods, 1):
        sep_vals = results[method]['separability_stats']['all_values']
        
        fig.add_trace(
            go.Histogram(
                x=sep_vals,
                nbinsx=15,
                marker=dict(color=COLORS.get(method, '#6C757D'), 
                           line=dict(color='white', width=1)),
                opacity=0.8,
                showlegend=False,
                hovertemplate='Sep: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=i
        )
        
        # Add threshold line
        fig.add_vline(
            x=0.8,
            line=dict(color=COLORS['threshold'], width=2, dash='dash'),
            row=1, col=i
        )
        
        # Add mean line
        fig.add_vline(
            x=results[method]['mean_sep'],
            line=dict(color='#06A77D', width=2),
            row=1, col=i
        )
        
        fig.update_xaxes(title_text='Separability', row=1, col=i, range=[0.3, 0.9])
        fig.update_yaxes(title_text='Count' if i == 1 else '', row=1, col=i)
    
    # Row 2: Bar chart comparison
    x_pos = np.arange(len(methods))
    mean_seps = [results[m]['mean_sep'] for m in methods]
    percent_mixed = [results[m]['percent_mixed']/100 for m in methods]
    
    for i, (method, mean_sep, pct) in enumerate(zip(methods, mean_seps, percent_mixed)):
        # Mean separability bars
        fig.add_trace(
            go.Bar(
                x=[method],
                y=[mean_sep],
                name='Mean Separability' if i == 0 else '',
                marker=dict(color=COLORS.get(method, '#6C757D')),
                opacity=0.7,
                showlegend=(i == 0),
                hovertemplate=f'{method}<br>Mean Sep: {mean_sep:.3f}<extra></extra>',
                legendgroup='sep'
            ),
            row=2, col=1
        )
        
        # Percent mixed bars
        fig.add_trace(
            go.Bar(
                x=[method],
                y=[pct],
                name='% Mixed (normalized)' if i == 0 else '',
                marker=dict(color='coral'),
                opacity=0.7,
                showlegend=(i == 0),
                hovertemplate=f'{method}<br>Mixed: {pct*100:.1f}%<extra></extra>',
                legendgroup='mixed'
            ),
            row=2, col=1
        )
    
    # Threshold line on bar chart
    fig.add_hline(
        y=0.8,
        line=dict(color=COLORS['threshold'], width=2, dash='dash'),
        row=2, col=1
    )
    
    # Hide other bar chart columns
    for col in range(2, len(methods)+1):
        fig.update_xaxes(visible=False, row=2, col=col)
        fig.update_yaxes(visible=False, row=2, col=col)
    
    fig.update_xaxes(title_text='<b>Method</b>', row=2, col=1)
    fig.update_yaxes(title_text='<b>Value</b>', row=2, col=1, range=[0, 1])
    
    fig.update_layout(
        title=dict(
            text='<b>Method Comparison: Mixed Selectivity Performance</b><br>' +
                 '<sub>Comparing different neural population generation methods</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=10),
        barmode='group',
        width=350*len(methods),
        height=900,
        margin=dict(l=70, r=70, t=120, b=70)
    )
    
    filepath = Path(save_dir) / 'method_comparison.html'
    fig.write_html(str(filepath))
    print(f"  ✓ Saved: {filepath.name}")


def _get_interpretation(mean_sep: float, percent_mixed: float, method: str = '') -> str:
    """Generate interpretation text based on results."""
    if mean_sep < 0.5:
        text = (f"Strong non-separability across population.\n"
                f"      Neurons exhibit robust mixed selectivity,\n"
                f"      suitable for flexible computation.")
    elif mean_sep < 0.65:
        text = (f"Moderate-strong mixed selectivity detected.\n"
                f"      Population shows conjunctive encoding\n"
                f"      of orientation and location.")
    elif mean_sep < 0.8:
        text = (f"Moderate mixed selectivity detected.\n"
                f"      Population shows conjunctive encoding\n"
                f"      with some separable neurons.")
    else:
        text = (f"Population shows predominantly separable tuning.\n"
                f"      Limited evidence of mixed selectivity.\n"
                f"      Consider using 'direct' method.")
    
    # Add method-specific note
    if method == 'gp_interaction' and mean_sep < 0.7:
        text += f"\n\n      Note: GP method achieved non-separability\n"
        text += f"      through parameter coupling alone\n"
        text += f"      (no artificial conjunction injection)."
    
    return text


# ========================================
# UTILITY: Separability analysis function
# ========================================

def analyze_population_separability(
    tuning_curves: np.ndarray,
    show_progress: bool = False
) -> Dict:
    """
    Analyze separability of neural tuning curves using SVD.
    
    For each neuron, compute:
        Separability = σ₁² / Σᵢ σᵢ²
    
    where σᵢ are singular values of the (orientations × locations) matrix.
    
    Args:
        tuning_curves: Array of shape (n_neurons, n_orientations, n_locations)
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with statistics: mean, std, median, min, max, 
        percent_mixed, all_values
    """
    n_neurons = tuning_curves.shape[0]
    separability_values = np.zeros(n_neurons)
    
    iterator = range(n_neurons)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Computing separability")
        except ImportError:
            pass
    
    for i in iterator:
        # Compute SVD
        U, S, Vt = np.linalg.svd(tuning_curves[i], full_matrices=False)
        
        # Separability = variance explained by first component
        separability_values[i] = (S[0]**2) / (np.sum(S**2) + 1e-10)
    
    # Compute statistics
    threshold = 0.8
    percent_mixed = 100 * np.mean(separability_values < threshold)
    
    return {
        'mean': np.mean(separability_values),
        'std': np.std(separability_values),
        'median': np.median(separability_values),
        'min': np.min(separability_values),
        'max': np.max(separability_values),
        'percent_mixed': percent_mixed,
        'all_values': separability_values
    }


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == '__main__':
    """
    Example usage:
    
    # Test single method
    results = run_experiment1(
        n_neurons=50,
        n_orientations=20,
        n_locations=4,
        method='gp_interaction',
        seed=42
    )
    
    # Compare methods
    results = run_experiment1(
        n_neurons=50,
        method='compare',
        seed=42
    )
    """
    
    print("\nExample: Running experiment with GP interaction method...")
    results = run_experiment1(
        n_neurons=20,
        n_orientations=20,
        n_locations=4,
        method='gp_interaction',
        plot=True,
        seed=42
    )
    
    print(f"\n✓ Experiment complete!")
    print(f"  Success: {results['success']}")
    print(f"  Mean separability: {results['separability_stats']['mean']:.3f}")