# Mixed Selectivity in Visual Working Memory

A computational investigation of mixed selectivity in visual working memory using Gaussian Processes, implementing biologically plausible neural population models for the prefrontal cortex.

## Quick Start

```python
from gaussian_process import NeuralGaussianProcess
from separability import analyze_population_separability

# Generate neurons with mixed selectivity (three methods available)
gp = NeuralGaussianProcess(
    n_orientations=20,
    n_locations=4,
    method='simple_conjunctive'  # or 'direct' or 'gp_interaction'
)

# Sample a population
neurons = gp.sample_neurons(n_neurons=50)

# Validate mixed selectivity
results = analyze_population_separability(neurons)
print(f"Mean separability: {results['mean']:.3f}")
print(f"Percent mixed: {results['percent_mixed']:.1f}%")
```

## Core Concept: Why Mixed Selectivity Matters

### The Problem
Visual working memory must bind features (orientation, location) into unified representations. Pure selectivity fails at this binding problem.

### The Solution: Mixed Selectivity
Neurons respond to **conjunctions** of features, not features independently:
- **Pure selectivity**: Response = f(orientation) × g(location) 
- **Mixed selectivity**: Response ≠ f(orientation) × g(location)

This non-factorizable response enables:
- Feature binding in working memory
- Context-dependent computation
- High-dimensional neural representations
- Solution to non-linearly separable problems

## Installation

```bash
# Clone repository
git clone <repository-url>
cd mixed_selectivity_vwm

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy tqdm matplotlib
```

## Three Methods for Generating Mixed Selectivity

### 1. Simple Conjunctive (Educational)
**Principle**: Preferred orientation shifts with location
```python
preferred_ori = base + rotation_rate * location
```
- Clearest demonstration of concept
- Minimal complexity
- Perfect for understanding fundamentals

### 2. Direct Construction (Experimental)
Four mechanisms ensuring non-separability:
- Rotating preferences across space
- Multiplicative feature interactions  
- XOR-like conjunction responses
- Phase-coupled oscillations

**Use for**: Testing algorithms, guaranteed mixed selectivity

### 3. GP Interaction (Biological)
Smooth, realistic tuning via:
- Location-varying orientation kernels
- Gain modulation mechanisms
- Conjunction hot-spots

**Use for**: Modeling real neural data

## Key Files

```
gaussian_process.py     # Core neural population generator
separability.py        # SVD-based mixed selectivity analysis
exp1_validation.py     # Validation experiments
kernels.py            # GP kernel implementations
```

## Understanding the Pipeline

### Step 1: Define Stimulus Space
Create grid of (orientation, location) combinations:
```python
Grid: 20 orientations × 4 locations = 80 conditions
```

### Step 2: Generate Neural Responses
Each method produces different response patterns:
- Simple: Location-dependent tuning
- Direct: Engineered interactions
- GP: Smooth probabilistic sampling

### Step 3: Measure Separability
Using SVD to quantify mixed selectivity:
```python
Separability = σ₁² / Σσᵢ²
```
- **≈ 1.0**: Pure selectivity (BAD)
- **< 0.8**: Mixed selectivity (TARGET)
- **< 0.6**: Strong mixed selectivity

### Step 4: Validate
```python
Success = mean(separability) < 0.8
```

## Experiments

### Experiment 1: Validation
```python
from exp1_validation import run_experiment1

results = run_experiment1(
    n_neurons=50,
    method='compare',  # Compare all methods
    plot=True
)
```

## Deep Technical Insights

### Why Product Kernels Create Mixed Selectivity
```python
k((θ,x), (θ',x')) = k_θ(θ,θ') × k_x(x,x')
```
Need BOTH features similar for high correlation → conjunction coding

### The Cholesky Transform
```python
K = L @ L^T
sample = L @ z  # z ~ N(0,I) → sample ~ N(0,K)
```
Transforms white noise into structured, correlated patterns

### Critical Data Transformation Chain
1. **Continuous → Discrete**: Make infinite finite (grid)
2. **Independent → Correlated**: Apply kernel structure
3. **Mathematical → Biological**: Ensure positive rates
4. **Vector → Matrix**: Impose 2D structure
5. **Matrix → Scalar**: Quantify separability

## Biological Relevance

### Prefrontal Cortex Properties Captured
- Heterogeneous tuning curves
- Gain modulation
- Location-dependent preferences
- Conjunction selectivity

### Working Memory Mechanisms
- **Binding**: "Red square on left" as unified representation
- **Context**: Same feature, different meaning by location
- **Flexibility**: Reconfigure for different tasks

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| n_orientations | 20 | Resolution of orientation space |
| n_locations | 4 | Number of spatial positions |
| n_neurons | 20-50 | Population size |
| theta_lengthscale | 0.3 | Orientation tuning width (GP) |
| spatial_lengthscale | 1.5 | Spatial tuning width (GP) |

## Method Selection Guide

| Goal | Method | Separability | Biological Realism |
|------|--------|--------------|-------------------|
| Understand concept | simple_conjunctive | ~0.6-0.7 | Medium |
| Test algorithms | direct | <0.6 | Low |
| Model real data | gp_interaction | ~0.7-0.8 | High |

## Common Issues

**Problem**: Separability too high (>0.8)
- Increase rotation rate
- Add more conjunction centers
- Decrease kernel lengthscales

**Problem**: Negative firing rates
- Apply softplus or ReLU
- Add baseline activity

**Problem**: Cholesky decomposition fails
- Add regularization: `K += 0.01 * I`
- Check kernel parameters

## Citation

If using this code, please cite:
```
Rigotti et al. (2013) "The importance of mixed selectivity in complex cognitive tasks" Nature
Fusi et al. (2016) "Why neurons mix: high dimensionality for higher cognition" Neuron
```

## License

MIT License - See LICENSE file

## Author

Yash Bharti - MPhil Student, Computational and Biological Learning Lab, Cambridge

## Further Reading

- Mixed selectivity theory: Rigotti et al. (2013) Nature
- High-dimensional coding: Fusi et al. (2016) Neuron
- Working memory mechanisms: Bouchacourt & Buschman (2019) Nature Neuroscience
- Gaussian Processes: Rasmussen & Williams (2006) MIT Press