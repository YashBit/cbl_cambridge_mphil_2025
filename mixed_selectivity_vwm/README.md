# Mixed Selectivity in Visual Working Memory

A computational investigation of mixed selectivity in visual working memory using Gaussian Processes.

## Project Structure
```
mixed_selectivity_vwm/
├── mixed_selectivity/      # Main package code
├── notebooks/              # Jupyter notebooks for experiments
├── scripts/                # Standalone execution scripts  
├── tests/                  # Unit tests
├── data/                   # Data storage
├── figures/                # Generated figures
├── configs/                # Configuration files
└── docs/                   # Documentation
```

## Installation
```bash
# Clone the repository
git clone <repository-url>
cd mixed_selectivity_vwm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Quick Start
```python
from mixed_selectivity.core import NeuralGaussianProcess
from mixed_selectivity.experiments import run_experiment1

# Run first experiment
results = run_experiment1(config_path='configs/default_config.yaml')
```

## Experiments

1. **Experiment 1**: Validation of Mixed Selectivity
2. **Experiment 2**: Behavioral Error Patterns
3. **Experiment 3**: Fisher Information Analysis
4. **Experiment 4**: Comparison with Pure Selectivity

## Authors

Yash Bharti - MPhil Student, CBL Cambridge

## License

MIT License - see LICENSE file for details


# Deep Technical Rationale for Each Pipeline Step

Let me explain the fundamental technical necessity and implications of each step in our pipeline.

## **Step 1: PARAMETERS**
### Why These Specific Numbers?

**Technical Necessity:**
- **20 neurons**: Minimum for statistical validity while computationally tractable
- **20×4 grid**: Balances resolution vs. computational cost (80×80 matrix operations)

**Core Implication:**
```python
# Matrix sizes cascade through pipeline:
Grid points = 20 × 4 = 80
Kernel matrix = 80 × 80 = 6,400 elements
Covariance storage = O(n²) memory
Cholesky = O(n³) computation
```

**Without this step**: Can't define problem scope; infinite dimensional space is computationally intractable.

---

## **Step 2: INPUT GRID**
### Why Discretize Continuous Space?

**Technical Necessity:**
Gaussian Processes are defined on continuous spaces, but computers can only handle discrete representations.

```python
# The mathematical GP:
f ~ GP(μ(θ,x), k((θ,x), (θ',x')))  # Infinite dimensional

# The computational reality:
f_discrete ~ N(μ_vec, K_matrix)  # Finite 80-dimensional
```

**Core Implication:**
The grid IS our function representation. We're not evaluating a function at points; the values AT these points ARE the function.

**Without this step**: Cannot represent functions on computers; GP would remain abstract mathematical object.

---

## **Step 3: KERNEL MATRIX**
### Why Compute All Pairwise Correlations?

**Technical Necessity:**
The kernel matrix IS the covariance structure of our multivariate Gaussian:

```python
# Mathematically:
Cov[f(x_i), f(x_j)] = k(x_i, x_j)

# Product kernel creates conjunction:
k((θ,x), (θ',x')) = k_θ(θ,θ') × k_x(x,x')

# This multiplication is KEY:
If k_θ = 0.9 (similar orientation) and k_x = 0.1 (different location)
Then k_total = 0.09 (low correlation)
# Need BOTH features similar for high correlation → mixed selectivity
```

**Core Implication:**
The kernel matrix completely determines the statistical dependencies between all points. This IS where mixed selectivity is encoded.

**Without this step**: No correlation structure; would get independent random values at each point (white noise, not smooth functions).

---

## **Step 4: CHOLESKY DECOMPOSITION**
### Why This Specific Matrix Factorization?

**Technical Necessity:**
We need to sample from N(0, K), but computers can only generate N(0, I).

```python
# The problem:
Want: samples ~ N(0, K)
Have: z ~ N(0, I)

# The solution via Cholesky:
K = L @ L^T  (Cholesky decomposition)
If z ~ N(0, I), then:
Lz ~ N(0, L @ I @ L^T) = N(0, L @ L^T) = N(0, K) ✓
```

**Core Implication:**
L encodes the "correlation structure" in a form we can use for sampling. It's like a "correlation filter" that transforms independent noise into structured patterns.

**Without this step**: Cannot generate correlated samples; would need to use much slower eigendecomposition or couldn't sample at all.

---

## **Step 5: RANDOM SAMPLING**
### Why Transform Random Noise?

**Technical Necessity:**
This is where individual neurons get their unique identities while sharing structure:

```python
# Each neuron:
z_neuron_i = [0.3, -1.2, 0.8, ...]  # Different random values
sample_i = L @ z_neuron_i           # Same L (shared structure)

# The genius: L (from kernel) ensures:
- Smooth tuning curves (nearby points correlated)
- Conjunctive responses (product kernel structure)
- But each z makes each neuron unique
```

**Core Implication:**
The randomness creates heterogeneity; the transformation through L creates structure. This mimics biology: neurons vary but follow organizational principles.

**Without this step**: Either all neurons identical (no z variation) or all neurons random (no L structure).

---

## **Step 6: NEURAL TUNING CURVES**
### Why Reshape and Apply Softplus?

**Technical Necessity:**
```python
# From math to biology:
sample = [-0.5, 1.2, -0.3, ...]  # Can be negative (GP gives real values)
firing_rates = softplus(sample)   # Must be positive (neurons don't fire negative)

# Reshape: vector → matrix
flat_vector[80] → matrix[20,4]
# This imposes 2D structure on 1D sample
```

**Core Implication:**
This transforms abstract mathematical objects into biological quantities (firing rates). The reshape is crucial: it defines which dimensions of our vector correspond to orientation vs. location.

**Without this step**: Would have negative "firing rates" (non-biological) and no 2D structure for analysis.

---

## **Step 7: SVD ANALYSIS**
### Why Singular Value Decomposition Specifically?

**Technical Necessity:**
SVD provides the optimal low-rank approximation (Eckart-Young theorem):

```python
# For matrix M:
M = U @ S @ V^T

# Rank-1 approximation:
M_rank1 = s[0] * u[:,0] @ v[:,0]^T

# If separable (pure selectivity):
M = f(θ) @ g(x)^T  # Already rank-1!
# So s[1], s[2], ... ≈ 0

# If non-separable (mixed):
# Need multiple components: s[0], s[1], s[2] all significant
```

**Core Implication:**
SVD quantifies the "intrinsic dimensionality" of the tuning surface. This directly tests our hypothesis about mixed selectivity.

**Without this step**: No quantitative measure of mixed vs. pure selectivity; couldn't validate our core claim.

---

## **Step 8: POPULATION STATISTICS**
### Why Aggregate Across Neurons?

**Technical Necessity:**
Single neurons are noisy samples; population statistics reveal the true distribution:

```python
# Individual neurons vary:
sep = [0.73, 0.45, 0.82, 0.61, ...]  # Noisy samples

# Population mean converges to true value:
mean(sep) → E[separability|kernel_parameters]  # Stable estimate
```

**Core Implication:**
The mean tells us about the kernel's effect; the variance tells us about heterogeneity. Both are biologically important.

**Without this step**: Can't make population-level claims; individual variation might mask systematic effects.

---

## **Step 9: SUCCESS CRITERION**
### Why Threshold at 0.8?

**Technical Necessity:**
Based on empirical studies of PFC neurons and mathematical analysis:

```python
# Pure selectivity (theoretical):
sep = 1.0  # Perfect factorization

# Real PFC neurons (empirical):
sep ≈ 0.4-0.7  # Strong mixed selectivity

# Our threshold:
sep < 0.8  # Clear departure from pure selectivity
```

**Core Implication:**
This validates that our method produces biologically realistic neurons, not mathematical artifacts.

---

## **The Deep Technical Chain of Necessity**

Each step MUST happen because:

1. **GP Theory → Discrete Implementation**: Steps 1-2 make infinite finite
2. **Statistical Structure → Correlation**: Steps 3-4 encode mixed selectivity  
3. **Mathematical → Biological**: Steps 5-6 create realistic neurons
4. **Hypothesis → Test**: Steps 7-9 validate our claims

**The key insight**: The product kernel in Step 3 propagates through the entire pipeline to create mixed selectivity in Step 7. Every other step exists to enable this core transformation.

Without ANY of these steps, the pipeline fails either:
- **Computationally** (can't be implemented)
- **Mathematically** (wrong distribution)
- **Biologically** (unrealistic results)
- **Scientifically** (can't test hypothesis)