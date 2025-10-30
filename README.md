# Comprehensive README: Mixed Selectivity Population Coding Model for Visual Working Memory

## Table of Contents
1. [Project Overview](#project-overview)
2. [Background & Motivation](#background--motivation)
3. [Bays (2014) Population Coding Model](#bays-2014-population-coding-model)
4. [Our Extension: Mixed Selectivity Model](#our-extension-mixed-selectivity-model)
5. [Mathematical Framework](#mathematical-framework)
6. [Gaussian Processes for Neural Tuning](#gaussian-processes-for-neural-tuning)
7. [Implementation Guide](#implementation-guide)
8. [Literature & Resources](#literature--resources)

---

## Project Overview

This project extends the Bays (2014) population coding model of visual working memory to incorporate **mixed selectivity** as observed in prefrontal cortex. Rather than having independent subpopulations encoding each stimulus location, we model a unified population where all neurons respond to all stimuli through conjunctive tuning.

### Key Features
- **Mixed selectivity**: Neurons encode multiple stimulus features simultaneously
- **Divisive normalization**: Maintains constant total population activity
- **Poisson variability**: Stochastic spike generation introduces errors
- **Gaussian Process framework**: Flexible, principled approach to modeling tuning curves

---

## Background & Motivation

### The Problem: Visual Working Memory Capacity Limits

Visual working memory (VWM) exhibits systematic capacity limitations:
- **Recall precision decreases** as the number of stored items increases
- **Error distributions** deviate from normality (excess kurtosis, heavy tails)
- **Attention modulates** storage precision through gain control

### Current Models

**Bays (2014) Model:**
- ✅ Accounts for error patterns through Poisson noise in neural populations
- ✅ Implements divisive normalization for capacity limits
- ❌ Assumes **independent subpopulations** per stimulus location
- ❌ Does not capture **mixed selectivity** observed in prefrontal cortex

**Limitation:** Prefrontal cortex neurons show **conjunctive coding** - responding to combinations of features across multiple items, not just one stimulus in isolation.

---

## Bays (2014) Population Coding Model

### Architecture

Each stimulus location $j$ is encoded by an **independent subpopulation** of $M$ neurons with orientation tuning:

```latex
f_{ij}(\theta_j) = \exp\left(\omega^{-1}(\cos(\phi_{ij} - \theta_j) - 1)\right)
```

Where:
- $\theta_j$ = orientation of stimulus at location $j$
- $\phi_{ij}$ = preferred orientation of neuron $i$ in subpopulation $j$
- $\omega$ = tuning width parameter

### Divisive Normalization

Post-normalization firing rate:

```latex
r_{ij}(\theta_j) = \frac{\gamma \cdot \alpha_j \cdot f_{ij}(\theta_j)}{\sum_{m,n} \alpha_n f_{mn}(\theta_n)}
```

Where:
- $\gamma$ = total population gain (constant)
- $\alpha_j$ = attentional gain factor for location $j$

**Result:** As $N$ (number of stimuli) increases, activity per stimulus decreases → increased variability.

### Poisson Spiking

```latex
P(n_{ij} | \theta_j, T) = \frac{(r_{ij} T)^{n_{ij}}}{n_{ij}!} \exp(-r_{ij} T)
```

### Maximum Likelihood Decoding

```latex
\hat{\theta}_p = \arg\max_{\theta_p} \sum_{i=1}^{M} n_{ip} \log(r_{ip}(\theta_p))
```

### Key Results

1. **Non-normal error distributions** emerge naturally at low signal-to-noise ratios
2. **Supralinear variance scaling** with memory load (exponent > 1)
3. **Optimal attention allocation** matches human behavior

---

## Our Extension: Mixed Selectivity Model

### Core Insight

Instead of independent subpopulations, **all neurons respond to all stimuli** through mixed selectivity.

### Key Differences from Bays

| Aspect | Bays (2014) | Our Model |
|--------|-------------|-----------|
| **Architecture** | Independent subpopulations per location | Unified population |
| **Neuron selectivity** | Pure (one location) | Mixed (all locations) |
| **Tuning** | Fixed bell-shaped curves | GP-sampled or heterogeneous |
| **Biological realism** | Sensory cortex | Prefrontal cortex |

### Why Mixed Selectivity?

**Empirical observations in PFC:**
- Neurons encode **conjunctions** of task variables (item identity × feature × location)
- Enables **flexible, context-dependent** computation
- Higher **representational capacity** than pure selectivity

**Reference:** Rigotti et al. (2013) - "The importance of mixed selectivity in complex cognitive tasks"

---

## Mathematical Framework

### Model Equations

#### 1. Driving Input (Mixed Selectivity)

Each neuron $i$ receives input from **all** stimuli:

```latex
f_i(\boldsymbol{\theta}, \mathbf{x}) = \sum_{j=1}^{N} w_{ij} \exp\left(\omega^{-1}(\cos(\phi_i - \theta_j) - 1)\right) g_i(\mathbf{x}_j)
```

**Parameters:**
- $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_N)$ = orientations of all stimuli
- $\mathbf{x} = (\mathbf{x}_1, \ldots, \mathbf{x}_N)$ = locations of all stimuli
- $w_{ij}$ = connection weight from stimulus $j$ to neuron $i$
- $g_i(\mathbf{x}_j)$ = spatial tuning of neuron $i$ to location $\mathbf{x}_j$

#### 2. Divisive Normalization

```latex
r_i(\boldsymbol{\theta}, \mathbf{x}) = \frac{\gamma \cdot \alpha_i \cdot f_i(\boldsymbol{\theta}, \mathbf{x})}{\sum_{k=1}^{M} f_k(\boldsymbol{\theta}, \mathbf{x}) + \sigma}
```

**Constraint:**
```latex
\sum_{i=1}^{M} r_i(\boldsymbol{\theta}, \mathbf{x}) = \Lambda = \text{constant}
```

#### 3. Poisson Spiking

```latex
n_i \sim \text{Poisson}(r_i \cdot T_d)
```

#### 4. Maximum Likelihood Decoding

```latex
\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^{M} P(n_i | \boldsymbol{\theta}, \mathbf{x}, T_d)
```

---

## Gaussian Processes for Neural Tuning

### What are Gaussian Processes?

A **Gaussian Process (GP)** is a distribution over functions, providing a principled way to model neural tuning curves without specifying their functional form.

**Definition:**
```latex
f(\theta) \sim \mathcal{GP}(m(\theta), k(\theta, \theta'))
```

Where:
- $m(\theta)$ = mean function (often 0)
- $k(\theta, \theta')$ = covariance (kernel) function

### Key Property: Smoothness & Correlation

**The kernel determines how neural responses covary:**

```latex
k(\theta, \theta') = \sigma^2 \exp\left(-\frac{2\sin^2((\theta - \theta')/2)}{\ell^2}\right)
```

**Interpretation:**
- Neurons with **similar preferred orientations** have **correlated tuning**
- $\ell$ (length scale) controls tuning width
- Naturally captures **heterogeneous tuning** across population

### Why GPs for Mixed Selectivity?

#### 1. **Natural Conjunctive Coding**

Multi-dimensional kernel for mixed selectivity:

```latex
k((\theta_j, \mathbf{x}_j), (\theta_{j'}, \mathbf{x}_{j'})) = k_\theta(\theta_j, \theta_{j'}) \cdot k_\mathbf{x}(\mathbf{x}_j, \mathbf{x}_{j'})
```

This automatically generates neurons that respond to **combinations** of orientation and location.

#### 2. **Principled Uncertainty Quantification**

GPs provide both **mean prediction** and **uncertainty**:

```latex
f(\theta^*) \sim \mathcal{N}(\mu(\theta^*), \sigma^2(\theta^*))
```

**Connection to decoding:** The GP posterior variance directly relates to your **error distributions** and Fisher Information!

```latex
\text{Var}[\hat{\theta}] \geq \frac{1}{I(\theta)}
```

#### 3. **Flexibility: Learn Tuning from Data**

Unlike fixed tuning curves:
```latex
f_i(\theta) = \exp(\omega^{-1}(\cos(\phi_i - \theta) - 1))  \quad \text{(fixed)}
```

GPs **adapt** to the structure of neural responses:
```latex
\mathbf{f} \sim \mathcal{GP}(0, K)  \quad \text{(learned)}
```

#### 4. **Models Noise Correlations**

Bays (2014) models short-range correlations as:
```latex
c_{i,j} = c_0 \exp(-|\phi_i - \phi_j|)
```

**This is exactly a GP kernel!** GPs naturally incorporate correlated noise.

### Key Advantages Summary

| Feature | Benefit |
|---------|---------|
| **Smooth tuning curves** | Biological realism |
| **Heterogeneous widths** | Captures PFC diversity |
| **Principled uncertainty** | Links to Fisher Information |
| **Conjunctive coding** | Natural mixed selectivity |
| **Correlation structure** | Models noise dependencies |
| **Non-parametric** | Flexible, data-driven |

---

## Implementation Guide

### Step 1: Install Dependencies

```bash
pip install torch gpytorch numpy matplotlib scipy
```

### Step 2: Define GP Model for Mixed Selectivity

```python
import gpytorch
import torch
import numpy as np

class MixedSelectivityGP(gpytorch.models.ExactGP):
    """
    Gaussian Process model for neural tuning functions with mixed selectivity.
    
    Each neuron's tuning is drawn from a GP with a product kernel over:
    - Orientation (periodic kernel)
    - Location (RBF kernel)
    """
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        
        # Mean function (typically zero for neural tuning)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Product kernel for mixed selectivity
        # k((θ, x), (θ', x')) = k_θ(θ, θ') * k_x(x, x')
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(
                # Periodic kernel for orientation (circular variable)
                gpytorch.kernels.PeriodicKernel(),
                # RBF kernel for spatial location
                gpytorch.kernels.RBFKernel()
            )
        )
    
    def forward(self, x):
        """
        Forward pass: compute GP prior.
        
        Args:
            x: [n_points, 2] tensor of (orientation, location) pairs
        
        Returns:
            MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

### Step 3: Sample Neural Tuning Functions

```python
def sample_tuning_curves(n_neurons=100, n_orientations=50, n_locations=8):
    """
    Sample tuning functions for population of neurons.
    
    Args:
        n_neurons: Number of neurons in population
        n_orientations: Resolution of orientation grid
        n_locations: Number of stimulus locations
    
    Returns:
        tuning_curves: [n_neurons, n_orientations, n_locations] tensor
    """
    
    # Create grid of inputs
    theta_grid = torch.linspace(-np.pi, np.pi, n_orientations)
    x_grid = torch.linspace(0, 2*np.pi, n_locations)  # locations on circle
    
    # Cartesian product: all (theta, x) combinations
    theta_mesh, x_mesh = torch.meshgrid(theta_grid, x_grid, indexing='ij')
    inputs = torch.stack([theta_mesh.flatten(), x_mesh.flatten()], dim=-1)
    
    # Initialize GP
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # Use dummy training data (prior sampling)
    train_x = torch.zeros(1, 2)
    train_y = torch.zeros(1)
    model = MixedSelectivityGP(train_x, train_y, likelihood)
    
    # Set to eval mode for sampling
    model.eval()
    likelihood.eval()
    
    # Sample tuning functions from GP prior
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model(inputs)
        # Sample n_neurons functions
        f_samples = f_dist.sample(sample_shape=torch.Size([n_neurons]))
    
    # Reshape to [n_neurons, n_orientations, n_locations]
    tuning_curves = f_samples.reshape(n_neurons, n_orientations, n_locations)
    
    # Apply softplus to ensure non-negativity (firing rates > 0)
    tuning_curves = torch.nn.functional.softplus(tuning_curves)
    
    return tuning_curves, theta_grid, x_grid
```

### Step 4: Implement Divisive Normalization

```python
def divisive_normalize(f, gamma=100.0, sigma=1e-6):
    """
    Apply divisive normalization to driving inputs.
    
    Args:
        f: [n_neurons, n_stimuli] driving inputs
        gamma: Total population gain (Hz)
        sigma: Semi-saturation constant
    
    Returns:
        r: [n_neurons, n_stimuli] normalized firing rates
    """
    # Sum across all neurons (normalization pool)
    total_input = f.sum(dim=0, keepdim=True) + sigma
    
    # Normalized firing rate
    r = gamma * f / total_input
    
    return r
```

### Step 5: Generate Poisson Spikes

```python
def generate_spikes(r, T_decode=0.1):
    """
    Generate Poisson spike counts from firing rates.
    
    Args:
        r: [n_neurons, n_stimuli] firing rates (Hz)
        T_decode: Decoding window duration (seconds)
    
    Returns:
        n: [n_neurons, n_stimuli] spike counts
    """
    # Poisson parameter: lambda = rate * time
    lam = r * T_decode
    
    # Sample from Poisson distribution
    n = torch.poisson(lam)
    
    return n
```

### Step 6: Maximum Likelihood Decoding

```python
def ml_decode(n, tuning_curves, theta_grid, stimulus_idx, T_decode=0.1):
    """
    Decode stimulus orientation using maximum likelihood.
    
    Args:
        n: [n_neurons] observed spike counts
        tuning_curves: [n_neurons, n_orientations, n_locations]
        theta_grid: [n_orientations] orientation values
        stimulus_idx: Which stimulus location to decode
        T_decode: Decoding window duration
    
    Returns:
        theta_hat: Decoded orientation (radians)
    """
    # Get tuning curves for this stimulus location
    r = tuning_curves[:, :, stimulus_idx]  # [n_neurons, n_orientations]
    
    # Log-likelihood: sum over neurons
    # log P(n|θ) = sum_i [n_i * log(r_i(θ)) - r_i(θ) * T]
    log_likelihood = (n[:, None] * torch.log(r + 1e-10)).sum(dim=0)
    log_likelihood -= (r * T_decode).sum(dim=0)
    
    # Find orientation with maximum likelihood
    max_idx = torch.argmax(log_likelihood)
    theta_hat = theta_grid[max_idx]
    
    return theta_hat
```

### Step 7: Complete Simulation Pipeline

```python
def simulate_trial(tuning_curves, theta_true, x_idx, 
                   theta_grid, gamma=100.0, T_decode=0.1):
    """
    Simulate a single trial: encode → spike → decode.
    
    Args:
        tuning_curves: [n_neurons, n_orientations, n_locations]
        theta_true: True orientation (radians)
        x_idx: Stimulus location index
        theta_grid: Orientation grid
        gamma: Population gain
        T_decode: Decoding window
    
    Returns:
        theta_hat: Decoded orientation
        error: Angular error (radians)
    """
    # Find closest orientation in grid
    theta_idx = torch.argmin(torch.abs(theta_grid - theta_true))
    
    # Get driving inputs for this orientation and location
    f = tuning_curves[:, theta_idx, x_idx]  # [n_neurons]
    
    # Apply divisive normalization (expand dims for broadcasting)
    r = divisive_normalize(f.unsqueeze(1), gamma=gamma).squeeze()
    
    # Generate spikes
    n = generate_spikes(r.unsqueeze(1), T_decode=T_decode).squeeze()
    
    # Decode
    theta_hat = ml_decode(n, tuning_curves, theta_grid, x_idx, T_decode)
    
    # Compute error (circular distance)
    error = torch.angle(torch.exp(1j * (theta_hat - theta_true)))
    
    return theta_hat.item(), error.item()
```

### Step 8: Run Experiment and Analyze

```python
def run_experiment(n_trials=1000, set_sizes=[1, 2, 4, 8]):
    """
    Run memory experiment across different set sizes.
    
    Returns:
        results: Dictionary with errors and statistics
    """
    # Sample tuning curves once
    tuning_curves, theta_grid, x_grid = sample_tuning_curves()
    
    results = {}
    
    for N in set_sizes:
        errors = []
        
        for trial in range(n_trials):
            # Random orientation and location
            theta_true = torch.rand(1).item() * 2 * np.pi - np.pi
            x_idx = torch.randint(0, len(x_grid), (1,)).item()
            
            # Simulate with set size N (simplified: just decode one item)
            _, error = simulate_trial(
                tuning_curves, theta_true, x_idx, 
                theta_grid, gamma=100.0 / N  # Decrease gain with set size
            )
            errors.append(error)
        
        # Store results
        results[N] = {
            'errors': np.array(errors),
            'variance': np.var(errors),
            'mean_abs_error': np.mean(np.abs(errors))
        }
    
    return results

# Run and visualize
results = run_experiment()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Error distributions
for N in [1, 2, 4, 8]:
    axes[0].hist(results[N]['errors'], bins=50, alpha=0.5, 
                 label=f'N={N}', density=True)
axes[0].set_xlabel('Error (radians)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_title('Error Distributions')

# Variance vs. set size
set_sizes = list(results.keys())
variances = [results[N]['variance'] for N in set_sizes]
axes[1].loglog(set_sizes, variances, 'o-')
axes[1].set_xlabel('Set Size')
axes[1].set_ylabel('Variance')
axes[1].set_title('Variance Scaling')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

## Literature & Resources

### Key Papers

**Population Coding & Working Memory:**
1. Bays, P. M. (2014). "Noise in neural populations accounts for errors in working memory." *Journal of Neuroscience*, 34(10), 3632-3645.
2. Ma, W. J., et al. (2014). "Bayesian inference with probabilistic population codes." *Nature Neuroscience*, 9(11), 1432-1438.

**Mixed Selectivity:**
3. Rigotti, M., et al. (2013). "The importance of mixed selectivity in complex cognitive tasks." *Nature*, 497, 585-590.
4. Fusi, S., et al. (2016). "Why neurons mix: high dimensionality for higher cognition." *Current Opinion in Neurobiology*, 37, 66-74.

**Divisive Normalization:**
5. Carandini, M., & Heeger, D. J. (2012). "Normalization as a canonical neural computation." *Nature Reviews Neuroscience*, 13, 51-62.

**Gaussian Processes:**
6. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
7. Wilson, A., et al. (2020). "Gaussian process kernels for pattern discovery and extrapolation." *ICML*.

### Online Resources

- **GPyTorch Documentation:** https://gpytorch.ai/
- **Theoretical Neuroscience (Dayan & Abbott):** Chapter on population codes
- **Perplexity Search Terms:** See section above for comprehensive list

---

## Scientific Objective (Slide Bullet Points)

**Research Goal: Extending Neural Population Models to Prefrontal Cortex**

### Objective

**Develop a biologically realistic population coding model that accounts for mixed selectivity in prefrontal cortex during visual working memory tasks**

### Approach

- **Foundation:** Extend Bays (2014) population coding framework
  - Preserves divisive normalization and Poisson variability
  - Maintains capacity for behavioral error prediction

- **Innovation:** Incorporate mixed selectivity architecture
  - Unified population: all neurons respond to all stimuli
  - Conjunctive tuning: neurons encode feature × location combinations
  - Reflects prefrontal cortex neurophysiology

- **Methods:** Gaussian Process framework for flexible tuning
  - Non-parametric modeling of heterogeneous neural responses
  - Principled uncertainty quantification for decoding
  - Natural correlation structure captures noise dependencies

### Expected Outcomes

- **Behavioral:** Reproduce Bays (2014) error patterns with mixed architecture
- **Neural:** Match PFC recordings showing conjunctive coding
- **Theoretical:** Connect GP posterior variance to Fisher Information bounds

---

