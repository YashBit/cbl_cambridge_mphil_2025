# Technical Documentation: Mixed Selectivity Implementation

## Core Concept: From Separable to Conjunctive Coding

### The Fundamental Problem
Visual working memory in the prefrontal cortex must solve a binding problem: maintaining multiple features (orientation, location) as unified objects. Pure selectivity (independent feature encoding) fails at this task.

## Mathematical Foundation

### Separability Definition
For a neural response matrix R(θ, x):
- **Separable (Pure Selectivity)**: R(θ, x) = f(θ) × g(x)
- **Non-separable (Mixed Selectivity)**: R(θ, x) ≠ f(θ) × g(x) for any f, g

### SVD-Based Measurement
```
Separability Index = σ₁² / Σᵢσᵢ²
```
Where σᵢ are singular values from SVD decomposition.

- **Separability ≈ 1.0**: Pure selectivity (BAD for working memory)
- **Separability < 0.8**: Mixed selectivity (GOOD - our target)
- **Separability < 0.6**: Strong mixed selectivity (EXCELLENT)

## Three Implementation Methods

### 1. Simple Conjunctive Method
**Core Principle**: Location-dependent orientation preference

```python
preferred_ori = base_preference + rotation_rate * location * π / n_locations
```

This single line breaks separability because the preferred orientation is now a function of location.

**Data Transformations**:
1. **Input**: (orientation, location) pairs
2. **Transform**: Shift gaussian center based on location
3. **Output**: Non-factorizable response matrix

**Why It Works**: 
- Cannot write as f(θ) × g(x) because the θ preference depends on x
- Mimics how receptive fields change across visual space
- Simplest demonstration of conjunction coding

### 2. Direct Construction Method
**Core Principle**: Four engineered mechanisms ensuring non-separability

**Mechanism Breakdown**:

#### Rotating Preference
```python
pref_theta = base_phase + rotation_rate * loc * π / n_locations
```
- Systematically shifts orientation preference with location
- Creates diagonal structure in response matrix

#### Multiplicative Interactions
```python
interaction = exp(-0.3 * (orient_dist * loc_dist))
```
- Product of distances creates genuine interaction terms
- Cannot be factorized mathematically

#### XOR Conjunctions
```python
if orient_group != loc_group:
    response += value
```
- Implements canonical non-linearly separable function
- Fundamental to neural network theory

#### Phase-Coupled Oscillations
```python
response = (1 + sin(f₁θ + coupling×loc)) × (1 + cos(f₂loc - coupling×θ))
```
- Cross-dimensional phase coupling
- Models temporal binding mechanisms

**Data Flow**:
1. Initialize zero matrix [n_orientations × n_locations]
2. Add each mechanism's contribution
3. Normalize and add baseline
4. Apply neuron-specific nonlinearity

### 3. Gaussian Process Interaction Method
**Core Principle**: Biologically plausible smooth tuning with interactions

**Key Innovation**: Location-dependent kernel parameters
```python
lengthscale = base_lengthscale + 0.5 * base_lengthscale * sin(loc * π / n_locations)
```

**Three-Phase Process**:

**Phase 1**: Location-varying orientation tuning
- Sample from GP at each location with different lengthscales
- Creates heterogeneous orientation selectivity

**Phase 2**: Orientation-varying spatial tuning
- Multiplicative gain modulation
- `tuning[orient, :] *= sigmoid(spatial_sample)`

**Phase 3**: Conjunction hot-spots
- Add specific (ori, loc) preferences
- Implement surround suppression

**Why GP Method?**:
- Smooth, continuous tuning curves
- Captures gain modulation (key cortical computation)
- More biologically realistic than engineered methods

## Critical Data Transformations

### 1. Grid Creation (Discretization)
**Why**: Computers cannot handle continuous infinite-dimensional GPs
```python
continuous: f ~ GP(μ(θ,x), k((θ,x), (θ',x')))  # Infinite
discrete:   f ~ N(μ_vec, K_matrix)              # 80-dimensional
```

### 2. Product Kernel (Mixed Selectivity Encoding)
**Why**: Creates conjunction structure in covariance
```python
k_total = k_orientation × k_location
```
If only one dimension is similar → low correlation → mixed selectivity

### 3. Cholesky Decomposition (Sampling Mechanism)
**Why**: Transform white noise to structured samples
```python
K = L @ L^T
sample = L @ z  # z ~ N(0,I) → sample ~ N(0,K)
```

### 4. Reshape (Structure Imposition)
**Why**: Convert 1D vector to 2D response matrix
```python
flat_vector[80] → matrix[20, 4]
```
Defines which dimensions correspond to orientation vs location

### 5. Softplus (Biological Constraint)
**Why**: Neural firing rates must be non-negative
```python
firing_rate = softplus(GP_sample)  # Ensures positivity
```

## Validation Pipeline

### SVD Analysis
Quantifies "intrinsic dimensionality" of tuning surface:
- Rank-1 matrix = separable
- Higher rank = mixed selectivity

### Population Statistics
- **Mean separability**: Overall mixed selectivity level
- **Variance**: Population heterogeneity
- **Percent mixed**: Fraction below threshold

### Success Criteria
- Target: Mean separability < 0.8
- Based on empirical PFC recordings
- Validates biological relevance

## Computational Complexity

| Operation | Complexity | Memory |
|-----------|------------|--------|
| Grid Creation | O(n_ori × n_loc) | O(80) |
| Kernel Matrix | O(n²) | O(6,400) |
| Cholesky | O(n³) | O(6,400) |
| SVD per neuron | O(min(n_ori, n_loc)³) | O(80) |

## Biological Relevance

### Working Memory Requirements
1. **Feature Binding**: Maintain "red square at location 2"
2. **Context-Dependent Coding**: Same feature, different context
3. **High-Dimensional Representation**: Support flexible computation

### PFC Neural Properties Captured
- Heterogeneous tuning curves
- Gain modulation
- Conjunction coding
- Mixed selectivity

## Usage Recommendations

| Method | Use When | Avoid When |
|--------|----------|------------|
| Simple Conjunctive | Teaching concepts | Need complexity |
| Direct | Testing algorithms | Need biological realism |
| GP Interaction | Modeling real neurons | Need guarantees |

## Key Insights

1. **Mixed selectivity emerges from location-dependent preferences** - the simplest mechanism
2. **Multiple mechanisms converge on same computational goal** - robustness
3. **Product kernels naturally encode conjunctions** - mathematical elegance
4. **SVD provides principled measurement** - quantitative validation
5. **Population heterogeneity is feature, not bug** - computational diversity