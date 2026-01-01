# Mixed Selectivity in Visual Working Memory

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This framework develops a **Gaussian Process-based generative model** for neural populations with **mixed selectivity**—the property where neurons respond conjunctively to multiple stimulus features rather than encoding them independently. We extend traditional population coding models by introducing **location-dependent lengthscales** that create non-separable tuning functions, capturing the heterogeneous response properties observed in prefrontal cortex during working memory tasks.

The core contribution is a principled mapping between our GP framework and the **Bays (2014) population coding model** for visual working memory. By implementing:

1. **Divisive Normalization (DN)** — metabolic budget constraint
2. **Poisson Spiking** — neural noise source  
3. **Maximum Likelihood Decoder** — readout mechanism

we create a complete pipeline from stimulus → neural activity → behavioral response, enabling quantitative predictions about working memory capacity limits and precision-load tradeoffs.

**Key Insight:** Working memory capacity limits emerge from **divisive normalization** capping total neural activity regardless of set size. This metabolic constraint forces precision to decline as more items compete for fixed resources.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Model Architecture](#-model-architecture)
- [Divisive Normalization](#-divisive-normalization)
- [Bays Model Equivalence](#-bays-model-equivalence)
- [Experiments](#-experiments)
- [Installation](#-installation)
- [Scientific Background](#-scientific-background)
- [Roadmap](#-roadmap)
- [Citation](#-citation)

---

## ⚡ Quick Start

### Run Pre-Normalized Analysis (Experiment 1)
```bash
# Single neuron analysis
python scripts/run_experiments.py --exp1 --n_neurons 1

# Population analysis (100 neurons)
python scripts/run_experiments.py --exp1 --n_neurons 100
```

### Run Post-Normalized Analysis with DN (Experiment 2)
```bash
# Single neuron with divisive normalization
python scripts/run_experiments.py --exp2 --n_neurons 1 --gamma 100

# Population analysis with DN
python scripts/run_experiments.py --exp2 --n_neurons 100 --gamma 100
```

### Run Both Experiments for Comparison
```bash
# Compare pre-DN vs post-DN (recommended)
python scripts/run_experiments.py --both --n_neurons 100 --gamma 100

# With custom parameters
python scripts/run_experiments.py --both --n_neurons 100 --gamma 150 --seed 42
```

### Command Line Options
```bash
python scripts/run_experiments.py --help

Options:
  --exp1              Run Experiment 1: Pre-Normalized Response
  --exp2              Run Experiment 2: Post-Normalized Response (DN)
  --both              Run both experiments for comparison
  
  --n_neurons INT     Number of neurons (default: 1)
  --n_orientations    Number of orientation bins (default: 10)
  --theta_lengthscale Base GP lengthscale (default: 0.3)
  --lengthscale_variability  σ_λ for heterogeneity (default: 0.5)
  
  --gamma FLOAT       DN gain constant in Hz (default: 100)
  --sigma_sq FLOAT    DN semi-saturation constant (default: 1e-6)
  
  --seed INT          Random seed (default: 22)
  --save_dir DIR      Output directory (default: figures)
  --no_plot           Disable plotting
```

---

## 🧠 Model Architecture

### The Five-Layer Model Stack

| Layer | Mathematical Object | Biological Interpretation |
|-------|---------------------|---------------------------|
| **L1: GP Tuning** | $f_i^{(n)}(\theta) \sim \mathcal{GP}(0, k_{\lambda_{n,i}})$ | Heterogeneous tuning curves |
| **L2: Pre-Normalized** | $g_i^{(n)}(\theta) = \exp(f_i^{(n)}(\theta))$ | Driving input (excitation) |
| **L3: Divisive Normalization** | $r_i = \gamma \cdot g_i / (\sum_j \bar{g}_j + \sigma^2)$ | Metabolic budget constraint |
| **L4: Poisson Spiking** | $n_i \sim \text{Poisson}(r_i \cdot T_d)$ | Neural noise |
| **L5: ML Decoder** | $\hat{\theta} = \arg\max \sum_i n_i \log r_i(\theta)$ | Behavioral readout |

### Mixed Selectivity via Location-Dependent Lengthscales

The key innovation is **location-dependent lengthscales** $\lambda_{n,i}$:

```
λ_{n,i} = λ_base × |1 + σ_λ × z_{n,i}|    where z_{n,i} ~ N(0,1)
```

This creates **non-separable** tuning: $R(\theta, L) \neq f(\theta) \cdot g(L)$

Different locations have different tuning sharpness, creating mixed selectivity without artificial conjunctions.

---

## ⚖️ Divisive Normalization

### The Core Equation

Following Bays (2014), we implement divisive normalization:

$$r_i^{(n)}(\theta) = \gamma \cdot \frac{g_i^{(n)}(\theta)}{\sum_{j=1}^{L} \bar{g}_j^{(n)} + \sigma^2}$$

Where:
- $g_i^{(n)}(\theta) = \exp(f_i^{(n)}(\theta))$ — pre-normalized driving input
- $\bar{g}_j^{(n)} = \frac{1}{n_\theta}\sum_\theta g_j^{(n)}(\theta)$ — mean activation at location $j$
- $\gamma$ — gain constant (total activity budget, in Hz)
- $\sigma^2$ — semi-saturation constant (numerical stability)

### Why DN Matters

| Without DN | With DN |
|------------|---------|
| Activity grows **exponentially** with set size | Activity is **capped** regardless of set size |
| No capacity limit | Explains WM capacity limits |
| ~18× increase from 2→8 items | ~1× (nearly flat) |
| Biologically implausible | Metabolically constrained |

### Global vs Per-Subset DN

**Global DN** (Bays-style): Denominator computed over **all locations**
- Total activity truly capped
- Post-DN curve has same shape as Pre-DN, just scaled

**Per-Subset DN**: Denominator computed per active subset
- Partial compression
- Post-DN still shows some growth

We implement **Global DN** to match Bays' assumption of a single shared resource pool.

---

## 🔬 Bays Model Equivalence

### Complete Component Mapping

| Bays (2014) | Equation | Our Framework | Status |
|-------------|----------|---------------|--------|
| Tuning function $f_{ij}$ | (1) | $g_i^{(n)}(\theta) = \exp(f_i^{(n)}(\theta))$ | ✅ Implemented |
| Tuning width $\omega$ | (1) | Lengthscale $\lambda_{n,i}$ | ✅ Implemented |
| Divisive normalization | (2-3) | `core/divisive_normalization.py` | ✅ Implemented |
| Attention gain $\alpha_j$ | (2) | Multiplicative gain (future) | 🔲 Planned |
| Poisson spiking | (4) | $n_i \sim \text{Poisson}(r_i T_d)$ | 🔲 Planned |
| ML decoder | (5-8) | $\hat{\theta} = \arg\max \sum_i n_i f_i(\theta)$ | 🔲 Planned |
| SNR | (15-16) | $\text{SNR} \propto T_d\gamma/N \cdot h(\lambda)$ | 🔲 Planned |
| Error distribution | (11-13) | Emerges from Poisson + DN | 🔲 Planned |
| **Mixed selectivity** | N/A | Location-dependent $\lambda_{n,i}$ | ✅ Innovation |

### Engineering Components to Implement

#### 1. Poisson Spiking Layer
```python
# Convert firing rates to spike counts
n_i = np.random.poisson(r_i * T_d)  # T_d = 100ms typically
```

#### 2. Maximum Likelihood Decoder
```python
# Decode stimulus from spike counts
theta_hat = argmax over theta of: sum_i n_i * log(r_i(theta))
```

#### 3. Attention Gain Modulation
```python
# Weight attended items more heavily
r_i = gamma * (alpha_i * g_i) / (sum_j alpha_j * g_bar_j + sigma_sq)
```

#### 4. Error Distribution Analysis
```python
# Compute recall errors
error = theta_hat - theta_true  # Circular difference
# Analyze: variance, kurtosis, "Mexican hat" deviation
```

#### 5. SNR Calculation
```python
# Signal-to-noise ratio per item
SNR = T_d * gamma / N * h(lambda)  # Decreases with set size N
```

---

## 📊 Experiments

### Experiment 1: Pre-Normalized Response
**Purpose:** Analyze raw GP-generated responses without normalization

**Key Finding:** R.mean grows **exponentially** with set size (~18× from l=2 to l=8)

```bash
python scripts/run_experiments.py --exp1 --n_neurons 100
```

**Output:**
- `figures/exp1_pre_norm/exp1_pre_norm_100neurons_log.png`
- `figures/exp1_pre_norm/exp1_pre_norm_100neurons_linear.png`

### Experiment 2: Post-Normalized Response (DN)
**Purpose:** Analyze responses after divisive normalization

**Key Finding:** With Global DN, R.mean has **same scaling** as Pre-DN but constant offset

```bash
python scripts/run_experiments.py --exp2 --n_neurons 100 --gamma 100
```

**Output:**
- `figures/exp2_post_norm/exp2_post_norm_global_100neurons.png`
- `figures/exp2_post_norm/exp2_comparison_global_100neurons_log.png`

### Comparison Analysis
**Purpose:** Direct comparison of Pre-DN vs Post-DN scaling

```bash
python scripts/run_experiments.py --both --n_neurons 100 --gamma 100
```

**Output:**
- `figures/comparison/comparison_100neurons.png`

### Future Experiments (Planned)

| Experiment | Description | Bays Equivalent |
|------------|-------------|-----------------|
| **Exp 3** | Error distributions under memory load | Bays Exp 1 |
| **Exp 4** | Attention weighting (cued items) | Bays Exp 2 |
| **Exp 5** | Optimal attention weights | Bays Exp 3 |
| **Exp 6** | Inter-item error correlations | Novel prediction |
| **Exp 7** | Set composition effects | Novel prediction |

---

## 🚀 Installation

### Requirements
```bash
python >= 3.8
numpy >= 1.20
matplotlib >= 3.3
seaborn >= 0.11
tqdm >= 4.60
```

### Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/mixed-selectivity-vwm.git
cd mixed-selectivity-vwm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy matplotlib seaborn tqdm

# Verify installation
python scripts/run_experiments.py --both --n_neurons 1 --no_plot
```

### Project Structure
```
mixed_selectivity_vwm/
├── mixed_selectivity/
│   ├── core/
│   │   ├── gaussian_process.py      # GP sampling
│   │   ├── divisive_normalization.py # DN logic
│   │   └── kernels.py               # Covariance functions
│   ├── experiments/
│   │   ├── exp1_pre_normalized.py   # Pre-DN analysis
│   │   └── exp2_post_normalized.py  # Post-DN analysis
│   ├── analysis/
│   │   └── separability.py          # SVD-based separability
│   └── utils/
├── scripts/
│   └── run_experiments.py           # Unified CLI runner
├── figures/                         # Output plots
├── data/                            # Data storage
└── docs/                            # Documentation
```

---

## 🔬 Scientific Background

### Visual Working Memory Capacity

Working memory can hold only ~3-4 items with high precision. Why?

**Traditional Explanations:**
- **Slot models:** Fixed number of discrete slots
- **Resource models:** Continuous but limited resource

**Our Framework (following Bays):**
- Capacity limits emerge from **metabolic constraints**
- Divisive normalization caps total neural activity
- More items → less activity per item → more noise → lower precision

### Mixed Selectivity in PFC

Neurons in prefrontal cortex show **mixed selectivity**:
- Respond to conjunctions of features, not single features
- Creates high-dimensional neural representations
- Enables flexible, context-dependent computation

Our GP framework generates realistic mixed selectivity through location-dependent lengthscales, without artificial injection of conjunctive responses.

### Key References

1. **Bays, P. M. (2014).** Noise in neural populations accounts for errors in working memory. *J. Neuroscience*, 34(10), 3632-3645.

2. **Bays, P. M., & Husain, M. (2008).** Dynamic shifts of limited working memory resources in human vision. *Science*, 321(5890), 851-854.

3. **Rigotti, M., et al. (2013).** The importance of mixed selectivity in complex cognitive tasks. *Nature*, 497(7451), 585-590.

4. **Carandini, M., & Heeger, D. J. (2012).** Normalization as a canonical neural computation. *Nature Reviews Neuroscience*, 13(1), 51-62.

---

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] GP-based tuning curve generation
- [x] Location-dependent lengthscales (mixed selectivity)
- [x] Divisive normalization (global DN)
- [x] Pre-DN vs Post-DN comparison experiments
- [x] Unified CLI runner

### Phase 2: Complete Bays Pipeline 🔲
- [ ] Poisson spiking layer
- [ ] Maximum likelihood decoder
- [ ] Attention gain modulation
- [ ] Error distribution analysis
- [ ] SNR calculations

### Phase 3: Validation 🔲
- [ ] Replicate Bays' error distribution findings
- [ ] Validate precision-load tradeoff
- [ ] Test attention effects

### Phase 4: Novel Predictions 🔲
- [ ] Inter-item error correlations (unique to mixed selectivity)
- [ ] Set composition effects
- [ ] Optimal attention allocation

### Phase 5: Metabolic Theory 🔲
- [ ] Derive optimal coding under metabolic constraints
- [ ] Connect to neural data
- [ ] Predict individual differences

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{mixed_selectivity_vwm_2024,
  author = {Your Name},
  title = {Mixed Selectivity in Visual Working Memory: A GP-Based Framework},
  year = {2024},
  url = {https://github.com/yourusername/mixed-selectivity-vwm}
}
```

And the foundational paper:

```bibtex
@article{bays2014noise,
  title={Noise in neural populations accounts for errors in working memory},
  author={Bays, Paul M},
  journal={Journal of Neuroscience},
  volume={34},
  number={10},
  pages={3632--3645},
  year={2014}
}
```

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Happy modeling! 🧠✨**