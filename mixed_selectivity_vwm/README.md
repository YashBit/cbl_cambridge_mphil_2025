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
