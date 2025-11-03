#!/bin/bash

# Create Directory Structure for Mixed Selectivity VWM Project
# Run this script from your project root directory

echo "Creating Mixed Selectivity VWM Project Structure..."

# Define the project root
PROJECT_NAME="mixed_selectivity_vwm"

# Create main project directory
mkdir -p $PROJECT_NAME

# Navigate to project directory
cd $PROJECT_NAME

# Create main directories
mkdir -p notebooks
mkdir -p mixed_selectivity/{core,models,experiments,analysis,utils}
mkdir -p scripts
mkdir -p tests
mkdir -p data/{raw,processed,results}
mkdir -p figures/{exp1,exp2,exp3,exp4}
mkdir -p configs
mkdir -p docs

# Create Python package files (__init__.py)
touch mixed_selectivity/__init__.py
touch mixed_selectivity/core/__init__.py
touch mixed_selectivity/models/__init__.py
touch mixed_selectivity/experiments/__init__.py
touch mixed_selectivity/analysis/__init__.py
touch mixed_selectivity/utils/__init__.py

# Create core module files
touch mixed_selectivity/core/gaussian_process.py
touch mixed_selectivity/core/kernels.py
touch mixed_selectivity/core/population.py

# Create model files
touch mixed_selectivity/models/base_model.py
touch mixed_selectivity/models/mixed_selectivity_model.py
touch mixed_selectivity/models/bays_model.py

# Create experiment files
touch mixed_selectivity/experiments/exp1_validation.py
touch mixed_selectivity/experiments/exp2_behavior.py
touch mixed_selectivity/experiments/exp3_information.py
touch mixed_selectivity/experiments/exp4_comparison.py

# Create analysis files
touch mixed_selectivity/analysis/separability.py
touch mixed_selectivity/analysis/fisher_information.py
touch mixed_selectivity/analysis/error_metrics.py
touch mixed_selectivity/analysis/visualization.py

# Create utility files
touch mixed_selectivity/utils/circular_stats.py
touch mixed_selectivity/utils/data_management.py
touch mixed_selectivity/utils/config.py

# Create script files
touch scripts/run_all_experiments.py
touch scripts/generate_figures.py
touch scripts/reproduce_paper.py

# Create test files
touch tests/test_kernels.py
touch tests/test_population.py
touch tests/test_separability.py
touch tests/test_fisher.py

# Create notebook files
touch notebooks/01_explore_gp_kernels.ipynb
touch notebooks/02_experiment1_validation.ipynb
touch notebooks/03_experiment2_behavior.ipynb
touch notebooks/04_experiment3_fisher.ipynb
touch notebooks/05_experiment4_comparison.ipynb
touch notebooks/06_figures_publication.ipynb

# Create config files
touch configs/default_config.yaml
touch configs/experiment_configs.yaml
touch configs/hyperparameters.yaml

# Create documentation files
touch docs/theory.md
touch docs/implementation.md
touch docs/api_reference.md

# Create root level files
touch README.md
touch LICENSE
touch requirements.txt
touch setup.py
touch .gitignore

# Create requirements.txt with basic dependencies
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
torch>=2.0.0
gpytorch>=1.9.0
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.3.0

# Analysis
scikit-learn>=1.0.0
statsmodels>=0.13.0

# Utilities
pyyaml>=6.0
tqdm>=4.62.0
jupyter>=1.0.0
ipykernel>=6.0.0

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0

# Code quality
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.h5
*.npy
*.npz
*.pkl
*.pickle

# Figures
*.png
*.pdf
*.svg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
data/raw/*
data/processed/*
data/results/*
!data/*/.gitkeep
figures/**/*.png
figures/**/*.pdf
!figures/*/.gitkeep

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
EOF

# Create .gitkeep files to track empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/results/.gitkeep
touch figures/exp1/.gitkeep
touch figures/exp2/.gitkeep
touch figures/exp3/.gitkeep
touch figures/exp4/.gitkeep

# Create a basic README
cat > README.md << 'EOF'
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
EOF

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="mixed_selectivity_vwm",
    version="0.1.0",
    description="Mixed selectivity in visual working memory using Gaussian Processes",
    author="Yash Bharti",
    author_email="your.email@cambridge.ac.uk",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt')
        if line.strip() and not line.startswith('#')
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "📁 Project created at: $(pwd)"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate"
echo "4. pip install -r requirements.txt"
echo "5. pip install -e ."
echo ""
echo "Happy coding! 🚀"
