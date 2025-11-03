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
