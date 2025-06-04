#!/usr/bin/env python3
"""
SatelliteRL: Reinforcement Learning for Satellite Constellation Management
Author: Debanjan Shil
Institution: M.Tech Data Science Program
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="satellite-rl",
    version="0.1.0",
    author="Debanjan Shil",
    author_email="your.email@example.com",  # Update with your email
    description="Reinforcement Learning for Intelligent Satellite Constellation Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debanjan06/SatelliteRL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=2.0.0",
        ],
        "viz": [
            "dash>=2.10.0",
            "plotly>=5.14.0",
            "bokeh>=3.1.0",
        ],
        "distributed": [
            "ray[tune]>=2.4.0",
            "dask>=2023.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "satellite-rl-train=src.training.train_dqn:main",
            "satellite-rl-sim=src.simulation.run_basic_sim:main",
            "satellite-rl-dashboard=src.visualization.dashboard:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "reinforcement-learning",
        "satellite",
        "earth-observation", 
        "orbital-mechanics",
        "space-technology",
        "multi-agent",
        "deep-learning",
        "remote-sensing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/debanjan06/SatelliteRL/issues",
        "Source": "https://github.com/debanjan06/SatelliteRL",
        "Documentation": "https://satelliterl.readthedocs.io/",
    },
)