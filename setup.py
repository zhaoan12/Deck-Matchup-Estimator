from setuptools import setup, find_packages

setup(
    name="game_matchup_estimator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "requests>=2.25.0",
        "flask>=2.0.0",
    ],
)