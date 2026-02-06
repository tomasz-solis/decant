from setuptools import setup, find_packages

setup(
    name="decant",
    version="0.1.0",
    description="Decant - Taste, with confidence. A wine analytics and recommendation platform.",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
    ],
)
