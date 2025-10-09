"""Setup configuration for causal_ml package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="causal-ml-treatment-effects",
    version="0.1.0",
    author="AE Abdou",
    description="Causal Machine Learning for Estimating Treatment Effects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aeabdou/Causal-Machine-Learning-for-Estimating-Treatment-Effect",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
