"""
Causal Machine Learning for Estimating Treatment Effects

This package provides reproducible pipelines for causal inference using:
- Double ML for robust treatment effect estimation
- GenericML (GATES, CLAN) for heterogeneous treatment effect analysis
"""

__version__ = "0.1.0"

from .double_ml.double_ml_pipeline import DoubleMLPipeline
from .generic_ml.gates import GATES
from .generic_ml.clan import CLAN

__all__ = [
    "DoubleMLPipeline",
    "GATES",
    "CLAN",
]
