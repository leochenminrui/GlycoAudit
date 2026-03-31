"""
GlycoAudit: A Benchmark for Glycan-Binding Agent Interaction Prediction

A reproducible benchmark evaluating computational methods for predicting
lectin-glycan and antibody-glycan binding interactions using positive-unlabeled
(PU) learning framework.
"""

__version__ = "1.0.0"
__author__ = "GlycoAudit Contributors"

# Main components
from . import mirror
from . import features
from . import evaluation
from . import analysis

__all__ = [
    "mirror",
    "features",
    "evaluation",
    "analysis",
]
