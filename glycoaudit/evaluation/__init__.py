"""Leakage-controlled ranking and guarded secondary evaluations."""

from .cross_source import CrossSourceResult, evaluate_cross_source, greedy_size_match
from .explicit_nonbinding import ExplicitNonbindingResult, evaluate_explicit_nonbinding
from .pu_ranking import (
    EvaluationResult,
    NullControlResult,
    evaluate_leave_one_agent_out_popularity,
    evaluate_positive_kfold,
    holm_adjust,
    infer_feature_columns,
    run_null_controls,
)

__all__ = [
    "CrossSourceResult",
    "EvaluationResult",
    "ExplicitNonbindingResult",
    "NullControlResult",
    "evaluate_cross_source",
    "evaluate_explicit_nonbinding",
    "evaluate_leave_one_agent_out_popularity",
    "evaluate_positive_kfold",
    "greedy_size_match",
    "holm_adjust",
    "infer_feature_columns",
    "run_null_controls",
]

