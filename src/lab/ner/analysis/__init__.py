"""Persistent, evaluator-backed foundations for deep NER diagnostics."""

from lab.ner.analysis.api import EVALUATION_FILENAME, analyze_evaluation, load_evaluation
from lab.ner.analysis.config import AnalysisConfig, NormalizationConfig
from lab.ner.analysis.evaluation import evaluate_to_events
from lab.ner.analysis.exposure import ExposureIndex, threshold_on_rapidfuzz_scale
from lab.ner.analysis.features import audit_annotations
from lab.ner.analysis.models import AnalysisResult
from lab.ner.analysis.oracles import OracleSelection, oracle_error_budget, oracle_result
from lab.ner.analysis.parquet import SCHEMA_VERSION, evaluation_schema
from lab.ner.analysis.publication import PublicationConfig, PublicationReport, create_publication_report
from lab.ner.analysis.statistics import BootstrapConfig, bootstrap_intervals, paired_seen_zero_shot_difference

__all__ = [
    "AnalysisResult",
    "AnalysisConfig",
    "EVALUATION_FILENAME",
    "SCHEMA_VERSION",
    "NormalizationConfig",
    "BootstrapConfig",
    "OracleSelection",
    "PublicationConfig",
    "PublicationReport",
    "analyze_evaluation",
    "evaluate_to_events",
    "evaluation_schema",
    "ExposureIndex",
    "audit_annotations",
    "load_evaluation",
    "bootstrap_intervals",
    "create_publication_report",
    "oracle_error_budget",
    "oracle_result",
    "paired_seen_zero_shot_difference",
    "threshold_on_rapidfuzz_scale",
]
