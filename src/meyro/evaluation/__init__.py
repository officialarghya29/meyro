"""MEYRO evaluation suites (Phases 19-29)."""

from meyro.evaluation.ablation import AblationStudy
from meyro.evaluation.cold_start import ColdStartAnalysis
from meyro.evaluation.cross_subject import CrossSubjectStudy
from meyro.evaluation.drift import BaselineDriftAnalysis
from meyro.evaluation.efficiency import measure_efficiency
from meyro.evaluation.metrics import detection_metrics
from meyro.evaluation.neural_gap import NeuralGapDiagnosis
from meyro.evaluation.robustness import RobustnessTestSuite
from meyro.evaluation.significance import PersonalizationSignificanceStudy
from meyro.evaluation.statistics import bootstrap_ci, paired_wilcoxon, per_group_auroc

__all__ = [
    "AblationStudy",
    "BaselineDriftAnalysis",
    "ColdStartAnalysis",
    "CrossSubjectStudy",
    "NeuralGapDiagnosis",
    "PersonalizationSignificanceStudy",
    "RobustnessTestSuite",
    "bootstrap_ci",
    "detection_metrics",
    "measure_efficiency",
    "paired_wilcoxon",
    "per_group_auroc",
]
