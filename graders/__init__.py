"""Graders package for Elite-Trade-Sentry HFT environments.

Exports
-------
    EasyDetectionGrader        - Task 1: Forgiving penalties (0.1 FP / 0.2 FN).
    MediumClassificationGrader - Task 2: Standard HFT penalties (0.2 FP / 0.4 FN).
    HardFixGrader              - Task 3: Brutal adversarial penalties (0.4 FP / 0.8 FN).
"""

from graders.grader_detection import EasyDetectionGrader
from graders.grader_classification import MediumClassificationGrader
from graders.grader_fix import HardFixGrader

__all__ = [
    "EasyDetectionGrader", 
    "MediumClassificationGrader", 
    "HardFixGrader"
]