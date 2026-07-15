"""Unit tests for the ACS formula: known inputs -> expected score."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.collaboration import compute_acs
from config import settings


def test_acs_all_ones():
    assert math.isclose(compute_acs(1.0, 1.0, 1.0), 1.0, abs_tol=1e-9)


def test_acs_all_zeros():
    assert math.isclose(compute_acs(0.0, 0.0, 0.0), 0.0, abs_tol=1e-9)


def test_acs_weighted_combination():
    w = settings.acs_weights
    confidence, evidence_quality, task_completion = 0.8, 0.5, 0.9
    expected = (
        w.confidence_weight * confidence
        + w.evidence_weight * evidence_quality
        + w.completion_weight * task_completion
    )
    assert math.isclose(compute_acs(confidence, evidence_quality, task_completion), expected, abs_tol=1e-9)


def test_acs_weights_sum_to_one():
    w = settings.acs_weights
    total = w.confidence_weight + w.evidence_weight + w.completion_weight
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_acs_below_retry_threshold():
    # A deliberately weak output should fall below the retry threshold.
    score = compute_acs(0.3, 0.2, 0.4)
    assert score < settings.acs_retry_threshold


def test_acs_above_retry_threshold():
    score = compute_acs(0.9, 0.85, 0.95)
    assert score >= settings.acs_retry_threshold


if __name__ == "__main__":
    test_acs_all_ones()
    test_acs_all_zeros()
    test_acs_weighted_combination()
    test_acs_weights_sum_to_one()
    test_acs_below_retry_threshold()
    test_acs_above_retry_threshold()
    print("All ACS tests passed.")
