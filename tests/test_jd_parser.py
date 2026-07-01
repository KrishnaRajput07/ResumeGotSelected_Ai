"""
Tests for Step 1: JD Parser
Tests schema validation, weight normalization, and edge cases.
"""

import pytest
from unittest.mock import patch, MagicMock

from schemas.rubric import FrozenRubric, WeightedCriterion, HardGate
from pipeline.step1_jd_parser import _build_rubric_from_response


# ── Weight Normalization ───────────────────────────────────────────────────────

def test_weight_normalization_when_sum_off():
    """If LLM weights don't sum to 1.0, they should auto-normalize."""
    response = {
        "job_title": "Software Engineer",
        "hard_gates": [],
        "weighted_criteria": [
            {"id": "c1", "criterion": "Python", "category": "Hard Skills", "weight": 0.30, "rationale": "Core language"},
            {"id": "c2", "criterion": "SQL", "category": "Hard Skills", "weight": 0.30, "rationale": "Data access"},
            {"id": "c3", "criterion": "Communication", "category": "Soft Skills", "weight": 0.30, "rationale": "Team work"},
        ],
        "bonus_sensitivity": "medium",
    }
    rubric, warnings = _build_rubric_from_response(response)
    total = sum(c.weight for c in rubric.weighted_criteria)
    assert abs(total - 1.0) < 0.01, f"Expected weights to sum to 1.0, got {total}"
    assert any("normalized" in w.lower() for w in warnings), "Expected normalization warning"


def test_exact_weights_no_warning():
    """Exact weights (sum=1.0) should produce no normalization warning."""
    response = {
        "job_title": "ML Engineer",
        "hard_gates": [],
        "weighted_criteria": [
            {"id": "c1", "criterion": "PyTorch", "category": "Hard Skills", "weight": 0.40, "rationale": ""},
            {"id": "c2", "criterion": "MLOps", "category": "Domain Knowledge", "weight": 0.35, "rationale": ""},
            {"id": "c3", "criterion": "Research", "category": "Domain Knowledge", "weight": 0.25, "rationale": ""},
        ],
        "bonus_sensitivity": "medium",
    }
    rubric, warnings = _build_rubric_from_response(response)
    total = sum(c.weight for c in rubric.weighted_criteria)
    assert abs(total - 1.0) < 0.01
    norm_warnings = [w for w in warnings if "normalized" in w.lower()]
    assert len(norm_warnings) == 0


# ── Schema Validation ─────────────────────────────────────────────────────────

def test_frozen_rubric_validates_weights():
    """FrozenRubric should raise ValueError if weights don't sum to ~1.0."""
    with pytest.raises(Exception):
        FrozenRubric(
            job_title="Test",
            weighted_criteria=[
                WeightedCriterion(id="c1", criterion="Test", category="Hard Skills",
                                  weight=0.50, rationale=""),
            ],
            # weight=0.50, which is not ~1.0 — should fail
        )


def test_frozen_rubric_accepts_valid_weights():
    """Valid rubric with weights summing to 1.0 should pass validation."""
    rubric = FrozenRubric(
        job_title="Data Scientist",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python", category="Hard Skills",
                              weight=0.50, rationale=""),
            WeightedCriterion(id="c2", criterion="Statistics", category="Domain Knowledge",
                              weight=0.50, rationale=""),
        ],
    )
    assert rubric.job_title == "Data Scientist"
    assert len(rubric.weighted_criteria) == 2


def test_rubric_freeze_sets_timestamp():
    """freeze() should set frozen_at timestamp."""
    rubric = FrozenRubric(
        job_title="Test Role",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python", category="Hard Skills",
                              weight=1.0, rationale=""),
        ],
    )
    assert rubric.frozen_at is None
    frozen = rubric.freeze()
    assert frozen.frozen_at is not None


# ── Hard Gates ────────────────────────────────────────────────────────────────

def test_hard_gate_strict_default():
    """Hard gates default to strict=True."""
    gate = HardGate(criterion="Must have valid work permit")
    assert gate.strict is True


def test_hard_gate_flexible_can_be_set():
    gate = HardGate(criterion="Preferred MBA", strict=False)
    assert gate.strict is False


# ── Rubric to Query Text ──────────────────────────────────────────────────────

def test_rubric_to_query_text():
    """to_query_text() should produce a non-empty string with job title and criteria."""
    rubric = FrozenRubric(
        job_title="Backend Engineer",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Go language", category="Hard Skills",
                              weight=0.60, rationale=""),
            WeightedCriterion(id="c2", criterion="Kubernetes", category="Hard Skills",
                              weight=0.40, rationale=""),
        ],
        hard_gates=[HardGate(criterion="5+ years experience")],
    )
    query = rubric.to_query_text()
    assert "Backend Engineer" in query
    assert "Go language" in query
    assert "Kubernetes" in query
    assert "5+ years experience" in query
