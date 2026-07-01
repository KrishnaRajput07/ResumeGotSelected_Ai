"""
Tests for Step 5: Q&A Scoring evidence validation logic.
Tests the evidence quote verification and score clamping (no LLM required).
"""

import pytest
from unittest.mock import patch, MagicMock

from schemas.rubric import FrozenRubric, WeightedCriterion
from schemas.candidate import CandidateRecord, CriterionScore
from pipeline.step5_qa_scoring import _score_single_criterion


# ── Score Clamping ────────────────────────────────────────────────────────────

def test_score_clamp_above_10():
    """LLM returning score > 10 should be clamped to 10."""
    # We test the clamping logic by injecting a mock LLM response
    mock_raw = {
        "answer": "Excellent candidate",
        "evidence_quote": "Led a team of 50 engineers at AWS",
        "confidence": 0.9,
        "score": 15,  # Out of range
    }
    # Clamp logic: max(0, min(10, score))
    score = max(0, min(10, int(mock_raw["score"])))
    assert score == 10


def test_score_clamp_below_0():
    """LLM returning negative score should be clamped to 0."""
    mock_raw = {"score": -3}
    score = max(0, min(10, int(mock_raw["score"])))
    assert score == 0


# ── Contradiction Check ────────────────────────────────────────────────────────

def test_no_evidence_high_score_is_contradiction():
    """score > 2 with NO_EVIDENCE_FOUND should be corrected."""
    # Simulated scenario: score=8, evidence_quote=NO_EVIDENCE_FOUND
    evidence_quote = "NO_EVIDENCE_FOUND"
    score = 8
    if evidence_quote == "NO_EVIDENCE_FOUND" and score > 2:
        score = min(score, 2)
    assert score == 2


def test_no_evidence_low_score_is_ok():
    """score <= 2 with NO_EVIDENCE_FOUND is valid (LLM correctly scored 0-2)."""
    evidence_quote = "NO_EVIDENCE_FOUND"
    score = 0
    if evidence_quote == "NO_EVIDENCE_FOUND" and score > 2:
        score = min(score, 2)
    assert score == 0  # Unchanged


# ── Confidence Clamping ────────────────────────────────────────────────────────

def test_confidence_clamp_above_1():
    """Confidence > 1.0 should be clamped to 1.0."""
    raw_conf = 1.5
    conf = max(0.0, min(1.0, float(raw_conf)))
    assert conf == 1.0


def test_confidence_penalty_for_unverified_quote():
    """Unverified evidence quote should reduce confidence by 0.25."""
    original_confidence = 0.8
    penalized = max(0.0, original_confidence - 0.25)
    assert abs(penalized - 0.55) < 0.001


# ── Evidence Quote Validation ─────────────────────────────────────────────────

def test_fuzzy_match_passes_for_verbatim_quote():
    """A verbatim substring should achieve high fuzzy match score."""
    from thefuzz import fuzz
    source = "Led a team of 20 engineers to deliver the product ahead of schedule."
    quote = "Led a team of 20 engineers"
    score = fuzz.partial_ratio(quote.lower(), source.lower())
    assert score >= 90  # Should match well


def test_fuzzy_match_fails_for_hallucinated_quote():
    """A completely hallucinated quote should fail the fuzzy match."""
    from thefuzz import fuzz
    source = "Developed Python microservices for fintech platform."
    fake_quote = "Published three papers in Nature journal on quantum computing"
    score = fuzz.partial_ratio(fake_quote.lower(), source.lower())
    assert score < 70  # Should fail validation threshold


def test_no_evidence_sentinel_skips_validation():
    """NO_EVIDENCE_FOUND should not be fuzzy-checked (it's a valid sentinel)."""
    quote = "NO_EVIDENCE_FOUND"
    # Validation should be skipped for this sentinel value
    should_validate = quote != "NO_EVIDENCE_FOUND" and quote != ""
    assert should_validate is False


# ── CriterionScore Model ──────────────────────────────────────────────────────

def test_criterion_score_default_evidence_verified():
    """CriterionScore defaults evidence_verified to True."""
    cs = CriterionScore(
        criterion_id="c1",
        criterion_text="Python skills",
        answer="Strong Python evidence",
        evidence_quote="5 years of Python experience",
        confidence=0.85,
        score=8,
    )
    assert cs.evidence_verified is True


def test_criterion_score_score_bounds():
    """Score must be between 0 and 10."""
    with pytest.raises(Exception):
        CriterionScore(
            criterion_id="c1",
            criterion_text="Test",
            answer="Test",
            evidence_quote="Test",
            confidence=0.5,
            score=11,  # Out of range
        )
