"""
Tests for Step 7: Weight Engine
Tests deterministic scoring math, confidence aggregation, and gate exclusion.
"""

import pytest
from schemas.rubric import FrozenRubric, WeightedCriterion, HardGate
from schemas.candidate import CandidateRecord, CriterionScore, OpenSignal, GateResult
from pipeline.step7_weight_engine import compute_candidate_score, rank_candidates


def make_rubric(**kwargs) -> FrozenRubric:
    """Helper: create a minimal valid rubric."""
    return FrozenRubric(
        job_title=kwargs.get("job_title", "Test Role"),
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Skill A", category="Hard Skills",
                              weight=0.60, rationale=""),
            WeightedCriterion(id="c2", criterion="Skill B", category="Soft Skills",
                              weight=0.40, rationale=""),
        ],
        hard_gates=kwargs.get("hard_gates", []),
        bonus_sensitivity=kwargs.get("bonus_sensitivity", "medium"),
    )


def make_candidate(
    candidate_id="cand_0001",
    scores: dict | None = None,
    open_signals: list | None = None,
    gate_status: str = "passed",
    excluded: bool = False,
) -> CandidateRecord:
    """Helper: create a candidate with given criterion scores."""
    scores = scores or {"c1": 8, "c2": 6}
    criterion_scores = [
        CriterionScore(
            criterion_id=cid,
            criterion_text=f"Criterion {cid}",
            answer="Test answer",
            evidence_quote="Some evidence found here",
            confidence=0.8,
            score=score,
        )
        for cid, score in scores.items()
    ]

    sigs = open_signals or []

    return CandidateRecord(
        candidate_id=candidate_id,
        name=f"Candidate {candidate_id}",
        file_path="/fake/path.pdf",
        criterion_scores=criterion_scores,
        open_signals=sigs,
        gate_status=gate_status,
        excluded=excluded,
    )


# ── Base Score Math ───────────────────────────────────────────────────────────

def test_base_score_weighted_sum():
    """base_score = Σ(weight_i × score_i)"""
    rubric = make_rubric()  # c1=0.60, c2=0.40
    candidate = make_candidate(scores={"c1": 10, "c2": 10})
    result = compute_candidate_score(candidate, rubric)
    assert abs(result.base_score - 10.0) < 0.01, f"Expected 10.0, got {result.base_score}"


def test_base_score_partial():
    """With scores c1=8, c2=6 and weights 0.6/0.4: base = 0.6×8 + 0.4×6 = 4.8+2.4 = 7.2"""
    rubric = make_rubric()
    candidate = make_candidate(scores={"c1": 8, "c2": 6})
    result = compute_candidate_score(candidate, rubric)
    assert abs(result.base_score - 7.2) < 0.01, f"Expected 7.2, got {result.base_score}"


def test_zero_score_candidate():
    """All zero scores → base_score = 0.0"""
    rubric = make_rubric()
    candidate = make_candidate(scores={"c1": 0, "c2": 0})
    result = compute_candidate_score(candidate, rubric)
    assert result.base_score == 0.0


# ── Bonus Math ────────────────────────────────────────────────────────────────

def test_bonus_applied_with_medium_sensitivity():
    """Medium sensitivity = 0.6× signal strength, capped at 1.0"""
    rubric = make_rubric(bonus_sensitivity="medium")
    signals = [OpenSignal(signal="Won national hackathon", evidence_quote="Won IIT hackathon", strength=0.9)]
    candidate = make_candidate(scores={"c1": 8, "c2": 6}, open_signals=signals)
    result = compute_candidate_score(candidate, rubric)
    expected_bonus = min(0.9 * 0.6, 1.0)  # 0.54
    assert abs(result.bonus_applied - expected_bonus) < 0.01


def test_bonus_off_sensitivity():
    """bonus_sensitivity='off' → no bonus regardless of signals."""
    rubric = make_rubric(bonus_sensitivity="off")
    signals = [OpenSignal(signal="Published Nature paper", evidence_quote="Published in Nature", strength=1.0)]
    candidate = make_candidate(scores={"c1": 5, "c2": 5}, open_signals=signals)
    result = compute_candidate_score(candidate, rubric)
    assert result.bonus_applied == 0.0


def test_bonus_cap_enforced():
    """Bonus is capped at bonus_cap (1.0) even with multiple high-strength signals."""
    rubric = make_rubric(bonus_sensitivity="high")
    signals = [
        OpenSignal(signal="Signal 1", evidence_quote="Evidence 1", strength=0.9),
        OpenSignal(signal="Signal 2", evidence_quote="Evidence 2", strength=0.9),
        OpenSignal(signal="Signal 3", evidence_quote="Evidence 3", strength=0.9),
    ]
    candidate = make_candidate(scores={"c1": 5, "c2": 5}, open_signals=signals)
    result = compute_candidate_score(candidate, rubric)
    assert result.bonus_applied <= 1.0


def test_final_score_capped_at_10():
    """final_score must never exceed 10.0."""
    rubric = make_rubric(bonus_sensitivity="high")
    signals = [OpenSignal(signal="Exceptional", evidence_quote="Top award", strength=1.0)]
    candidate = make_candidate(scores={"c1": 10, "c2": 10}, open_signals=signals)
    result = compute_candidate_score(candidate, rubric)
    assert result.final_score <= 10.0


# ── Confidence ────────────────────────────────────────────────────────────────

def test_high_confidence_with_good_evidence():
    """High individual confidence + all evidence found → High label."""
    rubric = make_rubric()
    candidate = make_candidate(scores={"c1": 8, "c2": 7})
    # Default scores have confidence=0.8, evidence_verified=True, evidence found
    result = compute_candidate_score(candidate, rubric)
    assert result.confidence in ("High", "Medium")  # Should be high or medium


def test_low_confidence_when_no_evidence():
    """Many NO_EVIDENCE_FOUND → confidence penalized → Low label."""
    rubric = make_rubric()
    candidate = CandidateRecord(
        candidate_id="test",
        name="Test Candidate",
        file_path="/test.pdf",
        criterion_scores=[
            CriterionScore(
                criterion_id="c1", criterion_text="Skill A",
                answer="No evidence", evidence_quote="NO_EVIDENCE_FOUND",
                confidence=0.3, score=0,
            ),
            CriterionScore(
                criterion_id="c2", criterion_text="Skill B",
                answer="No evidence", evidence_quote="NO_EVIDENCE_FOUND",
                confidence=0.3, score=0,
            ),
        ],
        gate_status="passed",
    )
    result = compute_candidate_score(candidate, rubric)
    assert result.confidence == "Low"


# ── Ranking ────────────────────────────────────────────────────────────────────

def test_ranking_order_by_final_score():
    """Higher final_score candidate should rank #1."""
    rubric = make_rubric()
    cand_a = make_candidate("cand_a", scores={"c1": 9, "c2": 8})
    cand_b = make_candidate("cand_b", scores={"c1": 5, "c2": 4})
    records = {"cand_a": cand_a, "cand_b": cand_b}
    ranked, excluded = rank_candidates(["cand_a", "cand_b"], records, rubric)
    assert ranked[0].candidate_id == "cand_a"
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_excluded_candidates_not_ranked():
    """Gate-failed candidates go to excluded list, not ranked list."""
    rubric = make_rubric(hard_gates=[HardGate(criterion="Work permit", strict=True)])
    cand_pass = make_candidate("pass", scores={"c1": 7, "c2": 6}, gate_status="passed")
    cand_fail = make_candidate("fail", scores={"c1": 9, "c2": 9},
                               gate_status="failed", excluded=True)
    records = {"pass": cand_pass, "fail": cand_fail}
    ranked, excluded = rank_candidates(["pass", "fail"], records, rubric)
    assert len(ranked) == 1
    assert ranked[0].candidate_id == "pass"
    assert len(excluded) == 1
    assert excluded[0].candidate_id == "fail"
