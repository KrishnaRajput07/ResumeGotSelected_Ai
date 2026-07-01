# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Weight Engine — Deterministic Final Scoring
# ─────────────────────────────────────────────────────────────────────────────
# Pure math. Zero LLM calls in this step.
# Every number is a function of named, weighted, capped parts.
# Formula:
#   base_score = Σ(weight_i × score_i)       [0-10 scale]
#   raw_bonus  = Σ(signal_strength) × sensitivity_multiplier
#   bonus      = min(raw_bonus, bonus_cap)
#   final_score= min(base_score + bonus, 10.0)
#   excluded   = True if gate_status == "failed" (score still computed for audit)
#
# Confidence aggregation:
#   avg_confidence × (1 - 0.5 × no_evidence_ratio)
#   → penalizes records with many "NO_EVIDENCE_FOUND" criterion scores
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Literal

from core.config import settings
from schemas.rubric import FrozenRubric
from schemas.candidate import CandidateRecord, CriterionScore

logger = logging.getLogger(__name__)

NO_EVIDENCE_SENTINEL = "NO_EVIDENCE_FOUND"


def compute_candidate_score(
    candidate: CandidateRecord,
    rubric: FrozenRubric,
) -> CandidateRecord:
    """
    Compute final_score, base_score, bonus, and confidence label.
    Mutates and returns the candidate record.
    
    Hard-gate excluded candidates still get a score for audit transparency,
    but are marked excluded=True and kept in the excluded list.
    
    Args:
        candidate: CandidateRecord with criterion_scores and open_signals populated
        rubric: Frozen rubric (weights + bonus_sensitivity)
    
    Returns:
        Updated CandidateRecord with scoring fields populated
    """
    # ── Base Score: Weighted Sum ──────────────────────────────────────────────
    # Build lookup: criterion_id → CriterionScore
    score_map = {cs.criterion_id: cs for cs in candidate.criterion_scores}

    base_score = 0.0
    scored_criteria_count = 0

    for criterion in rubric.weighted_criteria:
        if criterion.id in score_map:
            cs = score_map[criterion.id]
            base_score += criterion.weight * cs.score
            scored_criteria_count += 1
        else:
            # Missing criterion score → treat as 0 (candidate not evaluated on it)
            logger.debug(
                f"[{candidate.candidate_id}] Criterion {criterion.id} not scored — "
                f"treating as 0"
            )

    # base_score is already on 0-10 scale (weights sum to 1.0, scores are 0-10)

    # ── Bonus: Open Signals ────────────────────────────────────────────────────
    sensitivity_multiplier = settings.bonus_sensitivity_map.get(
        rubric.bonus_sensitivity, 0.6
    )
    raw_bonus = sum(sig.strength for sig in candidate.open_signals) * sensitivity_multiplier
    bonus_applied = min(raw_bonus, settings.bonus_cap)

    # ── Final Score ────────────────────────────────────────────────────────────
    final_score = min(base_score + bonus_applied, 10.0)

    # ── Confidence Aggregation ─────────────────────────────────────────────────
    confidence_label = _compute_confidence(candidate)

    # ── Write Back ────────────────────────────────────────────────────────────
    candidate.base_score = round(base_score, 4)
    candidate.bonus_applied = round(bonus_applied, 4)
    candidate.final_score = round(final_score, 4)
    candidate.confidence = confidence_label

    logger.debug(
        f"[{candidate.candidate_id}] '{candidate.name}': "
        f"base={base_score:.3f} + bonus={bonus_applied:.3f} = {final_score:.3f} "
        f"({confidence_label}) | gate={candidate.gate_status}"
    )

    return candidate


def _compute_confidence(candidate: CandidateRecord) -> Literal["High", "Medium", "Low"]:
    """
    Aggregate per-criterion confidence values into a run-level label.
    
    Two penalty factors:
    1. Low individual confidence scores (LLM uncertain about retrieval completeness)
    2. High ratio of NO_EVIDENCE_FOUND criteria (sparse resume coverage)
    """
    if not candidate.criterion_scores:
        return "Low"

    avg_confidence = sum(
        cs.confidence for cs in candidate.criterion_scores
    ) / len(candidate.criterion_scores)

    no_evidence_count = sum(
        1 for cs in candidate.criterion_scores
        if cs.evidence_quote == NO_EVIDENCE_SENTINEL
    )
    no_evidence_ratio = no_evidence_count / len(candidate.criterion_scores)

    # Penalize: each 10% no-evidence → -5% confidence
    adjusted = avg_confidence * (1.0 - 0.5 * no_evidence_ratio)

    # Also penalize unverified evidence quotes
    unverified_count = sum(
        1 for cs in candidate.criterion_scores
        if not cs.evidence_verified
    )
    unverified_ratio = unverified_count / len(candidate.criterion_scores)
    adjusted = adjusted * (1.0 - 0.3 * unverified_ratio)

    if adjusted >= settings.confidence_high_floor:
        return "High"
    elif adjusted >= settings.confidence_low_ceiling:
        return "Medium"
    else:
        return "Low"


def rank_candidates(
    shortlisted_ids: list[str],
    records: dict[str, CandidateRecord],
    rubric: FrozenRubric,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """
    Score all shortlisted candidates and produce ranked/excluded lists.
    
    Args:
        shortlisted_ids: Candidates that passed Step 4
        records: All candidate records (with Steps 5+6 data populated)
        rubric: Frozen rubric
    
    Returns:
        (ranked_candidates, excluded_candidates)
        ranked_candidates: Sorted by final_score desc, rank assigned
        excluded_candidates: Candidates that failed hard gates (with scores for audit)
    """
    logger.info(f"Step 7: Computing scores for {len(shortlisted_ids)} candidates")

    ranked = []
    excluded = []

    for candidate_id in shortlisted_ids:
        if candidate_id not in records:
            continue
        
        candidate = records[candidate_id]
        
        # Compute score (even for excluded candidates — audit transparency)
        candidate = compute_candidate_score(candidate, rubric)
        records[candidate_id] = candidate

        if candidate.excluded or candidate.gate_status == "failed":
            excluded.append(candidate)
        else:
            ranked.append(candidate)

    # Sort by final_score descending
    ranked.sort(key=lambda c: c.final_score, reverse=True)

    # Assign ranks
    for i, candidate in enumerate(ranked):
        candidate.rank = i + 1

    logger.info(
        f"Step 7 complete: {len(ranked)} ranked | {len(excluded)} excluded | "
        f"Top score: {ranked[0].final_score:.3f} ({ranked[0].name})" if ranked else "No ranked candidates"
    )

    return ranked, excluded
