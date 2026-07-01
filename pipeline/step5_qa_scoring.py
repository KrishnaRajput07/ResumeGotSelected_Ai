# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Smart Resume Q&A — Evidence-Grounded Scoring (Core Intelligence)
# ─────────────────────────────────────────────────────────────────────────────
# For each rubric criterion × each shortlisted candidate:
#   1. Retrieve top-N most relevant resume chunks via FAISS
#   2. Call LLM with retrieved chunks + score anchoring rubric embedded
#   3. Parse structured output: {answer, evidence_quote, confidence, score}
#   4. Validate evidence_quote exists verbatim in source (fuzzy match)
#      If validation fails → mark evidence_verified=False, penalize confidence
#
# Hallucination prevention:
#   - LLM sees ONLY retrieved chunks (not full resume)
#   - Evidence quote is validated post-hoc
#   - Score 0 / "NO_EVIDENCE_FOUND" is a named, explicit output path
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Callable

from thefuzz import fuzz

from core.llm_client import llm_client
from core.embedding_client import embedding_client
from core.reranker_client import reranker_client
from core.vector_store import CandidateVectorStore
from core.config import settings
from prompts.criterion_qa import get_criterion_qa_system_prompt, get_criterion_qa_user_prompt
from schemas.rubric import FrozenRubric, WeightedCriterion
from schemas.candidate import CandidateRecord, CriterionScore

logger = logging.getLogger(__name__)

NO_EVIDENCE_SENTINEL = "NO_EVIDENCE_FOUND"


def score_candidate_all_criteria(
    candidate: CandidateRecord,
    store: CandidateVectorStore,
    rubric: FrozenRubric,
) -> CandidateRecord:
    """
    Run evidence-grounded Q&A for all rubric criteria on one candidate.
    Mutates and returns the candidate record with criterion_scores filled.
    
    Args:
        candidate: CandidateRecord (raw_text must be populated)
        store: FAISS index for this candidate's resume chunks
        rubric: Frozen rubric with weighted_criteria to evaluate
    
    Returns:
        Updated CandidateRecord with criterion_scores populated
    """
    logger.info(
        f"[{candidate.candidate_id}] Scoring {len(rubric.weighted_criteria)} criteria "
        f"for '{candidate.name}'"
    )

    criterion_scores = []
    for criterion in rubric.weighted_criteria:
        score = _score_single_criterion(
            candidate=candidate,
            store=store,
            criterion=criterion,
        )
        criterion_scores.append(score)
        logger.debug(
            f"  [{candidate.candidate_id}] {criterion.criterion[:40]}: "
            f"score={score.score}, conf={score.confidence:.2f}, "
            f"verified={score.evidence_verified}"
        )

    candidate.criterion_scores = criterion_scores
    return candidate


def _score_single_criterion(
    candidate: CandidateRecord,
    store: CandidateVectorStore,
    criterion: WeightedCriterion,
) -> CriterionScore:
    """
    Score one criterion against one candidate's resume.
    
    Returns CriterionScore with answer, evidence_quote, confidence, score.
    """
    # ── Step 1: Retrieve relevant chunks via FAISS ──────────────────────────
    query_emb = embedding_client.embed_query(criterion.criterion)
    retrieve_n = max(settings.qa_retrieve_top_n, settings.reranker_top_n * 4)
    retrieved = store.retrieve(query_emb, top_n=retrieve_n)
    
    if not retrieved:
        # No chunks at all (ingestion failed?) → return zero score
        return CriterionScore(
            criterion_id=criterion.id,
            criterion_text=criterion.criterion,
            answer="No resume content available for this candidate.",
            evidence_quote=NO_EVIDENCE_SENTINEL,
            confidence=0.0,
            score=0,
            evidence_verified=True,
        )

    retrieved = reranker_client.rerank(
        query=criterion.criterion,
        passages=retrieved,
        top_n=settings.qa_retrieve_top_n,
    )
    chunk_texts = [chunk.text for chunk, _ in retrieved]

    # ── Step 2: LLM Q&A call ────────────────────────────────────────────────
    system_prompt = get_criterion_qa_system_prompt(criterion.criterion)
    user_prompt = get_criterion_qa_user_prompt(criterion.criterion, chunk_texts)

    try:
        raw = llm_client.structured_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_keys=["answer", "evidence_quote", "score"],
        )
    except Exception as e:
        logger.error(
            f"[{candidate.candidate_id}] LLM call failed for criterion "
            f"'{criterion.criterion}': {e}"
        )
        return CriterionScore(
            criterion_id=criterion.id,
            criterion_text=criterion.criterion,
            answer=f"Evaluation failed: {str(e)[:100]}",
            evidence_quote=NO_EVIDENCE_SENTINEL,
            confidence=0.0,
            score=0,
            evidence_verified=False,
        )

    # ── Step 3: Parse and clamp values ──────────────────────────────────────
    raw_quote = str(raw.get("evidence_quote", NO_EVIDENCE_SENTINEL)).strip()
    raw_score = raw.get("score", 0)
    raw_confidence = raw.get("confidence", 0.5)
    raw_answer = str(raw.get("answer", "")).strip()

    # Clamp score to 0-10
    score = max(0, min(10, int(raw_score) if isinstance(raw_score, (int, float)) else 0))
    # Clamp confidence to 0.0-1.0
    confidence = max(0.0, min(1.0, float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 0.5))

    # ── Step 4: Evidence Quote Validation ───────────────────────────────────
    # Verify the quote actually exists in the retrieved chunks (anti-hallucination)
    evidence_verified = True
    if raw_quote != NO_EVIDENCE_SENTINEL and raw_quote:
        source_text = " ".join(chunk_texts)
        # Use partial ratio: quote might be slightly paraphrased
        fuzzy_score = fuzz.partial_ratio(raw_quote.lower(), source_text.lower())
        
        if fuzzy_score < settings.evidence_quote_min_fuzzy_score:
            logger.warning(
                f"[{candidate.candidate_id}] Evidence quote failed validation "
                f"(fuzzy={fuzzy_score}): '{raw_quote[:60]}'"
            )
            evidence_verified = False
            # Penalize confidence when quote can't be verified
            confidence = max(0.0, confidence - 0.25)

    # If score > 0 but no evidence found, that's a contradiction — fix it
    if raw_quote == NO_EVIDENCE_SENTINEL and score > 2:
        logger.warning(
            f"[{candidate.candidate_id}] Contradiction: score={score} but "
            f"evidence_quote=NO_EVIDENCE_FOUND. Capping score at 2."
        )
        score = min(score, 2)
        confidence = max(0.0, confidence - 0.2)

    return CriterionScore(
        criterion_id=criterion.id,
        criterion_text=criterion.criterion,
        answer=raw_answer[:500],          # Truncate runaway answers
        evidence_quote=raw_quote[:200],   # Truncate runaway quotes
        confidence=round(confidence, 3),
        score=score,
        evidence_verified=evidence_verified,
    )


def score_all_candidates(
    shortlisted_ids: list[str],
    records: dict[str, CandidateRecord],
    stores: dict[str, CandidateVectorStore],
    rubric: FrozenRubric,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, CandidateRecord]:
    """
    Run Step 5 for all shortlisted candidates.
    
    Args:
        shortlisted_ids: Candidates to score (from Step 4)
        records: All candidate records
        stores: Per-candidate FAISS stores
        rubric: Frozen rubric
        progress_callback: Optional UI progress hook
    
    Returns:
        Updated records dict with criterion_scores populated
    """
    total = len(shortlisted_ids)
    logger.info(f"Step 5: Scoring {total} candidates across {len(rubric.weighted_criteria)} criteria")

    for i, candidate_id in enumerate(shortlisted_ids):
        if progress_callback:
            name = records[candidate_id].name if candidate_id in records else candidate_id
            progress_callback(i + 1, total, name)

        if candidate_id not in records or candidate_id not in stores:
            logger.warning(f"Missing record or store for {candidate_id} — skipping")
            continue

        records[candidate_id] = score_candidate_all_criteria(
            candidate=records[candidate_id],
            store=stores[candidate_id],
            rubric=rubric,
        )

    logger.info("Step 5 complete.")
    return records
