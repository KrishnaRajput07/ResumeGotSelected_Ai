# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Embedding-Based Shortlisting (Scale Layer)
# ─────────────────────────────────────────────────────────────────────────────
# This is a RECALL step, not a ranking step.
# It filters out clearly irrelevant candidates before expensive LLM reasoning.
# Deliberately generous — err toward keeping borderline candidates.
# The real judgment happens in Step 5.
# ─────────────────────────────────────────────────────────────────────────────

import logging

import numpy as np

from core.embedding_client import embedding_client
from core.vector_store import CrossCandidateIndex
from core.config import settings
from schemas.rubric import FrozenRubric
from schemas.candidate import CandidateRecord

logger = logging.getLogger(__name__)


def shortlist_candidates(
    rubric: FrozenRubric,
    records: dict[str, CandidateRecord],
    cross_index: CrossCandidateIndex,
    top_k: int | None = None,
    similarity_floor: float | None = None,
) -> tuple[list[str], list[str]]:
    """
    Filter candidates by embedding similarity to the rubric profile.
    
    Engineering note:
    We embed the rubric as a COMPOSITE QUERY representing the ideal candidate.
    The rubric's to_query_text() method produces a single string like:
    "Job: Senior ML Engineer | Python proficiency (weight: 0.20) | Required: PhD"
    This is then embedded with the BGE query prefix for asymmetric retrieval.
    
    Args:
        rubric: Frozen rubric (used to build the composite query vector)
        records: All ingested candidate records {candidate_id: CandidateRecord}
        cross_index: Cross-candidate FAISS index (populated in Step 3)
        top_k: Max candidates to shortlist (defaults to settings.shortlist_top_k)
        similarity_floor: Min cosine similarity (defaults to settings.shortlist_similarity_floor)
    
    Returns:
        (shortlisted_ids, filtered_out_ids)
    """
    top_k = top_k or settings.shortlist_top_k
    similarity_floor = similarity_floor or settings.shortlist_similarity_floor

    total_candidates = cross_index.total_candidates()
    logger.info(
        f"Step 4: Shortlisting {total_candidates} candidates → top {top_k} "
        f"(floor={similarity_floor:.2f})"
    )

    if total_candidates == 0:
        logger.warning("No candidates in cross-candidate index.")
        return [], []

    # If total candidates ≤ top_k, skip filtering — everyone goes through
    if total_candidates <= top_k:
        logger.info(
            f"Total candidates ({total_candidates}) ≤ top_k ({top_k}). "
            f"Skipping embedding filter — all candidates proceed to Step 5."
        )
        all_ids = list(records.keys())
        for cid in all_ids:
            records[cid].embedding_score = 1.0  # Mark as unfiltered
        return all_ids, []

    # ── Build composite rubric query embedding ─────────────────────────────────
    query_text = rubric.to_query_text()
    logger.debug(f"Rubric query text: {query_text[:200]}")
    
    query_embedding = embedding_client.embed_query(query_text)  # shape: (1, dim)

    # ── FAISS shortlisting ────────────────────────────────────────────────────
    shortlisted_pairs = cross_index.shortlist(
        query_embedding=query_embedding,
        top_k=top_k,
        similarity_floor=similarity_floor,
    )

    shortlisted_ids = [cid for cid, score in shortlisted_pairs]
    
    # Annotate records with embedding scores
    shortlisted_set = set()
    for cid, score in shortlisted_pairs:
        if cid in records:
            records[cid].embedding_score = score
            shortlisted_set.add(cid)

    filtered_out_ids = [
        cid for cid in records.keys()
        if cid not in shortlisted_set
    ]

    logger.info(
        f"Step 4 complete: {len(shortlisted_ids)} shortlisted, "
        f"{len(filtered_out_ids)} filtered out by embedding similarity"
    )

    # Log top candidates for debugging
    for cid, score in shortlisted_pairs[:5]:
        name = records[cid].name if cid in records else cid
        logger.debug(f"  Shortlisted: {name} (similarity={score:.3f})")

    return shortlisted_ids, filtered_out_ids
