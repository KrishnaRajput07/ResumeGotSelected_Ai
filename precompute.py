#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# precompute.py — Redrob Hackathon Pre-computation Pipeline
#
# Runs once (no time limit). Produces artifacts in data/precomputed/ that
# rank.py reads to produce the submission CSV in <10 seconds.
#
# Stages:
#   A. Behavioral scoring on ALL 100K candidates          (no LLM, ~3 min)
#   B. Embedding similarity shortlist → top 500           (no LLM, ~1 min)
#   C. LLM deep scoring on top 200 candidates             (Ollama, ~30–90 min)
#
# Usage:
#   python precompute.py --candidates candidates.jsonl
#   python precompute.py --candidates candidates.jsonl --llm-top-n 150
#   python precompute.py --candidates candidates.jsonl --skip-llm
#   python precompute.py --candidates candidates.jsonl --stage behavioral
#   python precompute.py --candidates candidates.jsonl --stage embedding
#   python precompute.py --candidates candidates.jsonl --stage llm
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import settings
from core.embedding_client import embedding_client
from pipeline.step2_json_ingestion import load_jsonl, ingest_json_candidates
from pipeline.step4_shortlisting import shortlist_candidates
from pipeline.step5_qa_scoring import score_all_candidates
from pipeline.step6_parallel_passes import run_all_parallel_passes
from pipeline.step7_weight_engine import rank_candidates
from pipeline.step_behavioral_scorer import score_all_structured_candidates
from schemas.candidate import StructuredCandidateScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")

PRECOMPUTED_DIR = _ROOT / "data" / "precomputed"
BEHAVIORAL_FILE = PRECOMPUTED_DIR / "behavioral_scores.json"
EMBEDDING_FILE  = PRECOMPUTED_DIR / "embedding_shortlist.json"
LLM_SCORES_FILE = PRECOMPUTED_DIR / "llm_scores.json"


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — Behavioral scoring (all candidates, deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def run_behavioral(candidates_path: Path, force: bool = False) -> list[dict]:
    if BEHAVIORAL_FILE.exists() and not force:
        logger.info(f"[Stage A] Loading cached behavioral scores: {BEHAVIORAL_FILE}")
        return json.loads(BEHAVIORAL_FILE.read_text(encoding="utf-8"))

    logger.info(f"[Stage A] Loading candidates from {candidates_path}")
    t0 = time.time()
    candidates = load_jsonl(candidates_path)
    logger.info(f"[Stage A] Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    def progress(current, total, name):
        if current % 10000 == 0 or current == total:
            pct = current / total * 100
            logger.info(f"[Stage A] Behavioral scoring: {current:,}/{total:,} ({pct:.0f}%)")

    t0 = time.time()
    scores = score_all_structured_candidates(candidates, progress_callback=progress)
    elapsed = time.time() - t0

    honeypots = sum(1 for s in scores if s.is_honeypot)
    logger.info(
        f"[Stage A] Done in {elapsed:.1f}s — "
        f"{len(scores):,} scored, {honeypots:,} honeypots detected"
    )

    # Serialize to dicts
    records = [s.model_dump() for s in scores]

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    BEHAVIORAL_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info(f"[Stage A] Saved → {BEHAVIORAL_FILE}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — Embedding shortlist (BGE-small similarity vs JD rubric, top 500)
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding(
    candidates_path: Path,
    behavioral_records: list[dict],
    top_n: int = 500,
    force: bool = False,
) -> list[str]:
    """
    Returns ordered list of candidate_ids (top_n by embedding similarity).
    Uses behavioral scores to pre-filter to top 2000 non-honeypots before embedding
    to avoid embedding all 100K profiles.
    """
    if EMBEDDING_FILE.exists() and not force:
        logger.info(f"[Stage B] Loading cached embedding shortlist: {EMBEDDING_FILE}")
        return json.loads(EMBEDDING_FILE.read_text(encoding="utf-8"))

    # Pre-filter: take top 2000 non-honeypots by behavioral score
    clean = [r for r in behavioral_records if not r["is_honeypot"]]
    clean.sort(key=lambda r: r["composite_score"], reverse=True)
    buffer_ids = {r["candidate_id"] for r in clean[:2000]}

    logger.info(f"[Stage B] Embedding top {len(buffer_ids):,} non-honeypot candidates")

    t0 = time.time()
    all_candidates = load_jsonl(candidates_path)
    buffer_candidates = [c for c in all_candidates if c.candidate_id in buffer_ids]
    logger.info(f"[Stage B] Loaded {len(buffer_candidates):,} candidates for embedding")

    run_dir = PRECOMPUTED_DIR / "_embed_tmp"
    run_dir.mkdir(parents=True, exist_ok=True)

    def ingest_progress(current, total, name):
        if current % 200 == 0 or current == total:
            logger.info(f"[Stage B] Embedding: {current:,}/{total:,}")

    records, stores, cross_index = ingest_json_candidates(
        candidates=buffer_candidates,
        run_dir=run_dir,
        progress_callback=ingest_progress,
    )
    logger.info(f"[Stage B] Embedded {len(records):,} candidates in {time.time()-t0:.1f}s")

    # Build rubric from JD for embedding query
    from pipeline.step1_jd_parser import parse_jd_to_rubric
    jd_path = _ROOT / "India_runs_data_and_ai_challenge" / "job_description.docx"
    if not jd_path.exists():
        # Try txt variant
        jd_path = _ROOT / "India_runs_data_and_ai_challenge" / "job_description.txt"

    logger.info(f"[Stage B] Parsing JD for embedding query: {jd_path.name}")
    jd_result = parse_jd_to_rubric(jd_path)
    rubric = jd_result.rubric

    shortlisted_ids, _ = shortlist_candidates(
        rubric=rubric,
        records=records,
        cross_index=cross_index,
        top_k=top_n,
        similarity_floor=0.10,   # Very low floor — we want top_n by rank, not threshold
    )

    logger.info(f"[Stage B] Embedding shortlist: {len(shortlisted_ids):,} candidates")

    # Annotate with similarity scores
    result = [
        {"candidate_id": cid, "embedding_score": records[cid].embedding_score}
        for cid in shortlisted_ids
        if cid in records
    ]

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"[Stage B] Saved → {EMBEDDING_FILE}")

    # Save rubric for rank.py to reuse
    rubric_path = PRECOMPUTED_DIR / "rubric.json"
    rubric_path.write_text(
        json.dumps(rubric.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    logger.info(f"[Stage B] Rubric saved → {rubric_path}")

    return [r["candidate_id"] for r in result]


# ─────────────────────────────────────────────────────────────────────────────
# Stage C — LLM deep scoring (top N candidates via Ollama, evidence-grounded)
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_scoring(
    candidates_path: Path,
    behavioral_records: list[dict],
    embedding_ids: list[str],
    llm_top_n: int = 200,
    force: bool = False,
) -> dict:
    """
    Runs full Steps 5+6+7 on the top llm_top_n candidates.
    Returns dict of candidate_id → llm scoring result.
    Saves incrementally so you can resume if interrupted.
    """
    if LLM_SCORES_FILE.exists() and not force:
        logger.info(f"[Stage C] Loading cached LLM scores: {LLM_SCORES_FILE}")
        return json.loads(LLM_SCORES_FILE.read_text(encoding="utf-8"))

    # Select top llm_top_n: prioritise embedding shortlist order, fill from behavioral
    behavioral_map = {r["candidate_id"]: r for r in behavioral_records}

    # Start from embedding order (already top by similarity), non-honeypots first
    candidates_for_llm = [
        cid for cid in embedding_ids
        if cid in behavioral_map and not behavioral_map[cid]["is_honeypot"]
    ][:llm_top_n]

    # Top up from behavioral-only if needed
    if len(candidates_for_llm) < llm_top_n:
        remaining = llm_top_n - len(candidates_for_llm)
        in_set = set(candidates_for_llm)
        behavioral_only = [
            r["candidate_id"] for r in behavioral_records
            if not r["is_honeypot"] and r["candidate_id"] not in in_set
        ]
        candidates_for_llm += behavioral_only[:remaining]

    logger.info(f"[Stage C] LLM deep scoring {len(candidates_for_llm)} candidates")

    # Load the actual candidate objects
    t0 = time.time()
    all_candidates = load_jsonl(candidates_path)
    cand_set = set(candidates_for_llm)
    llm_candidates = [c for c in all_candidates if c.candidate_id in cand_set]
    logger.info(f"[Stage C] Loaded {len(llm_candidates):,} candidates for LLM scoring")

    # Ingest: JSON → text → chunk → embed → FAISS
    run_dir = PRECOMPUTED_DIR / "_llm_tmp"
    run_dir.mkdir(parents=True, exist_ok=True)

    def ingest_progress(current, total, name):
        if current % 20 == 0 or current == total:
            logger.info(f"[Stage C] Ingesting: {current}/{total} — {name}")

    records, stores, cross_index = ingest_json_candidates(
        candidates=llm_candidates,
        run_dir=run_dir,
        progress_callback=ingest_progress,
    )

    # Load rubric
    rubric_path = PRECOMPUTED_DIR / "rubric.json"
    if not rubric_path.exists():
        from pipeline.step1_jd_parser import parse_jd_to_rubric
        jd_path = _ROOT / "India_runs_data_and_ai_challenge" / "job_description.docx"
        jd_result = parse_jd_to_rubric(jd_path)
        rubric = jd_result.rubric
        rubric_path.write_text(
            json.dumps(rubric.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    else:
        from schemas.rubric import FrozenRubric
        rubric = FrozenRubric.model_validate(
            json.loads(rubric_path.read_text(encoding="utf-8"))
        )

    # Attach behavioral scores to records
    for cid, record in records.items():
        if cid in behavioral_map:
            beh = behavioral_map[cid]
            record.embedding_score = beh.get("composite_score", 0.0)

    # Step 5: Q&A scoring
    logger.info(f"[Stage C] Step 5 — Evidence Q&A scoring {len(candidates_for_llm)} × {len(rubric.weighted_criteria)} criteria")

    def qa_progress(current, total, name):
        logger.info(f"[Stage C] Q&A: {current}/{total} — {name}")

    records = score_all_candidates(
        shortlisted_ids=candidates_for_llm,
        records=records,
        stores=stores,
        rubric=rubric,
        progress_callback=qa_progress,
    )

    # Step 6: Parallel passes
    logger.info(f"[Stage C] Step 6 — Parallel passes (signals, flags, gates)")

    def passes_progress(current, total, name):
        logger.info(f"[Stage C] Passes: {current}/{total} — {name}")

    records = run_all_parallel_passes(
        shortlisted_ids=candidates_for_llm,
        records=records,
        rubric=rubric,
        progress_callback=passes_progress,
    )

    # Step 7: Ranking math
    logger.info(f"[Stage C] Step 7 — Weight engine")
    ranked, excluded = rank_candidates(
        shortlisted_ids=candidates_for_llm,
        records=records,
        rubric=rubric,
    )

    elapsed = time.time() - t0
    logger.info(
        f"[Stage C] LLM scoring complete in {elapsed/60:.1f} min — "
        f"{len(ranked)} ranked, {len(excluded)} excluded"
    )

    # Serialize all records (ranked + excluded) keyed by candidate_id
    llm_results = {}
    for cand in ranked + excluded:
        llm_results[cand.candidate_id] = {
            "final_score": cand.final_score,
            "base_score": cand.base_score,
            "bonus_applied": cand.bonus_applied,
            "confidence": cand.confidence,
            "excluded": cand.excluded,
            "exclusion_reason": cand.exclusion_reason,
            "criterion_scores": [cs.model_dump() for cs in cand.criterion_scores],
            "open_signals": [s.model_dump() for s in cand.open_signals],
            "red_flags": [f.model_dump() for f in cand.red_flags],
            "gate_results": [g.model_dump() for g in cand.gate_results],
            "gate_status": cand.gate_status,
        }

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    LLM_SCORES_FILE.write_text(json.dumps(llm_results, indent=2), encoding="utf-8")
    logger.info(f"[Stage C] Saved → {LLM_SCORES_FILE}")
    return llm_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Pre-computation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python precompute.py --candidates candidates.jsonl
  python precompute.py --candidates candidates.jsonl --llm-top-n 150
  python precompute.py --candidates candidates.jsonl --skip-llm
  python precompute.py --candidates candidates.jsonl --stage behavioral
  python precompute.py --candidates candidates.jsonl --stage llm --force
        """,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("India_runs_data_and_ai_challenge/candidates.jsonl"),
        help="Path to candidates.jsonl (default: India_runs_data_and_ai_challenge/candidates.jsonl)",
    )
    parser.add_argument(
        "--llm-top-n",
        type=int,
        default=200,
        help="Number of candidates to LLM deep-score (default: 200)",
    )
    parser.add_argument(
        "--embedding-top-n",
        type=int,
        default=500,
        help="Embedding shortlist size (default: 500)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run only behavioral + embedding stages, skip LLM (faster for testing)",
    )
    parser.add_argument(
        "--stage",
        choices=["behavioral", "embedding", "llm", "all"],
        default="all",
        help="Run a specific stage only (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached files exist",
    )

    args = parser.parse_args()

    if not args.candidates.exists():
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Redrob Hackathon — Pre-computation Pipeline")
    logger.info(f"Candidates: {args.candidates}")
    logger.info(f"LLM top-N:  {args.llm_top_n}")
    logger.info(f"Output dir: {PRECOMPUTED_DIR}")
    logger.info("=" * 60)

    run_stage_behavioral = args.stage in ("behavioral", "all")
    run_stage_embedding  = args.stage in ("embedding", "all")
    run_stage_llm        = args.stage in ("llm", "all") and not args.skip_llm

    # ── Stage A ───────────────────────────────────────────────────────────────
    behavioral_records = []
    if run_stage_behavioral or run_stage_embedding or run_stage_llm:
        behavioral_records = run_behavioral(args.candidates, force=args.force)
        logger.info(f"[Stage A] ✓ {len(behavioral_records):,} behavioral scores ready")

    # ── Stage B ───────────────────────────────────────────────────────────────
    embedding_ids = []
    if run_stage_embedding or run_stage_llm:
        embedding_ids = run_embedding(
            args.candidates,
            behavioral_records,
            top_n=args.embedding_top_n,
            force=args.force,
        )
        logger.info(f"[Stage B] ✓ {len(embedding_ids):,} embedding shortlist ready")

    # ── Stage C ───────────────────────────────────────────────────────────────
    if run_stage_llm:
        llm_results = run_llm_scoring(
            args.candidates,
            behavioral_records,
            embedding_ids,
            llm_top_n=args.llm_top_n,
            force=args.force,
        )
        ranked_count = sum(1 for v in llm_results.values() if not v["excluded"])
        logger.info(f"[Stage C] ✓ {ranked_count} LLM-scored candidates ready")

    logger.info("=" * 60)
    logger.info("Pre-computation complete. Run: python rank.py --candidates <path> --out submission.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
