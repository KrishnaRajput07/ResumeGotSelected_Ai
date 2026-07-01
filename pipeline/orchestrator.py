# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Orchestrator — Runs all 8 steps end-to-end
# ─────────────────────────────────────────────────────────────────────────────
# Single entry-point called by the Streamlit dashboard.
# Handles run directory creation, step sequencing, and progress reporting.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from core.config import settings
from schemas.rubric import FrozenRubric
from schemas.candidate import RankedShortlist

from pipeline.step1_jd_parser import parse_jd_to_rubric, JDParseResult
from pipeline.step3_resume_ingestion import ingest_all_resumes
from pipeline.step4_shortlisting import shortlist_candidates
from pipeline.step5_qa_scoring import score_all_candidates
from pipeline.step6_parallel_passes import run_all_parallel_passes
from pipeline.step7_weight_engine import rank_candidates
from pipeline.step8_output import assemble_output

if TYPE_CHECKING:
    from schemas.candidate import StructuredCandidateScore

logger = logging.getLogger(__name__)


@dataclass
class ProgressUpdate:
    """Emitted by the orchestrator to report pipeline progress to the UI."""
    step: int                           # 1-8
    step_name: str
    current: int = 0
    total: int = 0
    detail: str = ""
    done: bool = False
    error: Optional[str] = None


@dataclass
class PipelineRun:
    """Complete result of one pipeline execution."""
    run_id: str
    run_dir: Path
    jd_parse_result: Optional[JDParseResult] = None
    frozen_rubric: Optional[FrozenRubric] = None
    shortlist: Optional[RankedShortlist] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    completed: bool = False
    structured_scores: list = field(default_factory=list)  # list[StructuredCandidateScore]


ProgressCallback = Callable[[ProgressUpdate], None]


def run_step1_only(jd_file_path: str | Path) -> JDParseResult:
    """
    Run only Step 1 (JD parsing → rubric proposal).
    Called by the dashboard to show the rubric before recruiter calibration.
    """
    return parse_jd_to_rubric(jd_file_path)


def run_full_pipeline(
    jd_file_path: str | Path,
    resume_file_paths: list[str | Path],
    frozen_rubric: FrozenRubric,
    progress_cb: ProgressCallback | None = None,
    run_id: str | None = None,
) -> PipelineRun:
    """
    Execute the complete pipeline (Steps 3-8) with a pre-frozen rubric.
    Step 1 and Step 2 (recruiter calibration) happen BEFORE this is called.
    
    Args:
        jd_file_path: Path to JD file (used for metadata only at this point)
        resume_file_paths: All uploaded resume file paths
        frozen_rubric: Rubric locked after recruiter calibration
        progress_cb: Optional callback for real-time progress updates
        run_id: Optional run ID; generated if not provided
    
    Returns:
        PipelineRun with all results and output paths
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = Path(settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run = PipelineRun(run_id=run_id, run_dir=run_dir, frozen_rubric=frozen_rubric)

    def emit(step: int, name: str, current: int = 0, total: int = 0,
             detail: str = "", done: bool = False, error: str | None = None):
        if progress_cb:
            progress_cb(ProgressUpdate(
                step=step, step_name=name, current=current,
                total=total, detail=detail, done=done, error=error
            ))

    try:
        total_resumes = len(resume_file_paths)
        logger.info(
            f"Pipeline run {run_id} starting | "
            f"{total_resumes} resumes | "
            f"rubric: {frozen_rubric.job_title}"
        )

        # ── Step 3: Resume Ingestion ─────────────────────────────────────────
        emit(3, "Resume Ingestion & Indexing", 0, total_resumes, "Starting...")

        def ingest_progress(current, total, name):
            emit(3, "Resume Ingestion & Indexing", current, total,
                 f"Parsing: {name}")

        records, stores, cross_index = ingest_all_resumes(
            file_paths=resume_file_paths,
            run_dir=run_dir,
            progress_callback=ingest_progress,
        )

        if not records:
            raise RuntimeError("No resumes could be parsed. Check file formats.")

        emit(3, "Resume Ingestion & Indexing", total_resumes, total_resumes,
             f"Indexed {len(records)} candidates", done=True)

        # ── Step 4: Embedding Shortlisting ───────────────────────────────────
        emit(4, "Embedding Shortlisting", 0, len(records), "Ranking by rubric similarity...")

        shortlisted_ids, filtered_out_ids = shortlist_candidates(
            rubric=frozen_rubric,
            records=records,
            cross_index=cross_index,
        )

        if not shortlisted_ids:
            raise RuntimeError(
                "No candidates passed the embedding similarity filter. "
                "Try lowering the similarity_floor in settings or uploading more diverse resumes."
            )

        emit(4, "Embedding Shortlisting", len(shortlisted_ids), len(records),
             f"{len(shortlisted_ids)} candidates shortlisted for deep scoring", done=True)

        # ── Step 5: Evidence-Grounded Q&A Scoring ────────────────────────────
        total_qa_calls = len(shortlisted_ids) * len(frozen_rubric.weighted_criteria)
        emit(5, "Evidence-Grounded Scoring", 0, total_qa_calls, "Starting Q&A scoring...")

        scored_count = [0]

        def qa_progress(current, total, name):
            scored_count[0] = current
            emit(5, "Evidence-Grounded Scoring",
                 current * len(frozen_rubric.weighted_criteria),
                 total_qa_calls,
                 f"Scoring '{name}' ({current}/{total})")

        records = score_all_candidates(
            shortlisted_ids=shortlisted_ids,
            records=records,
            stores=stores,
            rubric=frozen_rubric,
            progress_callback=qa_progress,
        )

        emit(5, "Evidence-Grounded Scoring", total_qa_calls, total_qa_calls,
             "All criteria scored", done=True)

        # ── Step 6: Parallel Passes ───────────────────────────────────────────
        emit(6, "Parallel Analysis", 0, len(shortlisted_ids),
             "Running open signals, red flags, hard gates...")

        def passes_progress(current, total, name):
            emit(6, "Parallel Analysis", current, total,
                 f"Analysing '{name}'")

        records = run_all_parallel_passes(
            shortlisted_ids=shortlisted_ids,
            records=records,
            rubric=frozen_rubric,
            progress_callback=passes_progress,
        )

        emit(6, "Parallel Analysis", len(shortlisted_ids), len(shortlisted_ids),
             "Signals, flags, gates complete", done=True)

        # ── Step 7: Weight Engine — Final Ranking ─────────────────────────────
        emit(7, "Final Scoring & Ranking", 0, len(shortlisted_ids), "Computing scores...")

        ranked_candidates, excluded_candidates = rank_candidates(
            shortlisted_ids=shortlisted_ids,
            records=records,
            rubric=frozen_rubric,
        )

        emit(7, "Final Scoring & Ranking", len(shortlisted_ids), len(shortlisted_ids),
             f"{len(ranked_candidates)} ranked, {len(excluded_candidates)} excluded", done=True)

        # ── Step 8: Output Assembly ───────────────────────────────────────────
        emit(8, "Output Generation", 0, 3, "Generating reports...")

        shortlist = assemble_output(
            run_id=run_id,
            rubric=frozen_rubric,
            ranked_candidates=ranked_candidates,
            excluded_candidates=excluded_candidates,
            total_submitted=total_resumes,
            run_dir=run_dir,
        )

        emit(8, "Output Generation", 3, 3, "Reports ready", done=True)

        run.shortlist = shortlist
        run.completed = True
        run.warnings = []

        logger.info(
            f"Pipeline run {run_id} COMPLETE | "
            f"{len(ranked_candidates)} ranked | "
            f"{len(excluded_candidates)} excluded | "
            f"Top: {ranked_candidates[0].name} ({ranked_candidates[0].final_score:.2f})"
            if ranked_candidates else f"Pipeline run {run_id} complete but no ranked candidates"
        )

    except Exception as e:
        logger.exception(f"Pipeline run {run_id} FAILED: {e}")
        run.errors.append(str(e))
        emit(0, "Error", error=str(e))

    return run


def run_jsonl_pipeline(
    jd_file_path: str | Path,
    jsonl_path: str | Path,
    frozen_rubric: FrozenRubric,
    progress_cb: ProgressCallback | None = None,
    run_id: str | None = None,
    max_candidates: int | None = None,
    llm_deep_score_top_n: int | None = None,
) -> PipelineRun:
    """
    Execute the full pipeline for a structured JSONL candidate pool.

    Pipeline summary:
      1. Load candidates.jsonl → list[StructuredCandidate]
      2. Behavioral scoring (deterministic, no LLM) on ALL candidates
      3. Take top llm_deep_score_top_n × 3 by composite_score as LLM buffer
      4. Convert those to CandidateRecord via JSON → text → chunk → embed → FAISS
      5. Step 4 shortlisting on the buffer → llm_deep_score_top_n candidates
      6. Steps 5, 6, 7, 8 on the shortlisted candidates
      7. Merge LLM scores (70%) + behavioral composite (30%) for top candidates
      8. Return full ranked list (LLM-scored at top, behavioral-only in tail)

    Args:
        jd_file_path: Path to JD file (for metadata).
        jsonl_path: Path to candidates.jsonl file.
        frozen_rubric: Rubric locked after recruiter calibration.
        progress_cb: Optional callback for real-time progress updates.
        run_id: Optional run ID; generated if not provided.
        max_candidates: Optional cap on candidates to load (for testing).
        llm_deep_score_top_n: How many candidates receive full LLM scoring.
                               Defaults to settings.jsonl_llm_deep_score_top_n.

    Returns:
        PipelineRun with shortlist and structured_scores populated.
    """
    from pipeline.step2_json_ingestion import load_jsonl, ingest_json_candidates
    from pipeline.step_behavioral_scorer import score_all_structured_candidates

    llm_deep_score_top_n = llm_deep_score_top_n or settings.jsonl_llm_deep_score_top_n
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = Path(settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run = PipelineRun(run_id=run_id, run_dir=run_dir, frozen_rubric=frozen_rubric)

    def emit(
        step: int, name: str, current: int = 0, total: int = 0,
        detail: str = "", done: bool = False, error: str | None = None,
    ):
        if progress_cb:
            progress_cb(ProgressUpdate(
                step=step, step_name=name, current=current,
                total=total, detail=detail, done=done, error=error,
            ))

    try:
        # ── Step 2.1: Load JSONL ─────────────────────────────────────────────
        emit(2, "JSON Loading", 0, 0, "Loading candidates.jsonl...")

        def load_progress(current, total, name):
            emit(2, "JSON Loading", current, total, f"Loading: {current}/{total}")

        candidates = load_jsonl(jsonl_path, max_candidates=max_candidates)
        total_loaded = len(candidates)

        emit(2, "JSON Loading", total_loaded, total_loaded,
             f"{total_loaded} candidates loaded", done=True)

        if not candidates:
            raise RuntimeError("No candidates loaded from JSONL file.")

        # ── Step 2.2: Behavioral Scoring (all candidates, no LLM) ────────────
        emit(2, "Behavioral Scoring", 0, total_loaded,
             "Fast deterministic scoring...")

        def behavioral_progress(current, total, name):
            if current % 5000 == 0 or current == total:
                emit(2, "Behavioral Scoring", current, total,
                     f"Scored {current}/{total}")

        all_scores = score_all_structured_candidates(
            candidates=candidates,
            progress_callback=behavioral_progress,
        )
        run.structured_scores = all_scores

        emit(2, "Behavioral Scoring", total_loaded, total_loaded,
             f"{total_loaded} candidates behaviorally scored", done=True)

        # ── Step 2.3: Shortlist buffer for LLM ──────────────────────────────
        llm_buffer_n = min(llm_deep_score_top_n * 3, total_loaded)
        # Non-honeypots only, sorted by composite descending
        non_honeypot_scores = [s for s in all_scores if not s.is_honeypot]
        top_score_ids = {
            s.candidate_id for s in non_honeypot_scores[:llm_buffer_n]
        }

        emit(2, "Shortlist Buffer", llm_buffer_n, total_loaded,
             f"Top {llm_buffer_n} candidates selected for LLM scoring buffer",
             done=True)

        # Build a lookup for StructuredCandidate by ID
        cand_by_id = {c.candidate_id: c for c in candidates}
        llm_candidates = [
            cand_by_id[cid] for cid in top_score_ids if cid in cand_by_id
        ]

        # ── Step 3: JSON → Text → Chunk → Embed → FAISS ─────────────────────
        emit(3, "Embedding Buffer Candidates", 0, len(llm_candidates),
             "Building embeddings for LLM buffer...")

        def ingest_progress(current, total, name):
            emit(3, "Embedding Buffer Candidates", current, total,
                 f"Embedding: {name} ({current}/{total})")

        records, stores, cross_index = ingest_json_candidates(
            candidates=llm_candidates,
            run_dir=run_dir,
            progress_callback=ingest_progress,
        )

        if not records:
            raise RuntimeError("No candidates could be embedded.")

        emit(3, "Embedding Buffer Candidates", len(records), len(records),
             f"{len(records)} candidates embedded", done=True)

        # Attach structured_score to each CandidateRecord
        score_map = {s.candidate_id: s for s in all_scores}
        for cid, record in records.items():
            if cid in score_map:
                record.structured_score = score_map[cid]

        # ── Step 4: Embedding Shortlisting ────────────────────────────────────
        emit(4, "Embedding Shortlisting", 0, len(records),
             "Ranking by rubric similarity...")

        shortlisted_ids, filtered_out_ids = shortlist_candidates(
            rubric=frozen_rubric,
            records=records,
            cross_index=cross_index,
            top_k=llm_deep_score_top_n,
        )

        if not shortlisted_ids:
            raise RuntimeError(
                "No candidates passed the embedding similarity filter for LLM scoring."
            )

        emit(4, "Embedding Shortlisting", len(shortlisted_ids), len(records),
             f"{len(shortlisted_ids)} candidates shortlisted for LLM scoring",
             done=True)

        # ── Step 5: Evidence-Grounded Q&A Scoring ────────────────────────────
        total_qa = len(shortlisted_ids) * len(frozen_rubric.weighted_criteria)
        emit(5, "Evidence-Grounded Scoring", 0, total_qa,
             "Starting Q&A scoring...")

        def qa_progress(current, total, name):
            emit(5, "Evidence-Grounded Scoring",
                 current * len(frozen_rubric.weighted_criteria), total_qa,
                 f"Scoring '{name}' ({current}/{total})")

        records = score_all_candidates(
            shortlisted_ids=shortlisted_ids,
            records=records,
            stores=stores,
            rubric=frozen_rubric,
            progress_callback=qa_progress,
        )

        emit(5, "Evidence-Grounded Scoring", total_qa, total_qa,
             "All criteria scored", done=True)

        # ── Step 6: Parallel Passes ───────────────────────────────────────────
        emit(6, "Parallel Analysis", 0, len(shortlisted_ids),
             "Running open signals, red flags, hard gates...")

        def passes_progress(current, total, name):
            emit(6, "Parallel Analysis", current, total, f"Analysing '{name}'")

        records = run_all_parallel_passes(
            shortlisted_ids=shortlisted_ids,
            records=records,
            rubric=frozen_rubric,
            progress_callback=passes_progress,
        )

        emit(6, "Parallel Analysis", len(shortlisted_ids), len(shortlisted_ids),
             "Signals, flags, gates complete", done=True)

        # ── Step 7: Weight Engine with behavioral merge ───────────────────────
        emit(7, "Final Scoring & Ranking", 0, len(shortlisted_ids),
             "Computing merged LLM + behavioral scores...")

        ranked_candidates, excluded_candidates = rank_candidates(
            shortlisted_ids=shortlisted_ids,
            records=records,
            rubric=frozen_rubric,
        )

        # Merge LLM final_score (70%) + behavioral composite (30%)
        for record in ranked_candidates:
            bscore = score_map.get(record.candidate_id)
            if bscore is not None:
                llm_component = record.final_score / 10.0  # normalise to 0-1
                merged = llm_component * 0.70 + bscore.composite_score * 0.30
                record.final_score = round(min(merged * 10.0, 10.0), 4)

        # Re-sort after merge
        ranked_candidates.sort(key=lambda c: c.final_score, reverse=True)
        for i, cand in enumerate(ranked_candidates):
            cand.rank = i + 1

        emit(7, "Final Scoring & Ranking", len(shortlisted_ids), len(shortlisted_ids),
             f"{len(ranked_candidates)} ranked, {len(excluded_candidates)} excluded",
             done=True)

        # ── Step 8: Output Assembly ───────────────────────────────────────────
        emit(8, "Output Generation", 0, 3, "Generating reports...")

        shortlist = assemble_output(
            run_id=run_id,
            rubric=frozen_rubric,
            ranked_candidates=ranked_candidates,
            excluded_candidates=excluded_candidates,
            total_submitted=total_loaded,
            run_dir=run_dir,
        )

        # Save structured scores for UI persistence
        import json as _json
        scores_file = run_dir / "structured_scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            _json.dump([s.model_dump() for s in all_scores], f, indent=2)

        emit(8, "Output Generation", 3, 3, "Reports ready", done=True)

        run.shortlist = shortlist
        run.completed = True

        logger.info(
            f"JSONL pipeline run {run_id} COMPLETE | "
            f"{total_loaded} total candidates | "
            f"{len(ranked_candidates)} LLM-ranked | "
            f"{len(all_scores)} behaviorally scored"
        )

    except Exception as exc:
        logger.exception(f"JSONL pipeline run {run_id} FAILED: {exc}")
        run.errors.append(str(exc))
        emit(0, "Error", error=str(exc))

    return run
