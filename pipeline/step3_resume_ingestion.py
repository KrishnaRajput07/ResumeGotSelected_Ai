# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Resume Ingestion & Indexing
# ─────────────────────────────────────────────────────────────────────────────
# Per-candidate pipeline:
#   parse file → clean text → section-aware chunking → embed chunks → FAISS index
# Also builds cross-candidate index (Step 4 prerequisite).
# ─────────────────────────────────────────────────────────────────────────────

import logging
import re
from pathlib import Path

import numpy as np

from core.pdf_parser import parse_file
from core.chunker import chunk_resume
from core.embedding_client import embedding_client
from core.vector_store import CandidateVectorStore, CrossCandidateIndex
from schemas.candidate import CandidateRecord

logger = logging.getLogger(__name__)


def _extract_name_from_text(text: str, filename: str) -> str:
    """
    Heuristic: try to extract candidate name from first 3 lines of resume.
    Falls back to filename (minus extension) if extraction fails.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()][:5]
    for line in lines:
        # Name-like: 1-4 words, mostly alpha, no special chars, not an email/phone
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r"^[A-Za-z\.\-']+$", w) for w in words):
            return line
    # Fallback: use filename
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def ingest_resume(
    file_path: str | Path,
    candidate_id: str,
    cross_candidate_index: CrossCandidateIndex,
    run_dir: Path | None = None,
) -> tuple[CandidateRecord, CandidateVectorStore]:
    """
    Ingest a single resume: parse → chunk → embed → index.
    
    Args:
        file_path: Path to resume file (PDF, DOCX, TXT)
        candidate_id: Unique ID for this candidate
        cross_candidate_index: Shared cross-candidate index (mutated in-place)
        run_dir: Optional directory to save the candidate index for reuse
    
    Returns:
        (CandidateRecord with raw_text filled, CandidateVectorStore ready for retrieval)
    """
    path = Path(file_path)
    warnings = []

    # ── Parse ─────────────────────────────────────────────────────────────────
    logger.info(f"[{candidate_id}] Parsing: {path.name}")
    parsed = parse_file(path)
    
    if parsed.warnings:
        warnings.extend(parsed.warnings)

    name = _extract_name_from_text(parsed.raw_text, path.name)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_resume(parsed.raw_text, candidate_id)
    
    if not chunks:
        logger.warning(f"[{candidate_id}] No chunks produced — skipping embedding")
        # Still return a record so we can track the failure
        record = CandidateRecord(
            candidate_id=candidate_id,
            name=name,
            file_path=str(path),
            raw_text=parsed.raw_text,
            gate_status="pending",
        )
        store = CandidateVectorStore(candidate_id)
        return record, store

    # ── Embed ─────────────────────────────────────────────────────────────────
    chunk_texts = [c.text for c in chunks]
    logger.info(f"[{candidate_id}] Embedding {len(chunks)} chunks...")
    embeddings = embedding_client.embed_passages(chunk_texts)

    # ── Per-Candidate Index ───────────────────────────────────────────────────
    store = CandidateVectorStore(candidate_id)
    store.add_chunks(chunks, embeddings)

    # Save index if run_dir provided
    if run_dir:
        idx_path = run_dir / "indices" / candidate_id
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(idx_path)

    # ── Cross-Candidate Index ─────────────────────────────────────────────────
    mean_emb = store.get_mean_embedding()
    if mean_emb is not None:
        cross_candidate_index.add_candidate(candidate_id, mean_emb)

    # ── Build CandidateRecord ─────────────────────────────────────────────────
    record = CandidateRecord(
        candidate_id=candidate_id,
        name=name,
        file_path=str(path),
        raw_text=parsed.raw_text,
        gate_status="pending",
    )

    logger.info(
        f"[{candidate_id}] Ingested: '{name}' | "
        f"{len(chunks)} chunks | "
        f"confidence={parsed.extraction_confidence:.2f}"
    )
    return record, store


def ingest_all_resumes(
    file_paths: list[str | Path],
    run_dir: Path | None = None,
    progress_callback=None,
) -> tuple[dict[str, CandidateRecord], dict[str, CandidateVectorStore], CrossCandidateIndex]:
    """
    Ingest all resumes in a batch.
    
    Args:
        file_paths: List of resume file paths
        run_dir: Optional directory to persist indices
        progress_callback: Optional callable(current, total, name) for UI progress
    
    Returns:
        - records: {candidate_id: CandidateRecord}
        - stores: {candidate_id: CandidateVectorStore}
        - cross_index: CrossCandidateIndex ready for Step 4
    """
    records: dict[str, CandidateRecord] = {}
    stores: dict[str, CandidateVectorStore] = {}
    cross_index = CrossCandidateIndex()
    failed = []

    for i, file_path in enumerate(file_paths):
        candidate_id = f"cand_{i+1:04d}"
        
        if progress_callback:
            progress_callback(i + 1, len(file_paths), Path(file_path).name)

        try:
            record, store = ingest_resume(
                file_path=file_path,
                candidate_id=candidate_id,
                cross_candidate_index=cross_index,
                run_dir=run_dir,
            )
            records[candidate_id] = record
            stores[candidate_id] = store
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            failed.append((file_path, str(e)))

    if failed:
        logger.warning(f"{len(failed)} files failed to ingest:")
        for fp, err in failed:
            logger.warning(f"  {fp}: {err}")

    logger.info(
        f"Ingestion complete: {len(records)}/{len(file_paths)} successful | "
        f"{cross_index.total_candidates()} in cross-candidate index"
    )
    return records, stores, cross_index
