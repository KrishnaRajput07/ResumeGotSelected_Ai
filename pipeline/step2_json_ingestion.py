"""
Step 2: JSON Candidate Ingestion
Converts structured JSON candidate profiles into:
  (a) Rich natural language text → feeds into existing chunker/embedding pipeline
  (b) StructuredCandidate objects → feeds into behavioral/feature scoring

This module bridges the structured JSONL format used in the Redrob Challenge
with the existing resume-ingestion pipeline (step3_resume_ingestion.py).
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from core.chunker import chunk_resume
from core.embedding_client import embedding_client
from core.vector_store import CandidateVectorStore, CrossCandidateIndex
from schemas.candidate import (
    CandidateRecord,
    StructuredCandidate,
    CareerEntry,
    EducationEntry,
    SkillEntry,
    CertificationEntry,
    RedrobSignals,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Text Conversion
# ─────────────────────────────────────────────────────────────────────────────


def json_to_resume_text(candidate: StructuredCandidate) -> str:
    """
    Convert a StructuredCandidate into natural language resume text.

    The output format deliberately mirrors common resume section headers that
    the existing chunker (SECTION_HEADERS regex) recognises, ensuring each
    section becomes its own semantic chunk for retrieval.

    Args:
        candidate: A fully-parsed StructuredCandidate object.

    Returns:
        Multi-section natural language string ready for chunking/embedding.
    """
    lines: list[str] = []

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    lines.append("SUMMARY")
    if candidate.headline:
        lines.append(candidate.headline)
    if candidate.summary:
        lines.append(candidate.summary)

    # Add location / experience metadata as a compact line for retrieval
    meta_parts = []
    if candidate.location:
        meta_parts.append(f"Location: {candidate.location}")
    if candidate.country:
        meta_parts.append(f"Country: {candidate.country}")
    if candidate.years_of_experience:
        meta_parts.append(f"Years of experience: {candidate.years_of_experience:.1f}")
    if meta_parts:
        lines.append(" | ".join(meta_parts))

    lines.append("")

    # ── WORK EXPERIENCE ───────────────────────────────────────────────────────
    if candidate.career_history:
        lines.append("WORK EXPERIENCE")
        for entry in candidate.career_history:
            end_label = entry.end_date if entry.end_date else "Present"
            yrs = entry.duration_months // 12
            mos = entry.duration_months % 12
            duration_str = f"{yrs}yr {mos}mo" if yrs else f"{mos}mo"

            lines.append(
                f"{entry.title} at {entry.company} "
                f"({entry.start_date} – {end_label}) | {duration_str}"
            )
            if entry.industry or entry.company_size:
                lines.append(
                    f"Industry: {entry.industry} | Company size: {entry.company_size}"
                )
            if entry.description:
                lines.append(entry.description)
            lines.append("")

    # ── EDUCATION ─────────────────────────────────────────────────────────────
    if candidate.education:
        lines.append("EDUCATION")
        for edu in candidate.education:
            grade_str = f" | {edu.grade}" if edu.grade else ""
            lines.append(
                f"{edu.degree} in {edu.field_of_study}, "
                f"{edu.institution} "
                f"({edu.start_year}–{edu.end_year}) | {edu.tier}{grade_str}"
            )
        lines.append("")

    # ── SKILLS ───────────────────────────────────────────────────────────────
    if candidate.skills:
        lines.append("SKILLS")
        # Group by proficiency level
        by_level: dict[str, list[str]] = {
            "expert": [],
            "advanced": [],
            "intermediate": [],
            "beginner": [],
        }
        for sk in candidate.skills:
            level = sk.proficiency.lower()
            if level in by_level:
                by_level[level].append(sk.name)

        for level in ("expert", "advanced", "intermediate", "beginner"):
            names = by_level[level]
            if names:
                lines.append(f"{level.capitalize()}: {', '.join(names)}")
        lines.append("")

    # ── CERTIFICATIONS ────────────────────────────────────────────────────────
    if candidate.certifications:
        lines.append("CERTIFICATIONS")
        for cert in candidate.certifications:
            lines.append(f"{cert.name} from {cert.issuer} ({cert.year})")
        lines.append("")

    # ── LANGUAGES ─────────────────────────────────────────────────────────────
    if candidate.languages:
        lines.append("LANGUAGES")
        for lang in candidate.languages:
            proficiency = lang.get("proficiency", "")
            language = lang.get("language", "")
            if language:
                lines.append(f"{language}: {proficiency}" if proficiency else language)
        lines.append("")

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# JSONL Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(
    path: str | Path,
    max_candidates: int | None = None,
) -> list[StructuredCandidate]:
    """
    Load and parse a candidates.jsonl file into StructuredCandidate objects.

    Malformed records are logged and skipped — never raises on bad data.
    Progress is logged every 10,000 records.

    Args:
        path: Path to the .jsonl or .json file.
        max_candidates: If set, stop after this many successfully parsed candidates.

    Returns:
        List of StructuredCandidate objects.
    """
    path = Path(path)
    candidates: list[StructuredCandidate] = []
    failed = 0
    line_num = 0

    logger.info(f"Loading JSON/JSONL candidates from: {path}")

    # Try reading as a standard single JSON array first
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    candidate = StructuredCandidate.model_validate(item)
                    candidates.append(candidate)
                    if max_candidates and len(candidates) >= max_candidates:
                        break
                except Exception as exc:
                    failed += 1
                    if failed <= 10:
                        logger.warning(f"JSON array item {i}: parse failed — {exc}")
            logger.info(
                f"JSON array load complete: {len(candidates)} candidates loaded, "
                f"{failed} failed"
            )
            return candidates
    except Exception as e:
        logger.info(f"File is not a valid JSON array ({e}), falling back to line-by-line JSONL reader.")

    # Fallback to line-by-line JSONL parsing
    failed = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line_num += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
                candidate = StructuredCandidate.model_validate(data)
                candidates.append(candidate)
            except Exception as exc:
                failed += 1
                if failed <= 10:  # Log only first 10 to avoid spam
                    logger.warning(
                        f"JSONL line {line_num}: parse failed — {exc}"
                    )

            if line_num % 10_000 == 0:
                logger.info(
                    f"JSONL progress: {line_num} lines read, "
                    f"{len(candidates)} parsed, {failed} failed"
                )

            if max_candidates and len(candidates) >= max_candidates:
                logger.info(f"Reached max_candidates={max_candidates}, stopping.")
                break

    logger.info(
        f"JSONL load complete: {len(candidates)} candidates loaded, "
        f"{failed} failed, {line_num} total lines"
    )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion (JSON → CandidateRecord + VectorStore)
# ─────────────────────────────────────────────────────────────────────────────


def ingest_json_candidate(
    candidate: StructuredCandidate,
    cross_candidate_index: CrossCandidateIndex,
    run_dir: Optional[Path] = None,
) -> tuple[CandidateRecord, CandidateVectorStore]:
    """
    Ingest a single StructuredCandidate into the vector pipeline.

    Converts the structured profile to natural language text, chunks it,
    embeds the chunks, and indexes them — identical to ingest_resume() but
    without file I/O.

    The file_path in the returned CandidateRecord is set to
    ``jsonl://{candidate_id}`` so downstream code can distinguish JSONL
    candidates from file-based ones.

    Args:
        candidate: Parsed StructuredCandidate object.
        cross_candidate_index: Shared cross-candidate FAISS index (mutated in-place).
        run_dir: Optional directory to persist the candidate's per-candidate index.

    Returns:
        (CandidateRecord, CandidateVectorStore)
    """
    cid = candidate.candidate_id
    logger.debug(f"[{cid}] Ingesting structured candidate: {candidate.name}")

    # Convert to natural language text
    resume_text = json_to_resume_text(candidate)

    # Chunk
    chunks = chunk_resume(resume_text, cid)

    record = CandidateRecord(
        candidate_id=cid,
        name=candidate.name,
        file_path=f"jsonl://{cid}",
        raw_text=resume_text,
        gate_status="pending",
    )

    store = CandidateVectorStore(cid)

    if not chunks:
        logger.warning(f"[{cid}] No chunks produced for {candidate.name}")
        return record, store

    # Embed
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_client.embed_passages(chunk_texts)

    # Index
    store.add_chunks(chunks, embeddings)

    if run_dir:
        idx_path = run_dir / "indices" / cid
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(idx_path)

    mean_emb = store.get_mean_embedding()
    if mean_emb is not None:
        cross_candidate_index.add_candidate(cid, mean_emb)

    return record, store


def ingest_json_candidates(
    candidates: list[StructuredCandidate],
    run_dir: Optional[Path] = None,
    progress_callback=None,
) -> tuple[dict[str, CandidateRecord], dict[str, CandidateVectorStore], CrossCandidateIndex]:
    """
    Ingest a list of StructuredCandidate objects in batch.

    Args:
        candidates: List of StructuredCandidate objects to ingest.
        run_dir: Optional run directory for persisting indices.
        progress_callback: Optional callable(current, total, name) for UI progress.

    Returns:
        - records: {candidate_id: CandidateRecord}
        - stores:  {candidate_id: CandidateVectorStore}
        - cross_index: CrossCandidateIndex ready for Step 4 shortlisting
    """
    records: dict[str, CandidateRecord] = {}
    stores: dict[str, CandidateVectorStore] = {}
    cross_index = CrossCandidateIndex()
    failed: list[tuple[str, str]] = []
    total = len(candidates)

    logger.info(f"Step 2: Ingesting {total} structured JSON candidates")

    for i, candidate in enumerate(candidates):
        if progress_callback:
            progress_callback(i + 1, total, candidate.name)

        try:
            record, store = ingest_json_candidate(
                candidate=candidate,
                cross_candidate_index=cross_index,
                run_dir=run_dir,
            )
            records[candidate.candidate_id] = record
            stores[candidate.candidate_id] = store
        except Exception as exc:
            logger.error(f"Failed to ingest {candidate.candidate_id}: {exc}")
            failed.append((candidate.candidate_id, str(exc)))

    if failed:
        logger.warning(f"{len(failed)} candidates failed ingestion:")
        for cid, err in failed[:5]:
            logger.warning(f"  {cid}: {err}")

    logger.info(
        f"Step 2 complete: {len(records)}/{total} ingested | "
        f"{cross_index.total_candidates()} in cross-candidate index"
    )
    return records, stores, cross_index
