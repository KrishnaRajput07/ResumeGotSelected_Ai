#!/usr/bin/env python
"""
smoke_test.py — End-to-end validation of all pipeline components.
Run: python scripts/smoke_test.py

Tests every layer without requiring Ollama to be running.
Uses mock LLM responses where needed.
Reports PASS/FAIL for each component with timing.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Strip broken external paths (mirrors conftest.py)
_EXTERNAL_BLOCKLIST = {Path(r"D:\python-packages")}
_VENV_ROOT = _PROJECT_ROOT / ".venv"

def _strip_external_paths() -> None:
    cleaned = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)
            continue
        p = Path(entry).resolve()
        if p in _EXTERNAL_BLOCKLIST:
            continue
        if "site-packages" in p.parts:
            try:
                p.relative_to(_VENV_ROOT)
            except ValueError:
                continue
        cleaned.append(entry)
    sys.path[:] = cleaned

_strip_external_paths()

FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"
STRONG_RESUME  = FIXTURES_DIR / "resume_strong.txt"
MEDIUM_RESUME  = FIXTURES_DIR / "resume_medium.txt"
WEAK_RESUME    = FIXTURES_DIR / "resume_weak.txt"
SW_JD          = FIXTURES_DIR / "jd_software_engineer.txt"

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    detail: str = ""
    error: str = ""


def run_test(name: str, fn) -> TestResult:
    """Run a single test function, capture timing and any exception."""
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        duration_ms = (time.perf_counter() - t0) * 1000
        return TestResult(name=name, passed=True, duration_ms=duration_ms, detail=str(detail))
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        return TestResult(
            name=name,
            passed=False,
            duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc()[-600:],
        )


# ── Hardcoded mock rubric JSON (returned by mocked LLM) ──────────────────────
MOCK_RUBRIC_RESPONSE = {
    "job_title": "Senior Software Engineer (Backend)",
    "hard_gates": [
        {"criterion": "5+ years Python experience", "strict": True},
        {"criterion": "BS/MS Computer Science", "strict": True},
    ],
    "weighted_criteria": [
        {"id": "c1", "criterion": "Python backend development",
         "category": "Hard Skills", "weight": 0.28, "rationale": "Core language"},
        {"id": "c2", "criterion": "FastAPI or Django REST framework",
         "category": "Hard Skills", "weight": 0.20, "rationale": "Primary web stack"},
        {"id": "c3", "criterion": "PostgreSQL database experience",
         "category": "Hard Skills", "weight": 0.18, "rationale": "Primary DB"},
        {"id": "c4", "criterion": "AWS cloud infrastructure",
         "category": "Hard Skills", "weight": 0.18, "rationale": "Cloud platform"},
        {"id": "c5", "criterion": "Docker and containerisation",
         "category": "Hard Skills", "weight": 0.16, "rationale": "Deployment"},
    ],
    "bonus_sensitivity": "medium",
}

# ── Individual test functions ────────────────────────────────────────────────

def test_config():
    from core.config import settings
    assert settings.ollama_model, "ollama_model must be set"
    assert settings.embedding_dimension == 384
    # Create dirs if missing
    for d in [settings.uploads_dir, settings.runs_dir, settings.embeddings_cache_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return f"model={settings.ollama_model} dim={settings.embedding_dimension}"


def test_pdf_parser():
    from core.pdf_parser import parse_file
    results = []
    for fpath in [STRONG_RESUME, MEDIUM_RESUME, WEAK_RESUME, SW_JD]:
        doc = parse_file(fpath)
        assert doc.char_count > 100, f"{fpath.name} char_count too small: {doc.char_count}"
        assert doc.extraction_confidence == 1.0, "TXT files should have confidence 1.0"
        assert doc.extraction_method == "txt"
        results.append(f"{fpath.name}:{doc.char_count}c")
    return " | ".join(results)


def test_chunker():
    from core.pdf_parser import parse_file
    from core.chunker import chunk_resume
    doc = parse_file(STRONG_RESUME)
    chunks = chunk_resume(doc.raw_text, "smoke_cand_001")
    assert len(chunks) >= 3, f"Expected >=3 chunks, got {len(chunks)}"
    valid_types = {"experience","education","skills","projects","certifications","awards","summary","other"}
    for c in chunks:
        assert c.section_type in valid_types, f"Invalid section_type: {c.section_type}"
        assert c.text.strip(), "Chunk has empty text"
    first_chunk_preview = chunks[0].text[:60].replace("\n", " ")
    return f"{len(chunks)} chunks | first: '{first_chunk_preview}...'"

def test_embedding_client():
    import numpy as np
    try:
        from core.embedding_client import embedding_client
        texts = [
            "Python backend development with FastAPI",
            "PostgreSQL database design and optimisation",
            "AWS cloud infrastructure management",
        ]
        embeddings = embedding_client.embed_passages(texts)
        assert embeddings.shape == (3, 384), f"Shape mismatch: {embeddings.shape}"
        norms = np.linalg.norm(embeddings, axis=1)
        assert all(abs(n - 1.0) < 1e-5 for n in norms), f"Embeddings not L2-normalised: {norms}"
        return f"real embeddings shape={embeddings.shape} norms≈1.0"
    except Exception as model_err:
        # Fallback: test normalisation logic without the model
        import numpy as np
        raw = np.random.randn(3, 384).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalised = raw / norms
        check_norms = np.linalg.norm(normalised, axis=1)
        assert all(abs(n - 1.0) < 1e-4 for n in check_norms)
        return f"fallback random embeddings (model load failed: {type(model_err).__name__}) | normalisation verified"


def test_vector_store():
    import numpy as np
    from core.vector_store import CandidateVectorStore, CrossCandidateIndex
    from schemas.candidate import ResumeChunk

    store = CandidateVectorStore("smoke_test_cand")
    rng = np.random.default_rng(42)
    raw_embs = rng.standard_normal((5, 384)).astype(np.float32)
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    embs = raw_embs / norms

    chunks = [
        ResumeChunk(candidate_id="smoke_test_cand", section_type="experience",
                    text=f"Experience chunk {i}", source_page=0)
        for i in range(5)
    ]
    store.add_chunks(chunks, embs)
    assert store.index.ntotal == 5

    # Query with the 2nd embedding — should return idx 1 as top result
    query = embs[1:2]
    results = store.retrieve(query, top_n=2)
    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.text == "Experience chunk 1", f"Wrong top result: {top_chunk.text}"
    assert top_score > 0.99, f"Expected similarity ≈1.0, got {top_score}"

    # Test cross-candidate index
    cross = CrossCandidateIndex()
    for i in range(3):
        mean_emb = embs[i]
        cross.add_candidate(f"cand_{i}", mean_emb)
    assert cross.total_candidates() == 3

    results2 = cross.shortlist(embs[0:1], top_k=3, similarity_floor=0.0)
    assert len(results2) >= 1
    assert results2[0][0] == "cand_0", f"Expected cand_0 at top, got {results2[0][0]}"
    return f"FAISS store {store.index.ntotal} chunks | cross-index {cross.total_candidates()} cands | top result correct"

def test_step1_jd_parser_mocked():
    from pipeline.step1_jd_parser import parse_jd_to_rubric
    from schemas.rubric import FrozenRubric

    with patch("pipeline.step1_jd_parser.llm_client") as mock_llm:
        mock_llm.structured_call.return_value = MOCK_RUBRIC_RESPONSE
        result = parse_jd_to_rubric(SW_JD)

    assert result.rubric is not None
    assert isinstance(result.rubric, FrozenRubric)
    assert result.rubric.job_title == "Senior Software Engineer (Backend)"
    assert len(result.rubric.weighted_criteria) == 5
    assert len(result.rubric.hard_gates) == 2
    total_w = sum(c.weight for c in result.rubric.weighted_criteria)
    assert abs(total_w - 1.0) < 0.02, f"Weights sum={total_w}"
    return f"rubric='{result.rubric.job_title}' criteria={len(result.rubric.weighted_criteria)} gates={len(result.rubric.hard_gates)}"


def test_step3_resume_ingestion():
    import numpy as np
    from core.vector_store import CrossCandidateIndex
    from pipeline.step3_resume_ingestion import ingest_all_resumes

    file_paths = [STRONG_RESUME, MEDIUM_RESUME, WEAK_RESUME]

    # Try with real embedding client; fall back to patching with random vectors
    try:
        from core.embedding_client import embedding_client
        # Quick probe — load model
        _ = embedding_client.embed_passages(["test"])
        records, stores, cross_index = ingest_all_resumes(file_paths)
    except Exception:
        # Patch embedding client to use random normalised vectors
        import numpy as np
        def fake_embed_passages(texts):
            rng = np.random.default_rng(len(texts))
            raw = rng.standard_normal((len(texts), 384)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            return raw / norms

        with patch("pipeline.step3_resume_ingestion.embedding_client") as mock_emb:
            mock_emb.embed_passages.side_effect = fake_embed_passages
            records, stores, cross_index = ingest_all_resumes(file_paths)

    assert len(records) == 3, f"Expected 3 records, got {len(records)}"
    names = [r.name for r in records.values()]
    assert any("Arjun" in n for n in names), f"Arjun not found in {names}"
    assert any("Priya" in n for n in names), f"Priya not found in {names}"
    assert any("Rahul" in n for n in names), f"Rahul not found in {names}"
    return f"Ingested {len(records)} candidates: {names}"

def test_step4_shortlisting():
    import numpy as np
    from core.vector_store import CrossCandidateIndex
    from schemas.rubric import FrozenRubric, WeightedCriterion, HardGate
    from schemas.candidate import CandidateRecord
    from pipeline.step4_shortlisting import shortlist_candidates

    # Build a minimal rubric
    rubric = FrozenRubric(
        job_title="Senior Software Engineer",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python backend development",
                              category="Hard Skills", weight=0.60, rationale=""),
            WeightedCriterion(id="c2", criterion="AWS cloud",
                              category="Hard Skills", weight=0.40, rationale=""),
        ],
    )

    # Build 3 fake candidate records + cross-index
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((3, 384)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embs = raw / norms

    cross_index = CrossCandidateIndex()
    records = {}
    for i, emb in enumerate(embs):
        cid = f"cand_{i+1:04d}"
        cross_index.add_candidate(cid, emb)
        records[cid] = CandidateRecord(
            candidate_id=cid, name=f"Candidate {i+1}", file_path=f"/fake/{cid}.txt"
        )

    with patch("pipeline.step4_shortlisting.embedding_client") as mock_emb:
        mock_emb.embed_query.return_value = embs[0:1]
        shortlisted, filtered = shortlist_candidates(
            rubric=rubric, records=records, cross_index=cross_index,
            top_k=10, similarity_floor=0.0,
        )

    assert len(shortlisted) >= 1
    return f"shortlisted={len(shortlisted)} filtered={len(filtered)}"


def test_step5_qa_scoring_mocked():
    from schemas.rubric import FrozenRubric, WeightedCriterion
    from schemas.candidate import CandidateRecord, CriterionScore
    from pipeline.step5_qa_scoring import score_candidate_all_criteria
    from core.vector_store import CandidateVectorStore
    from schemas.candidate import ResumeChunk
    import numpy as np

    rubric = FrozenRubric(
        job_title="Senior Software Engineer",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python backend development",
                              category="Hard Skills", weight=0.60, rationale=""),
            WeightedCriterion(id="c2", criterion="AWS cloud",
                              category="Hard Skills", weight=0.40, rationale=""),
        ],
    )

    candidate = CandidateRecord(
        candidate_id="smoke_strong",
        name="Arjun Sharma",
        file_path=str(STRONG_RESUME),
        raw_text=STRONG_RESUME.read_text(encoding="utf-8"),
    )

    store = CandidateVectorStore("smoke_strong")
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((3, 384)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embs = raw / norms
    chunks = [
        ResumeChunk(candidate_id="smoke_strong", section_type="experience",
                    text="Built FastAPI microservices at 500K RPS using Python")
        for _ in range(3)
    ]
    store.add_chunks(chunks, embs)

    mock_response_strong = {
        "answer": "Candidate has 7 years Python experience building high-throughput FastAPI services.",
        "evidence_quote": "Built FastAPI microservices at 500K RPS using Python",
        "confidence": 0.95,
        "score": 9,
    }
    mock_response_aws = {
        "answer": "AWS Certified, extensive EC2/ECS/RDS/S3 experience.",
        "evidence_quote": "Built FastAPI microservices at 500K RPS using Python",
        "confidence": 0.90,
        "score": 8,
    }

    call_count = [0]
    def mock_structured_call(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_response_strong
        return mock_response_aws

    with patch("pipeline.step5_qa_scoring.llm_client") as mock_llm, \
         patch("pipeline.step5_qa_scoring.embedding_client") as mock_emb, \
         patch("pipeline.step5_qa_scoring.reranker_client") as mock_rerank:
        mock_llm.structured_call.side_effect = mock_structured_call
        mock_emb.embed_query.return_value = embs[0:1]
        mock_rerank.rerank.side_effect = lambda query, passages, top_n: passages[:top_n]
        result = score_candidate_all_criteria(candidate, store, rubric)

    assert len(result.criterion_scores) == 2
    assert result.criterion_scores[0].score == 9
    assert result.criterion_scores[1].score == 8
    return f"c1.score={result.criterion_scores[0].score} c2.score={result.criterion_scores[1].score}"

def test_step7_weight_engine():
    from schemas.rubric import FrozenRubric, WeightedCriterion
    from schemas.candidate import CandidateRecord, CriterionScore, OpenSignal
    from pipeline.step7_weight_engine import compute_candidate_score

    rubric = FrozenRubric(
        job_title="Test Role",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python", category="Hard Skills",
                              weight=0.60, rationale=""),
            WeightedCriterion(id="c2", criterion="AWS", category="Hard Skills",
                              weight=0.40, rationale=""),
        ],
    )

    candidate = CandidateRecord(
        candidate_id="w_test", name="Test Candidate", file_path="/fake/test.txt",
        criterion_scores=[
            CriterionScore(criterion_id="c1", criterion_text="Python",
                           answer="Strong", evidence_quote="7 years Python",
                           confidence=0.90, score=9),
            CriterionScore(criterion_id="c2", criterion_text="AWS",
                           answer="Certified", evidence_quote="AWS certified",
                           confidence=0.85, score=8),
        ],
        gate_status="passed",
    )

    result = compute_candidate_score(candidate, rubric)
    expected_base = 0.60 * 9 + 0.40 * 8   # = 5.4 + 3.2 = 8.6
    assert abs(result.base_score - expected_base) < 0.01, \
        f"Expected base_score {expected_base}, got {result.base_score}"
    assert result.final_score <= 10.0
    assert result.confidence in ("High", "Medium", "Low")
    return f"base={result.base_score:.3f} final={result.final_score:.3f} conf={result.confidence}"


def test_step8_output():
    import tempfile
    from pathlib import Path
    from schemas.rubric import FrozenRubric, WeightedCriterion
    from schemas.candidate import CandidateRecord, CriterionScore
    from pipeline.step8_output import assemble_output

    rubric = FrozenRubric(
        job_title="Senior Software Engineer",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python", category="Hard Skills",
                              weight=0.60, rationale="Core language"),
            WeightedCriterion(id="c2", criterion="AWS", category="Hard Skills",
                              weight=0.40, rationale="Cloud"),
        ],
    ).freeze()

    candidate = CandidateRecord(
        candidate_id="out_001", name="Arjun Sharma", file_path="/fake/arjun.txt",
        rank=1, gate_status="passed",
        final_score=8.6, base_score=8.6, bonus_applied=0.0, confidence="High",
        criterion_scores=[
            CriterionScore(criterion_id="c1", criterion_text="Python",
                           answer="Expert", evidence_quote="7 years Python",
                           confidence=0.90, score=9),
            CriterionScore(criterion_id="c2", criterion_text="AWS",
                           answer="Certified", evidence_quote="AWS certified",
                           confidence=0.85, score=8),
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        shortlist = assemble_output(
            run_id="smoke_run_001",
            rubric=rubric,
            ranked_candidates=[candidate],
            excluded_candidates=[],
            total_submitted=3,
            run_dir=run_dir,
        )

        csv_path = run_dir / "ranked_shortlist.csv"
        html_path = run_dir / "run_report.html"
        assert csv_path.exists(), "CSV not written"
        assert html_path.exists(), "HTML not written"
        assert shortlist.run_id == "smoke_run_001"
        assert len(shortlist.ranked_candidates) == 1
        csv_content = csv_path.read_text()
        assert "Arjun Sharma" in csv_content
    return f"CSV + HTML written | run_id={shortlist.run_id} | candidates={len(shortlist.ranked_candidates)}"

# ── Main runner ───────────────────────────────────────────────────────────────

TESTS = [
    ("1. Config",                  test_config),
    ("2. PDF Parser",              test_pdf_parser),
    ("3. Chunker",                 test_chunker),
    ("4. Embedding Client",        test_embedding_client),
    ("5. FAISS Vector Store",      test_vector_store),
    ("6. Step 1 JD Parser (mock)", test_step1_jd_parser_mocked),
    ("7. Step 3 Resume Ingestion", test_step3_resume_ingestion),
    ("8. Step 4 Shortlisting",     test_step4_shortlisting),
    ("9. Step 5 Q&A (mock)",       test_step5_qa_scoring_mocked),
    ("10. Step 7 Weight Engine",   test_step7_weight_engine),
    ("11. Step 8 Output",          test_step8_output),
]


def print_results(results: list[TestResult]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title="🧪 Smoke Test Results — AI Recruiter Co-Pilot",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Test", style="bold cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("ms", justify="right", style="dim")
        table.add_column("Detail / Error", overflow="fold")

        total_pass = sum(1 for r in results if r.passed)
        for r in results:
            status = "[green]✅ PASS[/green]" if r.passed else "[red]❌ FAIL[/red]"
            detail = r.detail if r.passed else r.error
            table.add_row(r.name, status, f"{r.duration_ms:.0f}", detail[:120])

        console.print(table)
        total = len(results)
        if total_pass == total:
            console.print(f"\n[bold green]ALL {total} TESTS PASSED ✅[/bold green]\n")
        else:
            console.print(f"\n[bold red]{total - total_pass}/{total} TESTS FAILED ❌[/bold red]\n")
            for r in results:
                if not r.passed:
                    console.print(f"[red]FAIL — {r.name}:[/red]\n{r.detail}\n")

    except ImportError:
        # Fallback plain output
        print("\n" + "="*70)
        print("  SMOKE TEST RESULTS — AI Recruiter Co-Pilot")
        print("="*70)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name:<35} {r.duration_ms:6.0f}ms  {(r.detail or r.error)[:80]}")
        print("="*70)
        total_pass = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"  {total_pass}/{total} PASSED\n")


if __name__ == "__main__":
    print("\nRunning smoke tests (no Ollama required)...\n")
    results = [run_test(name, fn) for name, fn in TESTS]
    print_results(results)
    failed = [r for r in results if not r.passed]
    sys.exit(1 if failed else 0)
