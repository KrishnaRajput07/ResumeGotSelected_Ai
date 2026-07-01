"""
Integration tests — validate real component behavior without LLM.

These tests DO use actual embedding model, PDF parser, chunker, FAISS.
Only the LLM (Ollama) is mocked.

Run: python -m pytest tests/test_integration.py -v
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from schemas.rubric import FrozenRubric, WeightedCriterion, HardGate
from schemas.candidate import CandidateRecord, CriterionScore, ResumeChunk

# ── Fixtures dir ──────────────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "fixtures"
STRONG_RESUME = FIXTURES_DIR / "resume_strong.txt"
MEDIUM_RESUME = FIXTURES_DIR / "resume_medium.txt"
WEAK_RESUME   = FIXTURES_DIR / "resume_weak.txt"
SW_JD         = FIXTURES_DIR / "jd_software_engineer.txt"


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def strong_resume_text():
    return STRONG_RESUME.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def medium_resume_text():
    return MEDIUM_RESUME.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def weak_resume_text():
    return WEAK_RESUME.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sw_jd_text():
    return SW_JD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def strong_rubric():
    return FrozenRubric(
        job_title="Senior Software Engineer (Backend)",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Python backend development",
                              category="Hard Skills", weight=0.28, rationale="Core"),
            WeightedCriterion(id="c2", criterion="FastAPI or Django REST framework",
                              category="Hard Skills", weight=0.20, rationale="API"),
            WeightedCriterion(id="c3", criterion="PostgreSQL database experience",
                              category="Hard Skills", weight=0.18, rationale="DB"),
            WeightedCriterion(id="c4", criterion="AWS cloud infrastructure",
                              category="Hard Skills", weight=0.18, rationale="Cloud"),
            WeightedCriterion(id="c5", criterion="Docker containerisation",
                              category="Hard Skills", weight=0.16, rationale="Deploy"),
        ],
        hard_gates=[
            HardGate(criterion="5+ years Python experience", strict=True),
        ],
    )

# ── Parsing tests ─────────────────────────────────────────────────────────────

def test_parse_txt_resume_strong():
    from core.pdf_parser import parse_file
    doc = parse_file(STRONG_RESUME)
    assert doc.char_count > 500, f"Expected char_count>500, got {doc.char_count}"
    assert doc.extraction_confidence == 1.0, "TXT should have confidence=1.0"
    assert doc.extraction_method == "txt"


def test_parse_txt_resume_produces_name(strong_resume_text):
    from pipeline.step3_resume_ingestion import _extract_name_from_text
    name = _extract_name_from_text(strong_resume_text, "resume_strong.txt")
    assert "Arjun" in name and "Sharma" in name, \
        f"Expected 'Arjun Sharma' in name, got '{name}'"


def test_parse_medium_resume():
    from core.pdf_parser import parse_file
    doc = parse_file(MEDIUM_RESUME)
    assert doc.char_count > 300
    assert doc.extraction_method == "txt"


def test_parse_weak_resume():
    from core.pdf_parser import parse_file
    doc = parse_file(WEAK_RESUME)
    assert doc.char_count > 200
    assert doc.extraction_method == "txt"


# ── Chunking tests ────────────────────────────────────────────────────────────

VALID_SECTION_TYPES = {
    "experience", "education", "skills", "projects",
    "certifications", "awards", "summary", "other"
}


def test_chunker_produces_multiple_chunks(strong_resume_text):
    from core.chunker import chunk_resume
    chunks = chunk_resume(strong_resume_text, "integ_cand_01")
    assert len(chunks) >= 3, f"Expected >=3 chunks, got {len(chunks)}"


def test_chunker_section_types_valid(strong_resume_text):
    from core.chunker import chunk_resume
    chunks = chunk_resume(strong_resume_text, "integ_cand_02")
    for chunk in chunks:
        assert chunk.section_type in VALID_SECTION_TYPES, \
            f"Invalid section_type: '{chunk.section_type}'"


def test_chunker_no_empty_chunks(strong_resume_text):
    from core.chunker import chunk_resume
    chunks = chunk_resume(strong_resume_text, "integ_cand_03")
    for i, chunk in enumerate(chunks):
        assert chunk.text.strip(), f"Chunk {i} has empty/whitespace-only text"


def test_chunker_experience_chunks_present(strong_resume_text):
    from core.chunker import chunk_resume
    chunks = chunk_resume(strong_resume_text, "integ_cand_04")
    exp_chunks = [c for c in chunks if c.section_type == "experience"]
    assert len(exp_chunks) >= 1, \
        f"Expected >=1 experience chunk in strong resume, got {len(exp_chunks)}"


def test_chunker_weak_resume(weak_resume_text):
    from core.chunker import chunk_resume
    chunks = chunk_resume(weak_resume_text, "integ_weak_01")
    # Weak resume should still produce at least 1 chunk
    assert len(chunks) >= 1


def test_chunker_candidate_id_propagated(strong_resume_text):
    from core.chunker import chunk_resume
    cid = "integ_cand_id_test"
    chunks = chunk_resume(strong_resume_text, cid)
    for chunk in chunks:
        assert chunk.candidate_id == cid, \
            f"Expected candidate_id='{cid}', got '{chunk.candidate_id}'"

# ── Vector store tests ────────────────────────────────────────────────────────

def _make_normalised_embeddings(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def test_vector_store_add_and_retrieve():
    from core.vector_store import CandidateVectorStore
    store = CandidateVectorStore("integ_vs_01")
    embs = _make_normalised_embeddings(5, seed=99)
    chunks = [
        ResumeChunk(candidate_id="integ_vs_01", section_type="experience",
                    text=f"Chunk {i}", source_page=0)
        for i in range(5)
    ]
    store.add_chunks(chunks, embs)
    assert store.index.ntotal == 5

    # Query with 3rd embedding — should retrieve chunk 2 as top
    query = embs[2:3]
    results = store.retrieve(query, top_n=2)
    assert len(results) >= 1
    top_chunk, top_score = results[0]
    assert top_chunk.text == "Chunk 2", f"Expected 'Chunk 2', got '{top_chunk.text}'"
    assert top_score > 0.99


def test_cross_candidate_index_shortlisting():
    from core.vector_store import CrossCandidateIndex
    embs = _make_normalised_embeddings(3, seed=7)
    cross = CrossCandidateIndex()
    for i, emb in enumerate(embs):
        cross.add_candidate(f"cid_{i}", emb)

    assert cross.total_candidates() == 3
    # Query similar to first candidate
    results = cross.shortlist(embs[0:1], top_k=3, similarity_floor=0.0)
    assert len(results) >= 1
    assert results[0][0] == "cid_0", f"Expected cid_0 at top, got {results[0][0]}"


def test_vector_store_empty_returns_empty():
    from core.vector_store import CandidateVectorStore
    store = CandidateVectorStore("empty_test")
    emb = _make_normalised_embeddings(1, seed=1)
    results = store.retrieve(emb, top_n=5)
    assert results == []


# ── Weight engine tests ───────────────────────────────────────────────────────

def make_rubric_3crit() -> FrozenRubric:
    return FrozenRubric(
        job_title="Test Role",
        weighted_criteria=[
            WeightedCriterion(id="c1", criterion="Skill A", category="Hard Skills",
                              weight=0.50, rationale=""),
            WeightedCriterion(id="c2", criterion="Skill B", category="Soft Skills",
                              weight=0.30, rationale=""),
            WeightedCriterion(id="c3", criterion="Skill C", category="Experience",
                              weight=0.20, rationale=""),
        ],
    )


def test_missing_criterion_score_treated_as_zero():
    """Rubric has c1+c2+c3, candidate only scored on c1+c2 → c3 treated as 0."""
    from pipeline.step7_weight_engine import compute_candidate_score
    rubric = make_rubric_3crit()
    candidate = CandidateRecord(
        candidate_id="missing_c3", name="Test", file_path="/fake/test.txt",
        criterion_scores=[
            CriterionScore(criterion_id="c1", criterion_text="Skill A",
                           answer="Good", evidence_quote="Evidence A",
                           confidence=0.8, score=8),
            CriterionScore(criterion_id="c2", criterion_text="Skill B",
                           answer="OK", evidence_quote="Evidence B",
                           confidence=0.7, score=6),
            # c3 is deliberately missing
        ],
        gate_status="passed",
    )
    result = compute_candidate_score(candidate, rubric)
    # base = 0.50*8 + 0.30*6 + 0.20*0 = 4.0 + 1.8 + 0.0 = 5.8
    expected = 0.50 * 8 + 0.30 * 6 + 0.20 * 0
    assert abs(result.base_score - expected) < 0.01, \
        f"Expected base_score {expected}, got {result.base_score}"


def test_score_determinism():
    """compute_candidate_score twice on same input yields identical output."""
    from pipeline.step7_weight_engine import compute_candidate_score
    rubric = make_rubric_3crit()

    def make_cand():
        return CandidateRecord(
            candidate_id="determ_test", name="Det Candidate", file_path="/det.txt",
            criterion_scores=[
                CriterionScore(criterion_id="c1", criterion_text="Skill A",
                               answer="A", evidence_quote="Ev A", confidence=0.8, score=7),
                CriterionScore(criterion_id="c2", criterion_text="Skill B",
                               answer="B", evidence_quote="Ev B", confidence=0.75, score=5),
                CriterionScore(criterion_id="c3", criterion_text="Skill C",
                               answer="C", evidence_quote="Ev C", confidence=0.70, score=4),
            ],
            gate_status="passed",
        )

    result1 = compute_candidate_score(make_cand(), rubric)
    result2 = compute_candidate_score(make_cand(), rubric)
    assert result1.base_score == result2.base_score
    assert result1.final_score == result2.final_score
    assert result1.confidence == result2.confidence

# ── Step 1 mocked tests ───────────────────────────────────────────────────────

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
         "category": "Hard Skills", "weight": 0.20, "rationale": "Web stack"},
        {"id": "c3", "criterion": "PostgreSQL database experience",
         "category": "Hard Skills", "weight": 0.18, "rationale": "Primary DB"},
        {"id": "c4", "criterion": "AWS cloud infrastructure",
         "category": "Hard Skills", "weight": 0.18, "rationale": "Cloud platform"},
        {"id": "c5", "criterion": "Docker containerisation",
         "category": "Hard Skills", "weight": 0.16, "rationale": "Deployment"},
    ],
    "bonus_sensitivity": "medium",
}


def test_jd_parser_with_mocked_llm():
    from pipeline.step1_jd_parser import parse_jd_to_rubric
    with patch("pipeline.step1_jd_parser.llm_client") as mock_llm:
        mock_llm.structured_call.return_value = MOCK_RUBRIC_RESPONSE
        result = parse_jd_to_rubric(SW_JD)

    assert isinstance(result.rubric, FrozenRubric)
    assert result.rubric.job_title == "Senior Software Engineer (Backend)"
    assert len(result.rubric.weighted_criteria) == 5
    assert len(result.rubric.hard_gates) == 2
    total = sum(c.weight for c in result.rubric.weighted_criteria)
    assert abs(total - 1.0) < 0.02


def test_jd_parser_handles_weight_normalization():
    """Weights summing to 0.9 should auto-normalize to 1.0."""
    from pipeline.step1_jd_parser import parse_jd_to_rubric
    response = {**MOCK_RUBRIC_RESPONSE}
    # Scale weights to sum ~0.9
    criteria = [
        {**c, "weight": c["weight"] * 0.9}
        for c in MOCK_RUBRIC_RESPONSE["weighted_criteria"]
    ]
    response = {**MOCK_RUBRIC_RESPONSE, "weighted_criteria": criteria}

    with patch("pipeline.step1_jd_parser.llm_client") as mock_llm:
        mock_llm.structured_call.return_value = response
        result = parse_jd_to_rubric(SW_JD)

    total = sum(c.weight for c in result.rubric.weighted_criteria)
    assert abs(total - 1.0) < 0.02, f"Weights not normalized: {total}"
    assert any("normalized" in w.lower() for w in result.warnings), \
        "Expected normalization warning"


def test_jd_parser_handles_too_few_criteria():
    """Only 1 criterion returned → should raise ValueError (min is 3)."""
    from pipeline.step1_jd_parser import parse_jd_to_rubric
    response = {
        **MOCK_RUBRIC_RESPONSE,
        "weighted_criteria": [
            {"id": "c1", "criterion": "Python", "category": "Hard Skills",
             "weight": 1.0, "rationale": ""},
        ],
    }
    with patch("pipeline.step1_jd_parser.llm_client") as mock_llm:
        mock_llm.structured_call.return_value = response
        with pytest.raises(ValueError, match="Too few valid criteria"):
            parse_jd_to_rubric(SW_JD)

# ── Step 5 mocked tests ───────────────────────────────────────────────────────

def _make_store_with_chunks(candidate_id: str, texts: list[str]) -> "CandidateVectorStore":
    """Helper: build a CandidateVectorStore with fake embeddings for given texts."""
    from core.vector_store import CandidateVectorStore
    store = CandidateVectorStore(candidate_id)
    rng = np.random.default_rng(hash(candidate_id) % (2**32))
    raw = rng.standard_normal((len(texts), 384)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embs = raw / norms
    chunks = [
        ResumeChunk(candidate_id=candidate_id, section_type="experience",
                    text=t, source_page=0)
        for t in texts
    ]
    store.add_chunks(chunks, embs)
    return store


def test_qa_scoring_mocked():
    """Mock LLM returns valid JSON → CriterionScore populated correctly."""
    from pipeline.step5_qa_scoring import _score_single_criterion

    rubric_crit = WeightedCriterion(
        id="c1", criterion="Python backend development",
        category="Hard Skills", weight=1.0, rationale=""
    )
    chunk_text = "Arjun has 7 years of Python experience building FastAPI microservices at 500K RPS."
    candidate = CandidateRecord(
        candidate_id="qa_mock_01", name="Arjun Sharma",
        file_path=str(STRONG_RESUME),
        raw_text=chunk_text,
    )
    store = _make_store_with_chunks("qa_mock_01", [chunk_text])

    mock_response = {
        "answer": "Candidate has extensive Python backend experience at scale.",
        "evidence_quote": chunk_text[:60],
        "confidence": 0.93,
        "score": 9,
    }

    with patch("pipeline.step5_qa_scoring.llm_client") as mock_llm, \
         patch("pipeline.step5_qa_scoring.embedding_client") as mock_emb, \
         patch("pipeline.step5_qa_scoring.reranker_client") as mock_rerank:
        rng = np.random.default_rng(0)
        fake_emb = rng.standard_normal((1, 384)).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)
        mock_emb.embed_query.return_value = fake_emb
        mock_rerank.rerank.side_effect = lambda query, passages, top_n: passages[:top_n]
        mock_llm.structured_call.return_value = mock_response
        cs = _score_single_criterion(candidate, store, rubric_crit)

    assert cs.score == 9
    assert cs.criterion_id == "c1"
    assert cs.confidence > 0.5


def test_qa_scoring_no_evidence_caps_score():
    """Mock returns NO_EVIDENCE_FOUND with score=9 → score should be capped to 2."""
    from pipeline.step5_qa_scoring import _score_single_criterion

    rubric_crit = WeightedCriterion(
        id="c1", criterion="Kubernetes production experience",
        category="Hard Skills", weight=1.0, rationale=""
    )
    chunk_text = "Rahul worked on PHP websites with no cloud experience."
    candidate = CandidateRecord(
        candidate_id="qa_noev_01", name="Rahul Verma",
        file_path=str(WEAK_RESUME),
        raw_text=chunk_text,
    )
    store = _make_store_with_chunks("qa_noev_01", [chunk_text])

    mock_response = {
        "answer": "No evidence of Kubernetes experience found.",
        "evidence_quote": "NO_EVIDENCE_FOUND",
        "confidence": 0.9,
        "score": 9,   # Contradiction — should be capped to 2
    }

    with patch("pipeline.step5_qa_scoring.llm_client") as mock_llm, \
         patch("pipeline.step5_qa_scoring.embedding_client") as mock_emb, \
         patch("pipeline.step5_qa_scoring.reranker_client") as mock_rerank:
        rng = np.random.default_rng(1)
        fake_emb = rng.standard_normal((1, 384)).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)
        mock_emb.embed_query.return_value = fake_emb
        mock_rerank.rerank.side_effect = lambda query, passages, top_n: passages[:top_n]
        mock_llm.structured_call.return_value = mock_response
        cs = _score_single_criterion(candidate, store, rubric_crit)

    assert cs.score <= 2, \
        f"Expected score capped to 2 when NO_EVIDENCE_FOUND, got {cs.score}"
    assert cs.evidence_quote == "NO_EVIDENCE_FOUND"
