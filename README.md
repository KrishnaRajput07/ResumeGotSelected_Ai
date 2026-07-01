# AI Recruiter Co-Pilot

A fully local, evidence-grounded AI system that parses job descriptions, proposes evaluation rubrics, and ranks candidate resumes — all without sending data to any external API. Built for recruiters who need explainability and transparency at every step.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app/main.py)            │
│   Page 1: Upload  │  Page 2: Calibrate  │  Page 3: Processing  │
│                   │   Rubric Review     │   Live Progress       │
│   Page 4: Results │                                             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                        │
│                   (pipeline/orchestrator.py)                     │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┬───────┘
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
Step 1      Step 3     Step 4     Step 5     Step 6    Step 7+8
JD Parser  Resume    Embedding  Evidence   Parallel   Scoring &
           Ingest    Shortlist  Q&A Score  Passes     Output
           +FAISS    (recall)   (rank)     (signals,
                                          flags,
                                          gates)
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
Ollama      PDF        FAISS      Ollama     Ollama     CSV +
qwen3:8b   Parser     IndexIP    qwen3:8b   qwen3:8b   HTML +
           Chunker    BGE-small             BGE-rerank  JSON
```

---

## Stack

| Layer              | Technology                          | Purpose                          |
|--------------------|-------------------------------------|----------------------------------|
| LLM                | Ollama + qwen3:8b (local)           | JD parsing, Q&A scoring          |
| Embeddings         | BAAI/bge-small-en-v1.5 (local)      | Resume + rubric embedding        |
| Reranker           | BAAI/bge-reranker-base (local)      | Top-N chunk reranking            |
| Vector Search      | FAISS (IndexFlatIP)                 | Per-candidate + cross-candidate  |
| Document Parsing   | pdfplumber / PyMuPDF / python-docx  | PDF, DOCX, TXT                   |
| Schemas            | Pydantic v2                         | All data contracts               |
| Dashboard          | Streamlit + Plotly                  | 4-page interactive UI            |
| Data               | pandas, CSV, HTML                   | Export artifacts                 |
| Testing            | pytest + unittest.mock              | 50 tests total                   |

---

## Setup (Windows CMD / PowerShell)

### Prerequisites

1. Python 3.11 or 3.12 installed
2. [Ollama](https://ollama.ai) installed and running
3. At least 8 GB RAM; 16 GB recommended for qwen3:8b on CPU

### Step-by-step

**1. Clone and navigate to the project:**
```cmd
git clone <your-repo-url> D:\India_Run
cd D:\India_Run
```

**2. Create a virtual environment:**
```cmd
python -m venv .venv
```

**3. Activate the virtual environment:**
```cmd
.venv\Scripts\activate
```

**4. Install dependencies:**
```cmd
pip install -r requirements.txt
```

**5. Pull the required Ollama model:**
```cmd
ollama pull qwen3:8b
```

**6. Copy the example environment file:**
```cmd
copy .env.example .env
```

**7. (Optional) Edit `.env` to customise settings** — the defaults work out of the box.

**8. Verify setup with smoke tests (no Ollama required):**
```cmd
python scripts/smoke_test.py
```

**9. Run the app (Interactive UI):**
```cmd
streamlit run app/main.py
```

**10. Reproduce the Submission CSV (Ranking Step):**
To fulfill the Stage 3 reproduction requirement (runs in < 5 mins on CPU using pre-computed artifacts), use the exact command below:
```cmd
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```


---

## Running the Smoke Tests

Smoke tests validate every pipeline component without requiring Ollama:

```cmd
python scripts/smoke_test.py
```

Expected output: `ALL 11 TESTS PASSED ✅`

Tests cover: config loading, PDF/TXT parsing, section-aware chunking, embedding client (with fallback), FAISS vector store, Step 1 JD parser (mocked LLM), Step 3 resume ingestion, Step 4 shortlisting, Step 5 Q&A scoring (mocked LLM), Step 7 weight engine, Step 8 output assembly.

---

## Running the Validator Script

For visual inspection of what each stage produces:

```cmd
# Safe stages only (no Ollama needed)
python scripts/validate_pipeline.py --stage all

# Individual stages (no Ollama)
python scripts/validate_pipeline.py --stage parsing
python scripts/validate_pipeline.py --stage chunking
python scripts/validate_pipeline.py --stage embedding

# Ollama-dependent stages
python scripts/validate_pipeline.py --stage step1
python scripts/validate_pipeline.py --stage step5
python scripts/validate_pipeline.py --stage full
```

---

## Running All Tests

```cmd
python -m pytest tests/ -v
```

Expected: `50 passed`

| Test file                    | Count | What it covers                              |
|------------------------------|-------|---------------------------------------------|
| `test_jd_parser.py`          | 8     | Rubric schema, weight normalisation, gates  |
| `test_qa_scoring.py`         | 11    | Score clamping, evidence validation, fuzzy  |
| `test_weight_engine.py`      | 11    | Weighted scoring math, bonus cap, ranking   |
| `test_integration.py`        | 20    | Parsing, chunking, FAISS, mocked Step 1 & 5 |

---

## Running the Streamlit App

```cmd
streamlit run app/main.py
```

Navigate to `http://localhost:8501` in your browser.

**Page flow:**
1. **Upload** — drop a JD file + resumes → click "Parse JD & Generate Rubric"
2. **Calibrate** — review AI-proposed rubric, adjust weights, toggle gates → click "Evaluate"
3. **Processing** — live step-by-step progress tracker
4. **Results** — ranked table, score heatmap, per-candidate evidence breakdown, CSV/HTML export

---

## Configuration Reference

All settings are in `.env` (copy from `.env.example`). Every setting has a working default.

| Variable                      | Type    | Default                     | Description                                      |
|-------------------------------|---------|-----------------------------|--------------------------------------------------|
| `OLLAMA_BASE_URL`             | str     | `http://localhost:11434`    | Ollama API endpoint                              |
| `OLLAMA_MODEL`                | str     | `qwen3:8b`                  | Model for all LLM calls                          |
| `LLM_TEMP_STRUCTURED`         | float   | `0.0`                       | Temperature for JSON output (rubric, scoring)    |
| `LLM_TEMP_REASONING`          | float   | `0.1`                       | Temperature for free-text Q&A answers            |
| `OLLAMA_TIMEOUT`              | int     | `120`                       | Seconds before timeout per LLM call              |
| `EMBEDDING_MODEL`             | str     | `BAAI/bge-small-en-v1.5`    | Local HuggingFace embedding model                |
| `EMBEDDING_DIMENSION`         | int     | `384`                       | Embedding vector size (must match model)         |
| `EMBEDDING_DEVICE`            | str     | `cpu`                       | `cpu` or `cuda`                                  |
| `RERANKER_ENABLED`            | bool    | `True`                      | Enable BGE reranker in Step 5 retrieval          |
| `RERANKER_MODEL`              | str     | `BAAI/bge-reranker-base`    | Reranker model (local)                           |
| `SHORTLIST_TOP_K`             | int     | `40`                        | Max candidates entering LLM scoring (Step 4)     |
| `SHORTLIST_SIMILARITY_FLOOR`  | float   | `0.30`                      | Min cosine similarity to enter Step 5            |
| `QA_RETRIEVE_TOP_N`           | int     | `5`                         | Resume chunks retrieved per criterion Q&A        |
| `FULL_RESUME_MAX_CHARS`       | int     | `8000`                      | Resume truncation for hard gate check            |
| `MIN_WEIGHTED_CRITERIA`       | int     | `3`                         | Minimum rubric criteria (error if fewer)         |
| `MAX_WEIGHTED_CRITERIA`       | int     | `8`                         | Maximum rubric criteria (LLM output capped)      |
| `BONUS_CAP`                   | float   | `1.0`                       | Max bonus added to base score                    |
| `CONFIDENCE_HIGH_FLOOR`       | float   | `0.72`                      | Adjusted confidence ≥ this → "High"              |
| `CONFIDENCE_LOW_CEILING`      | float   | `0.48`                      | Adjusted confidence < this → "Low"               |
| `EVIDENCE_QUOTE_MIN_FUZZY_SCORE` | int  | `70`                        | Fuzzy match threshold for evidence validation    |
| `RUNS_DIR`                    | str     | `data/runs`                 | Where run output directories are written         |
| `UPLOADS_DIR`                 | str     | `data/uploads`              | Where uploaded files are temporarily stored      |
| `EMBEDDINGS_CACHE_DIR`        | str     | `data/embeddings_cache`     | Disk cache for computed embeddings               |

---

## Pipeline Step Descriptions

**Step 1 — JD Parser (`pipeline/step1_jd_parser.py`)**
Parses a job description file (PDF/DOCX/TXT) and calls the LLM with a structured prompt to produce a proposed rubric. The rubric contains 3–8 weighted criteria, optional hard gates (binary eligibility requirements), and a bonus sensitivity setting. Weights are automatically normalised to sum to 1.0. The output is shown to the recruiter for review before any candidate is scored.

**Step 2 — Recruiter Calibration (in `app/main.py`, page 2)**
A human-in-the-loop step. The recruiter adjusts criterion weights using sliders, marks priority criteria (×1.15 boost), toggles hard gate strictness, and selects bonus sensitivity. On clicking "Evaluate", the rubric is frozen with a UTC timestamp — this exact rubric governs all candidates in the run.

**Step 3 — Resume Ingestion (`pipeline/step3_resume_ingestion.py`)**
Each resume is parsed (PDF/DOCX/TXT), cleaned, and split into semantic chunks using a section-aware algorithm (experience → job entries, education, skills, etc.). Chunks are embedded with BAAI/bge-small-en-v1.5 and stored in a per-candidate FAISS index. A cross-candidate index (mean embedding per candidate) is also built for Step 4.

**Step 4 — Embedding Shortlisting (`pipeline/step4_shortlisting.py`)**
A recall-oriented filter. The frozen rubric is embedded as a composite query ("ideal candidate profile"). Each candidate's mean embedding is compared via cosine similarity. The top-K candidates above the similarity floor proceed to Step 5. Candidates below the floor are filtered out without LLM cost. Deliberately generous — borderline candidates are kept.

**Step 5 — Evidence-Grounded Q&A Scoring (`pipeline/step5_qa_scoring.py`)**
The core scoring step. For each shortlisted candidate × each rubric criterion, the system retrieves the top-N most relevant resume chunks from FAISS, optionally reranks them with BGE-reranker, then calls the LLM with a strict prompt: score 0–10, provide a verbatim evidence quote, and explain the reasoning. The evidence quote is validated post-hoc using fuzzy string matching. Contradictions (high score + no evidence) are corrected automatically.

**Step 6 — Parallel Passes (`pipeline/step6_parallel_passes.py`)**
Three independent LLM calls per candidate, each narrowly scoped: (A) Open Signal Mining — finds standout achievements outside the rubric criteria; (B) Red Flag Detection — detects objective plausibility concerns (employment gaps, date overlaps, credential mismatches) without speculation; (C) Hard Gate Evaluation — binary pass/fail on each strict eligibility gate. Gate failures mark the candidate as excluded.

**Step 7 — Weight Engine (`pipeline/step7_weight_engine.py`)**
Pure math, no LLM. Computes `base_score = Σ(weight_i × score_i)`, applies a capped bonus from open signals, and produces `final_score = min(base_score + bonus, 10.0)`. Confidence is aggregated from per-criterion scores with penalties for unverified evidence and no-evidence-found criteria. Candidates are sorted by final_score; hard-gate-excluded candidates are kept in an audit list with their scores.

**Step 8 — Output Assembly (`pipeline/step8_output.py`)**
Writes three artifacts per run: `ranked_shortlist.csv` (submission-ready with all scoring detail), `run_report.html` (self-contained visual report with colour-coded score matrix), and per-candidate `candidates/{id}.json` files for downstream use. Also writes `rubric.json` as an audit trail.

---

## Known Limitations

- **CPU-only performance**: qwen3:8b on CPU is slow (~2–5 min per candidate depending on criteria count). For faster results, use a smaller model or reduce `SHORTLIST_TOP_K`.
- **No OCR**: Image-based PDFs (scanned resumes) will produce near-empty text. Pre-process with an OCR tool before uploading.
- **Single language**: Optimised for English resumes. Non-English text will parse but embeddings and LLM reasoning quality will degrade.
- **Hallucination risk**: While evidence validation catches most LLM hallucinations, it cannot catch all cases. Always review evidence quotes before making hiring decisions.
- **Streamlit threading**: The pipeline runs in a background thread to avoid blocking the UI. On very long runs, the browser tab may time out. The 10-minute safety timeout will surface an error message in that case.
- **FAISS not GPU-accelerated**: The project uses faiss-cpu. For large-scale (500+ resumes) deployments, switch to faiss-gpu or a dedicated vector database.

---

## Troubleshooting

**`ImportError: numpy` — missing compiled C extensions**

This happens when a broken external numpy installation is on `PYTHONPATH`. The `conftest.py` strips `D:\python-packages` from `sys.path` automatically for pytest. For scripts, use:
```cmd
set PYTHONPATH=
python scripts/smoke_test.py
```

**`Ollama not reachable` in sidebar**

Ensure Ollama is running: `ollama serve` (or check the system tray). Then pull the model: `ollama pull qwen3:8b`

**`ModuleNotFoundError: No module named 'sentence_transformers'`**

The virtual environment is not active. Run `.venv\Scripts\activate` first.

**Embeddings download very slowly**

The BAAI/bge-small-en-v1.5 model (~130 MB) downloads from HuggingFace on first use. Set `HF_TOKEN` in `.env` for higher rate limits, or pre-download with:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
```

**`pdfplumber` extracts empty text from a PDF**

The PDF is likely image-based (scanned). Convert with OCR first:
```cmd
ocrmypdf input.pdf output.pdf
```

**Streamlit shows blank page / constant spinner**

Check that `app/main.py` is run from the project root, not from the `app/` subdirectory:
```cmd
streamlit run app/main.py
```
Not: `cd app && streamlit run main.py` — this breaks the `sys.path.insert` at the top of `main.py`.

**All candidates get similarity score 0.0 (all filtered in Step 4)**

If `total_candidates <= shortlist_top_k`, everyone passes through automatically (Step 4 is bypassed). If similarity scores are consistently near 0, check that the embedding model loaded correctly and that resumes contain English text.

---

## File Structure

```
D:\India_Run\
├── app/
│   └── main.py                  # Streamlit 4-page dashboard
├── core/
│   ├── config.py                # Central settings (pydantic-settings)
│   ├── llm_client.py            # Ollama client with retry logic
│   ├── embedding_client.py      # BGE-small embedding + disk cache
│   ├── reranker_client.py       # BGE-reranker-base
│   ├── pdf_parser.py            # pdfplumber → PyMuPDF → DOCX → TXT
│   ├── chunker.py               # Section-aware resume chunker
│   └── vector_store.py          # FAISS per-candidate + cross-candidate
├── pipeline/
│   ├── orchestrator.py          # End-to-end pipeline runner
│   ├── step1_jd_parser.py       # JD → rubric (LLM)
│   ├── step3_resume_ingestion.py# Parse + chunk + embed + index
│   ├── step4_shortlisting.py    # Embedding similarity filter
│   ├── step5_qa_scoring.py      # Evidence-grounded Q&A scoring (LLM)
│   ├── step6_parallel_passes.py # Open signals, red flags, gates (LLM)
│   ├── step7_weight_engine.py   # Deterministic scoring math
│   └── step8_output.py          # CSV + HTML + JSON output
├── prompts/
│   ├── jd_parsing.py            # JD → rubric prompt templates
│   ├── criterion_qa.py          # Q&A scoring prompt templates
│   └── parallel_passes.py       # Signals/flags/gates prompt templates
├── schemas/
│   ├── rubric.py                # FrozenRubric, WeightedCriterion, HardGate
│   └── candidate.py             # CandidateRecord, CriterionScore, etc.
├── scripts/
│   ├── smoke_test.py            # Offline smoke test (no Ollama needed)
│   └── validate_pipeline.py     # Visual stage-by-stage inspector
├── tests/
│   ├── fixtures/
│   │   ├── jd_software_engineer.txt
│   │   ├── jd_data_scientist.txt
│   │   ├── resume_strong.txt    # Arjun Sharma — 7yr Python, IIT+CMU
│   │   ├── resume_medium.txt    # Priya Mehta — 4yr Django, BITS Pilani
│   │   └── resume_weak.txt      # Rahul Verma — 2yr PHP, no Python
│   ├── test_jd_parser.py
│   ├── test_qa_scoring.py
│   ├── test_weight_engine.py
│   └── test_integration.py
├── data/
│   ├── runs/                    # One subdirectory per evaluation run
│   ├── uploads/                 # Temporary uploaded files
│   └── embeddings_cache/        # Disk cache for computed embeddings
├── conftest.py                  # Strips broken D:\python-packages from sys.path
├── pytest.ini                   # pytest configuration
├── requirements.txt
└── .env.example                 # Copy to .env and customise
```
