# ─────────────────────────────────────────────────────────────────────────────
# AI Recruiter Co-Pilot — Central Configuration
# ─────────────────────────────────────────────────────────────────────────────
# All tunable parameters live here. No magic numbers in pipeline code.

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
D_DRIVE_ROOT = Path(os.getenv("INDIA_RUN_ROOT", str(PROJECT_ROOT)))

# Keep all large local model/cache artifacts on D: by default.
os.environ.setdefault("HF_HOME", str(D_DRIVE_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(D_DRIVE_ROOT / ".cache" / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(D_DRIVE_ROOT / ".cache" / "sentence-transformers"))
os.environ.setdefault("OLLAMA_MODELS", str(D_DRIVE_ROOT / "models" / "ollama"))


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DETECTED_DEVICE = _detect_device()


class Settings(BaseSettings):
    """
    All configuration is sourced from environment variables (see .env)
    with sensible defaults for local/free operation.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama LLM ────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"          # Primary model for all LLM calls

    # Temperature strategy:
    #   0.0 = deterministic → used for structured JSON output (rubric gen, scoring)
    #   0.1 = near-deterministic → used for free-text answers in Q&A
    llm_temp_structured: float = 0.0
    llm_temp_reasoning: float = 0.1

    # Ollama call settings
    ollama_timeout: int = 120               # seconds; 8B models can be slow on CPU
    ollama_max_retries: int = 3
    ollama_retry_wait: float = 2.0          # seconds between retries

    # ── Embeddings (local HuggingFace) ────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384           # bge-small produces 384-dim vectors
    embedding_batch_size: int = 32           # Chunks per embedding batch
    embedding_device: str = DETECTED_DEVICE  # Uses cuda if available, otherwise cpu

    # Reranker (optional but enabled by default for better retrieval precision)
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = DETECTED_DEVICE
    reranker_top_n: int = 5

    # ── Pipeline Thresholds ───────────────────────────────────────────────────
    shortlist_top_k: int = 40               # Step 4: max candidates into LLM scoring
    shortlist_similarity_floor: float = 0.30 # Min cosine similarity to enter shortlist
    qa_retrieve_top_n: int = 5              # Step 5: resume chunks per criterion Q&A
    full_resume_max_chars: int = 8000       # Hard gate check: truncate to this length

    # ── Rubric Constraints ────────────────────────────────────────────────────
    min_weighted_criteria: int = 3
    max_weighted_criteria: int = 8
    max_hard_gates: int = 4                 # Warn if JD yields more than this
    weight_prior_delta_max: float = 0.08    # Max LLM can deviate from category prior

    # ── Scoring Math ──────────────────────────────────────────────────────────
    bonus_cap: float = 1.0                  # Max bonus added to base 0-10 score
    bonus_sensitivity_map: dict = Field(
        default={"off": 0.0, "low": 0.3, "medium": 0.6, "high": 1.0}
    )

    # ── Confidence Thresholds ─────────────────────────────────────────────────
    confidence_high_floor: float = 0.72
    confidence_low_ceiling: float = 0.48

    # ── Evidence Quote Validation ─────────────────────────────────────────────
    evidence_quote_min_fuzzy_score: int = 70  # thefuzz ratio threshold (0-100)

    # ── Data Paths ────────────────────────────────────────────────────────────
    project_root: str = str(D_DRIVE_ROOT)
    data_root: str = str(D_DRIVE_ROOT / "data")
    uploads_dir: str = str(D_DRIVE_ROOT / "data" / "uploads")
    runs_dir: str = str(D_DRIVE_ROOT / "data" / "runs")
    embeddings_cache_dir: str = str(D_DRIVE_ROOT / "data" / "embeddings_cache")
    models_dir: str = str(D_DRIVE_ROOT / "models")

    # ── Category Weight Priors ────────────────────────────────────────────────
    # These are the baseline weights per criterion category.
    # The LLM can deviate ±weight_prior_delta_max from these.
    # This prevents arbitrary weight generation while allowing JD-specific tuning.
    category_weight_priors: dict = Field(default={
        "Hard Skills":       0.20,
        "Experience":        0.18,
        "Domain Knowledge":  0.16,
        "Education":         0.12,
        "Leadership":        0.10,
        "Soft Skills":       0.08,
    })

    # ── JSONL Challenge Mode ──────────────────────────────────────────────────
    jsonl_llm_deep_score_top_n: int = 50      # Candidates that get full LLM scoring
    jsonl_behavioral_batch_size: int = 1000   # Candidates per behavioral scoring batch
    jsonl_embedding_batch_size: int = 64      # Larger batch for JSON text (shorter chunks)


# Singleton — import this everywhere
settings = Settings()
