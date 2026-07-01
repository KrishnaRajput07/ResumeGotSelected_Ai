#!/usr/bin/env python
r"""
download_models.py — Download all HuggingFace models to D:\India_Run\.cache\

Downloads:
  1. BAAI/bge-small-en-v1.5   (~133 MB)  — embedding model
  2. BAAI/bge-reranker-base    (~278 MB)  — reranker model

All artefacts land in D:\India_Run\.cache\ via the environment variables
set in core/config.py (HF_HOME, SENTENCE_TRANSFORMERS_HOME).

Run:
    .venv\Scripts\python.exe scripts\download_models.py
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
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Strip broken external packages (mirrors conftest.py) ─────────────────────
_EXTERNAL_BLOCKLIST = {Path(r"D:\python-packages")}
_VENV_ROOT = _ROOT / ".venv"

def _strip_external_paths():
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

# ── Load settings (this sets HF_HOME / ST_HOME env vars via os.environ.setdefault)
from core.config import settings

# ── Override with the D-drive paths explicitly ───────────────────────────────
os.environ["HF_HOME"] = str(_ROOT / ".cache" / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(_ROOT / ".cache" / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_ROOT / ".cache" / "sentence-transformers")
os.environ["HF_HUB_CACHE"] = str(_ROOT / ".cache" / "huggingface" / "hub")

# Create all directories upfront
for d in [
    _ROOT / ".cache" / "huggingface" / "hub",
    _ROOT / ".cache" / "transformers",
    _ROOT / ".cache" / "sentence-transformers",
    _ROOT / "models" / "ollama",
]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def check_disk_space(path: Path, required_gb: float) -> bool:
    import shutil
    stat = shutil.disk_usage(str(path))
    free_gb = stat.free / (1024 ** 3)
    print(f"  Free space on {path.drive}: {free_gb:.1f} GB  (need {required_gb:.1f} GB)")
    if free_gb < required_gb:
        print(f"  ⚠ WARNING: Low disk space. Model download may fail.")
        return False
    return True


def download_embedding_model():
    banner("1 / 2  —  Downloading BAAI/bge-small-en-v1.5  (~133 MB, embedding model)")

    cache_root = Path(os.environ["SENTENCE_TRANSFORMERS_HOME"])
    print(f"  Cache dir : {cache_root}")
    check_disk_space(_ROOT, 0.5)

    t0 = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading/downloading model (this shows a progress bar from HuggingFace)...")
        model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device="cpu",
            cache_folder=str(cache_root),
        )
        elapsed = time.perf_counter() - t0

        # Quick smoke test
        import numpy as np
        test_emb = model.encode(["test sentence"], normalize_embeddings=True)
        assert test_emb.shape == (1, 384), f"Shape mismatch: {test_emb.shape}"
        norm = float(np.linalg.norm(test_emb[0]))
        assert abs(norm - 1.0) < 1e-4, f"Not normalised: norm={norm}"

        print(f"\n  ✅ bge-small-en-v1.5 ready  ({elapsed:.1f}s)")
        print(f"     Test embed shape: {test_emb.shape}  norm: {norm:.6f}")

        # Show where files landed
        model_cache = cache_root / "models--BAAI--bge-small-en-v1.5"
        if model_cache.exists():
            blobs = list((model_cache / "blobs").glob("*"))
            total_mb = sum(b.stat().st_size for b in blobs) / 1e6
            print(f"     Files in cache: {len(blobs)} blobs  ({total_mb:.0f} MB)")
        return True

    except Exception as e:
        print(f"\n  ❌ FAILED: {e}")
        return False


def download_reranker_model():
    banner("2 / 2  —  Downloading BAAI/bge-reranker-base  (~278 MB, reranker model)")

    cache_root = Path(os.environ["SENTENCE_TRANSFORMERS_HOME"])
    print(f"  Cache dir : {cache_root}")
    check_disk_space(_ROOT, 1.0)

    t0 = time.perf_counter()
    try:
        from sentence_transformers import CrossEncoder
        print("  Loading/downloading model (this shows a progress bar from HuggingFace)...")
        model = CrossEncoder(
            "BAAI/bge-reranker-base",
            device="cpu",
        )
        elapsed = time.perf_counter() - t0

        # Quick smoke test
        score = model.predict([("What is Python?", "Python is a programming language.")])
        print(f"\n  ✅ bge-reranker-base ready  ({elapsed:.1f}s)")
        print(f"     Test rerank score: {float(score[0]):.4f}")

        # Show where files landed
        model_cache = cache_root / "models--BAAI--bge-reranker-base"
        if not model_cache.exists():
            # CrossEncoder uses HF hub cache instead
            hub_cache = Path(os.environ["HF_HUB_CACHE"]) / "models--BAAI--bge-reranker-base"
            if hub_cache.exists():
                blobs = list((hub_cache / "blobs").glob("*"))
                total_mb = sum(b.stat().st_size for b in blobs) / 1e6
                print(f"     Files in HF hub cache: {len(blobs)} blobs  ({total_mb:.0f} MB)")
        return True

    except Exception as e:
        print(f"\n  ❌ FAILED: {e}")
        return False


def print_ollama_instructions():
    banner("Ollama — Manual Installation Required")
    print("""
  Ollama is NOT installed on this machine. The LLM (qwen3:8b) requires Ollama.

  STEP 1 — Before installing, set OLLAMA_MODELS to D:\\ so models land there:

    Open System Properties → Environment Variables → User Variables → New:
      Variable name:  OLLAMA_MODELS
      Variable value: D:\\India_Run\\models\\ollama

    OR in PowerShell (run once):
      [System.Environment]::SetEnvironmentVariable(
          "OLLAMA_MODELS",
          "D:\\India_Run\\models\\ollama",
          "User"
      )

  STEP 2 — Download and install Ollama:
    https://ollama.com/download/OllamaSetup.exe
    (Windows installer, ~60 MB)

  STEP 3 — Open a NEW terminal after installation and verify:
    ollama --version

  STEP 4 — Pull the model (it will land in D:\\India_Run\\models\\ollama):
    ollama pull qwen3:8b

    qwen3:8b = ~5.2 GB on disk. Takes 5–15 min depending on internet speed.

  STEP 5 — Verify the model works:
    ollama run qwen3:8b "What is Python? Answer in one sentence."

  STEP 6 — Keep Ollama running in the background, then launch the app:
    streamlit run app\\main.py
""")


def print_summary(emb_ok: bool, rerank_ok: bool):
    banner("DOWNLOAD SUMMARY")
    print(f"  HuggingFace models cache : D:\\India_Run\\.cache\\")
    print(f"  Ollama models dir        : D:\\India_Run\\models\\ollama\\")
    print()
    status = lambda ok: "✅ Ready" if ok else "❌ Failed"
    print(f"  bge-small-en-v1.5  (embedding)  : {status(emb_ok)}")
    print(f"  bge-reranker-base  (reranker)   : {status(rerank_ok)}")
    print(f"  qwen3:8b           (LLM)         : ⏳ Needs Ollama installed first")
    print()
    if emb_ok and rerank_ok:
        print("  HuggingFace models are ready. Install Ollama and pull qwen3:8b to complete setup.")
    else:
        print("  Some models failed. Check errors above and re-run this script.")
    print()


if __name__ == "__main__":
    print("\n🤖 AI Recruiter Co-Pilot — Model Downloader")
    print(f"   All models will be saved under: D:\\India_Run\\.cache\\ and D:\\India_Run\\models\\")
    print(f"   Python: {sys.executable}")
    print(f"   HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"   SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")
    print(f"   OLLAMA_MODELS: {os.environ.get('OLLAMA_MODELS', 'not set')}")

    emb_ok = download_embedding_model()
    rerank_ok = download_reranker_model()
    print_ollama_instructions()
    print_summary(emb_ok, rerank_ok)
