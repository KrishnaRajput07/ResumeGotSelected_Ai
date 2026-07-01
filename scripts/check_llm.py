r"""
check_llm.py -- Verify qwen3:8b is reachable and producing correct output.
Run: .venv\Scripts\python.exe scripts\check_llm.py
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path = [p for p in sys.path if "python-packages" not in p]

os.environ["OLLAMA_MODELS"] = str(_ROOT / "models" / "ollama")

from core.llm_client import llm_client
from core.config import settings

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def sep(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)

results = []

# ── Test 1: Connectivity ──────────────────────────────────
sep("1 / 3  Connectivity + model presence")
ok = llm_client.check_connectivity()
status = PASS if ok else FAIL
print(f"  Ollama running:  {status}")
print(f"  Model:           {settings.ollama_model}")
print(f"  Base URL:        {settings.ollama_base_url}")
results.append(ok)

# ── Test 2: Structured JSON call ─────────────────────────
sep("2 / 3  Structured JSON call (JD parsing pattern)")
try:
    t0 = time.perf_counter()
    result = llm_client.structured_call(
        system_prompt=(
            "You extract job titles from job postings. "
            "Return ONLY valid JSON with exactly one key: job_title (string)."
        ),
        user_prompt="Job posting: We are hiring a Senior Python Backend Engineer.",
        expected_keys=["job_title"],
    )
    elapsed = time.perf_counter() - t0
    title = result.get("job_title", "")
    ok2 = bool(title) and len(title) > 3
    print(f"  Response:   {result}")
    print(f"  Latency:    {elapsed:.1f}s")
    print(f"  Valid JSON: {PASS if ok2 else FAIL}")
    results.append(ok2)
except Exception as e:
    print(f"  {FAIL}: {e}")
    results.append(False)

# ── Test 3: Reasoning / free-text call ───────────────────
sep("3 / 3  Reasoning call (Q&A scoring pattern)")
try:
    t0 = time.perf_counter()
    answer = llm_client.reasoning_call(
        system_prompt=(
            "/no_think\n\n"
            "You score candidates on a single criterion. "
            "Answer in exactly two sentences."
        ),
        user_prompt=(
            "Criterion: Python backend development experience.\n"
            "Resume excerpt: '7 years building FastAPI microservices at 500K RPS.'\n"
            "Give a score 0-10 and justify it briefly."
        ),
    )
    elapsed = time.perf_counter() - t0
    ok3 = len(answer) > 20
    print(f"  Response:   {answer[:300]}")
    print(f"  Latency:    {elapsed:.1f}s")
    print(f"  Non-empty:  {PASS if ok3 else FAIL}")
    results.append(ok3)
except Exception as e:
    print(f"  {FAIL}: {e}")
    results.append(False)

# ── Summary ───────────────────────────────────────────────
sep("SUMMARY")
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print(f"\n  ✅ qwen3:8b is fully integrated and working.\n")
    print(f"  Storage: D:\\India_Run\\models\\ollama\\")
    print(f"  Model:   {settings.ollama_model}")
    print(f"\n  Ready to run: streamlit run app\\main.py\n")
else:
    print(f"\n  ❌ Some tests failed. Check Ollama is running:\n")
    print(f"     ollama serve\n")

sys.exit(0 if passed == total else 1)
