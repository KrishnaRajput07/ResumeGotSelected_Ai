"""Quick smoke test for the new behavioral scorer and JSON ingestion."""
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
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if "python-packages" not in p]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.step2_json_ingestion import load_jsonl
from pipeline.step_behavioral_scorer import score_all_structured_candidates

candidates = load_jsonl(
    r"D:\India_Run\India_runs_data_and_ai_challenge\candidates.jsonl",
    max_candidates=100,
)
print(f"Loaded {len(candidates)} candidates")

scores = score_all_structured_candidates(candidates)
honeypots = [s for s in scores if s.is_honeypot]
print(f"Scored {len(scores)} | Honeypots detected: {len(honeypots)}")
print()
print("Top 5 candidates:")
for s in scores[:5]:
    title = s.current_title[:28]
    name = s.name[:22]
    print(f"  #{s.rank} {name:<22} | {title:<28} | composite={s.composite_score:.3f} | conf={s.confidence}")
    print(f"     {s.reasoning[:100]}")

print()
print("Honeypots found:")
for s in honeypots[:3]:
    print(f"  {s.name} | {s.honeypot_reasons[0][:80]}")

print()
print("Score distribution:")
high   = sum(1 for s in scores if s.composite_score >= 0.60)
medium = sum(1 for s in scores if 0.35 <= s.composite_score < 0.60)
low    = sum(1 for s in scores if s.composite_score < 0.35)
print(f"  High (>=0.60): {high} | Medium (0.35-0.60): {medium} | Low (<0.35): {low}")
