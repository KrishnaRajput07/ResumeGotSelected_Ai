"""
analyze_challenge.py - Analyze the India_runs_data_and_ai_challenge folder
and assess alignment with current system.
"""
import sys, os, json
from pathlib import Path

sys.path = [p for p in sys.path if "python-packages" not in p]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHALLENGE_DIR = Path(r"D:\India_Run\India_runs_data_and_ai_challenge")

# 1. Count candidates
print("=" * 60)
print("CHALLENGE DATA ANALYSIS")
print("=" * 60)

jsonl_path = CHALLENGE_DIR / "candidates.jsonl"
lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
print(f"\nTotal candidates in candidates.jsonl: {len(lines)}")

# Load first 5 to understand structure
candidates = [json.loads(l) for l in lines[:5]]
print(f"Sample IDs: {[c['candidate_id'] for c in candidates]}")
print(f"Fields per candidate: {list(candidates[0].keys())}")

# 2. Check sample submission
print("\n" + "=" * 60)
print("SAMPLE SUBMISSION (expected output format)")
print("=" * 60)
sample_csv = CHALLENGE_DIR / "sample_submission.csv"
rows = sample_csv.read_text().strip().splitlines()
print(f"Header: {rows[0]}")
print(f"Row 1: {rows[1]}")
print(f"Row 2: {rows[2]}")
print(f"Total rows (excl header): {len(rows)-1}")

# 3. Score range in sample
import csv, io
reader = csv.DictReader(io.StringIO(sample_csv.read_text()))
sample_rows = list(reader)
scores = [float(r["score"]) for r in sample_rows]
print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
print(f"Scoring formula from reasoning column: e.g. '{sample_rows[0]['reasoning']}'")

# 4. Understand AI skills used in scoring
print("\n" + "=" * 60)
print("WHAT THE SAMPLE SUBMISSION SCORES ON")
print("=" * 60)
for r in sample_rows[:5]:
    print(f"  Rank {r['rank']}: {r['candidate_id']} score={r['score']} | {r['reasoning']}")

# 5. Check what AI core skills are referenced
print("\n" + "=" * 60)
print("CANDIDATE DATA STRUCTURE (first candidate)")
print("=" * 60)
c = json.loads(lines[0])
print(f"  candidate_id: {c['candidate_id']}")
print(f"  name: {c['profile']['anonymized_name']}")
print(f"  title: {c['profile']['current_title']}")
print(f"  years_exp: {c['profile']['years_of_experience']}")
print(f"  skills ({len(c['skills'])}): {[s['name'] for s in c['skills'][:6]]}")
print(f"  redrob_signals keys: {list(c['redrob_signals'].keys())}")
print(f"  recruiter_response_rate: {c['redrob_signals']['recruiter_response_rate']}")
print(f"  open_to_work: {c['redrob_signals']['open_to_work_flag']}")

# 6. What the current system does vs what's needed
print("\n" + "=" * 60)
print("ALIGNMENT ASSESSMENT")
print("=" * 60)
print("""
WHAT THE CHALLENGE NEEDS:
  Input:  candidates.jsonl (100,000 structured JSON profiles)  
  Output: submission.csv (top 100 ranked candidates)
  Format: candidate_id, rank, score, reasoning
  Constraint: Must run in <5 min on CPU, no GPU, no API calls
  Scoring: NDCG@10 (50%) + NDCG@50 (30%) + MAP (15%) + P@10 (5%)
  
WHAT OUR CURRENT SYSTEM DOES:
  Input:  PDF/DOCX/TXT files (resume documents)
  Output: ranked_shortlist.csv + HTML report
  Method: PDF parse -> chunk -> embed -> LLM Q&A scoring per criterion
  Speed:  ~2-5 min PER CANDIDATE (way too slow for 100K candidates)
  
GAP ANALYSIS:
  1. INPUT FORMAT MISMATCH - Challenge uses structured JSON profiles,
     not PDF files. Our PDF parser is irrelevant here.
     
  2. SPEED MISMATCH (CRITICAL) - Our LLM-per-candidate approach would take
     100,000 × ~3 min = ~200,000 minutes = impossible.
     The challenge requires a FAST SCORER: embeddings + rule-based math.
     
  3. OUTPUT FORMAT - Our output needs to become the challenge CSV format.
  
  4. SCORING SIGNALS - Challenge uses redrob_signals (behavioral data)
     as a key ranking factor. Our system ignores these.
     
  5. HONEYPOT DETECTION - Challenge has ~80 trap candidates with
     impossible profiles. Must detect and exclude them.
     
WHAT NEEDS TO BE BUILT (new rank.py):
  - Fast feature extractor from JSON (no LLM needed)
  - AI skills matcher against JD requirements  
  - Behavioral signal scorer (recruiter_response_rate etc)
  - Honeypot detector (impossible dates, skill durations)
  - Score combiner -> top 100 -> CSV output
  - Target: <5 min for 100K candidates on CPU
""")
