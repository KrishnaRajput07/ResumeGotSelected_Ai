import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.step1_jd_parser import parse_jd_to_rubric
from pipeline.orchestrator import run_jsonl_pipeline, ProgressUpdate
from schemas.rubric import FrozenRubric

def progress_cb(update: ProgressUpdate):
    print(f"[{update.step_name}] {update.detail}", flush=True)

def main():
    jd_path = Path("India_runs_data_and_ai_challenge/job_description.docx")
    jsonl_path = Path("India_runs_data_and_ai_challenge/candidates.jsonl")

    if not jd_path.exists():
        print(f"Error: JD not found at {jd_path}")
        sys.exit(1)
    if not jsonl_path.exists():
        print(f"Error: JSONL not found at {jsonl_path}")
        sys.exit(1)

    print(f"Parsing JD: {jd_path}")
    result = parse_jd_to_rubric(jd_path)
    rubric = result.rubric

    frozen = FrozenRubric(
        job_title=rubric.job_title,
        hard_gates=rubric.hard_gates,
        weighted_criteria=rubric.weighted_criteria,
        bonus_sensitivity=rubric.bonus_sensitivity,
    ).freeze()

    print(f"Rubric job title: {frozen.job_title}")
    
    print(f"Running JSONL Pipeline on {jsonl_path}...")
    run = run_jsonl_pipeline(
        jd_file_path=jd_path,
        jsonl_path=jsonl_path,
        frozen_rubric=frozen,
        progress_cb=progress_cb,
        llm_deep_score_top_n=50
    )

    if run.errors:
        print(f"\nPipeline completed with errors: {run.errors}")
    else:
        print("\nPipeline complete!")
        print(f"Run dir: {run.run_dir}")

if __name__ == "__main__":
    main()
