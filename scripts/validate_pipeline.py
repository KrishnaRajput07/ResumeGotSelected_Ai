#!/usr/bin/env python
"""
validate_pipeline.py — Deep manual inspection of each pipeline stage.

Use this to visually inspect what each stage produces.
Requires Ollama to be running with qwen3:8b for step1/step5/full stages.

Run: python scripts/validate_pipeline.py --stage all
     python scripts/validate_pipeline.py --stage parsing
     python scripts/validate_pipeline.py --stage chunking
     python scripts/validate_pipeline.py --stage embedding
     python scripts/validate_pipeline.py --stage step1
     python scripts/validate_pipeline.py --stage step5
     python scripts/validate_pipeline.py --stage full
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
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Strip broken external paths
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

RESUMES = [
    ("Arjun Sharma (strong)", STRONG_RESUME),
    ("Priya Mehta (medium)", MEDIUM_RESUME),
    ("Rahul Verma (weak)",   WEAK_RESUME),
]

def get_console():
    from rich.console import Console
    return Console()


def stage_parsing():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from core.pdf_parser import parse_file
    from pipeline.step3_resume_ingestion import _extract_name_from_text

    console = get_console()
    console.print(Panel("[bold cyan]STAGE: PARSING[/bold cyan]", expand=False))

    table = Table(title="Resume Parsing Results", show_lines=True)
    table.add_column("File", style="bold")
    table.add_column("Extracted Name", style="green")
    table.add_column("Chars", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Method")
    table.add_column("First 300 chars")

    for label, fpath in RESUMES:
        doc = parse_file(fpath)
        name = _extract_name_from_text(doc.raw_text, fpath.name)
        preview = doc.raw_text[:300].replace("\n", " ").strip()
        table.add_row(
            fpath.name, name, str(doc.char_count),
            f"{doc.extraction_confidence:.2f}", doc.extraction_method,
            preview,
        )

    console.print(table)


def stage_chunking():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from core.pdf_parser import parse_file
    from core.chunker import chunk_resume

    console = get_console()
    console.print(Panel("[bold cyan]STAGE: CHUNKING[/bold cyan]", expand=False))

    for label, fpath in RESUMES:
        doc = parse_file(fpath)
        chunks = chunk_resume(doc.raw_text, "validate_01")
        console.print(f"\n[bold green]{label}[/bold green] — {len(chunks)} chunks")

        table = Table(show_lines=True, title=f"{fpath.name}")
        table.add_column("idx", justify="right", style="dim")
        table.add_column("section_type", style="cyan")
        table.add_column("chars", justify="right")
        table.add_column("first 80 chars")

        for i, chunk in enumerate(chunks):
            preview = chunk.text[:80].replace("\n", " ")
            table.add_row(str(i), chunk.section_type, str(len(chunk.text)), preview)
        console.print(table)

def stage_embedding():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    import numpy as np
    from core.embedding_client import embedding_client

    console = get_console()
    console.print(Panel("[bold cyan]STAGE: EMBEDDING[/bold cyan]", expand=False))

    criterion_texts = [
        "Python development experience",
        "Python programming skills",
        "AWS cloud infrastructure",
        "Docker containerisation",
        "Leadership and team management",
    ]

    console.print("Embedding 5 criterion strings...")
    embs = embedding_client.embed_passages(criterion_texts)
    console.print(f"Embeddings shape: [bold]{embs.shape}[/bold] (should be 5×384)")

    # Cosine similarity matrix (embeddings already L2-normalised)
    sim_matrix = embs @ embs.T

    table = Table(title="Cosine Similarity Matrix", show_lines=True)
    table.add_column("", style="bold cyan")
    for i in range(len(criterion_texts)):
        table.add_column(criterion_texts[i][:20], justify="right")

    for i, row_text in enumerate(criterion_texts):
        row_vals = [f"{sim_matrix[i, j]:.3f}" for j in range(len(criterion_texts))]
        table.add_row(row_text[:20], *row_vals)

    console.print(table)
    console.print(
        f"\n[bold]Note:[/bold] '{criterion_texts[0]}' vs '{criterion_texts[1]}' "
        f"similarity = [green]{sim_matrix[0, 1]:.3f}[/green] (should be high ≥ 0.85)"
    )


def stage_step1(jd_path=None):
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from pipeline.step1_jd_parser import parse_jd_to_rubric

    console = get_console()
    console.print(Panel("[bold cyan]STAGE: STEP 1 — JD PARSING (requires Ollama)[/bold cyan]", expand=False))

    jd_path = jd_path or SW_JD
    console.print(f"Parsing JD: [bold]{jd_path}[/bold]")

    result = parse_jd_to_rubric(jd_path)
    rubric = result.rubric

    console.print(f"\n[bold green]Role:[/bold green] {rubric.job_title}")
    console.print(f"[bold]Criteria:[/bold] {len(rubric.weighted_criteria)}")
    console.print(f"[bold]Hard Gates:[/bold] {len(rubric.hard_gates)}")
    console.print(f"[bold]Bonus Sensitivity:[/bold] {rubric.bonus_sensitivity}")

    if result.warnings:
        for w in result.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")

    table = Table(title="Weighted Criteria", show_lines=True)
    table.add_column("ID", style="dim")
    table.add_column("Criterion", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Weight", justify="right")
    table.add_column("Rationale")

    for c in rubric.weighted_criteria:
        table.add_row(c.id, c.criterion, c.category, f"{c.weight:.3f}", c.rationale[:60])
    console.print(table)

    if rubric.hard_gates:
        console.print("\n[bold]Hard Gates:[/bold]")
        for g in rubric.hard_gates:
            console.print(f"  {'🔴 Strict' if g.strict else '🟡 Flexible'}: {g.criterion}")


def stage_step5():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    import numpy as np
    from pipeline.step1_jd_parser import parse_jd_to_rubric
    from pipeline.step3_resume_ingestion import ingest_resume
    from core.vector_store import CrossCandidateIndex
    from pipeline.step5_qa_scoring import _score_single_criterion

    console = get_console()
    console.print(Panel("[bold cyan]STAGE: STEP 5 — Q&A SCORING (requires Ollama)[/bold cyan]", expand=False))

    console.print("Parsing SW Engineer JD...")
    jd_result = parse_jd_to_rubric(SW_JD)
    rubric = jd_result.rubric

    console.print("Ingesting strong resume...")
    cross_index = CrossCandidateIndex()
    record, store = ingest_resume(STRONG_RESUME, "validate_strong", cross_index)

    criterion = rubric.weighted_criteria[0]
    console.print(f"Scoring criterion: [bold]{criterion.criterion}[/bold]")

    cs = _score_single_criterion(record, store, criterion)

    console.print(Panel(
        f"[bold]Score:[/bold] {cs.score}/10\n"
        f"[bold]Confidence:[/bold] {cs.confidence:.2f}\n"
        f"[bold]Evidence Verified:[/bold] {cs.evidence_verified}\n"
        f"[bold]Evidence Quote:[/bold] {cs.evidence_quote}\n"
        f"[bold]Answer:[/bold] {cs.answer}",
        title=f"CriterionScore — {criterion.criterion}",
        border_style="green",
    ))

def stage_full():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from pipeline.orchestrator import run_step1_only, run_full_pipeline
    from schemas.rubric import FrozenRubric
    import tempfile
    from pathlib import Path

    console = get_console()
    console.print(Panel("[bold cyan]STAGE: FULL PIPELINE (requires Ollama)[/bold cyan]", expand=False))

    # Step 1
    console.print("[bold]Step 1:[/bold] Parsing JD...")
    jd_result = run_step1_only(SW_JD)
    frozen = FrozenRubric(
        job_title=jd_result.rubric.job_title,
        hard_gates=jd_result.rubric.hard_gates,
        weighted_criteria=jd_result.rubric.weighted_criteria,
        bonus_sensitivity=jd_result.rubric.bonus_sensitivity,
    ).freeze()

    console.print(f"Rubric: [bold]{frozen.job_title}[/bold] | {len(frozen.weighted_criteria)} criteria")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "validate_full"
        resume_paths = [str(STRONG_RESUME), str(MEDIUM_RESUME), str(WEAK_RESUME)]

        updates = []
        def progress_cb(update):
            updates.append(update)
            console.print(f"  [dim]Step {update.step}:[/dim] {update.detail}")

        console.print("\n[bold]Running full pipeline...[/bold]")
        run = run_full_pipeline(
            jd_file_path=SW_JD,
            resume_file_paths=resume_paths,
            frozen_rubric=frozen,
            progress_cb=progress_cb,
            run_id="validate_full_run",
        )

    if run.errors:
        console.print(f"[red]Pipeline errors:[/red] {run.errors}")
        return

    shortlist = run.shortlist
    if not shortlist:
        console.print("[red]No shortlist produced[/red]")
        return

    table = Table(title=f"Ranked Results — {shortlist.job_title}", show_lines=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Name", style="bold cyan")
    table.add_column("Final Score", justify="right", style="bold")
    table.add_column("Base", justify="right")
    table.add_column("Confidence")
    table.add_column("Gate")

    for c in shortlist.ranked_candidates:
        color = "green" if c.final_score >= 7 else "yellow" if c.final_score >= 5 else "red"
        table.add_row(
            str(c.rank), c.name,
            f"[{color}]{c.final_score:.2f}[/{color}]",
            f"{c.base_score:.2f}", c.confidence, c.gate_status,
        )
    for c in shortlist.excluded_candidates:
        table.add_row("EXCL", c.name, f"{c.final_score:.2f}", f"{c.base_score:.2f}",
                      c.confidence, "[red]failed[/red]")

    console.print(table)
    console.print(f"\n[bold]Total submitted:[/bold] {shortlist.total_resumes_submitted}")
    console.print(f"[bold]Ranked:[/bold] {len(shortlist.ranked_candidates)}")
    console.print(f"[bold]Excluded:[/bold] {len(shortlist.excluded_candidates)}")


STAGE_FUNCTIONS = {
    "parsing":   stage_parsing,
    "chunking":  stage_chunking,
    "embedding": stage_embedding,
    "step1":     stage_step1,
    "step5":     stage_step5,
    "full":      stage_full,
}

# "all" skips Ollama-dependent stages
ALL_SAFE_STAGES = ["parsing", "chunking", "embedding"]


def main():
    parser = argparse.ArgumentParser(description="Validate AI Recruiter Co-Pilot pipeline stages")
    parser.add_argument(
        "--stage",
        default="all",
        choices=list(STAGE_FUNCTIONS.keys()) + ["all"],
        help="Which stage to run (default: all → runs parsing/chunking/embedding only)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        stages_to_run = ALL_SAFE_STAGES
        print("\n[INFO] --stage=all runs only safe stages (no Ollama required).")
        print("       Use --stage step1/step5/full for LLM-dependent stages.\n")
    else:
        stages_to_run = [args.stage]

    for stage_name in stages_to_run:
        fn = STAGE_FUNCTIONS[stage_name]
        print(f"\n{'='*60}")
        print(f"Running stage: {stage_name.upper()}")
        print('='*60)
        try:
            fn()
        except ImportError as e:
            print(f"[SKIP] {stage_name}: missing dependency — {e}")
        except Exception as e:
            import traceback
            print(f"[ERROR] {stage_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
