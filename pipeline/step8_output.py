# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Output Assembly — CSV, JSON, and HTML Report
# ─────────────────────────────────────────────────────────────────────────────
# Three output artifacts per run:
#   1. ranked_shortlist.csv  — submission-ready ranked list
#   2. candidates/           — per-candidate full JSON breakdown
#   3. run_report.html       — human-readable rich report for the dashboard
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from schemas.rubric import FrozenRubric
from schemas.candidate import CandidateRecord, RankedShortlist

logger = logging.getLogger(__name__)


def assemble_output(
    run_id: str,
    rubric: FrozenRubric,
    ranked_candidates: list[CandidateRecord],
    excluded_candidates: list[CandidateRecord],
    total_submitted: int,
    run_dir: Path,
) -> RankedShortlist:
    """
    Assemble all output artifacts for a completed evaluation run.
    
    Creates:
    - run_dir/ranked_shortlist.csv
    - run_dir/candidates/{candidate_id}.json
    - run_dir/run_report.html
    - run_dir/rubric.json  (frozen rubric for audit)
    
    Args:
        run_id: Unique run identifier
        rubric: The frozen rubric used for this run
        ranked_candidates: Scored + ranked candidates (not excluded)
        excluded_candidates: Hard-gate-failed candidates
        total_submitted: Total resumes originally submitted
        run_dir: Directory to write all outputs
    
    Returns:
        RankedShortlist — the complete run record
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)

    logger.info(f"Step 8: Assembling output for run '{run_id}' → {run_dir}")

    shortlist = RankedShortlist(
        run_id=run_id,
        job_title=rubric.job_title,
        total_resumes_submitted=total_submitted,
        shortlisted_count=len(ranked_candidates),
        ranked_candidates=ranked_candidates,
        excluded_candidates=excluded_candidates,
        rubric_used=rubric.model_dump(mode="json"),
    )

    # ── 1. Frozen Rubric JSON (audit trail) ───────────────────────────────────
    rubric_path = run_dir / "rubric.json"
    rubric_path.write_text(
        json.dumps(rubric.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    logger.info(f"Rubric saved: {rubric_path}")

    # ── 2. Per-Candidate Breakdown JSONs ──────────────────────────────────────
    all_candidates = ranked_candidates + excluded_candidates
    for candidate in all_candidates:
        candidate_path = run_dir / "candidates" / f"{candidate.candidate_id}.json"
        # Exclude raw_text from JSON to keep files small
        record_dict = candidate.model_dump(mode="json", exclude={"raw_text"})
        candidate_path.write_text(
            json.dumps(record_dict, indent=2),
            encoding="utf-8",
        )

    logger.info(f"Per-candidate JSONs saved: {len(all_candidates)} files")

    # ── 3. Ranked Shortlist CSV ───────────────────────────────────────────────
    csv_path = run_dir / "ranked_shortlist.csv"
    _export_ranked_csv(ranked_candidates, excluded_candidates, rubric, csv_path)
    logger.info(f"CSV exported: {csv_path}")

    # ── 4. HTML Report ────────────────────────────────────────────────────────
    html_path = run_dir / "run_report.html"
    _export_html_report(shortlist, rubric, html_path)
    logger.info(f"HTML report generated: {html_path}")

    return shortlist


def _export_ranked_csv(
    ranked: list[CandidateRecord],
    excluded: list[CandidateRecord],
    rubric: FrozenRubric,
    output_path: Path,
) -> None:
    """Export submission-ready CSV with ranked candidates + excluded section."""
    rows = []

    for candidate in ranked:
        top_evidence = "; ".join(candidate.get_top_evidence(n=2))
        red_flag_summary = "; ".join(
            f"{f.type}({f.severity})" for f in candidate.red_flags
        ) or "None"
        open_signals_summary = "; ".join(
            s.signal for s in candidate.open_signals
        ) or "None"

        rows.append({
            "Rank": candidate.rank,
            "Status": "Ranked",
            "Candidate_ID": candidate.candidate_id,
            "Name": candidate.name,
            "Final_Score": round(candidate.final_score, 3),
            "Base_Score": round(candidate.base_score, 3),
            "Bonus_Applied": round(candidate.bonus_applied, 3),
            "Confidence": candidate.confidence,
            "Gate_Status": candidate.gate_status,
            "Embedding_Similarity": round(candidate.embedding_score, 3),
            "Top_Evidence": top_evidence,
            "Open_Signals": open_signals_summary,
            "Red_Flags": red_flag_summary,
            "Exclusion_Reason": "",
        })

    for candidate in excluded:
        rows.append({
            "Rank": "EXCLUDED",
            "Status": "Excluded",
            "Candidate_ID": candidate.candidate_id,
            "Name": candidate.name,
            "Final_Score": round(candidate.final_score, 3),
            "Base_Score": round(candidate.base_score, 3),
            "Bonus_Applied": round(candidate.bonus_applied, 3),
            "Confidence": candidate.confidence,
            "Gate_Status": candidate.gate_status,
            "Embedding_Similarity": round(candidate.embedding_score, 3),
            "Top_Evidence": "",
            "Open_Signals": "",
            "Red_Flags": "",
            "Exclusion_Reason": candidate.exclusion_reason or "Failed hard gate",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")


def _export_html_report(
    shortlist: RankedShortlist,
    rubric: FrozenRubric,
    output_path: Path,
) -> None:
    """Generate a clean, self-contained HTML report for human review."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Criteria header columns
    criteria_headers = "".join(
        f"<th>{c.criterion[:30]} (w={c.weight:.2f})</th>"
        for c in rubric.weighted_criteria
    )

    # Ranked rows
    ranked_rows_html = ""
    for c in shortlist.ranked_candidates:
        score_cells = ""
        score_map = {cs.criterion_id: cs for cs in c.criterion_scores}
        for crit in rubric.weighted_criteria:
            cs = score_map.get(crit.id)
            if cs:
                bg = _score_to_bg(cs.score)
                quote = cs.evidence_quote[:80] if cs.evidence_quote != "NO_EVIDENCE_FOUND" else "—"
                score_cells += (
                    f'<td style="background:{bg};text-align:center" '
                    f'title="{quote}">{cs.score}/10</td>'
                )
            else:
                score_cells += '<td style="text-align:center">—</td>'

        flags_html = " ".join(
            f'<span class="flag flag-{f.severity}">{f.type}</span>'
            for f in c.red_flags
        ) or "—"
        signals_html = "; ".join(s.signal[:60] for s in c.open_signals) or "—"

        conf_class = {"High": "conf-high", "Medium": "conf-med", "Low": "conf-low"}[c.confidence]

        ranked_rows_html += f"""
        <tr>
          <td><strong>#{c.rank}</strong></td>
          <td>{c.name}</td>
          <td style="text-align:center"><strong>{c.final_score:.2f}</strong></td>
          <td class="{conf_class}" style="text-align:center">{c.confidence}</td>
          {score_cells}
          <td style="font-size:0.85em">{flags_html}</td>
          <td style="font-size:0.85em;color:#555">{signals_html}</td>
        </tr>"""

    # Excluded rows
    excluded_rows_html = ""
    for c in shortlist.excluded_candidates:
        excluded_rows_html += f"""
        <tr style="background:#fff3f3">
          <td>{c.name}</td>
          <td style="color:#c00">{c.exclusion_reason or "Hard gate failure"}</td>
          <td>{c.final_score:.2f} (audit)</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Recruiter Co-Pilot — Run {shortlist.run_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #f5f7fa; color: #1a1a2e; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #6c63ff; padding-bottom: 10px; }}
  h2 {{ color: #6c63ff; margin-top: 32px; }}
  .meta {{ background: #fff; padding: 16px; border-radius: 8px; 
           border-left: 4px solid #6c63ff; margin-bottom: 24px; }}
  .meta span {{ margin-right: 24px; font-size: 0.95em; color: #555; }}
  .meta strong {{ color: #1a1a2e; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  th {{ background: #1a1a2e; color: #fff; padding: 10px 8px; 
        text-align: left; font-size: 0.82em; white-space: nowrap; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid #eee; font-size: 0.88em; vertical-align: top; }}
  tr:hover td {{ background: #f0f0ff; }}
  .flag {{ display: inline-block; padding: 2px 7px; border-radius: 12px;
           font-size: 0.78em; font-weight: 600; margin: 1px; }}
  .flag-info {{ background: #e3f2fd; color: #1565c0; }}
  .flag-review {{ background: #fff8e1; color: #e65100; }}
  .flag-concern {{ background: #fce4ec; color: #b71c1c; }}
  .conf-high {{ color: #2e7d32; font-weight: 700; }}
  .conf-med  {{ color: #e65100; font-weight: 700; }}
  .conf-low  {{ color: #c62828; font-weight: 700; }}
  .rubric-table td, .rubric-table th {{ font-size: 0.85em; }}
</style>
</head>
<body>
<h1>🤖 AI Recruiter Co-Pilot — Evaluation Report</h1>
<div class="meta">
  <span>Run ID: <strong>{shortlist.run_id}</strong></span>
  <span>Role: <strong>{shortlist.job_title}</strong></span>
  <span>Generated: <strong>{now}</strong></span>
  <span>Submitted: <strong>{shortlist.total_resumes_submitted}</strong></span>
  <span>Shortlisted: <strong>{shortlist.shortlisted_count}</strong></span>
  <span>Ranked: <strong>{len(shortlist.ranked_candidates)}</strong></span>
  <span>Excluded: <strong>{len(shortlist.excluded_candidates)}</strong></span>
</div>

<h2>📋 Rubric Used</h2>
<table class="rubric-table">
  <tr><th>Criterion</th><th>Category</th><th>Weight</th><th>Rationale</th></tr>
  {"".join(f"<tr><td>{c.criterion}</td><td>{c.category}</td><td>{c.weight:.3f}</td><td>{c.rationale}</td></tr>" for c in rubric.weighted_criteria)}
</table>
{"<br><strong>Hard Gates:</strong> " + ", ".join(f"{g.criterion} (strict={g.strict})" for g in rubric.hard_gates) if rubric.hard_gates else ""}

<h2>🏆 Ranked Candidates</h2>
<table>
  <tr>
    <th>Rank</th><th>Name</th><th>Final Score</th><th>Confidence</th>
    {criteria_headers}
    <th>Red Flags</th><th>Open Signals</th>
  </tr>
  {ranked_rows_html}
</table>

{"<h2>🚫 Excluded Candidates</h2><table><tr><th>Name</th><th>Exclusion Reason</th><th>Score (audit)</th></tr>" + excluded_rows_html + "</table>" if shortlist.excluded_candidates else ""}

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def _score_to_bg(score: int) -> str:
    """Map 0-10 score to a background colour for the HTML table."""
    if score >= 8:
        return "#c8e6c9"  # Green
    elif score >= 6:
        return "#fff9c4"  # Yellow
    elif score >= 4:
        return "#ffe0b2"  # Orange
    else:
        return "#ffcdd2"  # Red
