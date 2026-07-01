# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Parallel Passes — Open Signals, Red Flags, Hard Gates
# ─────────────────────────────────────────────────────────────────────────────
# Three SEPARATE, narrow-scoped LLM calls per candidate.
# Each has a single purpose and a tight prompt — keeping prompts narrow
# is what keeps each pass reliable.
#
# These run INDEPENDENTLY of criterion scoring (Step 5):
#   - Open Signals: additive bonus, unbounded by rubric
#   - Red Flags: plausibility check only, never speculative
#   - Hard Gates: binary eligibility, applied as exclusion before ranking
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Callable

from core.llm_client import llm_client
from core.config import settings
from prompts.parallel_passes import (
    OPEN_SIGNALS_SYSTEM,
    get_open_signals_user_prompt,
    RED_FLAG_SYSTEM,
    get_red_flag_user_prompt,
    get_hard_gate_system_prompt,
    get_hard_gate_user_prompt,
)
from schemas.rubric import FrozenRubric
from schemas.candidate import CandidateRecord, OpenSignal, RedFlag, GateResult

logger = logging.getLogger(__name__)


# ── Pass A: Open Signal Mining ────────────────────────────────────────────────

def run_open_signal_pass(candidate: CandidateRecord) -> list[OpenSignal]:
    """
    Scan for standout achievements outside rubric criteria.
    Returns list of OpenSignal objects (may be empty).
    """
    if not candidate.raw_text.strip():
        return []

    try:
        raw = llm_client.structured_call(
            system_prompt=OPEN_SIGNALS_SYSTEM,
            user_prompt=get_open_signals_user_prompt(candidate.raw_text),
            expected_keys=["signals"],
        )
    except Exception as e:
        logger.warning(f"[{candidate.candidate_id}] Open signal pass failed: {e}")
        return []

    signals = []
    for item in raw.get("signals", []):
        try:
            strength = float(item.get("strength", 0))
            if strength < 0.2:
                continue  # Below threshold
            signals.append(OpenSignal(
                signal=str(item.get("signal", ""))[:200],
                evidence_quote=str(item.get("evidence_quote", ""))[:150],
                strength=round(min(1.0, max(0.0, strength)), 3),
            ))
        except Exception:
            continue

    logger.debug(f"[{candidate.candidate_id}] Open signals: {len(signals)}")
    return signals[:5]  # Cap at 5 signals


# ── Pass B: Red Flag Detection ────────────────────────────────────────────────

def run_red_flag_pass(candidate: CandidateRecord) -> list[RedFlag]:
    """
    Detect objective plausibility concerns in the resume timeline.
    Returns list of RedFlag objects (may be empty).
    """
    if not candidate.raw_text.strip():
        return []

    VALID_TYPES = {
        "employment_gap", "title_inconsistency", "date_overlap",
        "short_tenure", "credential_mismatch"
    }
    VALID_SEVERITIES = {"info", "review", "concern"}

    try:
        raw = llm_client.structured_call(
            system_prompt=RED_FLAG_SYSTEM,
            user_prompt=get_red_flag_user_prompt(candidate.raw_text),
            expected_keys=["flags"],
        )
    except Exception as e:
        logger.warning(f"[{candidate.candidate_id}] Red flag pass failed: {e}")
        return []

    flags = []
    for item in raw.get("flags", []):
        try:
            flag_type = str(item.get("type", "employment_gap"))
            if flag_type not in VALID_TYPES:
                flag_type = "employment_gap"  # Fallback to valid type
            
            severity = str(item.get("severity", "info"))
            if severity not in VALID_SEVERITIES:
                severity = "info"

            flags.append(RedFlag(
                type=flag_type,
                detail=str(item.get("detail", ""))[:300],
                severity=severity,
            ))
        except Exception:
            continue

    logger.debug(f"[{candidate.candidate_id}] Red flags: {len(flags)}")
    return flags


# ── Pass C: Hard Gate Evaluation ─────────────────────────────────────────────

def run_hard_gate_pass(
    candidate: CandidateRecord,
    rubric: FrozenRubric,
) -> tuple[list[GateResult], str]:
    """
    Evaluate all hard gates for a candidate.
    
    Returns:
        (gate_results, gate_status) where gate_status = "passed" | "failed"
    """
    if not rubric.hard_gates:
        return [], "passed"

    if not candidate.raw_text.strip():
        # No text → fail all gates
        results = [
            GateResult(
                criterion=gate.criterion,
                passed=False,
                evidence_quote="NOT_FOUND",
                failure_reason="Resume text could not be extracted.",
            )
            for gate in rubric.hard_gates
        ]
        return results, "failed"

    gate_results = []
    overall_passed = True

    for gate in rubric.hard_gates:
        try:
            raw = llm_client.structured_call(
                system_prompt=get_hard_gate_system_prompt(),
                user_prompt=get_hard_gate_user_prompt(
                    gate.criterion,
                    candidate.raw_text[:settings.full_resume_max_chars],
                ),
                expected_keys=["passed"],
            )
        except Exception as e:
            logger.warning(
                f"[{candidate.candidate_id}] Hard gate eval failed for "
                f"'{gate.criterion}': {e}"
            )
            # Fail-safe: if we can't evaluate, treat as inconclusive (not failed)
            gate_results.append(GateResult(
                criterion=gate.criterion,
                passed=True,  # Benefit of doubt
                evidence_quote="EVALUATION_ERROR",
                failure_reason=f"Evaluation error: {str(e)[:100]}",
            ))
            continue

        passed = bool(raw.get("passed", False))
        
        if gate.strict and not passed:
            overall_passed = False

        gate_results.append(GateResult(
            criterion=gate.criterion,
            passed=passed,
            evidence_quote=str(raw.get("evidence_quote", "NOT_FOUND"))[:150],
            failure_reason=str(raw.get("failure_reason", "")) if not passed else None,
        ))

    gate_status = "passed" if overall_passed else "failed"
    
    if gate_status == "failed":
        failed_gates = [r.criterion for r in gate_results if not r.passed]
        logger.info(
            f"[{candidate.candidate_id}] '{candidate.name}' EXCLUDED — "
            f"failed gates: {failed_gates}"
        )

    return gate_results, gate_status


# ── Run All Passes for All Candidates ─────────────────────────────────────────

def run_all_parallel_passes(
    shortlisted_ids: list[str],
    records: dict[str, CandidateRecord],
    rubric: FrozenRubric,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, CandidateRecord]:
    """
    Run all three parallel passes for all shortlisted candidates.
    Mutates records in-place with open_signals, red_flags, gate_results.
    
    Args:
        shortlisted_ids: Candidates that passed Step 4
        records: All candidate records (mutated in-place)
        rubric: Frozen rubric (needed for hard gate criteria)
        progress_callback: Optional UI progress hook(current, total, name)
    
    Returns:
        Updated records dict
    """
    total = len(shortlisted_ids)
    logger.info(f"Step 6: Running parallel passes on {total} candidates")

    for i, candidate_id in enumerate(shortlisted_ids):
        if candidate_id not in records:
            continue

        candidate = records[candidate_id]

        if progress_callback:
            progress_callback(i + 1, total, candidate.name)

        logger.info(f"[{candidate_id}] Running parallel passes for '{candidate.name}'")

        # Pass A: Open Signals
        candidate.open_signals = run_open_signal_pass(candidate)

        # Pass B: Red Flags
        candidate.red_flags = run_red_flag_pass(candidate)

        # Pass C: Hard Gates
        gate_results, gate_status = run_hard_gate_pass(candidate, rubric)
        candidate.gate_results = gate_results
        candidate.gate_status = gate_status

        if gate_status == "failed":
            failed = [r for r in gate_results if not r.passed]
            reasons = "; ".join(
                f"'{r.criterion}': {r.failure_reason or 'requirement not met'}"
                for r in failed
            )
            candidate.excluded = True
            candidate.exclusion_reason = f"Failed hard gate(s): {reasons}"

        records[candidate_id] = candidate

    excluded_count = sum(1 for cid in shortlisted_ids if records[cid].excluded)
    logger.info(
        f"Step 6 complete: {excluded_count} candidates excluded by hard gates"
    )
    return records
