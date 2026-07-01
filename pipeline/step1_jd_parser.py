# ─────────────────────────────────────────────────────────────────────────────
# Step 1: JD Parser → AI-Proposed Rubric
# ─────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path

from core.llm_client import llm_client
from core.pdf_parser import parse_file
from core.config import settings
from prompts.jd_parsing import get_jd_parsing_prompt, get_jd_user_prompt
from schemas.rubric import FrozenRubric, WeightedCriterion, HardGate

logger = logging.getLogger(__name__)


class JDParseResult:
    """Container for Step 1 output shown to recruiter for review."""
    def __init__(self, rubric: FrozenRubric, jd_text: str, warnings: list[str]):
        self.rubric = rubric
        self.jd_text = jd_text
        self.warnings = warnings


def parse_jd_to_rubric(jd_file_path: str | Path) -> JDParseResult:
    """
    Parse a job description file into a structured, AI-proposed rubric.
    
    Pipeline:
    1. Extract text from JD file (PDF/DOCX/TXT)
    2. Call LLM with structured output prompt
    3. Parse JSON → Pydantic FrozenRubric
    4. Validate and auto-normalize weights
    5. Return rubric + warnings for recruiter review
    
    Args:
        jd_file_path: Path to JD file (PDF, DOCX, or TXT)
    
    Returns:
        JDParseResult with the proposed rubric and any warnings
    """
    warnings = []

    # ── Extract JD text ───────────────────────────────────────────────────────
    logger.info(f"Parsing JD file: {jd_file_path}")
    parsed_doc = parse_file(jd_file_path)
    
    if parsed_doc.extraction_confidence < 0.3:
        warnings.append(
            f"JD extraction confidence is low ({parsed_doc.extraction_confidence:.2f}). "
            f"Results may be inaccurate. Consider using a cleaner PDF."
        )
    
    jd_text = parsed_doc.raw_text
    if len(jd_text) < 100:
        raise ValueError(f"JD text too short ({len(jd_text)} chars). Check file format.")

    logger.info(f"JD text extracted: {len(jd_text)} chars via {parsed_doc.extraction_method}")

    # ── LLM Call: JD → Rubric JSON ───────────────────────────────────────────
    system_prompt = get_jd_parsing_prompt()
    user_prompt = get_jd_user_prompt(jd_text)

    logger.info("Calling LLM to generate rubric...")
    raw_response = llm_client.structured_call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_keys=["job_title", "weighted_criteria"],
    )

    # ── Build Rubric from LLM Response ────────────────────────────────────────
    rubric, rubric_warnings = _build_rubric_from_response(raw_response)
    warnings.extend(rubric_warnings)

    logger.info(
        f"Rubric generated: '{rubric.job_title}' | "
        f"{len(rubric.weighted_criteria)} criteria | "
        f"{len(rubric.hard_gates)} hard gates"
    )

    return JDParseResult(rubric=rubric, jd_text=jd_text, warnings=warnings)


def _build_rubric_from_response(response: dict) -> tuple[FrozenRubric, list[str]]:
    """
    Parse the LLM's JSON response into a validated FrozenRubric.
    Handles normalization and validation with informative warnings.
    """
    warnings = []

    # ── Parse weighted criteria ────────────────────────────────────────────────
    raw_criteria = response.get("weighted_criteria", [])
    if not raw_criteria:
        raise ValueError("LLM returned no weighted criteria. Check JD content.")

    criteria = []
    for i, raw in enumerate(raw_criteria[:settings.max_weighted_criteria]):
        try:
            criterion = WeightedCriterion(
                id=raw.get("id", f"c{i+1}"),
                criterion=raw.get("criterion", f"Criterion {i+1}"),
                category=raw.get("category", "Hard Skills"),
                weight=float(raw.get("weight", 0.15)),
                rationale=raw.get("rationale", ""),
            )
            criteria.append(criterion)
        except Exception as e:
            warnings.append(f"Skipped malformed criterion #{i+1}: {e}")

    if len(criteria) < settings.min_weighted_criteria:
        raise ValueError(
            f"Too few valid criteria ({len(criteria)}). "
            f"Minimum required: {settings.min_weighted_criteria}"
        )

    # ── Parse hard gates ───────────────────────────────────────────────────────
    raw_gates = response.get("hard_gates", [])
    gates = []
    for raw in raw_gates:
        try:
            gates.append(HardGate(
                criterion=raw.get("criterion", ""),
                strict=raw.get("strict", True),
            ))
        except Exception as e:
            warnings.append(f"Skipped malformed hard gate: {e}")

    if len(gates) > settings.max_hard_gates:
        warnings.append(
            f"JD produced {len(gates)} hard gates (max recommended: {settings.max_hard_gates}). "
            f"Consider marking some as non-strict to avoid over-filtering candidates."
        )

    # ── Build rubric (unvalidated first for normalization) ─────────────────────
    rubric_data = {
        "job_title": response.get("job_title", "Unknown Role"),
        "hard_gates": gates,
        "weighted_criteria": criteria,
        "bonus_sensitivity": response.get("bonus_sensitivity", "medium"),
    }

    # Create without validation to allow normalization
    temp_rubric = FrozenRubric.model_construct(**rubric_data)
    temp_rubric.weighted_criteria = criteria
    temp_rubric.hard_gates = gates

    # ── Weight normalization ───────────────────────────────────────────────────
    weight_total = sum(c.weight for c in criteria)
    if not (0.97 <= weight_total <= 1.03):
        warnings.append(
            f"LLM weights summed to {weight_total:.3f} (expected 1.0). "
            f"Auto-normalized proportionally."
        )
        for c in criteria:
            c.weight = round(c.weight / weight_total, 6)

    # ── Final validated rubric ─────────────────────────────────────────────────
    rubric = FrozenRubric(
        job_title=rubric_data["job_title"],
        hard_gates=gates,
        weighted_criteria=criteria,
        bonus_sensitivity=rubric_data["bonus_sensitivity"],
    )

    return rubric, warnings
