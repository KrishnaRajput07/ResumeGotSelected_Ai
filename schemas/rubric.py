# ─────────────────────────────────────────────────────────────────────────────
# Pydantic v2 Schemas — Rubric data contracts
# Every LLM output boundary is typed and validated here.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
from datetime import datetime, timezone
import uuid


class HardGate(BaseModel):
    """
    A hard gate is a binary eligibility requirement.
    strict=True  → failing candidate is EXCLUDED from ranking entirely.
    strict=False → failing candidate receives a heavy score penalty but stays in list.
    """
    criterion: str
    strict: bool = True


class WeightedCriterion(BaseModel):
    """
    A single scoreable evaluation axis derived from the JD.
    Weight is the fraction of the 0-10 final score this criterion contributes.
    """
    id: str = Field(default_factory=lambda: f"c{uuid.uuid4().hex[:4]}")
    criterion: str
    category: Literal[
        "Hard Skills",
        "Soft Skills",
        "Experience",
        "Education",
        "Domain Knowledge",
        "Leadership"
    ]
    weight: float = Field(ge=0.0, le=1.0)
    rationale: str   # LLM explains why this weight was chosen


class FrozenRubric(BaseModel):
    """
    The immutable contract for an evaluation run.
    Frozen at the moment the recruiter clicks "Evaluate".
    Every candidate in this run is scored against this exact object.
    """
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_title: str
    hard_gates: list[HardGate] = Field(default_factory=list)
    weighted_criteria: list[WeightedCriterion]
    bonus_sensitivity: Literal["off", "low", "medium", "high"] = "medium"
    frozen_at: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'FrozenRubric':
        """Weights must sum to 1.0 (±1% tolerance for float rounding)."""
        if not self.weighted_criteria:
            raise ValueError("Rubric must have at least one weighted criterion.")
        total = sum(c.weight for c in self.weighted_criteria)
        if not (0.97 <= total <= 1.03):
            raise ValueError(
                f"Criterion weights must sum to 1.0, got {total:.4f}. "
                f"Auto-normalize before freezing."
            )
        return self

    def normalize_weights(self) -> 'FrozenRubric':
        """
        Re-scale all weights proportionally so they sum to exactly 1.0.
        Call this before validation if the LLM returned slightly off weights.
        """
        total = sum(c.weight for c in self.weighted_criteria)
        if total == 0:
            raise ValueError("All weights are zero — cannot normalize.")
        normalized = self.model_copy(deep=True)
        for criterion in normalized.weighted_criteria:
            criterion.weight = round(criterion.weight / total, 6)
        return normalized

    def freeze(self) -> 'FrozenRubric':
        """Mark the rubric as frozen with current timestamp."""
        frozen = self.model_copy(deep=True)
        frozen.frozen_at = datetime.now(timezone.utc)
        return frozen

    def to_query_text(self) -> str:
        """
        Converts the rubric into a single text blob for embedding-based
        shortlisting (Step 4). Represents 'the ideal candidate profile.'
        """
        parts = [f"Job: {self.job_title}"]
        for c in self.weighted_criteria:
            parts.append(f"{c.criterion} (weight: {c.weight:.2f})")
        for g in self.hard_gates:
            parts.append(f"Required: {g.criterion}")
        return " | ".join(parts)
