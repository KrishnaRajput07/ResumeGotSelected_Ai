"""
Prompt templates for JD parsing → rubric generation (Step 1).

Engineering principles:
- Temperature 0 call: must produce deterministic, parseable JSON
- Category weight priors injected into system prompt (prevents arbitrary weights)
- Explicit schema example prevents model from inventing fields
- /no_think prefix disables Qwen3 thinking tokens (not needed for structured extraction)
"""

from core.config import settings


def get_jd_parsing_prompt(category_priors: dict | None = None) -> str:
    """
    System prompt for JD → rubric extraction.
    Injects the category weight priors to bound LLM weight generation.
    """
    priors = category_priors or settings.category_weight_priors
    prior_lines = "\n".join(
        f"  - {cat}: default weight {weight:.2f}"
        for cat, weight in priors.items()
    )
    delta = settings.weight_prior_delta_max

    return f"""You are an expert technical recruiter. Analyze a job description and extract a structured evaluation rubric.

OUTPUT: Respond with ONLY valid JSON. No prose, no markdown, no explanation.

SCHEMA (follow exactly):
{{
  "job_title": "string — role title from JD",
  "hard_gates": [
    {{
      "criterion": "string — exact dealbreaker requirement",
      "strict": true
    }}
  ],
  "weighted_criteria": [
    {{
      "id": "c1",
      "criterion": "string — specific, distinct evaluation criterion",
      "category": "Hard Skills|Soft Skills|Experience|Education|Domain Knowledge|Leadership",
      "weight": 0.20,
      "rationale": "string — why this weight (1 sentence)"
    }}
  ],
  "bonus_sensitivity": "medium"
}}

RULES:
1. Extract 5-8 DISTINCT weighted criteria. Avoid synonyms or overlapping criteria.
2. Hard gates = true dealbreakers ONLY (e.g., legal work authorization, mandatory degree).
   Max 4 hard gates. If nothing is truly mandatory, set hard_gates to [].
3. Weights MUST sum to exactly 1.00. No exceptions.
4. Category default weights (your starting point):
{prior_lines}
5. You may deviate from defaults by ±{delta:.2f} based on JD emphasis language
   (words like "must", "required", "critical", "essential" = higher weight;
   "nice to have", "preferred", "bonus" = lower weight).
6. Assign sequential IDs: "c1", "c2", "c3", etc.
7. criterion strings must be specific and actionable, not vague
   (BAD: "communication skills"; GOOD: "stakeholder communication in Agile team settings")
8. bonus_sensitivity: default "medium" unless JD explicitly values exceptional achievement.
"""


def get_jd_user_prompt(jd_text: str) -> str:
    """User-turn prompt: passes the JD text to parse."""
    return f"""Job Description:
---
{jd_text[:6000]}
---

Extract the evaluation rubric JSON now."""
