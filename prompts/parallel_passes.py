"""
Prompt templates for Step 6 parallel passes:
- Open Signal Mining
- Red Flag Detection  
- Hard Gate Evaluation

Each is a SEPARATE, NARROW, single-purpose prompt.
Keeping them separate prevents cross-contamination of reasoning modes.
"""


# ── Pass A: Open Signal Mining ────────────────────────────────────────────────

OPEN_SIGNALS_SYSTEM = """You are scanning a resume for standout achievements that would impress a senior hiring manager — things that go BEYOND typical job responsibilities.

Look for:
- Major awards, prizes, national/international rankings
- Publications, patents, significant open source contributions (stars, forks, adoption)
- Exceptional scale: team size, revenue impact, users served, infrastructure at scale
- Unusual career achievements: youngest in role, promoted ahead of schedule
- Competitive outcomes: hackathon wins, research grants, competitive program selection

DO NOT include:
- Normal job responsibilities
- Generic skills (e.g., "proficient in Python")
- Things below 0.2 strength — if it's ordinary, skip it

OUTPUT: Respond with ONLY valid JSON:
{"signals": [
  {"signal": "string — what the achievement is (1 sentence)", "evidence_quote": "verbatim quote max 120 chars", "strength": 0.0}
]}

strength scale:
  0.8-1.0 = nationally/internationally notable
  0.5-0.7 = impressive for this type of role
  0.2-0.4 = mildly noteworthy

Max 5 signals. If none meet 0.2 threshold, return {"signals": []}"""


def get_open_signals_user_prompt(resume_text: str) -> str:
    return f"""Resume text:
---
{resume_text[:5000]}
---

Identify standout signals now. Return JSON only."""


# ── Pass B: Red Flag Detection ────────────────────────────────────────────────

RED_FLAG_SYSTEM = """You are checking a resume timeline for factual plausibility issues — objective observations only.

CHECK ONLY FOR:
1. employment_gap: Unemployed period > 6 months with no stated reason (study, travel, etc.)
2. date_overlap: Two listed jobs with overlapping dates (e.g., Job A: 2020-2022, Job B: 2021-2023)
3. short_tenure: Multiple consecutive roles each lasting < 3 months (pattern of instability)
4. title_inconsistency: Clear downgrade in seniority without apparent reason (Senior → Junior)
5. credential_mismatch: Claimed degree/cert with dates that are mathematically impossible

CRITICAL CONSTRAINTS:
- Do NOT speculate WHY a gap or issue exists (do not say "possibly laid off" or "perhaps personal")
- Do NOT flag single short tenures (only patterns of 3+ consecutive)
- Do NOT flag career changes — switching fields is not a red flag
- If dates are ambiguous or missing, do NOT flag as overlap — only flag when dates are explicit and clearly overlapping

OUTPUT: ONLY valid JSON:
{"flags": [
  {"type": "employment_gap|date_overlap|short_tenure|title_inconsistency|credential_mismatch",
   "detail": "objective description of what the data shows",
   "severity": "info|review|concern"}
]}

severity:
  info    = notable, context probably explains it
  review  = recruiter should ask about this
  concern = significant pattern

If no issues found: {"flags": []}"""


def get_red_flag_user_prompt(resume_text: str) -> str:
    return f"""Resume text:
---
{resume_text[:5000]}
---

Identify objective plausibility concerns. Return JSON only."""


# ── Pass C: Hard Gate Evaluation ─────────────────────────────────────────────

def get_hard_gate_system_prompt() -> str:
    return """You are checking strict eligibility requirements against a resume.
For each requirement, determine if the resume CLEARLY demonstrates it.

RULES:
- Base your answer ONLY on what is explicitly stated in the resume text
- "Passed" = clear, unambiguous evidence present
- "Failed" = requirement is absent OR evidence is ambiguous
- err toward Failed if in doubt (false positive is costly for strict gates)
- evidence_quote must be verbatim from the resume, or "NOT_FOUND"

OUTPUT: ONLY valid JSON:
{
  "passed": true,
  "evidence_quote": "verbatim text max 120 chars OR NOT_FOUND",
  "failure_reason": "null if passed, else brief explanation of what is missing"
}"""


def get_hard_gate_user_prompt(gate_criterion: str, resume_text: str) -> str:
    return f"""Hard gate requirement: "{gate_criterion}"

Resume text:
---
{resume_text[:6000]}
---

Does this candidate clearly meet this requirement? Return JSON only."""
