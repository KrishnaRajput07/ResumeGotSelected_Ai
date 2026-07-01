"""
Prompt templates for per-criterion evidence-grounded Q&A (Step 5).

This is the core intelligence prompt. Key engineering decisions:
1. Score anchors embedded in system prompt — prevents score drift across candidates
2. LLM receives ONLY retrieved chunks (not full resume) — prevents hallucination
3. evidence_quote must be VERBATIM — post-processing validates this
4. "NO_EVIDENCE_FOUND" is an explicit, named output — avoids forced low scores
5. Confidence = certainty that all relevant evidence was captured, NOT score certainty
"""

# ── Score Anchoring Rubric (included in every Q&A prompt) ───────────────────
# This is the key mechanism that makes scores comparable across candidates.
# Without anchors, LLMs cluster scores in the 5-8 range.
SCORE_ANCHOR_RUBRIC = """
SCORE ANCHORING (use these anchors to calibrate your score):
  10: Exceptional, quantified evidence. Direct, strong match. (e.g., "Led team of 20 engineers at Google for 3 years")
  8-9: Clear, specific evidence found. Solid match, not fully quantified. 
  6-7: Reasonable evidence present. Partial match or indirect indicator.
  4-5: Weak or tangential evidence. Speculative connection to criterion.
  2-3: Barely relevant evidence found. Very loose association.
  1:   No relevant evidence, but adjacent topic exists in resume.
  0:   Criterion topic is COMPLETELY absent from resume excerpts.
"""


def get_criterion_qa_system_prompt(criterion: str) -> str:
    """
    System prompt for scoring a single criterion.
    The criterion is baked into the system prompt for clarity.
    """
    return f"""You are evaluating a candidate's resume for one specific criterion:
CRITERION: "{criterion}"

You have access ONLY to the resume excerpts provided by the user.
You must base your entire evaluation on those excerpts — nothing else.
ONLY cite facts present in this data, do not invent or extrapolate anything. Do not make assumptions or fill in missing details.

{SCORE_ANCHOR_RUBRIC}

OUTPUT: Respond with ONLY valid JSON matching this exact schema:
{{
  "answer": "string — 1-3 sentence reasoning explaining your score",
  "evidence_quote": "string — VERBATIM excerpt (max 150 chars) OR the literal string NO_EVIDENCE_FOUND",
  "confidence": 0.85,
  "score": 7
}}

CRITICAL RULES:
1. evidence_quote MUST be copied verbatim from the excerpts. Do not paraphrase or summarize.
2. If no relevant evidence exists, set evidence_quote to exactly: NO_EVIDENCE_FOUND and score to 0.
3. Do NOT infer from the criterion name alone (e.g., if criterion is "Python" but resume says nothing about it, score 0).
4. confidence = your certainty that the excerpts captured ALL relevant info (0.0-1.0).
   Low confidence if resume sections appear cut off or incomplete.
5. answer should explain the score in 1-3 concise sentences. Reference the evidence.
"""


def get_criterion_qa_user_prompt(
    criterion: str,
    retrieved_chunks: list[str],
) -> str:
    """
    User-turn: provides the retrieved resume chunks for this criterion.
    
    Args:
        criterion: The criterion being evaluated
        retrieved_chunks: List of resume chunk texts (top-N from vector search)
    """
    chunks_text = "\n\n[CHUNK BREAK]\n\n".join(retrieved_chunks)
    return f"""Evaluate this candidate for: "{criterion}"

Resume excerpts (use ONLY these):
---
{chunks_text}
---

Provide your JSON evaluation now."""
