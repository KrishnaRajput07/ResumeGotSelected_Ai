"""
Behavioral Scorer v2 — Deterministic feature-based scoring for StructuredCandidate.

Architecture (three-layer, no double-counting):
  Layer 1 — Honeypot Detection     : 12 checks, 2 tiers (hard / soft), CHK_F+G guard
  Layer 2 — behavioral_composite   : JD-agnostic candidate quality
                                     (skill · career · experience · platform_engagement)
  Layer 3 — jd_alignment_score     : Exclusively owns ALL JD-specific signals
                                     (location · title · product-co · retrieval-domain ·
                                      availability · ML-foundations)
  Final    — 0.75 × behavioral + 0.25 × jd_alignment  (run on all ~99,980 clean candidates)

No LLM calls.  Pure Python math.  <5s on 100 K candidates.
Designed for the Redrob AI Challenge (100 K candidate JSONL pool).
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from schemas.candidate import (
    StructuredCandidate,
    StructuredCandidateScore,
    RedrobSignals,
)

logger = logging.getLogger(__name__)

CURRENT_YEAR = 2026

# ─────────────────────────────────────────────────────────────────────────────
# JD Alignment Config  (JD-specific params, generalisable via JD-parser v2)
# ─────────────────────────────────────────────────────────────────────────────

# City-tier scores — based verbatim on JD preference language
# "Pune/Noida preferred · Delhi NCR / Mumbai / Hyderabad welcome · case-by-case otherwise"
LOCATION_TIERS: dict[str, float] = {
    "pune":      1.00, "noida":     1.00,
    "delhi":     0.88, "ncr":       0.88,
    "gurgaon":   0.88, "gurugram":  0.88,
    "new delhi": 0.88,
    "mumbai":    0.88, "hyderabad": 0.88,
    "bangalore": 0.72, "bengaluru": 0.72,
    "chennai":   0.72, "kolkata":   0.65,
}
OTHER_INDIA_SCORE   = 0.65
REMOTE_INDIA_SCORE  = 0.62   # India-based remote — deprioritised slightly vs on-site
REMOTE_UNKNOWN_SCORE = 0.30  # "Remote" with no India signal → treated as outside-India
OUTSIDE_INDIA_SCORE = 0.25

# Target titles (from JD)
TARGET_TITLES: list[str] = [
    "ai engineer", "machine learning engineer", "ml engineer",
    "search engineer", "nlp engineer", "applied scientist",
    "deep learning engineer", "retrieval engineer",
    "research scientist", "recommendation engineer",
]
ADJACENT_TITLES: list[str] = [
    "data scientist", "software engineer", "platform engineer",
    "backend engineer", "applied ml", "senior engineer",
]

# Retrieval/ranking domain keywords — scoped to career descriptions only
RETRIEVAL_KEYWORDS: list[str] = [
    "ranking model", "ranking system", "search relevance", "search ranking",
    "recommendation system", "recommender system", "retrieval system",
    "reranking", "reranker", "learning to rank", "query understanding",
    "click-through", "ctr optimisation", "ctr optimization",
    "ndcg", "mrr", "recall@", "precision@",
    "personalization", "relevance ranking", "semantic search system",
    "dense retrieval", "two-tower", "dual encoder",
    "surfacing", "item ranking", "candidate retrieval",
]

# ML foundations — not LLM-wrapper only
ML_FOUNDATION_KEYWORDS: list[str] = [
    "pytorch", "tensorflow", "jax",
    "fine-tuning", "fine tuning", "model training", "training loop",
    "gradient descent", "backpropagation",
    "scikit-learn", "sklearn", "xgboost", "gradient boosting",
    "neural network", "deep learning",
    "transformers", "bert", "t5", "llama",
    "lora", "qlora", "peft", "rlhf",
]

# LLM-framework-only signals (penalise if present without ML foundations)
FRAMEWORK_ONLY_KEYWORDS: list[str] = [
    "langchain", "llamaindex", "autogpt", "auto-gpt", "crewai",
]

# ─────────────────────────────────────────────────────────────────────────────
# Skill Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

CORE_AI_SKILLS: dict[str, list[str]] = {
    "embeddings": [
        "embedding", "sentence-transformer", "sentence transformer",
        "bge", "e5", "openai embedding", "vector embedding",
        "dense retrieval", "semantic search",
    ],
    "vector_db": [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "vector database", "vector store",
        "ann", "hnsw",
    ],
    "retrieval": [
        "retrieval", "rag", "retrieval augmented", "bm25",
        "hybrid search", "information retrieval", "ir system",
        "ranking", "reranking", "reranker",
    ],
    "llm": [
        "llm", "large language model", "gpt", "bert", "transformer",
        "fine-tuning", "fine tuning", "lora", "qlora", "peft",
        "instruction tuning",
    ],
    "nlp": [
        "nlp", "natural language processing", "text classification",
        "named entity", "ner", "sentiment", "text mining", "language model",
    ],
    "ml_core": [
        "machine learning", "deep learning", "neural network",
        "pytorch", "tensorflow", "scikit-learn", "sklearn",
        "xgboost", "gradient boosting",
    ],
    "mlops": [
        "mlops", "ml pipeline", "model deployment", "model serving",
        "inference", "bentoml", "mlflow", "wandb", "weights & biases",
        "kubeflow",
    ],
    "eval": [
        "ndcg", "mrr", "map", "a/b test", "a/b testing",
        "evaluation framework", "offline evaluation",
        "online evaluation", "ranking evaluation",
    ],
    "python_eng": [
        "python", "fastapi", "flask", "django", "rest api", "microservice",
    ],
    "data": [
        "sql", "spark", "airflow", "dbt", "data pipeline",
        "kafka", "streaming", "pandas", "numpy",
    ],
}

NICE_TO_HAVE_SKILLS: dict[str, list[str]] = {
    "ltr": [
        "learning to rank", "lambdamart", "ranknet", "listnet", "xgboost rank",
    ],
    "open_source": ["open source", "github", "contributions", "maintainer"],
    "hr_tech": ["hr tech", "recruiting", "talent", "ats", "applicant tracking"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Firm Lists
# ─────────────────────────────────────────────────────────────────────────────

CONSULTING_FIRMS: list[str] = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "l&t infotech", "hexaware",
    "mindtree", "niit technologies", "virtusa", "syntel", "zensar",
    "mastech", "igate", "patni",
]

PRODUCT_COMPANIES: list[str] = [
    "google", "meta", "microsoft", "amazon", "apple", "netflix",
    "uber", "linkedin", "airbnb", "flipkart", "zomato", "swiggy",
    "phonepe", "razorpay", "zerodha", "cred", "freshworks",
    "browserstack", "chargebee", "postman", "groww", "meesho",
    "paytm", "ola", "byju", "unacademy", "sharechat", "daily hunt",
    "slice", "jupiter", "niyo", "open financial", "setu",
    "hasura", "dgraph", "atlan", "rubrik", "druva",
    "walmart", "target", "adobe", "salesforce", "oracle",
    "servicenow", "workday", "atlassian", "datadog", "elastic",
    "databricks", "snowflake", "confluent",
]

TITLE_MISMATCH_KEYWORDS: list[str] = [
    "marketing", "sales", "hr manager", "human resources",
    "finance", "accounting", "operations manager", "administrator",
    "admin", "recruiter", "procurement",
]

# ─────────────────────────────────────────────────────────────────────────────
# Date Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _days_since(date_str: str) -> int:
    """Return days elapsed since a date string (ISO-8601 or YYYY-MM-DD)."""
    try:
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return 9999
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _years_since(date_str: str) -> float:
    """Return fractional years elapsed since a date string."""
    return _days_since(date_str) / 365.25


def _log_scale(value: float, max_val: float = 500.0) -> float:
    """Logarithmic 0–1 scale."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(max_val), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot Detection — v2 (12 checks, 2 tiers)
# ─────────────────────────────────────────────────────────────────────────────

def _is_consulting_firm(company_name: str) -> bool:
    cn = company_name.lower().strip()
    return any(firm in cn for firm in CONSULTING_FIRMS)


def _count_expert_skills(candidate: StructuredCandidate) -> int:
    return sum(1 for sk in candidate.skills if sk.proficiency == "expert")


def _total_career_months(candidate: StructuredCandidate) -> int:
    return sum(e.duration_months for e in candidate.career_history)


def _earliest_career_year(candidate: StructuredCandidate) -> Optional[int]:
    years = []
    for e in candidate.career_history:
        try:
            yr = int(str(e.start_date)[:4])
            if 1980 <= yr <= CURRENT_YEAR:
                years.append(yr)
        except (ValueError, TypeError, AttributeError):
            pass
    return min(years) if years else None


# Hard signals — 1 hit → honeypot (100% precision, mathematically impossible for real candidates)
def _chk_a_salary_inversion(c: StructuredCandidate) -> bool:
    """salary min > max is logically impossible."""
    try:
        sal = c.redrob_signals.expected_salary_range_inr_lpa
        if sal is None:
            return False
        sal_min = sal.get("min", 0) if isinstance(sal, dict) else 0
        sal_max = sal.get("max", float("inf")) if isinstance(sal, dict) else float("inf")
        return sal_min > 0 and sal_max > 0 and sal_min > sal_max
    except Exception:
        return False


def _chk_b_github_inactive_paradox(c: StructuredCandidate) -> bool:
    """GitHub activity ≥ 80 but platform inactive for 2+ years — impossible combo."""
    try:
        gas = c.redrob_signals.github_activity_score
        if gas is None or gas < 80:
            return False
        return _years_since(c.redrob_signals.last_active_date) > 2
    except Exception:
        return False


HARD_SIGNAL_CHECKS = {
    "CHK_A_salary_inversion":        _chk_a_salary_inversion,
    "CHK_B_github_inactive_paradox": _chk_b_github_inactive_paradox,
}


# Soft signals — need 2+ independent hits → honeypot
def _chk_1_skill_duration_impossible(c: StructuredCandidate) -> bool:
    """Skill duration_months > career length + buffer."""
    cap = c.years_of_experience * 12 + 24
    return any(
        sk.duration_months > 0 and sk.duration_months > cap
        for sk in c.skills
    )


def _chk_2_career_yoe_mismatch(c: StructuredCandidate) -> bool:
    """Total documented career months >> declared YoE (20% slack + 24m)."""
    total = _total_career_months(c)
    declared = c.years_of_experience * 12
    return total > declared * 1.2 + 24


def _chk_3_expert_zero_endorsements(c: StructuredCandidate) -> bool:
    """≥3 skills listed as expert with 0 endorsements AND 0 duration → fabricated."""
    count = sum(
        1 for sk in c.skills
        if sk.proficiency == "expert"
        and sk.endorsements == 0
        and sk.duration_months == 0
    )
    return count >= 3


def _chk_4_education_date_inversion(c: StructuredCandidate) -> bool:
    """Education end_year < start_year — impossible."""
    return any(
        edu.end_year < edu.start_year
        for edu in c.education
        if edu.end_year and edu.start_year
    )


def _chk_6_yoe_exceeds_career_start(c: StructuredCandidate) -> bool:
    """Declared YoE is impossible given earliest documented career start year."""
    earliest = _earliest_career_year(c)
    if earliest is None:
        return False
    max_possible_yoe = (CURRENT_YEAR - earliest) + 2
    return c.years_of_experience > max_possible_yoe + 3


def _chk_c_profile_completeness_paradox(c: StructuredCandidate) -> bool:
    """100% complete profile + near-zero network activity is paradoxical."""
    s = c.redrob_signals
    return (
        s.profile_completeness_score == 100
        and s.connection_count < 5
        and s.endorsements_received == 0
    )


def _chk_d_assessment_zero_endorsement(c: StructuredCandidate) -> bool:
    """Perfect assessment score with zero endorsements anywhere."""
    s = c.redrob_signals
    if not s.skill_assessment_scores or s.endorsements_received > 0:
        return False
    return any(v >= 100 for v in s.skill_assessment_scores.values())


def _chk_e_career_gap_impossibility(c: StructuredCandidate) -> bool:
    """Documented career months < half of declared YoE — major fabrication gap."""
    if c.years_of_experience <= 0:
        return False
    return _total_career_months(c) < c.years_of_experience * 6


def _chk_f_new_signup_high_yoe(c: StructuredCandidate) -> bool:
    """Joined platform < 30 days ago but claims 5+ years experience — suspicious."""
    try:
        days = _days_since(c.redrob_signals.signup_date)
        return days < 30 and c.years_of_experience > 5
    except Exception:
        return False


def _chk_g_expert_no_assessments(c: StructuredCandidate) -> bool:
    """≥5 expert skills but zero platform assessments taken."""
    s = c.redrob_signals
    return (
        _count_expert_skills(c) >= 5
        and not s.skill_assessment_scores  # empty dict or None
    )


SOFT_SIGNAL_CHECKS = {
    "CHK_1_skill_duration_impossible":  _chk_1_skill_duration_impossible,
    "CHK_2_career_yoe_mismatch":        _chk_2_career_yoe_mismatch,
    "CHK_3_expert_zero_endorsements":   _chk_3_expert_zero_endorsements,
    "CHK_4_education_date_inversion":   _chk_4_education_date_inversion,
    "CHK_6_yoe_exceeds_career_start":   _chk_6_yoe_exceeds_career_start,
    "CHK_C_profile_completeness_paradox": _chk_c_profile_completeness_paradox,
    "CHK_D_assessment_zero_endorsement":  _chk_d_assessment_zero_endorsement,
    "CHK_E_career_gap_impossibility":     _chk_e_career_gap_impossibility,
    "CHK_F_new_signup_high_yoe":          _chk_f_new_signup_high_yoe,
    "CHK_G_expert_no_assessments":        _chk_g_expert_no_assessments,
}


def detect_honeypot(candidate: StructuredCandidate) -> tuple[bool, list[str]]:
    """
    Tiered honeypot detection — 12 checks, 2 tiers.

    Tier 1 (HARD): 1 hit on any hard signal → honeypot.  100% precision.
    Tier 2 (SOFT): 2+ soft hits → honeypot.  CHK_F+G combination is guarded
                   (they naturally co-occur for legit new senior users).

    Returns:
        (is_honeypot: bool, reasons: list[str])
    """
    reasons: list[str] = []

    # --- HARD tier ---
    for name, check in HARD_SIGNAL_CHECKS.items():
        try:
            if check(candidate):
                reasons.append(name)
        except Exception as exc:
            logger.debug(f"Honeypot check {name} failed for {candidate.candidate_id}: {exc}")

    if reasons:
        return True, reasons

    # --- SOFT tier ---
    triggered: list[str] = []
    for name, check in SOFT_SIGNAL_CHECKS.items():
        try:
            if check(candidate):
                triggered.append(name)
        except Exception as exc:
            logger.debug(f"Honeypot check {name} failed for {candidate.candidate_id}: {exc}")

    # CHK_F+G guard: both naturally co-occur for legitimate new senior users who haven't
    # had time to take assessments yet.  Only suppress when they are the ONLY two triggers.
    if set(triggered) == {"CHK_F_new_signup_high_yoe", "CHK_G_expert_no_assessments"}:
        return False, []

    if len(triggered) >= 2:
        return True, triggered

    return False, triggered  # 0 or 1 soft hits → clean (borderline noted but not flagged)


# ─────────────────────────────────────────────────────────────────────────────
# Negative Signal Detection
# ─────────────────────────────────────────────────────────────────────────────

VISION_ONLY_KEYWORDS: list[str] = [
    "computer vision", "image recognition", "object detection",
    "speech recognition", "asr", "robotics", "ros",
]
NLP_RETRIEVAL_KEYWORDS: list[str] = [
    "nlp", "natural language", "text", "retrieval", "search",
    "ranking", "embedding", "language model",
]


def detect_negative_signals(candidate: StructuredCandidate) -> list[str]:
    signals: list[str] = []
    all_text = " ".join([
        candidate.summary.lower(),
        " ".join(sk.name.lower() for sk in candidate.skills),
        " ".join(e.description.lower() for e in candidate.career_history),
    ])
    skill_names_lower = [sk.name.lower() for sk in candidate.skills]

    # All roles at consulting firms
    if candidate.career_history and all(
        _is_consulting_firm(e.company) for e in candidate.career_history
    ):
        signals.append("Consulting-only background: all roles at outsourcing/services firms")

    # Title mismatch
    title_lower = candidate.current_title.lower()
    if any(kw in title_lower for kw in TITLE_MISMATCH_KEYWORDS):
        signals.append(f"Title mismatch: '{candidate.current_title}' is not an AI/ML role")

    # CV/vision-only — no NLP/IR background
    if (any(kw in all_text for kw in VISION_ONLY_KEYWORDS)
            and not any(kw in all_text for kw in NLP_RETRIEVAL_KEYWORDS)):
        signals.append("Background purely CV/speech/robotics — no NLP or IR signals")

    # Framework-only (LangChain without ML foundations)
    if (any(kw in all_text for kw in FRAMEWORK_ONLY_KEYWORDS)
            and not any(kw in all_text for kw in ML_FOUNDATION_KEYWORDS)):
        signals.append("Framework-only engineer: LangChain/LlamaIndex without ML foundations")

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Behavioral Composite Component Scorers
# (JD-agnostic — no location / notice / openness / title signals here)
# ─────────────────────────────────────────────────────────────────────────────

def score_ai_skills(candidate: StructuredCandidate) -> tuple[float, int]:
    """Score AI skill match against CORE_AI_SKILLS taxonomy. Returns (score, matched_core_count)."""
    blob = " ".join([
        candidate.summary.lower(),
        candidate.headline.lower(),
        " ".join(sk.name.lower() for sk in candidate.skills),
        " ".join(e.description.lower() for e in candidate.career_history),
    ])
    matched_core = sum(
        1 for _cat, kws in CORE_AI_SKILLS.items()
        if any(kw in blob for kw in kws)
    )
    matched_nice = sum(
        1 for _cat, kws in NICE_TO_HAVE_SKILLS.items()
        if any(kw in blob for kw in kws)
    )
    n_core = len(CORE_AI_SKILLS)
    n_nice = len(NICE_TO_HAVE_SKILLS)
    core_score = min(matched_core / n_core, 1.0)
    nice_bonus = (matched_nice / n_nice) if n_nice else 0.0
    return round(core_score * 0.80 + nice_bonus * 0.20, 4), matched_core


RELEVANT_TITLE_KEYWORDS: list[str] = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "research scientist", "nlp engineer",
    "applied scientist", "deep learning", "search engineer",
    "recommendation engineer", "ranking engineer",
]


def score_career_quality(candidate: StructuredCandidate) -> float:
    """Score career trajectory quality — product-company bias, title progression.
    Consulting-only careers score 0.  Mixed careers are NOT penalised."""
    if not candidate.career_history:
        return 0.2

    companies_lower = [e.company.lower() for e in candidate.career_history]
    industries_lower = [e.industry.lower() for e in candidate.career_history]

    # Hard zero for consulting-only career
    if all(_is_consulting_firm(e.company) for e in candidate.career_history):
        return 0.0

    score = 0.0

    # At least one non-consulting role
    non_consulting = [e for e in candidate.career_history if not _is_consulting_firm(e.company)]
    if non_consulting:
        score += 0.30

    # Known product company
    if any(pc in comp for comp in companies_lower for pc in PRODUCT_COMPANIES):
        score += 0.20
    else:
        ai_signal = any(
            kw in comp or kw in ind
            for comp in companies_lower
            for ind in industries_lower
            for kw in ("ai", "ml", "data", "analytics", "search", "tech", "product")
        )
        if ai_signal:
            score += 0.10

    # Title progression (senior/lead in recent roles, junior in earlier roles)
    titles = [e.title.lower() for e in candidate.career_history]
    has_senior_recent = any(
        ("senior" in t or "lead" in t or "principal" in t or "staff" in t)
        for t in titles[:2]
    )
    has_junior_earlier = any(
        ("junior" in t or "associate" in t or "intern" in t)
        for t in titles[-2:]
    )
    if has_senior_recent and has_junior_earlier:
        score += 0.10

    # Current title relevance
    title_lower = candidate.current_title.lower()
    if any(kw in title_lower for kw in RELEVANT_TITLE_KEYWORDS):
        score += 0.30
    elif "engineer" in title_lower or "scientist" in title_lower:
        score += 0.15
    elif "analyst" in title_lower or "developer" in title_lower:
        score += 0.10

    return round(min(score, 1.0), 4)


def score_experience(candidate: StructuredCandidate) -> float:
    """JD sweet spot: 5–9 years (6–8 best). <2 disqualifying, >12 overqualified."""
    yoe = candidate.years_of_experience
    if yoe < 2:   return 0.0
    if yoe < 4:   return 0.30
    if yoe < 5:   return 0.60
    if yoe <= 9:  return 1.00
    if yoe <= 12: return 0.80
    return 0.60


CS_ML_FIELDS: list[str] = [
    "computer science", "software engineering", "artificial intelligence",
    "machine learning", "data science", "statistics", "mathematics",
    "information technology", "electrical engineering", "electronics",
    "computational", "information systems",
]
HIGHER_DEGREES: list[str] = ["phd", "ph.d", "doctorate", "master", "m.s.", "m.tech", "msc"]


def score_education(candidate: StructuredCandidate) -> float:
    """Score by institution tier × field relevance. Higher degrees get a bonus."""
    if not candidate.education:
        return 0.20
    best = 0.0
    for edu in candidate.education:
        tier    = (edu.tier or "").lower()
        field   = (edu.field_of_study or "").lower()
        degree  = (edu.degree or "").lower()
        is_cs   = any(kw in field for kw in CS_ML_FIELDS)
        if is_cs:
            base = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60, "tier_4": 0.40}.get(tier, 0.50)
        else:
            base = 0.50 if tier == "tier_1" else 0.30
        if any(hd in degree for hd in HIGHER_DEGREES):
            base = min(base + 0.10, 1.0)
        best = max(best, base)
    return round(best, 4)


def score_platform_engagement(signals: RedrobSignals) -> float:
    """
    Pure platform-engagement quality score — JD-agnostic.

    Does NOT include: open_to_work_flag, notice_period_days, recruiter_response_rate.
    Those three are exclusively owned by jd_alignment_score's availability factor.

    Components:
      ENGAGEMENT  (0.40): interview completion · offer acceptance · response time
      QUALITY     (0.35): profile completeness · verifications · GitHub · LinkedIn
      ACTIVITY    (0.25): search appearances · saves · views · assessments
    """
    # ── ENGAGEMENT ────────────────────────────────────────────────────────────
    interview_compl = max(0.0, min(signals.interview_completion_rate, 1.0))
    oar = signals.offer_acceptance_rate
    offer_acc = oar if oar >= 0 else 0.5  # -1 = no history → neutral

    rt_hrs = signals.avg_response_time_hours
    if rt_hrs < 24:    rt_score = 1.0
    elif rt_hrs < 48:  rt_score = 0.80
    elif rt_hrs < 72:  rt_score = 0.60
    elif rt_hrs < 168: rt_score = 0.40
    else:              rt_score = 0.20

    engagement = (interview_compl * 0.50 + offer_acc * 0.30 + rt_score * 0.20)

    # ── QUALITY ───────────────────────────────────────────────────────────────
    completeness = max(0.0, min(signals.profile_completeness_score / 100.0, 1.0))
    verification = (
        (0.50 if signals.verified_email else 0.0)
        + (0.50 if signals.verified_phone else 0.0)
    )
    linkedin_bonus = 0.10 if signals.linkedin_connected else 0.0

    gas = signals.github_activity_score
    github_score = (gas / 100.0) if gas >= 0 else 0.20  # -1 = no GitHub → slight negative

    quality = (
        completeness  * 0.35
        + verification  * 0.30
        + github_score  * 0.25
        + min(linkedin_bonus, 0.10)
    )
    quality = min(quality, 1.0)

    # ── ACTIVITY ─────────────────────────────────────────────────────────────
    search_score = _log_scale(signals.search_appearance_30d, max_val=500)
    saved_score  = min(signals.saved_by_recruiters_30d / 10.0, 1.0)
    views_score  = min(signals.profile_views_received_30d / 50.0, 1.0)

    ass_scores = signals.skill_assessment_scores
    if ass_scores:
        assessment_score = min(sum(ass_scores.values()) / (len(ass_scores) * 100.0), 1.0)
    else:
        assessment_score = 0.30  # no assessments → slightly below neutral

    activity = (
        search_score      * 0.30
        + saved_score     * 0.30
        + views_score     * 0.20
        + assessment_score * 0.20
    )

    # ── WEIGHTED COMBINATION ──────────────────────────────────────────────────
    return round(max(0.0, min(
        engagement * 0.40 + quality * 0.35 + activity * 0.25,
        1.0
    )), 4)


def behavioral_composite(
    skill_score: float,
    career_score: float,
    exp_score: float,
    edu_score: float,
    platform_score: float,
    consulting_multiplier: float = 1.0,
) -> float:
    """
    Layer 2 — JD-agnostic candidate quality composite.
    Weights: skill 40% · career 30% · experience 20% · edu 5% · engagement 5%.
    Education deliberately low — this is an engineering role, not academia.
    """
    raw = (
        skill_score    * 0.40
        + career_score * 0.30
        + exp_score    * 0.20
        + edu_score    * 0.05
        + platform_score * 0.05
    )
    return round(max(0.0, min(raw * consulting_multiplier, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — JD Alignment Score
# Exclusively owns ALL JD-specific signals.  No overlap with behavioral_composite.
# ─────────────────────────────────────────────────────────────────────────────

def score_location_tiered(location_str: str) -> float:
    """City-tiered location score based on JD preference language."""
    if not location_str:
        return OUTSIDE_INDIA_SCORE
    loc = location_str.lower()

    # Handle remote first
    if "remote" in loc:
        if "india" in loc:
            return REMOTE_INDIA_SCORE
        return REMOTE_UNKNOWN_SCORE

    # City tiers
    for key, score in LOCATION_TIERS.items():
        if key in loc:
            return score

    # Generic India
    if "india" in loc:
        return OTHER_INDIA_SCORE

    return OUTSIDE_INDIA_SCORE


def score_title_match(current_title: str) -> float:
    """How closely the candidate's title matches the JD's target role."""
    t = current_title.lower()
    if any(tgt in t for tgt in TARGET_TITLES):
        return 1.0
    if any(adj in t for adj in ADJACENT_TITLES):
        return 0.55
    if "engineer" in t or "scientist" in t:
        return 0.35
    return 0.10


def product_company_aware_penalty(career_history) -> float:
    """
    0.0 = no penalty (has product company experience).
    0.5 = partial penalty (mixed career — consulting dominant but some product).
    1.0 = full penalty (pure consulting — should only occur via career_quality=0 path).

    This function returns a 0–1 PENALTY, NOT a score.
    The jd_alignment factor subtracts from 1: (1 - penalty).
    """
    if not career_history:
        return 0.0
    total = len(career_history)
    consulting_count = sum(1 for e in career_history if _is_consulting_firm(e.company))

    if consulting_count == 0:
        return 0.0
    if consulting_count == total:
        return 1.0
    # Partial: fraction consulting, capped at 0.5
    return round(min(0.50, (consulting_count / total) * 0.6), 4)


def retrieval_domain_signal(career_history) -> float:
    """
    Score evidence of ranking/search/recommendation work.

    Scoped EXCLUSIVELY to career_history descriptions — not company names,
    not skill list, not headline. Prevents company-name false positives.
    Partial-credit scaling: 1 hit → 0.40, 3 hits → 0.70, 5+ → 1.0.
    """
    if not career_history:
        return 0.0
    # Only career descriptions — not company names or skill names
    text = " ".join(
        r.description.lower()
        for r in career_history
        if r.description
    )
    if not text:
        return 0.0
    hits = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in text)
    if hits == 0:
        return 0.0
    # Gradual scale: 0.25 base + 0.15 per hit, capped at 1.0
    # 1 hit → 0.40 · 2 → 0.55 · 3 → 0.70 · 5 → 1.00
    return round(min(0.25 + 0.15 * hits, 1.0), 4)


def availability_score(notice_period_days: int, open_to_work: bool) -> float:
    """
    Availability signal — exclusively owns notice period and open-to-work.
    JD: 'love ≤30d notice, can buy out 30d, 30+ still in scope but bar higher'.
    """
    if notice_period_days <= 15:   np_score = 1.00
    elif notice_period_days <= 30: np_score = 0.90
    elif notice_period_days <= 60: np_score = 0.70
    elif notice_period_days <= 90: np_score = 0.50
    else:                          np_score = 0.30

    otw_boost = 0.15 if open_to_work else 0.0
    return round(min(np_score * 0.85 + otw_boost, 1.0), 4)


def ml_foundations_signal(career_history, skills) -> float:
    """
    Distinguishes genuine ML engineers from LLM-framework wrappers.
    High score = has ML foundations.  Penalised if framework-only.
    """
    skill_blob = " ".join(sk.name.lower() for sk in skills)
    career_blob = " ".join(
        e.description.lower() for e in career_history if e.description
    )
    all_text = skill_blob + " " + career_blob

    has_foundations = any(kw in all_text for kw in ML_FOUNDATION_KEYWORDS)
    uses_frameworks = any(kw in all_text for kw in FRAMEWORK_ONLY_KEYWORDS)

    if has_foundations and not uses_frameworks:
        return 1.0   # Pure ML foundations — ideal
    if has_foundations and uses_frameworks:
        return 0.75  # ML + frameworks — good
    if uses_frameworks and not has_foundations:
        return 0.20  # Framework-only — JD red flag
    return 0.50      # No signal either way — neutral


def recruiter_responsiveness_signal(recruiter_response_rate: float) -> float:
    """
    Platform-side responsiveness — separate from availability_score.
    Kept in jd_alignment because JD explicitly says 'high response rate candidates preferred'.
    Not in behavioral_composite (platform_engagement) to avoid double-counting.
    """
    rr = max(0.0, min(recruiter_response_rate, 1.0))
    if rr >= 0.80:   return 1.00
    if rr >= 0.60:   return 0.80
    if rr >= 0.40:   return 0.55
    if rr >= 0.20:   return 0.30
    return 0.10


def jd_alignment_score(candidate: StructuredCandidate) -> float:
    """
    Layer 3 — Exclusively owns ALL JD-specific signals.

    Weights (sum = 1.00):
      location       : 0.25  (city-tiered, India-aware, remote-handled)
      availability   : 0.15  (notice_period + open_to_work — owned here only)
      title_match    : 0.15  (exact/adjacent/generic)
      product_company: 0.15  (career-history-aware, not just current role)
      retrieval_domain: 0.15 (scoped to career descriptions, partial credit)
      ml_foundations : 0.10  (ML vs framework-only signal)
      responsiveness : 0.05  (recruiter response rate — JD-explicit preference)
    """
    loc    = score_location_tiered(candidate.location)
    avail  = availability_score(
        candidate.redrob_signals.notice_period_days,
        candidate.redrob_signals.open_to_work_flag,
    )
    title  = score_title_match(candidate.current_title)
    prod   = 1.0 - product_company_aware_penalty(candidate.career_history)
    ret    = retrieval_domain_signal(candidate.career_history)
    ml     = ml_foundations_signal(candidate.career_history, candidate.skills)
    resp   = recruiter_responsiveness_signal(candidate.redrob_signals.recruiter_response_rate)

    score = (
        loc   * 0.25
        + avail * 0.15
        + title * 0.15
        + prod  * 0.15
        + ret   * 0.15
        + ml    * 0.10
        + resp  * 0.05
    )
    return round(max(0.0, min(score, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Final Score
# ─────────────────────────────────────────────────────────────────────────────

def final_composite_score(b_composite: float, jd_align: float) -> float:
    """0.75 × behavioral_composite + 0.25 × jd_alignment_score."""
    return round(max(0.0, min(0.75 * b_composite + 0.25 * jd_align, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Generator — data-driven, no LLM, per-candidate
# ─────────────────────────────────────────────────────────────────────────────

def _get_matched_skills(skills, n: int = 4) -> list[str]:
    """Return top expert/advanced AI skills matched to CORE_AI_SKILLS taxonomy."""
    matched = []
    all_core_kws = [kw for kws in CORE_AI_SKILLS.values() for kw in kws]
    for sk in skills:
        if sk.proficiency in ("expert", "advanced") and any(kw in sk.name.lower() for kw in all_core_kws):
            matched.append(sk.name)
    return matched[:n]


def _location_note(location: str, country: str) -> str:
    loc_score = score_location_tiered(location)
    if loc_score >= 0.85:
        return f"{location} [preferred city]"
    if loc_score >= 0.60:
        return f"{location} [India]"
    if "remote" in location.lower():
        return f"{location} [remote]"
    return f"{location} [outside preferred region]"


def generate_reasoning(
    candidate: StructuredCandidate,
    jd_skill_labels: Optional[list[str]] = None,
    n_jd_criteria: int = 10,
) -> str:
    """
    Build a per-candidate, data-driven reasoning string without any LLM call.

    Uses 6 real data fields: title/company, location, matched skills,
    career highlight, availability, and concerns.
    """
    jd_skill_labels = jd_skill_labels or list(CORE_AI_SKILLS.keys())
    n_jd_criteria   = len(jd_skill_labels)

    parts: list[str] = []

    # 1 — Identity
    company_size = candidate.profile.get("current_company_size", "") or ""
    size_note = f" ({company_size})" if company_size else ""
    parts.append(
        f"{candidate.current_title} at {candidate.current_company}{size_note}"
    )

    # 2 — Experience + location
    loc_note = _location_note(candidate.location, candidate.country)
    parts.append(f"{candidate.years_of_experience:.1f} yrs | {loc_note}")

    # 3 — Matched skills
    matched = _get_matched_skills(candidate.skills)
    blob = " ".join([
        candidate.summary.lower(), candidate.headline.lower(),
        " ".join(sk.name.lower() for sk in candidate.skills),
        " ".join(e.description.lower() for e in candidate.career_history),
    ])
    n_matched_core = sum(
        1 for _cat, kws in CORE_AI_SKILLS.items()
        if any(kw in blob for kw in kws)
    )
    if matched:
        skills_str = ", ".join(matched)
        parts.append(
            f"Skills: {skills_str} — {n_matched_core}/{n_jd_criteria} JD criteria matched"
        )
    else:
        parts.append(f"Skills: {n_matched_core}/{n_jd_criteria} JD criteria matched")

    # 4 — Career highlight (product company context)
    non_consulting = [e for e in candidate.career_history if not _is_consulting_firm(e.company)]
    if non_consulting:
        top_role = non_consulting[0]
        industry = (top_role.industry or candidate.current_industry or "").strip()
        industry_note = f" · {industry}" if industry else ""
        parts.append(f"Product background: {top_role.company}{industry_note}")

    # 5 — Retrieval domain evidence
    ret_score = retrieval_domain_signal(candidate.career_history)
    if ret_score >= 0.70:
        parts.append("Strong ranking/search/recommendation experience evidenced")
    elif ret_score >= 0.40:
        parts.append("Some retrieval/ranking domain exposure")

    # 6 — Availability
    otw    = candidate.redrob_signals.open_to_work_flag
    np_d   = candidate.redrob_signals.notice_period_days
    rr     = candidate.redrob_signals.recruiter_response_rate
    gh     = candidate.redrob_signals.github_activity_score
    otw_str = "Open to work" if otw else "Passive candidate"
    parts.append(
        f"Availability: {otw_str}, {np_d}d notice, {rr:.0%} response rate"
    )

    # 7 — GitHub
    if gh is not None and gh >= 0:
        gh_note = "active" if gh >= 60 else ("moderate" if gh >= 30 else "low")
        parts.append(f"GitHub: {gh_note} (score {gh})".encode('ascii', 'ignore').decode('ascii'))

    # ── Concerns ──────────────────────────────────────────────────────────────
    concerns: list[str] = []

    if np_d > 60:
        concerns.append(f"{np_d}-day notice period (above 60d threshold)")

    pen = product_company_aware_penalty(candidate.career_history)
    if pen >= 0.8:
        concerns.append("consulting-dominated career history")
    elif pen >= 0.4:
        concerns.append("mixed consulting/product background")

    loc_s = score_location_tiered(candidate.location)
    if loc_s < 0.30:
        concerns.append(f"located outside India ({candidate.location})")

    if gh is None or gh < 0:
        concerns.append("no GitHub profile linked")

    if ret_score == 0.0:
        concerns.append("no explicit ranking/search/recommendation work found in career history")

    ml_s = ml_foundations_signal(candidate.career_history, candidate.skills)
    if ml_s <= 0.25:
        concerns.append("possible framework-only engineer — limited ML foundations signal")

    # Assemble
    reasoning = ". ".join(parts) + "."
    if concerns:
        reasoning += " Concerns: " + "; ".join(concerns) + "."
    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Per-Candidate Scoring  (called by batch scorer)
# ─────────────────────────────────────────────────────────────────────────────

def score_structured_candidate(
    candidate: StructuredCandidate,
) -> StructuredCandidateScore:
    """
    Produce a complete StructuredCandidateScore for one candidate.

    Pipeline:
      1. Honeypot detection (return early if flagged)
      2. Component scores (skill · career · experience · education · engagement)
      3. behavioral_composite (Layer 2 — JD-agnostic)
      4. jd_alignment_score  (Layer 3 — JD-specific, no double-counting)
      5. final_composite = 0.75 × behavioral + 0.25 × jd_alignment
      6. Reasoning generation
    """
    # Step 1 — Honeypot
    is_hp, hp_reasons = detect_honeypot(candidate)
    negative_signals  = detect_negative_signals(candidate)

    if is_hp:
        return StructuredCandidateScore(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            headline=candidate.headline,
            current_title=candidate.current_title,
            current_company=candidate.current_company,
            years_of_experience=candidate.years_of_experience,
            location=candidate.location,
            country=candidate.country,
            is_honeypot=True,
            honeypot_reasons=hp_reasons,
            negative_signals=negative_signals,
            composite_score=0.0,
            reasoning=f"HONEYPOT DETECTED: {'; '.join(hp_reasons[:3])}",
            confidence="Low",
        )

    # Step 2 — Component scores
    skill_score, matched_core = score_ai_skills(candidate)
    career_score  = score_career_quality(candidate)
    exp_score     = score_experience(candidate)
    edu_score     = score_education(candidate)
    platform_score = score_platform_engagement(candidate.redrob_signals)

    # Consulting multiplier (pure consulting = 0.5×, mixed = 1.0×)
    all_consulting = bool(candidate.career_history) and all(
        _is_consulting_firm(e.company) for e in candidate.career_history
    )
    consulting_multiplier = 0.50 if all_consulting else 1.0
    consulting_penalty    = 0.50 if all_consulting else 0.0

    # Step 3 — Layer 2
    b_composite = behavioral_composite(
        skill_score, career_score, exp_score, edu_score, platform_score,
        consulting_multiplier=consulting_multiplier,
    )

    # Step 4 — Layer 3
    jd_align = jd_alignment_score(candidate)

    # Step 5 — Final
    composite = final_composite_score(b_composite, jd_align)

    # Confidence label
    if composite >= 0.65:   confidence = "High"
    elif composite >= 0.40: confidence = "Medium"
    else:                   confidence = "Low"

    # Step 6 — Reasoning
    reasoning = generate_reasoning(candidate)

    return StructuredCandidateScore(
        candidate_id=candidate.candidate_id,
        name=candidate.name,
        headline=candidate.headline,
        current_title=candidate.current_title,
        current_company=candidate.current_company,
        years_of_experience=candidate.years_of_experience,
        location=candidate.location,
        country=candidate.country,
        skill_match_score=skill_score,
        career_quality_score=career_score,
        experience_score=exp_score,
        education_score=edu_score,
        behavioral_score=platform_score,
        is_honeypot=False,
        honeypot_reasons=[],
        consulting_only_penalty=consulting_penalty,
        negative_signals=negative_signals,
        composite_score=composite,
        reasoning=reasoning,
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_all_structured_candidates(
    candidates: list[StructuredCandidate],
    progress_callback=None,
) -> list[StructuredCandidateScore]:
    """
    Score all candidates deterministically.  Runs on ALL candidates (no prefilter).
    Sorts: honeypots last, then by composite_score descending.  Assigns ranks.

    Args:
        candidates: All StructuredCandidate objects from the JSONL pool.
        progress_callback: Optional callable(current, total, name) for UI progress.

    Returns:
        Sorted, ranked list of StructuredCandidateScore objects.
    """
    scores: list[StructuredCandidateScore] = []
    total  = len(candidates)
    failed = 0

    logger.info(f"Behavioral scoring v2: {total} candidates")

    for i, candidate in enumerate(candidates):
        if progress_callback and (i % 500 == 0 or i == total - 1):
            progress_callback(i + 1, total, candidate.name)
        try:
            scores.append(score_structured_candidate(candidate))
        except Exception as exc:
            failed += 1
            logger.warning(f"Scoring failed for {candidate.candidate_id}: {exc}")

    if failed:
        logger.warning(f"{failed} candidates failed scoring and were skipped")

    # Sort: honeypots last, then by composite_score descending
    scores.sort(key=lambda s: (s.is_honeypot, -s.composite_score))

    # Assign sequential ranks
    for i, s in enumerate(scores):
        s.rank = i + 1

    n_honeypots = sum(1 for s in scores if s.is_honeypot)
    n_clean     = len(scores) - n_honeypots
    logger.info(
        f"Scoring complete | total={len(scores)} | honeypots={n_honeypots} | "
        f"clean={n_clean} | "
        + (f"top composite={scores[0].composite_score:.4f} ({scores[0].name})" if scores else "no scores")
    )
    return scores
