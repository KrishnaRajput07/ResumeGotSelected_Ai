# ─────────────────────────────────────────────────────────────────────────────
# Pydantic v2 Schemas — Candidate data contracts
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Literal, Optional
import uuid


# ── Step 3: Resume Chunking ───────────────────────────────────────────────────

class ResumeChunk(BaseModel):
    """
    One atomic unit of a parsed resume. Section-aware chunking means each chunk
    maps to a logical block (one job entry, education record, skills section).
    This is what gets embedded and stored in FAISS.
    """
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    candidate_id: str
    section_type: Literal[
        "experience", "education", "skills", "projects",
        "certifications", "awards", "summary", "other"
    ]
    text: str
    source_page: int = 0
    char_start: int = 0
    char_end: int = 0


# ── Step 5: Evidence-Grounded Q&A ────────────────────────────────────────────

class CriterionScore(BaseModel):
    """
    LLM output for a single criterion Q&A call.
    evidence_quote MUST be a verbatim excerpt from the resume chunk text,
    or the literal string "NO_EVIDENCE_FOUND".
    """
    criterion_id: str
    criterion_text: str           # Denormalized for readability in output
    answer: str                   # 1-3 sentence reasoning from LLM
    evidence_quote: str           # Verbatim resume text or "NO_EVIDENCE_FOUND"
    confidence: float = Field(ge=0.0, le=1.0)
    score: int = Field(ge=0, le=10)
    evidence_verified: bool = True  # Set False if fuzzy match fails


# ── Step 6: Parallel Passes ───────────────────────────────────────────────────

class OpenSignal(BaseModel):
    """
    A standout achievement found outside the rubric criteria.
    These contribute bonus points, capped by bonus_sensitivity setting.
    """
    signal: str
    evidence_quote: str
    strength: float = Field(ge=0.0, le=1.0)


class RedFlag(BaseModel):
    """
    A plausibility concern detected from the resume timeline or structure.
    NEVER speculates on WHY — only states what is objectively present in the data.
    """
    type: Literal[
        "employment_gap",
        "title_inconsistency",
        "date_overlap",
        "short_tenure",
        "credential_mismatch"
    ]
    detail: str
    severity: Literal["info", "review", "concern"]


class GateResult(BaseModel):
    """Binary pass/fail on a single hard gate criterion."""
    criterion: str
    passed: bool
    evidence_quote: str
    failure_reason: Optional[str] = None


# ── Step 7-8: Final Candidate Record ─────────────────────────────────────────

class CandidateRecord(BaseModel):
    """
    Complete scored record for one candidate.
    This is the source of truth for ranking, display, and export.
    """
    candidate_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str
    file_path: str
    raw_text: str = ""            # Full resume text (for hard gate checks)
    embedding_score: float = 0.0  # Step 4: cosine similarity vs rubric profile

    # Step 6 outputs
    gate_results: list[GateResult] = Field(default_factory=list)
    gate_status: Literal["passed", "failed", "pending"] = "pending"
    gate_failure_reason: Optional[str] = None

    # Step 5 outputs
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    open_signals: list[OpenSignal] = Field(default_factory=list)
    red_flags: list[RedFlag] = Field(default_factory=list)

    # Step 7 outputs (scoring math)
    base_score: float = 0.0
    bonus_applied: float = 0.0
    final_score: float = 0.0
    confidence: Literal["High", "Medium", "Low"] = "Low"

    # Step 8: ranking
    rank: Optional[int] = None
    excluded: bool = False
    exclusion_reason: Optional[str] = None

    structured_score: Optional["StructuredCandidateScore"] = None  # Set when processing JSONL

    def get_top_evidence(self, n: int = 3) -> list[str]:
        """Return the top N evidence quotes by score for the summary view."""
        sorted_scores = sorted(
            self.criterion_scores,
            key=lambda x: x.score,
            reverse=True
        )
        quotes = []
        for cs in sorted_scores[:n]:
            if cs.evidence_quote != "NO_EVIDENCE_FOUND":
                quotes.append(f"[{cs.criterion_text}] {cs.evidence_quote[:120]}")
        return quotes


# ── Step 8: Run-Level Output ──────────────────────────────────────────────────

class RankedShortlist(BaseModel):
    """The complete output of one evaluation run."""
    run_id: str
    job_title: str
    total_resumes_submitted: int
    shortlisted_count: int         # After Step 4 embedding filter
    ranked_candidates: list[CandidateRecord]
    excluded_candidates: list[CandidateRecord]
    rubric_used: dict              # Serialized FrozenRubric for audit


# ── Structured JSON Candidate (Redrob Challenge) ──────────────────────────────

from typing import TYPE_CHECKING


class SkillEntry(BaseModel):
    """A single skill entry from the structured candidate profile."""
    name: str
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    endorsements: int = 0
    duration_months: int = 0


class CareerEntry(BaseModel):
    """A single career/work history entry from the structured candidate profile."""
    company: str
    title: str
    start_date: str  # ISO date string
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


class EducationEntry(BaseModel):
    """A single education entry from the structured candidate profile."""
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"  # tier_1 through tier_4 or unknown


class CertificationEntry(BaseModel):
    """A single certification entry from the structured candidate profile."""
    name: str
    issuer: str
    year: int


class RedrobSignals(BaseModel):
    """
    All 23 behavioral signals from the Redrob platform for a candidate.
    -1 values in github_activity_score and offer_acceptance_rate mean 'no data'.
    """
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: dict[str, float] = Field(default_factory=dict)
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float  # -1 = no GitHub linked
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float  # -1 = no offer history
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


class StructuredCandidate(BaseModel):
    """
    A candidate profile from the Redrob structured JSON dataset.
    Maps 1:1 to the candidate_schema.json format.
    """
    candidate_id: str
    profile: dict  # Raw profile dict — we access fields directly
    career_history: list[CareerEntry]
    education: list[EducationEntry]
    skills: list[SkillEntry]
    certifications: list[CertificationEntry] = Field(default_factory=list)
    languages: list[dict] = Field(default_factory=list)
    redrob_signals: RedrobSignals

    @property
    def name(self) -> str:
        return self.profile.get("anonymized_name", self.candidate_id)

    @property
    def headline(self) -> str:
        return self.profile.get("headline", "")

    @property
    def summary(self) -> str:
        return self.profile.get("summary", "")

    @property
    def years_of_experience(self) -> float:
        return float(self.profile.get("years_of_experience", 0))

    @property
    def current_title(self) -> str:
        return self.profile.get("current_title", "")

    @property
    def current_company(self) -> str:
        return self.profile.get("current_company", "")

    @property
    def current_industry(self) -> str:
        return self.profile.get("current_industry", "")

    @property
    def location(self) -> str:
        return self.profile.get("location", "")

    @property
    def country(self) -> str:
        return self.profile.get("country", "")


class StructuredCandidateScore(BaseModel):
    """
    Complete scoring breakdown for a StructuredCandidate.
    Stored alongside the CandidateRecord for display in dashboard.
    """
    candidate_id: str
    name: str
    headline: str
    current_title: str
    current_company: str
    years_of_experience: float
    location: str
    country: str

    # Component scores (all 0.0-1.0)
    skill_match_score: float = 0.0         # How well skills match JD AI requirements
    career_quality_score: float = 0.0      # Product company exp, title relevance
    experience_score: float = 0.0          # Years in sweet spot, progression
    education_score: float = 0.0           # Field relevance, institution tier
    behavioral_score: float = 0.0          # All 23 redrob signals combined

    # Penalty flags
    is_honeypot: bool = False              # Detected impossible profile
    honeypot_reasons: list[str] = Field(default_factory=list)
    consulting_only_penalty: float = 0.0   # 0.0-1.0 penalty multiplier
    negative_signals: list[str] = Field(default_factory=list)

    # Final
    composite_score: float = 0.0          # Weighted combination 0.0-1.0
    reasoning: str = ""                    # Human-readable 1-2 sentence explanation
    confidence: Literal["High", "Medium", "Low"] = "Low"

    # Ranking (assigned after batch scoring)
    rank: Optional[int] = None
