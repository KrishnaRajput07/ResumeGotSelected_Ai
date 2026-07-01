# ─────────────────────────────────────────────────────────────────────────────
# Section-Aware Resume Chunker
# ─────────────────────────────────────────────────────────────────────────────
# Strategy:
#   1. Detect section headers via regex (known heading vocabulary)
#   2. Split long experience sections by individual job entry
#      (employer + date pattern = atomic retrieval unit)
#   3. Fallback: sentence-boundary chunks with 20% overlap
#
# Why section-aware > fixed-size:
#   Fixed 500-char windows cut across job entries mid-sentence.
#   "The 2021 Infosys job" should be ONE retrieval unit, not split
#   across two chunks arbitrarily.
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging
from schemas.candidate import ResumeChunk

logger = logging.getLogger(__name__)

# ── Section Header Patterns ───────────────────────────────────────────────────
# Matches lines that are likely section headings in a resume.
# Case-insensitive. These become chunk boundaries.

SECTION_HEADERS = re.compile(
    r"^(?:"
    r"WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|"
    r"EDUCATION|ACADEMIC\s+BACKGROUND|QUALIFICATIONS|"
    r"SKILLS|TECHNICAL\s+SKILLS|CORE\s+COMPETENCIES|COMPETENCIES|"
    r"PROJECTS|PROJECT\s+EXPERIENCE|KEY\s+PROJECTS|"
    r"CERTIFICATIONS?|CERTIFICATES?|LICENSES?|"
    r"AWARDS?|ACHIEVEMENTS?|HONORS?|ACCOMPLISHMENTS?|"
    r"PUBLICATIONS?|RESEARCH|"
    r"SUMMARY|PROFILE|OBJECTIVE|ABOUT\s+ME|"
    r"LEADERSHIP|VOLUNTEERING?|EXTRACURRICULAR|"
    r"LANGUAGES?|INTERESTS?|HOBBIES|ACTIVITIES"
    r")(?:\s*[:.]?\s*)$",
    re.IGNORECASE | re.MULTILINE,
)

# ── Job Entry Sub-Chunk Pattern ────────────────────────────────────────────────
# Detects the start of an individual job entry within an experience section.
# Pattern: line with a 4-digit year (possibly with month) + some other context.
DATE_PATTERN = re.compile(
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}"
    r"|\d{1,2}[/\-]\d{4}"
    r"|\d{4}"
    r")"
    r"(?:\s*[-–—]\s*"
    r"(?:Present|Current|Now|Till\s+Date|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}"
    r"|\d{1,2}[/\-]\d{4}"
    r"|\d{4}"
    r"))?",
    re.IGNORECASE,
)

SECTION_TYPE_MAP = {
    "experience": ["experience", "employment", "work"],
    "education": ["education", "academic", "qualifications"],
    "skills": ["skills", "competencies", "technical"],
    "projects": ["projects", "project"],
    "certifications": ["certifications", "certificates", "licenses"],
    "awards": ["awards", "achievements", "honors", "accomplishments"],
    "summary": ["summary", "profile", "objective", "about"],
    "other": [],
}

# Max chars per chunk (fallback splitting threshold)
MAX_CHUNK_CHARS = 1200
# Overlap chars for fallback sentence-boundary chunking
OVERLAP_CHARS = 150


def chunk_resume(text: str, candidate_id: str) -> list[ResumeChunk]:
    """
    Split a resume's clean text into semantic chunks.
    
    Pipeline:
    1. Split into sections using SECTION_HEADERS regex
    2. For experience sections: sub-split by job entry (date detection)
    3. For all sections: apply max-length fallback if chunk still too large
    
    Args:
        text: Clean resume text (output of pdf_parser._clean_text)
        candidate_id: Unique candidate ID for chunk metadata
    
    Returns:
        List of ResumeChunk objects ready for embedding
    """
    if not text.strip():
        logger.warning(f"Empty text for candidate {candidate_id}")
        return []

    # Split text into sections
    sections = _split_into_sections(text)
    
    chunks = []
    chunk_index = 0

    for section_name, section_text in sections:
        section_type = _classify_section(section_name)
        
        if not section_text.strip():
            continue

        if section_type == "experience" and len(section_text) > MAX_CHUNK_CHARS:
            # Sub-split experience by job entry
            sub_chunks = _split_experience_by_entry(section_text)
        elif len(section_text) > MAX_CHUNK_CHARS:
            # Generic long section: sentence-boundary fallback
            sub_chunks = _sentence_boundary_split(section_text)
        else:
            sub_chunks = [section_text]

        for sub_text in sub_chunks:
            if not sub_text.strip():
                continue
            chunks.append(ResumeChunk(
                chunk_id=f"{candidate_id}_c{chunk_index:03d}",
                candidate_id=candidate_id,
                section_type=section_type,
                text=sub_text.strip(),
                source_page=0,        # Page tracking omitted for simplicity
                char_start=0,
                char_end=len(sub_text),
            ))
            chunk_index += 1

    # Fallback: if no sections detected, treat whole text as one chunk
    if not chunks:
        logger.warning(f"No sections detected for {candidate_id}. Using full text as single chunk.")
        chunks.append(ResumeChunk(
            chunk_id=f"{candidate_id}_c000",
            candidate_id=candidate_id,
            section_type="other",
            text=text[:MAX_CHUNK_CHARS * 3],  # Limit full-text fallback
            source_page=0,
        ))

    logger.debug(f"Chunked candidate {candidate_id}: {len(chunks)} chunks")
    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text on section headers. Returns list of (header_name, section_body) tuples.
    The preamble before the first detected header is labeled "summary".
    """
    lines = text.split("\n")
    sections = []
    current_header = "summary"
    current_lines = []

    for line in lines:
        if SECTION_HEADERS.match(line.strip()):
            # Save current section
            if current_lines:
                sections.append((current_header, "\n".join(current_lines)))
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections.append((current_header, "\n".join(current_lines)))

    return sections


def _classify_section(header: str) -> str:
    """Map a detected header string to a canonical section type."""
    header_lower = header.lower()
    for section_type, keywords in SECTION_TYPE_MAP.items():
        if any(kw in header_lower for kw in keywords):
            return section_type
    return "other"


def _split_experience_by_entry(text: str) -> list[str]:
    """
    Split an experience section into individual job entries.
    
    Heuristic: A new job entry starts when we find a DATE_PATTERN on a line
    that's likely a header (short line, or followed by a company/title line).
    """
    lines = text.split("\n")
    entries = []
    current_entry = []

    for line in lines:
        # Check if this line contains a date range (potential job entry start)
        if DATE_PATTERN.search(line) and len(line) < 120:
            if current_entry:
                entries.append("\n".join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)

    if current_entry:
        entries.append("\n".join(current_entry))

    # Apply max-length fallback to any entry that's still too long
    final_entries = []
    for entry in entries:
        if len(entry) > MAX_CHUNK_CHARS:
            final_entries.extend(_sentence_boundary_split(entry))
        else:
            final_entries.append(entry)

    return final_entries if final_entries else [text]


def _sentence_boundary_split(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text at sentence boundaries with overlap.
    Used as fallback when sections are too large to fit in one chunk.
    """
    # Simple sentence splitter: split on '. ' or '.\n'
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap (last OVERLAP_CHARS of previous)
            overlap = current_chunk[-OVERLAP_CHARS:] if current_chunk else ""
            current_chunk = overlap + " " + sentence if overlap else sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:max_chars]]
