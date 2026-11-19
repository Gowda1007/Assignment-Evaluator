"""
ocr_engine.py — Extract text from student answer-sheet images.

Default backend: EasyOCR (spec-compliant).
Parses the structured header to pull out Student ID, Question ID,
the question text, and the student's answer.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import easyocr

# ---------------------------------------------------------------------------
# EasyOCR reader (lazy singleton — model loads once on first call)
# ---------------------------------------------------------------------------
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_raw_text(image_path: str) -> list[str]:
    """Run EasyOCR on an image and return a list of detected text lines."""
    reader = _get_reader()
    results = reader.readtext(image_path, detail=0, paragraph=False)
    return results


def parse_answer_sheet(image_path: str) -> dict:
    """Extract structured fields from a student answer-sheet image.

    Returns:
        {
            "student_id":    "MCA001",
            "question_id":   "Q1",
            "question_text": "What is Natural Language Processing?",
            "student_answer": "NLP is a branch of AI ...",
            "raw_lines":     [<all OCR lines>],
            "ocr_confidence": "high" | "medium" | "low",
        }
    """
    raw_lines = extract_raw_text(image_path)
    full_text = "\n".join(raw_lines)

    student_id = _parse_student_id(raw_lines, full_text)
    question_id = _parse_question_id(raw_lines, full_text)
    question_text, student_answer = _parse_question_and_answer(raw_lines)

    # Confidence estimate based on how many fields we could parse
    parsed_count = sum(1 for v in [student_id, question_id, question_text, student_answer] if v)
    confidence = "high" if parsed_count == 4 else ("medium" if parsed_count >= 2 else "low")

    return {
        "student_id": student_id,
        "question_id": question_id,
        "question_text": question_text,
        "student_answer": student_answer,
        "raw_lines": raw_lines,
        "ocr_confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------
def _normalise_ocr_id(text: str) -> str:
    """Fix common OCR confusions in IDs: O↔0, l↔1, I↔1."""
    text = text.strip()
    # Normalize standard MCA prefix
    if text.upper().startswith("MCA"):
        prefix = "MCA"
        rest = text[3:]
        # Convert any confusing letters to digits in the numeric part
        trans = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "i": "1", "Z": "2", "S": "5", "s": "5"})
        digits = rest.translate(trans)
        return prefix + digits
    return text.upper()


def _parse_student_id(lines: list[str], full_text: str) -> str:
    """Find 'Student ID: MCA001' in OCR output with fault-tolerant parsing."""
    # Try structured line-by-line match
    for line in lines:
        m = re.search(r"[Ss]tudent\s*[IiLl1][Dd]\s*[:\-]?\s*([A-Za-z0-9]{5,10})", line)
        if m:
            return _normalise_ocr_id(m.group(1))

    # Fallback: look for MCA prefix anywhere in text
    m = re.search(r"\b(MCA[A-Za-z0-9]{2,6})\b", full_text, re.IGNORECASE)
    if m:
        return _normalise_ocr_id(m.group(1))

    return ""


def _parse_question_id(lines: list[str], full_text: str) -> str:
    """Find 'Question: Q1' or 'Q1' in OCR output."""
    # Look for explicit "Question: Q1" or "Question Q1" or "Question: 1"
    for line in lines:
        m = re.search(r"[Qq]uestion\s*[:\-]?\s*[Qq]?(\d+)", line)
        if m:
            return f"Q{m.group(1)}"

    # Fallback: standalone Q followed by digit
    m = re.search(r"\bQ(\d+)\b", full_text)
    if m:
        return f"Q{m.group(1)}"

    return ""


def _parse_question_and_answer(lines: list[str]) -> tuple[str, str]:
    """Split OCR lines into the question text and the student answer.

    Strategy:
        - Find the line(s) after 'Question:' header → question text
        - Find the line(s) after 'Student Answer:' header → student answer
        - Skip boilerplate lines (title, 'End of answer', etc.)
    """
    SKIP_PATTERNS = [
        r"MCA\s+DEPARTMENT",
        r"STUDENT\s+ANSWER\s+SHEET",
        r"End\s+of\s+answer",
        r"Student\s+ID",
        r"Question\s*:\s*Q?\d",  # the "Question: Q1" header line
    ]

    question_text = ""
    student_answer = ""
    section = None  # None → 'question' → 'answer'

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip boilerplate
        if any(re.search(p, stripped, re.IGNORECASE) for p in SKIP_PATTERNS):
            # But check if "Question:" label appears — next non-skip line is the question
            if re.search(r"^Question\s*:?\s*$", stripped, re.IGNORECASE):
                section = "question"
            continue

        # Detect section headers
        if re.search(r"Student\s+Answer\s*:", stripped, re.IGNORECASE):
            section = "answer"
            # Check if answer is on the same line after the colon
            after = re.sub(r"Student\s+Answer\s*:\s*", "", stripped, flags=re.IGNORECASE).strip()
            if after:
                student_answer = after
            continue

        if re.search(r"^Question\s*:\s*$", stripped, re.IGNORECASE):
            section = "question"
            continue

        # Accumulate text in current section
        if section == "question":
            question_text = (question_text + " " + stripped).strip()
            # Once we have the question, next meaningful section is answer
        elif section == "answer":
            student_answer = (student_answer + " " + stripped).strip()
        else:
            # Before any section header — could be the question text if the
            # OCR missed the "Question:" label.  Heuristic: lines ending with
            # "?" are likely question text.
            if stripped.endswith("?"):
                question_text = stripped
                section = "question"

    # Post-process answer text for known OCR artifacts
    # e.g., 'Al' -> 'AI' (capital A with lowercase L), strip trailing colons
    student_answer = re.sub(r"\bAl\b", "AI", student_answer)
    student_answer = student_answer.strip(" \t\n\r:.-")

    return question_text, student_answer
