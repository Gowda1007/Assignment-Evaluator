"""
utils.py — Dataset loading and shared helpers.

Loads questions, rubric and student-answer CSVs once,
and exposes simple lookup functions the rest of the app calls.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_unzipped")
QUESTIONS_PATH = os.path.join(DATASET_DIR, "questions.csv")
RUBRIC_PATH = os.path.join(DATASET_DIR, "marking_rubric.csv")
ANSWERS_PATH = os.path.join(DATASET_DIR, "student_answers.csv")
IMAGES_DIR = os.path.join(DATASET_DIR, "answer_images")


# ---------------------------------------------------------------------------
# Loaders (cached at module level so CSVs are read only once)
# ---------------------------------------------------------------------------
_questions_df: pd.DataFrame | None = None
_rubric_df: pd.DataFrame | None = None
_answers_df: pd.DataFrame | None = None


def load_questions() -> pd.DataFrame:
    """Return the faculty questions table (8 rows)."""
    global _questions_df
    if _questions_df is None:
        _questions_df = pd.read_csv(QUESTIONS_PATH)
    return _questions_df


def load_rubric() -> pd.DataFrame:
    """Return the marking rubric table (40 rows, 5 per question)."""
    global _rubric_df
    if _rubric_df is None:
        _rubric_df = pd.read_csv(RUBRIC_PATH)
    return _rubric_df


def load_student_answers() -> pd.DataFrame:
    """Return the validation-only student answers (24 rows).

    ⚠  Never feed this into the evaluation pipeline.
       Use it only to measure OCR accuracy and marks accuracy.
    """
    global _answers_df
    if _answers_df is None:
        _answers_df = pd.read_csv(ANSWERS_PATH)
    return _answers_df


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def get_question_data(question_id: str) -> dict:
    """Return {question_id, subject, question, reference_answer, max_marks}
    for a given question ID like 'Q1'."""
    df = load_questions()
    row = df[df["question_id"] == question_id]
    if row.empty:
        raise ValueError(f"Question '{question_id}' not found in questions.csv")
    r = row.iloc[0]
    return {
        "question_id": r["question_id"],
        "subject": r["subject"],
        "question": r["question"],
        "reference_answer": r["reference_answer"],
        "max_marks": int(r["maximum_marks"]),
    }


def get_rubric_concepts(question_id: str) -> list[dict]:
    """Return a list of concept dicts for the question.

    Each dict:  {"concept_raw": str, "variants": [str, ...], "marks": int}
    Variants are created by splitting the raw concept on '/'.
    """
    df = load_rubric()
    rows = df[df["question_id"] == question_id]
    if rows.empty:
        raise ValueError(f"No rubric entries for '{question_id}'")

    concepts = []
    for _, r in rows.iterrows():
        raw = r["concept_or_expected_point"]
        variants = [v.strip() for v in raw.split("/")]
        concepts.append({
            "concept_raw": raw,
            "variants": variants,
            "marks": int(r["marks"]),
        })
    return concepts


def get_all_image_paths() -> list[str]:
    """Return sorted list of absolute paths to all answer-sheet PNGs."""
    files = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return sorted(files)
