"""
rubric_evaluator.py — Concept matching, relevance gate, and marks calculation.

The grading pipeline:
    1.  For each rubric concept, try three matching layers:
            exact (word-boundary) → fuzzy → semantic
    2.  Apply a relevance gate: if the answer is off-topic, cap marks at 0.
    3.  Sum the marks from matched concepts.
"""

import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import nltk

# Ensure wordnet is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

stemmer = PorterStemmer()

from nlp_processor import compute_tfidf_similarity, compute_semantic_similarity, preprocess
from utils import get_question_data, get_rubric_concepts


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 82          # fuzzywuzzy partial_ratio
RELEVANCE_SIMILARITY = 0.28   # TF-IDF similarity threshold


def _get_synonyms(word: str) -> set[str]:
    """Retrieve synset lemmas for a word from WordNet."""
    syns = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            clean = lem.name().lower().replace("_", " ")
            syns.add(clean)
    return syns


# ---------------------------------------------------------------------------
# Single concept matching (Explainable 4-layer strategy)
# ---------------------------------------------------------------------------
def match_concept(concept: dict, answer_text: str) -> dict:
    """Check whether the student answer covers one rubric concept.

    Layers:
        1. Exact match (word-boundary regex)
        2. Stem / morphological match (PorterStemmer)
        3. Synonym match (NLTK WordNet synsets)
        4. Fuzzy partial match (fuzzywuzzy)
    """
    answer_lower = answer_text.lower()
    answer_words = re.findall(r"\b[a-z0-9]+\b", answer_lower)
    answer_stems = {stemmer.stem(w) for w in answer_words}

    for variant in concept["variants"]:
        variant_lower = variant.strip().lower()
        if not variant_lower:
            continue

        # --- Layer 1: exact word-boundary match ---
        pattern = r"\b" + re.escape(variant_lower) + r"\b"
        m = re.search(pattern, answer_lower)
        if m:
            return _hit(concept, "exact", variant, f'"{m.group()}"')

    # --- Layer 2: Stemmed / Morphological match ---
    # Check if all content words of the concept variant match stems in answer
    for variant in concept["variants"]:
        variant_lower = variant.strip().lower()
        v_words = [w for w in re.findall(r"\b[a-z0-9]+\b", variant_lower) if len(w) > 2]
        if v_words:
            v_stems = [stemmer.stem(w) for w in v_words]
            if all(s in answer_stems for s in v_stems):
                return _hit(concept, "stemmed", variant, f"stems={v_stems}")

    # --- Layer 3: Synonym matching via WordNet ---
    # e.g., 'splitting' in rubric vs 'divides' in answer
    for variant in concept["variants"]:
        variant_lower = variant.strip().lower()
        v_words = [w for w in re.findall(r"\b[a-z0-9]+\b", variant_lower) if len(w) > 2]
        for vw in v_words:
            syns = _get_synonyms(vw)
            overlap = syns & set(answer_words)
            if overlap:
                # If a key synonym is present and context matches
                if len(v_words) == 1 or any(stemmer.stem(w) in answer_stems for w in v_words if w != vw):
                    return _hit(concept, "synonym", variant, f"synonym '{list(overlap)[0]}' for '{vw}'")

    # --- Layer 4: Fuzzy partial match ---
    for variant in concept["variants"]:
        variant_lower = variant.strip().lower()
        if len(variant_lower) < 4:
            continue
        # Check partial ratio against answer
        score = fuzz.partial_ratio(variant_lower, answer_lower)
        if score >= FUZZY_THRESHOLD:
            return _hit(concept, "fuzzy", variant, f"fuzzy={score}%")

    return _miss(concept)


def _hit(concept, method, variant, phrase):
    return {
        "concept": concept["concept_raw"],
        "matched": True,
        "method": method,
        "variant": variant,
        "phrase": phrase,
        "marks": concept["marks"],
    }


def _miss(concept):
    return {
        "concept": concept["concept_raw"],
        "matched": False,
        "method": None,
        "variant": None,
        "phrase": None,
        "marks": 0,
    }


# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------
def check_relevance(
    student_answer: str,
    reference_answer: str,
    question_text: str,
    matched_results: list[dict],
) -> dict:
    """Decide whether the answer actually addresses the question.

    Returns:
        {
            "is_relevant":    bool,
            "reason":         str,
            "tfidf_sim":      float,
            "topic_overlap":  int,     # how many question keywords appear
            "concepts_hit":   int,
        }
    """
    tfidf_sim = compute_tfidf_similarity(student_answer, reference_answer)
    concepts_hit = sum(1 for r in matched_results if r["matched"])

    # Check if the answer mentions any topic words from the question
    q_prep = preprocess(question_text)
    a_lower = student_answer.lower()
    topic_overlap = sum(
        1 for kw in q_prep["keywords"]
        if re.search(r"\b" + re.escape(kw) + r"\b", a_lower)
    )

    # --- Relevance Gate Logic ---
    # Off-topic if:
    # 1. Zero concepts matched
    # 2. At most 1 concept matched AND tfidf_sim is low (< 0.28) AND zero question topic keywords
    # 3. Overall similarity is near zero (< 0.10) with at most 1 concept hit
    is_relevant = True
    reason = "Answer addresses the question."

    if concepts_hit == 0:
        is_relevant = False
        reason = f"Answer does not address expected rubric concepts (0 concepts matched, TF-IDF {tfidf_sim:.2f})."
    elif concepts_hit <= 1 and tfidf_sim < 0.28 and topic_overlap == 0:
        is_relevant = False
        reason = (
            f"Answer appears off-topic: question topic not addressed, "
            f"only {concepts_hit} concept matched peripherally, and low similarity ({tfidf_sim:.2f})."
        )
    elif tfidf_sim < 0.10 and concepts_hit <= 1:
        is_relevant = False
        reason = f"Answer has very low similarity ({tfidf_sim:.2f}) and minimal concept coverage ({concepts_hit})."

    return {
        "is_relevant": is_relevant,
        "reason": reason,
        "tfidf_sim": tfidf_sim,
        "topic_overlap": topic_overlap,
        "concepts_hit": concepts_hit,
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate(student_answer: str, question_id: str) -> dict:
    """Run the complete rubric-based evaluation.

    Returns:
        {
            "question_id":       str,
            "question":          str,
            "subject":           str,
            "reference_answer":  str,
            "max_marks":         int,
            "student_answer":    str,
            "preprocessing":     dict,       # from nlp_processor.preprocess
            "concept_results":   [dict, ...], # per-concept match results
            "relevance":         dict,        # from check_relevance
            "tfidf_similarity":  float,
            "total_marks":       int,
            "category":          str,         # "Relevant" / "Partially relevant" / "Irrelevant"
        }
    """
    # Look up question data and rubric
    q_data = get_question_data(question_id)
    concepts = get_rubric_concepts(question_id)

    # Preprocess student answer (for display / debugging)
    prep = preprocess(student_answer)

    # Match each rubric concept
    concept_results = [match_concept(c, student_answer) for c in concepts]

    # Relevance gate
    relevance = check_relevance(
        student_answer,
        q_data["reference_answer"],
        q_data["question"],
        concept_results,
    )

    # Compute TF-IDF similarity (shown as supporting info)
    tfidf_sim = relevance["tfidf_sim"]

    # Total marks (capped at 0 if irrelevant)
    if relevance["is_relevant"]:
        total_marks = sum(r["marks"] for r in concept_results)
    else:
        total_marks = 0

    # Category label
    if total_marks == 0:
        category = "Irrelevant"
    elif total_marks >= 4:
        category = "Relevant"
    else:
        category = "Partially relevant"

    return {
        "question_id": question_id,
        "question": q_data["question"],
        "subject": q_data["subject"],
        "reference_answer": q_data["reference_answer"],
        "max_marks": q_data["max_marks"],
        "student_answer": student_answer,
        "preprocessing": prep,
        "concept_results": concept_results,
        "relevance": relevance,
        "tfidf_similarity": tfidf_sim,
        "total_marks": min(total_marks, q_data["max_marks"]),
        "category": category,
    }
