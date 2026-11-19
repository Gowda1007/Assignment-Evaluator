"""
feedback_generator.py — Generate human-readable feedback from evaluation results.

Covers:
    • Concepts the student covered (with match method)
    • Concepts the student missed
    • A constructive improvement suggestion
    • Faculty review flag when OCR or matching confidence is low
"""


def generate_feedback(evaluation: dict) -> dict:
    """Create feedback from an evaluation dict (returned by rubric_evaluator.evaluate).

    Returns:
        {
            "covered":      [{"concept": str, "method": str, "phrase": str}, ...],
            "missing":      [str, ...],
            "suggestion":   str,
            "needs_review": bool,
            "review_reason": str | None,
        }
    """
    covered = []
    missing = []

    for r in evaluation["concept_results"]:
        if r["matched"]:
            covered.append({
                "concept": r["concept"],
                "method": r["method"],
                "phrase": r["phrase"],
            })
        else:
            missing.append(r["concept"])

    # --- Suggestion text ---
    if not evaluation["relevance"]["is_relevant"]:
        suggestion = (
            "Your answer does not appear to address the question asked. "
            "Please re-read the question and focus on the specific topic: "
            f'"{evaluation["question"]}".'
        )
    elif not missing:
        suggestion = (
            "Excellent! Your answer covers all the expected concepts. "
            "Well done."
        )
    elif len(missing) <= 2:
        concepts_str = " and ".join(f'"{c}"' for c in missing)
        suggestion = (
            f"Good attempt. To strengthen your answer, also include the concept(s): "
            f"{concepts_str}."
        )
    else:
        concepts_str = ", ".join(f'"{c}"' for c in missing)
        suggestion = (
            f"Your answer partially addresses the question. "
            f"Consider adding these concepts for a more complete answer: {concepts_str}."
        )

    # --- Faculty review flag ---
    needs_review = False
    review_reason = None

    # Flag if any concept matched only via semantic (looser match)
    semantic_matches = [r for r in evaluation["concept_results"]
                        if r["matched"] and r["method"] == "semantic"]
    if semantic_matches:
        needs_review = True
        review_reason = (
            f"{len(semantic_matches)} concept(s) matched via semantic similarity only — "
            "faculty should verify these are correct."
        )

    # Flag borderline relevance
    rel = evaluation["relevance"]
    if rel["is_relevant"] and rel["tfidf_sim"] < 0.20:
        needs_review = True
        review_reason = (
            f"Low overall similarity ({rel['tfidf_sim']:.2f}) — "
            "marks awarded based on concept matching but faculty should review."
        )

    return {
        "covered": covered,
        "missing": missing,
        "suggestion": suggestion,
        "needs_review": needs_review,
        "review_reason": review_reason,
    }
