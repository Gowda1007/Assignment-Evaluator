"""
validate.py — Run the full pipeline on all 24 images and report metrics.

Compares OCR output and awarded marks against the validation CSV.
Prints a summary table and computes MAE, accuracy, etc.

Usage:
    python validate.py
"""

import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_student_answers, get_all_image_paths
from ocr_engine import parse_answer_sheet
from rubric_evaluator import evaluate
from feedback_generator import generate_feedback


def run_validation():
    answers_df = load_student_answers()
    image_paths = get_all_image_paths()

    print("=" * 90)
    print("  INTELLIGENT STUDENT ASSIGNMENT EVALUATION — FULL VALIDATION")
    print("=" * 90)
    print(f"\n  Images found: {len(image_paths)}")
    print(f"  Validation rows: {len(answers_df)}\n")

    results = []
    total_time = 0

    for _, row in answers_df.iterrows():
        sid = row["student_id"]
        qid = row["question_id"]
        expected_text = row["student_answer_text_for_validation_only"]
        expected_marks = int(row["expected_marks_out_of_5"])
        expected_cat = row["expected_category"]

        # Find the matching image
        img_name = f"{sid}_{qid}.png"
        img_path = None
        for p in image_paths:
            if os.path.basename(p) == img_name:
                img_path = p
                break

        if img_path is None:
            print(f"  ⚠  Image not found for {sid} {qid}")
            continue

        # --- Run pipeline ---
        t0 = time.time()
        ocr_result = parse_answer_sheet(img_path)
        eval_result = evaluate(ocr_result["student_answer"], qid)
        feedback = generate_feedback(eval_result)
        elapsed = time.time() - t0
        total_time += elapsed

        # --- OCR text accuracy (simple word overlap) ---
        ocr_words = set(ocr_result["student_answer"].lower().split())
        expected_words = set(expected_text.lower().split())
        if expected_words:
            word_overlap = len(ocr_words & expected_words) / len(expected_words) * 100
        else:
            word_overlap = 0.0

        marks = eval_result["total_marks"]
        cat = eval_result["category"]

        results.append({
            "student_id": sid,
            "question_id": qid,
            "ocr_text": ocr_result["student_answer"][:60],
            "ocr_word_accuracy": word_overlap,
            "ocr_confidence": ocr_result["ocr_confidence"],
            "expected_marks": expected_marks,
            "awarded_marks": marks,
            "marks_error": abs(marks - expected_marks),
            "expected_category": expected_cat,
            "predicted_category": cat,
            "category_match": cat == expected_cat,
            "concepts_covered": sum(1 for c in eval_result["concept_results"] if c["matched"]),
            "concepts_total": len(eval_result["concept_results"]),
            "tfidf_sim": eval_result["tfidf_similarity"],
            "relevance": eval_result["relevance"]["is_relevant"],
            "needs_review": feedback["needs_review"],
            "time_s": elapsed,
        })

    # =====================================================================
    # Print detailed table
    # =====================================================================
    print("-" * 90)
    print(f"  {'Student':<10} {'Q':<4} {'OCR%':>6} {'Exp':>4} {'Got':>4} {'Err':>4} "
          f"{'Concepts':>10} {'TF-IDF':>7} {'Cat Match':>10} {'Review':>7} {'Time':>6}")
    print("-" * 90)

    for r in results:
        cat_match_str = "✅" if r["category_match"] else "❌"
        review_str = "⚠️" if r["needs_review"] else "—"
        print(
            f"  {r['student_id']:<10} {r['question_id']:<4} "
            f"{r['ocr_word_accuracy']:5.1f}% "
            f"{r['expected_marks']:>4} {r['awarded_marks']:>4} {r['marks_error']:>4} "
            f"  {r['concepts_covered']}/{r['concepts_total']:<6} "
            f"{r['tfidf_sim']:6.3f} "
            f"{cat_match_str:>10} {review_str:>7} "
            f"{r['time_s']:5.1f}s"
        )

    # =====================================================================
    # Aggregate metrics
    # =====================================================================
    n = len(results)
    if n == 0:
        print("\n  No results to summarise.")
        return

    avg_ocr = sum(r["ocr_word_accuracy"] for r in results) / n
    mae = sum(r["marks_error"] for r in results) / n
    cat_acc = sum(1 for r in results if r["category_match"]) / n * 100
    perfect = sum(1 for r in results if r["marks_error"] == 0) / n * 100
    within_one = sum(1 for r in results if r["marks_error"] <= 1) / n * 100
    reviews = sum(1 for r in results if r["needs_review"])

    print("\n" + "=" * 90)
    print("  SUMMARY METRICS")
    print("=" * 90)
    print(f"  Total images evaluated  : {n}")
    print(f"  Average OCR word accuracy : {avg_ocr:.1f}%")
    print(f"  Marks MAE               : {mae:.2f}")
    print(f"  Exact marks match       : {perfect:.1f}%")
    print(f"  Within ±1 mark          : {within_one:.1f}%")
    print(f"  Category accuracy       : {cat_acc:.1f}%")
    print(f"  Flagged for review      : {reviews}/{n}")
    print(f"  Total time              : {total_time:.1f}s")
    print(f"  Average time per image  : {total_time / n:.1f}s")
    print("=" * 90)


if __name__ == "__main__":
    run_validation()
