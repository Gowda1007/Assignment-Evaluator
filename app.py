"""
app.py — Intelligent Student Assignment Evaluation Assistant
Streamlit Web Application featuring:
  • Image & PDF Upload Support (renders PDF pages directly via pypdfium2)
  • Clean, modern, humanized UI layout (cards, badges, metric pill indicators)
  • 4-Layer Explainable Concept Matching (Exact, Stemmed, WordNet Synonym, Fuzzy)
  • Relevance Gate protection against deceptive off-topic answers
  • Interactive Faculty Review Panel (Editable marks, comments, CSV export/download)
  • Full 24-sample benchmark analytics tab
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import streamlit as st
import pandas as pd
from PIL import Image

try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from utils import (
    load_questions,
    load_rubric,
    load_student_answers,
    get_question_data,
    get_rubric_concepts,
    get_all_image_paths,
)
from ocr_engine import parse_answer_sheet
from rubric_evaluator import evaluate
from feedback_generator import generate_feedback

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Assignment Evaluator AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Clean & Modern Custom Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #FAFAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .header-title {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    /* Content Cards */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Badges */
    .badge {
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-relevant {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .badge-partial {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .badge-irrelevant {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .badge-info {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #DBEAFE;
    }

    /* Review banner */
    .review-banner {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        color: #92400E;
        font-size: 0.9rem;
    }
    .relevance-banner {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        color: #991B1B;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Navigation & System Profile
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📝 Assignment Evaluator")
    st.caption("NLP & Computer Vision Assistant")
    st.markdown("---")
    
    nav_choice = st.radio(
        "Navigation",
        ["🔍 Single Document Evaluation", "📊 24-Image Benchmark & Analytics"],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("##### ⚙️ System Configuration")
    st.caption("• **OCR Core**: EasyOCR (Spec default)")
    st.caption("• **Document Formats**: PNG, JPG, JPEG, PDF")
    st.caption("• **Concept Matching**: 4-Tier NLP Engine")
    st.caption("• **Relevance Gate**: Active (MCA003 trap protected)")
    st.caption("• **Supervision**: Faculty Final Review")
    
    st.markdown("---")
    st.markdown("##### 🔗 Project Repository")
    st.markdown("[GitHub Repository](https://github.com/Gowda1007/Assignment-Evaluator.git)")


# ---------------------------------------------------------------------------
# Helper: PDF to PIL Image conversion
# ---------------------------------------------------------------------------
def convert_pdf_page_to_image(pdf_bytes: bytes, page_number: int = 0) -> Image.Image:
    """Render a single page of a PDF document to a high-resolution PIL Image."""
    if not PDF_SUPPORT:
        raise RuntimeError("pypdfium2 is required for PDF evaluation.")
    pdf = pdfium.PdfDocument(pdf_bytes)
    if page_number >= len(pdf):
        page_number = 0
    page = pdf[page_number]
    # Render at 2.5x scale for sharp OCR reading
    pil_img = page.render(scale=2.5).to_pil()
    return pil_img


# ---------------------------------------------------------------------------
# VIEW 1: Single Document Evaluation
# ---------------------------------------------------------------------------
if nav_choice == "🔍 Single Document Evaluation":
    # Header Banner
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Intelligent Student Assignment Evaluation Assistant</h1>
        <p class="header-subtitle">Upload any student answer sheet in Image (PNG/JPG) or PDF format for automated OCR reading, rubric matching, and feedback generation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input Mode Tabs
    input_tab1, input_tab2 = st.tabs(["📤 Upload Student Sheet (Image or PDF)", "📂 Select Pre-loaded Sample (24 Available)"])

    active_image_path = None
    image_to_display = None
    temp_image_path = os.path.join(os.path.dirname(__file__), "temp_uploaded.png")

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Choose an answer sheet document",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Accepts high-resolution images or single-page PDF files."
        )
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            if filename.endswith(".pdf"):
                with st.spinner("Rendering PDF document page..."):
                    pdf_bytes = uploaded_file.getvalue()
                    try:
                        pil_img = convert_pdf_page_to_image(pdf_bytes, page_number=0)
                        pil_img.save(temp_image_path)
                        active_image_path = temp_image_path
                        image_to_display = pil_img
                        st.success(f"📄 Loaded PDF: '{uploaded_file.name}' (Page 1 rendered at 2.5x scale for OCR)")
                    except Exception as e:
                        st.error(f"Failed to process PDF: {e}")
            else:
                # Standard Image
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                active_image_path = temp_image_path
                image_to_display = Image.open(temp_image_path)

    with input_tab2:
        image_paths = get_all_image_paths()
        sample_options = ["-- Select a pre-loaded sample --"] + [os.path.basename(p) for p in image_paths]
        selected_sample = st.selectbox("Select sample sheet from dataset:", sample_options, index=0)
        
        if selected_sample != "-- Select a pre-loaded sample --":
            for p in image_paths:
                if os.path.basename(p) == selected_sample:
                    active_image_path = p
                    image_to_display = Image.open(p)
                    break

    # -----------------------------------------------------------------------
    # Main Evaluation Workspace
    # -----------------------------------------------------------------------
    if active_image_path and image_to_display:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Reset / Clear Button
        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.markdown("### 📑 Evaluation Pipeline Workspace")
        with col_hdr2:
            if st.button("🔄 Clear / Evaluate Another", use_container_width=True):
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
                st.rerun()

        # Step 1 & 2: Side-by-side Document Preview & OCR Text Extraction
        col_img, col_ocr = st.columns([1, 1], gap="medium")
        
        with col_img:
            st.markdown("##### 📄 Document Preview")
            st.image(image_to_display, use_container_width=True)

        with col_ocr:
            st.markdown("##### 🔍 Optical Character Recognition (EasyOCR)")
            with st.spinner("Performing OCR text extraction..."):
                ocr_data = parse_answer_sheet(active_image_path)

            # Metadata Pill Badges
            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                st.metric("Student ID", ocr_data["student_id"] or "Not Detected")
            with m_c2:
                st.metric("Question ID", ocr_data["question_id"] or "Not Detected")
            with m_c3:
                st.metric("OCR Confidence", ocr_data["ocr_confidence"].upper())

            st.text_input("Extracted Question", value=ocr_data["question_text"] or "(None)", disabled=True)
            st.text_area("Extracted Written Answer", value=ocr_data["student_answer"] or "(None)", height=110, disabled=True)

            with st.expander("🔎 View Raw OCR Text Lines"):
                st.write(ocr_data["raw_lines"])

        # Step 3 to 8: Rubric Evaluation & Report
        qid = ocr_data["question_id"]
        if not qid:
            st.error("⚠️ Unable to identify Question ID from the document header. Please ensure Question ID (e.g., Q1) is legible.")
        else:
            try:
                eval_data = evaluate(ocr_data["student_answer"], qid)
                feedback = generate_feedback(eval_data)
            except Exception as e:
                st.error(f"Evaluation error for {qid}: {e}")
                eval_data = None

            if eval_data:
                st.markdown("---")
                st.markdown("### 📊 Preliminary Evaluation Report")
                st.caption("All marks are **preliminary** and calculated according to rubric concept points. Final evaluation remains subject to faculty review.")

                # Metric Cards
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                marks = eval_data["total_marks"]
                max_marks = eval_data["max_marks"]
                cat = eval_data["category"]
                concepts_hit = sum(1 for c in eval_data["concept_results"] if c["matched"])
                total_concepts = len(eval_data["concept_results"])

                with kpi1:
                    st.metric("Preliminary Score", f"{marks} / {max_marks}")
                with kpi2:
                    st.metric("Concept Coverage", f"{concepts_hit} / {total_concepts}")
                with kpi3:
                    st.metric("TF-IDF Cosine Similarity", f"{eval_data['tfidf_similarity']:.3f}")
                with kpi4:
                    if cat == "Relevant":
                        badge_html = f'<div class="badge badge-relevant">✅ {cat}</div>'
                    elif cat == "Partially relevant":
                        badge_html = f'<div class="badge badge-partial">⚠️ {cat}</div>'
                    else:
                        badge_html = f'<div class="badge badge-irrelevant">❌ {cat}</div>'
                    st.markdown("**Category**")
                    st.markdown(badge_html, unsafe_allow_html=True)

                # Warning banners
                if feedback["needs_review"]:
                    st.markdown(f"""
                    <div class="review-banner">
                        ⚠️ <strong>Faculty Review Recommended:</strong> {feedback['review_reason']}
                    </div>
                    """, unsafe_allow_html=True)

                if not eval_data["relevance"]["is_relevant"]:
                    st.markdown(f"""
                    <div class="relevance-banner">
                        🚫 <strong>Relevance Gate Triggered:</strong> {eval_data['relevance']['reason']}
                    </div>
                    """, unsafe_allow_html=True)

                # Reference comparison
                with st.expander("📚 View Reference Question & Faculty Benchmark Answer", expanded=False):
                    st.markdown(f"**Question:** {eval_data['question']}")
                    st.markdown(f"**Subject:** `{eval_data['subject']}` | **Max Marks:** `{eval_data['max_marks']}`")
                    st.info(f"**Faculty Reference Answer:**\n{eval_data['reference_answer']}")

                # Per-Concept Match Breakdown
                st.markdown("#### 🎯 Rubric Concept Matching Breakdown")
                breakdown_records = []
                for idx, c in enumerate(eval_data["concept_results"], start=1):
                    breakdown_records.append({
                        "Point": f"Concept {idx}",
                        "Rubric Expected Concept": c["concept"],
                        "Status": "✅ Covered" if c["matched"] else "❌ Missing",
                        "Match Method": c["method"].upper() if c["method"] else "None",
                        "Matched Evidence": c["phrase"] or "—",
                        "Marks": f"{c['marks']} Mark"
                    })
                st.dataframe(pd.DataFrame(breakdown_records), use_container_width=True, hide_index=True)

                # Humanized Feedback
                st.markdown("#### 💬 Actionable Feedback & Suggestions")
                fb_col1, fb_col2 = st.columns([1, 1], gap="medium")
                with fb_col1:
                    st.markdown("**Concepts Covered:**")
                    if feedback["covered"]:
                        for cov in feedback["covered"]:
                            st.write(f"- ✅ **{cov['concept']}** *(detected via {cov['method']})*")
                    else:
                        st.write("*(None)*")

                with fb_col2:
                    st.markdown("**Concepts Missing:**")
                    if feedback["missing"]:
                        for mis in feedback["missing"]:
                            st.write(f"- ❌ {mis}")
                    else:
                        st.write("*(None — all concepts covered)*")

                st.info(f"💡 **Constructive Suggestion:** {feedback['suggestion']}")

                # Preprocessing Details
                with st.expander("⚙️ View NLP Preprocessing Stages", expanded=False):
                    prep = eval_data["preprocessing"]
                    st.json({
                        "1_lowercased_text": prep["lowered"],
                        "2_cleaned_text": prep["cleaned"],
                        "3_tokenized_words": prep["tokens"],
                        "4_stopwords_removed": prep["filtered"],
                        "5_unique_keywords": prep["keywords"],
                    })

                # Faculty Review Panel
                st.markdown("---")
                st.markdown("### 👨‍🏫 Faculty Supervisory Review Panel")
                st.caption("Instructors can override preliminary marks, provide final commentary, and download review logs.")

                with st.form("faculty_review_form"):
                    rf_col1, rf_col2 = st.columns([1, 2])
                    with rf_col1:
                        final_marks = st.number_input(
                            "Final Approved Score",
                            min_value=0,
                            max_value=max_marks,
                            value=int(marks),
                            step=1
                        )
                    with rf_col2:
                        remarks = st.text_input("Instructor Remarks", value="Assessment confirmed.")
                    
                    save_btn = st.form_submit_button("💾 Save Faculty Decision")
                    if save_btn:
                        review_entry = {
                            "student_id": ocr_data["student_id"],
                            "question_id": qid,
                            "preliminary_marks": marks,
                            "final_marks": final_marks,
                            "remarks": remarks,
                        }
                        review_csv_path = os.path.join(os.path.dirname(__file__), "faculty_reviews.csv")
                        new_row_df = pd.DataFrame([review_entry])
                        if not os.path.exists(review_csv_path):
                            new_row_df.to_csv(review_csv_path, index=False)
                        else:
                            new_row_df.to_csv(review_csv_path, mode="a", header=False, index=False)
                        st.success(f"Decision saved for {ocr_data['student_id']} ({qid})! Final Score: {final_marks}/{max_marks}")

                # Download Reviews CSV
                review_csv_path = os.path.join(os.path.dirname(__file__), "faculty_reviews.csv")
                if os.path.exists(review_csv_path):
                    with open(review_csv_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Faculty Reviews CSV",
                            data=f,
                            file_name="faculty_reviews.csv",
                            mime="text/csv",
                        )

    else:
        st.info("👆 Please upload an answer sheet (PNG, JPG, or PDF) or choose a pre-loaded sample from the tabs above to begin.")


# ---------------------------------------------------------------------------
# VIEW 2: 24-Sample Validation Benchmark
# ---------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">24-Image Benchmark & Validation Analytics</h1>
        <p class="header-subtitle">Evaluate the complete pipeline across all 24 dataset images against the ground truth benchmark (student_answers.csv).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    This test runs the end-to-end evaluation pipeline on all 24 answer sheets:
    **Document Image → EasyOCR Text Extraction → Tolerance Header Parser → NLP Preprocessing → 4-Layer Matcher → Relevance Gate**.
    """)

    if st.button("🚀 Run Complete 24-Sample Benchmark", type="primary"):
        answers_df = load_student_answers()
        image_paths = get_all_image_paths()
        results = []

        progress = st.progress(0)
        status_text = st.empty()

        for idx, (_, row) in enumerate(answers_df.iterrows()):
            sid = row["student_id"]
            qid = row["question_id"]
            exp_text = row["student_answer_text_for_validation_only"]
            exp_marks = int(row["expected_marks_out_of_5"])
            exp_cat = row["expected_category"]

            img_name = f"{sid}_{qid}.png"
            img_path = next((p for p in image_paths if os.path.basename(p) == img_name), None)

            status_text.text(f"Processing sheet {idx + 1} of 24: {sid} ({qid})...")

            if img_path:
                ocr_out = parse_answer_sheet(img_path)
                eval_out = evaluate(ocr_out["student_answer"], qid)
                fb_out = generate_feedback(eval_out)

                ocr_words = set(ocr_out["student_answer"].lower().split())
                exp_words = set(exp_text.lower().split())
                word_acc = (len(ocr_words & exp_words) / len(exp_words) * 100) if exp_words else 0

                pred_marks = eval_out["total_marks"]
                pred_cat = eval_out["category"]

                results.append({
                    "Student": sid,
                    "Question": qid,
                    "OCR Word Acc": f"{word_acc:.1f}%",
                    "Expected Marks": exp_marks,
                    "Awarded Marks": pred_marks,
                    "Error": abs(pred_marks - exp_marks),
                    "Expected Category": exp_cat,
                    "Predicted Category": pred_cat,
                    "Category Match": "✅" if pred_cat == exp_cat else "❌",
                    "TF-IDF": f"{eval_out['tfidf_similarity']:.2f}",
                    "Concepts Hit": f"{sum(1 for c in eval_out['concept_results'] if c['matched'])}/5",
                    "Review Flag": "⚠️ Yes" if fb_out["needs_review"] else "No"
                })

            progress.progress((idx + 1) / len(answers_df))

        status_text.empty()
        res_df = pd.DataFrame(results)

        # Aggregate Metrics
        n = len(res_df)
        mae = sum(res_df["Error"]) / n
        cat_acc = sum(1 for _, r in res_df.iterrows() if r["Category Match"] == "✅") / n * 100
        within_one = sum(1 for _, r in res_df.iterrows() if r["Error"] <= 1) / n * 100
        perfect_marks = sum(1 for _, r in res_df.iterrows() if r["Error"] == 0) / n * 100

        st.markdown("### 🏆 Aggregate Performance Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Marks MAE", f"{mae:.2f}")
        k2.metric("Category Accuracy", f"{cat_acc:.1f}%")
        k3.metric("Within ±1 Mark", f"{within_one:.1f}%")
        k4.metric("Exact Score Match", f"{perfect_marks:.1f}%")

        st.markdown("### 📋 Sample-by-Sample Validation Breakdown")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.markdown("### 📊 Predicted Category Distribution")
        st.bar_chart(res_df["Predicted Category"].value_counts())
