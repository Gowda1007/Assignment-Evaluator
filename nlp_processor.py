"""
nlp_processor.py — Text preprocessing and similarity computation.

Handles lowercasing, punctuation removal, tokenization, stop-word removal,
TF-IDF cosine similarity, and optional semantic similarity.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ---------------------------------------------------------------------------
# One-time NLTK data download
# ---------------------------------------------------------------------------
_nltk_ready = False


def _ensure_nltk_data():
    global _nltk_ready
    if not _nltk_ready:
        for resource in ["punkt", "punkt_tab", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        _nltk_ready = True


# ---------------------------------------------------------------------------
# Semantic similarity (optional — degrades gracefully)
# ---------------------------------------------------------------------------
_semantic_model = None
_semantic_available: bool | None = None


def _load_semantic_model():
    """Try to load a lightweight sentence-transformer for semantic matching."""
    global _semantic_model, _semantic_available
    if _semantic_available is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        _semantic_available = True
    except Exception:
        _semantic_available = False


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity between two texts using sentence embeddings.

    Returns 0.0 if the semantic model is unavailable.
    """
    _load_semantic_model()
    if not _semantic_available or _semantic_model is None:
        return 0.0
    embeddings = _semantic_model.encode([text_a, text_b])
    score = float(sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0])
    return max(0.0, score)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> dict:
    """Clean and tokenize text for downstream comparison.

    Returns a dict so the UI can display each stage:
        {
            "original": str,
            "lowered": str,
            "cleaned": str,       # punctuation removed
            "tokens": [str, ...],
            "filtered": [str, ...],  # stop words removed
            "keywords": [str, ...],  # unique meaningful words
        }
    """
    _ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)   # keep only letters, digits, spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()     # collapse whitespace

    tokens = word_tokenize(cleaned)
    filtered = [t for t in tokens if t not in stop_words and len(t) > 1]
    keywords = list(dict.fromkeys(filtered))  # unique, order-preserving

    return {
        "original": text,
        "lowered": lowered,
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered,
        "keywords": keywords,
    }


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity
# ---------------------------------------------------------------------------
def compute_tfidf_similarity(text_a: str, text_b: str) -> float:
    """Return TF-IDF cosine similarity between two texts (0.0 – 1.0)."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text_a, text_b])
    score = float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0][0])
    return max(0.0, score)
