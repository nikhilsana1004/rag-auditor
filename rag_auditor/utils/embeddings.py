"""
Embedding and similarity utilities.
Uses sentence-transformers locally — no API key required.
Falls back to TF-IDF cosine similarity if torch is unavailable.
"""

from typing import List
import numpy as np

_model = None


def get_embedder():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _model = "tfidf"
    return _model


def embed(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    if model == "tfidf":
        return _tfidf_embed(texts)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def rank_chunks_by_query(query: str, chunk_texts: List[str]) -> List[int]:
    """Return chunk indices sorted by descending similarity to query."""
    all_texts = [query] + chunk_texts
    vecs = embed(all_texts)
    q_vec = vecs[0]
    c_vecs = vecs[1:]
    scores = [cosine_similarity(q_vec, c) for c in c_vecs]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), scores


def _tfidf_embed(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=1024)
    mat = vec.fit_transform(texts).toarray()
    return mat.astype(np.float32)
