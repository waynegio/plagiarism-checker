import re

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from gensim.models import Word2Vec
    from gensim.downloader import downloader as gensim_downloader
except ImportError:
    Word2Vec = None
    gensim_downloader = None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\"']", "", text)
    return text


def basic_tokenise(text: str) -> list[str]:
    return text.lower().split()


def cosine_similarity_tfidf(vec, text1: str, text2: str) -> float:
    v1 = vec.transform([text1])
    v2 = vec.transform([text2])
    dot = v1.multiply(v2).sum()
    norm = np.sqrt(v1.multiply(v1).sum()) * np.sqrt(v2.multiply(v2).sum())
    if norm == 0:
        return 0.0
    return float(dot / norm)


def jaccard_similarity(text1: str, text2: str) -> float:
    t1 = set(basic_tokenise(text1))
    t2 = set(basic_tokenise(text2))
    if not t1 and not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def levenshtein_ratio(s1: str, s2: str) -> float:
    m, n = len(s1), len(s2)
    if m == 0 and n == 0:
        return 1.0
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    dist = prev[n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len) if max_len else 1.0


def ngram_set(text: str, n: int = 3) -> set:
    text = text.replace(" ", "_")
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    g1 = ngram_set(text1, n)
    g2 = ngram_set(text2, n)
    if not g1 and not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)


def sbert_similarity(text1: str, text2: str) -> float:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb1 = model.encode([text1], convert_to_numpy=True, show_progress_bar=False)
    emb2 = model.encode([text2], convert_to_numpy=True, show_progress_bar=False)
    v1, v2 = emb1[0], emb2[0]
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm else 0.0


def w2v_similarity(w2v_data: dict, text1: str, text2: str) -> float:
    vectors = w2v_data["vectors"]
    dim = w2v_data["dim"]

    def avg_vector(text: str) -> np.ndarray:
        tokens = basic_tokenise(text)
        vecs = [vectors[t] for t in tokens if t in vectors]
        if not vecs:
            return np.zeros(dim)
        return np.mean(vecs, axis=0)

    v1, v2 = avg_vector(text1), avg_vector(text2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm else 0.0


_GLOVE_CACHE = None
_GLOVE_DIM = 100


def _ensure_glove():
    global _GLOVE_CACHE, _GLOVE_DIM
    if _GLOVE_CACHE is not None:
        return _GLOVE_CACHE, _GLOVE_DIM

    try:
        wv = gensim_downloader.load(f"glove-wiki-gigaword-{_GLOVE_DIM}")
        _GLOVE_CACHE = {w: wv[w] for w in wv.key_to_index}
        return _GLOVE_CACHE, _GLOVE_DIM
    except Exception:
        _GLOVE_CACHE = {}
        return _GLOVE_CACHE, _GLOVE_DIM


def glove_similarity(text1: str, text2: str) -> float:
    glove, dim = _ensure_glove()

    def avg_vector(text: str) -> np.ndarray:
        tokens = basic_tokenise(text)
        vecs = [glove[t] for t in tokens if t in glove]
        if not vecs:
            return np.zeros(dim)
        return np.mean(vecs, axis=0)

    v1, v2 = avg_vector(text1), avg_vector(text2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm else 0.0


def extract_features(text1: str, text2: str, tfidf_vec, w2v_data: dict) -> np.ndarray:
    t1 = clean_text(text1)
    t2 = clean_text(text2)

    cosine = cosine_similarity_tfidf(tfidf_vec, t1, t2)
    jaccard = jaccard_similarity(t1, t2)
    lev = levenshtein_ratio(t1, t2)
    ngram = ngram_similarity(t1, t2, n=3)
    sbert = sbert_similarity(t1, t2)
    w2v = w2v_similarity(w2v_data, t1, t2)
    glove = glove_similarity(t1, t2)

    return np.array([[cosine, jaccard, lev, ngram, sbert, w2v, glove]], dtype=np.float64)
