import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from gensim.models import KeyedVectors, Word2Vec
except ImportError:
    KeyedVectors = None

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "MSRParaphraseCorpus"
MODEL_DIR  = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "msr_paraphrase_train.xlsx"


def load_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "id" in lc and "1" in lc:
            col_map[c] = "id1"
        elif "id" in lc and "2" in lc:
            col_map[c] = "id2"
        elif "string" in lc and "1" in lc:
            col_map[c] = "text1"
        elif "string" in lc and "2" in lc:
            col_map[c] = "text2"
        elif "quality" in lc or "label" in lc:
            col_map[c] = "label"
    df = df.rename(columns=col_map)
    return df[["text1", "text2", "label"]]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\"']", "", text)
    return text


def basic_tokenise(text: str) -> list[str]:
    return text.lower().split()


def build_tfidf(corpus1: list[str], corpus2: list[str]):
    all_sents = corpus1 + corpus2
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    vec.fit(all_sents)
    return vec


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


def load_sbert_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return SentenceTransformer("all-MiniLM-L6-v2")


def train_word2vec(sentences: list[str], dim: int = 100) -> dict:
    if KeyedVectors is None:
        raise RuntimeError("gensim is not installed.")

    tokenised = [basic_tokenise(s) for s in sentences if s.strip()]
    model = Word2Vec(
        sentences=tokenised,
        vector_size=dim,
        window=5,
        min_count=2,
        workers=4,
        epochs=20,
        seed=42,
    )
    return {
        "vectors": model.wv,
        "dim": dim,
    }


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


_GLOVE_CACHE: dict | None = None
_GLOVE_DIM: int = 100


def ensure_glove() -> tuple[dict, int]:
    global _GLOVE_CACHE, _GLOVE_DIM
    if _GLOVE_CACHE is not None:
        return _GLOVE_CACHE, _GLOVE_DIM

    try:
        import gensim.downloader as gd
        wv = gd.load(f"glove-wiki-gigaword-{_GLOVE_DIM}")
        _GLOVE_CACHE = {w: wv[w] for w in wv.key_to_index}
        return _GLOVE_CACHE, _GLOVE_DIM
    except Exception:
        _GLOVE_CACHE = {}
        return _GLOVE_CACHE, _GLOVE_DIM


def glove_similarity(text1: str, text2: str) -> float:
    glove, dim = ensure_glove()

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


FEATURE_NAMES = [
    "cosine_tfidf",
    "jaccard",
    "levenshtein",
    "ngram_3",
    "sbert",
    "word2vec",
    "glove",
]


def extract_features(
    df: pd.DataFrame,
    tfidf_vec,
    w2v_data: dict | None,
) -> pd.DataFrame:
    n = len(df)
    feat = {name: np.empty(n, dtype=np.float64) for name in FEATURE_NAMES}

    sbert_model = load_sbert_model()

    emb1 = sbert_model.encode(
        df["clean1"].tolist(), batch_size=256,
        convert_to_numpy=True, show_progress_bar=True,
    )
    emb2 = sbert_model.encode(
        df["clean2"].tolist(), batch_size=256,
        convert_to_numpy=True, show_progress_bar=True,
    )

    for idx in range(n):
        t1 = df.iloc[idx]["clean1"]
        t2 = df.iloc[idx]["clean2"]

        feat["cosine_tfidf"][idx] = cosine_similarity_tfidf(tfidf_vec, t1, t2)
        feat["jaccard"][idx] = jaccard_similarity(t1, t2)
        feat["levenshtein"][idx] = levenshtein_ratio(t1, t2)
        feat["ngram_3"][idx] = ngram_similarity(t1, t2, n=3)

        v1, v2 = emb1[idx], emb2[idx]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        feat["sbert"][idx] = float(dot / norm) if norm else 0.0

        if w2v_data is not None:
            feat["word2vec"][idx] = w2v_similarity(w2v_data, t1, t2)

        feat["glove"][idx] = glove_similarity(t1, t2)

    result = pd.DataFrame(feat)
    result["label"] = df["label"].values
    return result


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    accuracy_score(y_test, y_pred)
    roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])

    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    model_full = LogisticRegression(max_iter=1000)
    model_full.fit(X_full_s, y)

    return model_full, scaler_full


def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    tfidf_vec,
    w2v_data: dict | None,
    feature_names: list[str],
    out_dir: Path,
):
    import pickle
    import json

    artefacts = {
        "model":            model,
        "scaler":           scaler,
        "tfidf_vectoriser": tfidf_vec,
        "w2v_vectors":      w2v_data["vectors"] if w2v_data else None,
        "w2v_dim":          w2v_data["dim"] if w2v_data else None,
        "feature_names":    feature_names,
    }
    path = out_dir / "plagiarism_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(artefacts, f)

    meta = {
        "feature_names": feature_names,
        "model_type": "LogisticRegression",
        "description": (
            "Plagiarism similarity scores predicted by Logistic Regression "
            "trained on: cosine_tfidf, jaccard, levenshtein, ngram_3, "
            "sbert, word2vec, glove."
        ),
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    import warnings
    warnings.filterwarnings("ignore")

    df = load_data(TRAIN_FILE)

    df["clean1"] = df["text1"].apply(clean_text)
    df["clean2"] = df["text2"].apply(clean_text)

    df["clean1"] = df["clean1"].fillna("")
    df["clean2"] = df["clean2"].fillna("")

    mask = df["clean1"].eq("") & df["clean2"].eq("")
    if mask.sum():
        df = df[~mask].reset_index(drop=True)

    corpus1 = df["clean1"].tolist()
    corpus2 = df["clean2"].tolist()
    tfidf_vec = build_tfidf(corpus1, corpus2)

    w2v_data = train_word2vec(corpus1 + corpus2, dim=100)

    feat_df = extract_features(df, tfidf_vec, w2v_data)

    X = feat_df[FEATURE_NAMES].values
    y = feat_df["label"].values

    model_final, scaler_final = train_and_evaluate(X, y)

    save_model(model_final, scaler_final, tfidf_vec, w2v_data, FEATURE_NAMES, MODEL_DIR)


if __name__ == "__main__":
    main()