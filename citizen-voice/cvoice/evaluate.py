from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from . import config

TOKEN_PATTERN = r"(?u)[^\s]+"


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    strat = df["label"] + "_" + df["lang"]
    strat = strat.where(strat.map(strat.value_counts()) >= 2, df["label"])
    return train_test_split(df.index.values, test_size=test_size, random_state=seed, stratify=strat)


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=config.LABELS), 3)}


def scores_by_lang(df: pd.DataFrame, pred_col: str) -> dict:
    out = {"all": scores(df["label"], df[pred_col])}
    for lang in ("en", "ar"):
        d = df[df["lang"] == lang]
        if len(d):
            out[lang] = scores(d["label"], d[pred_col])
    return out


def joined(df):
    return df["tokens"].map(" ".join)


def bow(train_docs, all_docs, max_features=5000):
    v = CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=2, max_features=max_features)
    v.fit(train_docs)
    return v.transform(all_docs), v


def tfidf(train_docs, all_docs, max_features=5000, ngram_range=(1, 2)):
    v = TfidfVectorizer(token_pattern=TOKEN_PATTERN, min_df=2, max_features=max_features,
                        ngram_range=ngram_range, sublinear_tf=True)
    v.fit(train_docs)
    return v.transform(all_docs), v


def train_word2vec(token_lists, vector_size=100, window=5, sg=1, epochs=20, seed=42):
    from gensim.models import Word2Vec
    return Word2Vec(sentences=list(token_lists), vector_size=vector_size, window=window, min_count=2,
                    sg=sg, epochs=epochs, seed=seed, workers=1)


def w2v_doc_vectors(model, token_lists) -> np.ndarray:
    out = np.zeros((len(token_lists), model.vector_size), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [model.wv[t] for t in toks if t in model.wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return normalize(out)


def classify(X, y, train_idx, test_idx, C=1.0):
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    return clf.predict(X[test_idx]), clf


def precision_at_k(X, labels, k=5, n_queries=300, seed=42) -> float:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    Xn = normalize(X) if not isinstance(X, np.ndarray) else X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    qs = rng.choice(Xn.shape[0], size=min(n_queries, Xn.shape[0]), replace=False)
    hits = []
    for q in qs:
        sims = Xn @ Xn[q].T
        sims = np.asarray(sims.todense()).ravel() if hasattr(sims, "todense") else np.ravel(sims)
        sims[q] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        hits.append(np.mean(labels[top] == labels[q]))
    return round(float(np.mean(hits)), 3)


def compare_representations(df: pd.DataFrame, X_dense: np.ndarray, train_idx, test_idx, w2v_model) -> tuple[pd.DataFrame, dict]:
    docs = joined(df)
    y = df["label"].values
    reps = {}
    reps["BoW"], _ = bow(docs.iloc[train_idx], docs)
    reps["TF-IDF (1-2 grams)"], _ = tfidf(docs.iloc[train_idx], docs)
    reps["Word2Vec (trained here)"] = w2v_doc_vectors(w2v_model, df["tokens"].tolist())
    reps["BGE-M3 (pre-trained)"] = X_dense
    rows = []
    for name, X in reps.items():
        pred, _ = classify(X, y, train_idx, test_idx)
        rows.append({"representation": name, "dimensions": X.shape[1], **scores(y[test_idx], pred),
                     "precision@5": precision_at_k(X, y)})
    return pd.DataFrame(rows), reps


def cross_lingual_test(df: pd.DataFrame, pairs: list[tuple[str, str]], encode, tfidf_vec, X_tfidf,
                       X_dense, k: int = 10) -> pd.DataFrame:
    from .preprocess import tokens
    rows = []
    for en, ar in pairs:
        for name in ("TF-IDF", "BGE-M3"):
            res = {}
            for q in (en, ar):
                if name == "TF-IDF":
                    qv = tfidf_vec.transform([" ".join(tokens(q))])
                    sims = np.asarray((X_tfidf @ qv.T).todense()).ravel()
                else:
                    sims = X_dense @ encode([q])[0]
                res[q] = {i for i in np.argsort(-sims)[:k] if sims[i] > 0}
            ar_share = np.mean(df.iloc[list(res[en])]["lang"].values == "ar") if res[en] else 0.0
            rows.append({"query_en": en, "query_ar": ar, "method": name,
                         "overlap@10": round(len(res[en] & res[ar]) / k, 2), "arabic_share_for_en_query": round(ar_share, 2)})
    return pd.DataFrame(rows)
