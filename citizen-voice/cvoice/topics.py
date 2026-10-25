from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from . import config, llm


def choose_k(X: np.ndarray, k_range=range(6, 15), seed: int = 42, sample: int = 3000) -> tuple[int, dict]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    scores = {}
    for k in k_range:
        if k >= len(idx):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X[idx])
        scores[k] = float(silhouette_score(X[idx], labels, metric="cosine"))
    if not scores:
        return min(2, len(X)), scores
    best = max(scores, key=scores.get)
    return best, scores


def ctfidf_keywords(token_lists, labels, top_n: int = 12) -> dict[int, list[str]]:
    docs = pd.Series([" ".join(t) for t in token_lists]).groupby(np.asarray(labels)).apply(" ".join)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)[^\s]+")
    tf = cv.fit_transform(docs.values).toarray().astype(float)
    tf = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    avg_words = tf.sum() / len(docs)
    idf = np.log(1 + avg_words / np.maximum(tf.sum(axis=0), 1e-9))
    scores = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {int(c): [w for w in vocab[np.argsort(-scores[i])][:top_n]] for i, c in enumerate(docs.index)}


GENERIC = {"app", "application", "not", "uae", "pas", "pass", "use", "work", "please", "good", "bad", "time",
           "تطبيق", "برنامج", "لا", "جدا", "الله", "مشكله"}


def _keyword_label(keywords: list[str], n: int = 3) -> str:
    useful = [k for k in keywords if not set(k.split()) <= GENERIC]
    return ", ".join((useful or keywords)[:n])


def cluster_issues(df: pd.DataFrame, X: np.ndarray, k: int | None = None, seed: int = 42) -> dict:
    if k is None:
        k, sil = choose_k(X, seed=seed)
    else:
        sil = {}
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    kw = ctfidf_keywords(df["tokens"].tolist(), labels)
    clusters = []
    for c in range(k):
        m = labels == c
        sub = df[m]
        centroid = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
        sims = X[m] @ centroid
        reps = sub.iloc[np.argsort(-sims)[:8]]
        clusters.append({
            "id": c, "size": int(m.sum()), "keywords": kw.get(c, []),
            "avg_rating": round(float(sub["rating"].mean()), 2),
            "share_arabic": round(float((sub["lang"] == "ar").mean()), 2),
            "apps": sub["app"].value_counts().head(3).to_dict(),
            "examples": reps["text"].str.slice(0, 220).tolist(),
            "example_ids": reps["review_id"].tolist(),
            "label_en": _keyword_label(kw.get(c, [])), "label_ar": "", "description": "",
        })
    return {"k": k, "silhouette": sil, "assignments": dict(zip(df["review_id"], map(int, labels))),
            "clusters": clusters}


NAMING_PROMPT = """You are a product analyst for UAE government mobile apps.
Below are clusters of citizen complaints. Each has top keywords (English and/or Arabic) and example reviews.
For every cluster give a short, specific issue name (max 5 words) in English and in Arabic,
plus a one-sentence description of the underlying problem.

Return ONLY a JSON list: [{{"id": 0, "label_en": "...", "label_ar": "...", "description": "..."}}, ...]

CLUSTERS:
{clusters}"""


ONE_NAME_PROMPT = """You are a product analyst for UAE government mobile apps.
Here is one cluster of citizen complaints: its top keywords and example reviews.

KEYWORDS: {keywords}
EXAMPLES:
{examples}

Answer in exactly three lines, nothing else:
EN: <specific issue name in English, max 5 words>
AR: <the same name in Arabic>
DESC: <one English sentence describing the underlying problem>"""


def _name_one(c: dict) -> dict | None:
    examples = "\n".join(f"- {str(e)[:200]}" for e in c["examples"][:5])
    try:
        text = llm.ask(ONE_NAME_PROMPT.format(keywords=", ".join(c["keywords"][:10]), examples=examples),
                       temperature=0.0)
    except llm.LLMError:
        return None
    out = {}
    for line in text.splitlines():
        key, _, val = line.strip().strip("*").partition(":")
        key = key.strip("* ").upper()
        if key in ("EN", "AR", "DESC") and val.strip():
            out[{"EN": "label_en", "AR": "label_ar", "DESC": "description"}[key]] = val.strip().strip("*\"' ")
    return out if out.get("label_en") else None


def name_clusters(result: dict, log=print) -> dict:
    if not llm.available():
        return result
    payload = [{"id": c["id"], "keywords": c["keywords"], "examples": c["examples"][:5]}
               for c in result["clusters"]]
    by_id = {}
    try:
        names = llm.ask_json(NAMING_PROMPT.format(clusters=json.dumps(payload, ensure_ascii=False)))
        if isinstance(names, dict):
            names = names.get("clusters") or [names]
        for n in names:
            try:
                by_id[int(n["id"])] = n
            except (KeyError, TypeError, ValueError):
                pass
    except llm.LLMError as e:
        log(f"  cluster naming (one call) failed, naming clusters one by one: {str(e)[:150]}")
    named = 0
    for c in result["clusters"]:
        n = by_id.get(c["id"])
        if not (isinstance(n, dict) and n.get("label_en")):
            n = _name_one(c)
        if n:
            c["label_en"] = n.get("label_en") or c["label_en"]
            c["label_ar"] = n.get("label_ar") or ""
            c["description"] = n.get("description") or ""
            named += 1
    log(f"  {named}/{len(result['clusters'])} clusters named by the LLM")
    return result


def save(result: dict, path=None):
    path = path or config.CLUSTERS
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def load(path=None) -> dict | None:
    path = path or config.CLUSTERS
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
