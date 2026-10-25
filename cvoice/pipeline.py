from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
import pandas as pd

from . import collect, config, models, preprocess, topics

CRITICAL = re.compile(
    r"\b(money|paid|payment|charged|deducted|refund|fine|fines|visa|expired|locked|blocked|fraud|stolen)\b"
    r"|خصم|فلوس|مبلغ|دفع|مخالف|تاشير|تأشير|اقام|إقام|مقفل|محظور|احتيال|استرداد", re.IGNORECASE)

JSON_COLS = ["aspects", "svo", "entities"]


def urgency(row) -> float:
    anger_like = row.get("anger", 0) + row.get("disgust", 0) + row.get("fear", 0)
    social = min(1.0, math.log1p(row.get("thumbs_up", 0)) / math.log1p(50))
    score = 100 * (0.45 * row.get("negative", 0) + 0.35 * min(anger_like, 1.0) + 0.20 * social)
    if CRITICAL.search(str(row.get("text", ""))):
        score += 10
    return round(min(score, 100.0), 1)


def enrich(df: pd.DataFrame, log=print) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy().reset_index(drop=True)
    log(f"Language detection + preprocessing for {len(df)} reviews")
    df["lang"] = df["text"].map(preprocess.detect_lang)
    df = df[df["lang"] != "other"].reset_index(drop=True)
    df["clean"] = [preprocess.clean(t, l) for t, l in zip(df["text"], df["lang"])]
    df["tokens"] = [preprocess.tokens(t, l) for t, l in zip(df["text"], df["lang"])]
    df["label"] = df["rating"].map(config.rating_to_label)
    df["is_negative"] = (df["rating"] <= 2).astype(int)
    df["has_reply"] = df["reply_text"].notna() & (df["reply_text"].astype(str).str.len() > 0)

    log("Sentiment (RoBERTa for English, CAMeLBERT for Arabic, XLM-R for mixed)")
    s = models.sentiment_routed(df["text"].tolist(), df["lang"].tolist())
    df[config.LABELS] = s[config.LABELS].values
    df["sentiment"] = s["label"].values
    df["sentiment_model"] = s["model"].values
    if models.lora_available():
        log("Sentiment with our fine-tuned XLM-R LoRA model")
        df["sentiment_lora"] = models.sentiment_lora(df["text"].tolist())["label"].values

    log("Emotion detection")
    e = models.emotion(df["text"].tolist(), df["lang"].tolist())
    df[config.EMOTIONS] = e[config.EMOTIONS].values
    df["emotion"] = e["emotion"].values
    df["urgency"] = [urgency(r) for r in df.to_dict("records")]

    log("spaCy aspects, relations and entities (English)")
    from . import linguistic
    df["aspects"] = [[] for _ in range(len(df))]
    df["svo"] = [[] for _ in range(len(df))]
    df["entities"] = [[] for _ in range(len(df))]
    en_idx = df.index[df["lang"].isin(["en", "mixed"])]
    for i, a in zip(en_idx, linguistic.analyze_docs(df.loc[en_idx, "text"].tolist())):
        df.at[i, "aspects"] = [list(p) for p in a["aspects"]]
        df.at[i, "svo"] = [list(t) for t in a["svo"]]
        df.at[i, "entities"] = [list(x) for x in a["entities"]]
    ar_idx = df.index[df["lang"] == "ar"]
    if len(ar_idx):
        log("Arabic NER (CAMeLBERT)")
        for i, ents in zip(ar_idx, models.ner_ar(df.loc[ar_idx, "text"].tolist())):
            df.at[i, "entities"] = [list(x) for x in ents]

    log("BGE-M3 embeddings")
    X = models.embed(df["text"].tolist(), show_progress=True)
    return df, X


def build_clusters(df: pd.DataFrame, X: np.ndarray, log=print, k: int | None = None) -> pd.DataFrame:
    mask = ((df["rating"] <= 2) | (df["sentiment"] == "negative")).values
    if mask.sum() < 30:
        log("Not enough complaints to cluster yet")
        df["cluster"] = np.nan
        return df
    log(f"Clustering {mask.sum()} complaints")
    res = topics.cluster_issues(df[mask], X[mask], k=k)
    log(f"  k={res['k']} clusters; naming them with the LLM")
    res = topics.name_clusters(res, log=log)
    topics.save(res)
    df["cluster"] = df["review_id"].map(res["assignments"]).astype("float")
    names = {c["id"]: c["label_en"] for c in res["clusters"]}
    df["issue_label"] = df["cluster"].map(lambda c: names.get(int(c)) if pd.notna(c) else None)
    return df


def save(df: pd.DataFrame, X: np.ndarray):
    out = df.copy()
    for c in JSON_COLS:
        out[c] = out[c].map(lambda v: json.dumps(v, ensure_ascii=False))
    out["tokens"] = out["tokens"].map(" ".join)
    out.to_parquet(config.ENRICHED, index=False)
    np.save(config.EMBEDDINGS, X)


def load() -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    if not (config.ENRICHED.exists() and config.EMBEDDINGS.exists()):
        return None, None
    df = pd.read_parquet(config.ENRICHED)
    for c in JSON_COLS:
        df[c] = df[c].map(lambda s: json.loads(s) if isinstance(s, str) else [])
    df["tokens"] = df["tokens"].fillna("").map(lambda s: s.split())
    X = np.load(config.EMBEDDINGS)
    if len(X) != len(df):
        raise ValueError("enriched data and embeddings are out of sync; run `python -m cvoice.pipeline --full`")
    return df, X


def run(full: bool = False, log=print, k: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    raw = collect.load_reviews()
    if raw.empty:
        raise SystemExit("No reviews yet. Run `python -m cvoice.collect` first.")
    old_df, old_X = (None, None) if full else load()
    if old_df is not None:
        new = raw[~raw["review_id"].isin(old_df["review_id"])]
        log(f"{len(new)} new reviews to analyse ({len(old_df)} already done)")
        if len(new):
            new_df, new_X = enrich(new, log)
            df = pd.concat([old_df, new_df], ignore_index=True)
            X = np.vstack([old_X, new_X])
        else:
            df, X = old_df, old_X
    else:
        df, X = enrich(raw, log)
    df = build_clusters(df, X, log, k=k)
    save(df, X)
    log(f"Done: {len(df)} reviews analysed -> {config.ENRICHED}")
    return df, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="recompute everything")
    ap.add_argument("--k", type=int, default=None, help="number of issue clusters (default: auto)")
    a = ap.parse_args()
    run(full=a.full, k=a.k)


if __name__ == "__main__":
    main()
