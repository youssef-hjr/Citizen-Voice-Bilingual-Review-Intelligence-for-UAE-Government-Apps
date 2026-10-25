from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import llm, models
from .preprocess import detect_lang


class ReviewIndex:
    def __init__(self, df: pd.DataFrame, vectors: np.ndarray, encoder=None):
        assert len(df) == len(vectors), "embeddings must be row-aligned with the dataframe"
        self.df = df.reset_index(drop=True)
        self.X = vectors.astype(np.float32)
        self.encode = encoder or (lambda texts: models.embed(texts))

    def search(self, query: str, k: int = 10, apps=None, langs=None, sentiment=None,
               since=None) -> pd.DataFrame:
        q = self.encode([query])[0]
        mask = np.ones(len(self.df), dtype=bool)
        if apps:
            mask &= self.df["app"].isin(apps).values
        if langs:
            mask &= self.df["lang"].isin(langs).values
        if sentiment:
            mask &= self.df["sentiment"].isin(sentiment).values
        if since is not None:
            mask &= (pd.to_datetime(self.df["date"]) >= pd.Timestamp(since)).values
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return self.df.iloc[[]].assign(score=[])
        sims = self.X[idx] @ q
        top = idx[np.argsort(-sims)[:k]]
        out = self.df.iloc[top].copy()
        out["score"] = (self.X[top] @ q).round(3)
        return out


ANSWER_SYSTEM = ("You are Citizen Voice, an analyst that explains what residents say about UAE government apps. "
                 "Answer ONLY from the reviews provided. Cite reviews like [R3]. If the reviews do not contain "
                 "the answer, say so. Never invent numbers, policies or phone numbers.")

ANSWER_PROMPT = """Question: {question}

Reviews (id | app | stars | date | text):
{context}

Write a concise answer (max 150 words) in {language}. Use bullet points for distinct issues,
mention how many of the reviews support each point, and cite review ids in square brackets."""

JUDGE_PROMPT = """You are a strict fact-checker. Split the ANSWER into individual claims and check each one
against the REVIEWS. A claim is supported only if the cited reviews actually say it.

Return ONLY JSON: {{"claims": [{{"claim": "...", "supported": true, "evidence": ["R1"]}}], "faithfulness": 0.0-1.0}}

REVIEWS:
{context}

ANSWER:
{answer}"""


def _context(hits: pd.DataFrame) -> tuple[str, dict]:
    lines, ids = [], {}
    for i, (_, r) in enumerate(hits.iterrows(), 1):
        rid = f"R{i}"
        ids[rid] = r["review_id"]
        date = str(r["date"])[:10] if pd.notna(r.get("date")) else "?"
        text = str(r["text"]).replace("\n", " ")[:400]
        lines.append(f"{rid} | {r['app']} | {r['rating']}* | {date} | {text}")
    return "\n".join(lines), ids


def ask(index: ReviewIndex, question: str, k: int = 12, verify: bool = True, **filters) -> dict:
    hits = index.search(question, k=k, **filters)
    context, ids = _context(hits)
    language = "Arabic" if detect_lang(question) == "ar" else "English"
    answer = llm.ask(ANSWER_PROMPT.format(question=question, context=context, language=language),
                     system=ANSWER_SYSTEM)
    cited = sorted(set(re.findall(r"R\d+", answer)), key=lambda s: int(s[1:]))
    result = {"answer": answer, "hits": hits, "id_map": ids, "cited": cited,
              "invalid_citations": [c for c in cited if c not in ids], "model": llm.last_model_used}
    if verify:
        result["judge"] = judge(answer, context)
    return result


JUDGE_PLAIN_PROMPT = """You are a strict fact-checker. Split the ANSWER into its individual claims and check each one
against the REVIEWS. A claim is supported only if the reviews actually say it.
Write one line per claim, starting with SUPPORTED or UNSUPPORTED, then " | " and the claim. Nothing else.

REVIEWS:
{context}

ANSWER:
{answer}"""


def _judge_plain(answer: str, context: str) -> dict | None:
    try:
        text = llm.ask(JUDGE_PLAIN_PROMPT.format(context=context, answer=answer), temperature=0.0)
    except llm.LLMError:
        return None
    claims = []
    for line in text.splitlines():
        head, _, claim = line.strip().lstrip("-*0123456789. ").partition("|")
        head = head.strip().upper()
        if head.startswith("UNSUPPORTED") or head.startswith("SUPPORTED"):
            claims.append({"claim": claim.strip(), "supported": head.startswith("SUPPORTED")})
    return {"claims": claims} if claims else None


def judge(answer: str, context: str) -> dict:
    try:
        j = llm.ask_json(JUDGE_PROMPT.format(context=context, answer=answer), temperature=0.0)
    except llm.LLMError as e:
        j = _judge_plain(answer, context)
        if j is None:
            return {"error": str(e)}
    if isinstance(j, list):
        j = {"claims": j}
    claims = [c for c in j.get("claims", []) if isinstance(c, dict)]
    if "faithfulness" not in j and claims:
        j["faithfulness"] = sum(bool(c.get("supported")) for c in claims) / len(claims)
    try:
        j["faithfulness"] = float(j["faithfulness"])
    except (KeyError, TypeError, ValueError):
        j.pop("faithfulness", None)
    return j


def lexical_overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    sa, sb = set(a["review_id"]), set(b["review_id"])
    return len(sa & sb) / max(len(sa), 1)


def to_json(result: dict) -> str:
    r = {k: v for k, v in result.items() if k != "hits"}
    return json.dumps(r, ensure_ascii=False, indent=1, default=str)
