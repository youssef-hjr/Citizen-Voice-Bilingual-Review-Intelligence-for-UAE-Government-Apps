from __future__ import annotations

import math
import re

import pandas as pd


def _vkey(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", str(v))) or (0,)


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    n = len(p)
    order = p.sort_values()
    q = order * n / pd.Series(range(1, n + 1), index=order.index)
    q = q[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def version_table(df: pd.DataFrame, min_reviews: int = 20, fdr: float = 0.05,
                  min_lift: float = 0.10, neg_col: str = "is_negative") -> pd.DataFrame:
    rows = []
    for app, g in df[df["version"].notna()].groupby("app"):
        for ver, gv in g.groupby("version"):
            n1 = len(gv)
            if n1 < min_reviews:
                continue
            rest = g[g["version"] != ver]
            n2 = len(rest)
            if n2 < min_reviews:
                continue
            p1, p2 = gv[neg_col].mean(), rest[neg_col].mean()
            p = (gv[neg_col].sum() + rest[neg_col].sum()) / (n1 + n2)
            se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
            z = (p1 - p2) / se
            row = {"app": app, "version": ver, "reviews": n1, "neg_rate": round(p1, 3),
                   "baseline": round(p2, 3), "lift": round(p1 - p2, 3), "z": round(z, 2),
                   "avg_rating": round(gv["rating"].mean(), 2),
                   "p": 0.5 * math.erfc(z / math.sqrt(2)),
                   "first_seen": gv["date"].min()}
            if "cluster" in df.columns:
                c1 = gv["cluster"].dropna().value_counts(normalize=True)
                c2 = rest["cluster"].dropna().value_counts(normalize=True)
                growth = (c1 - c2.reindex(c1.index).fillna(0)).sort_values(ascending=False)
                row["top_growing_issue"] = int(growth.index[0]) if len(growth) else None
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = benjamini_hochberg(out["p"])
    out["flag"] = (out["q"] < fdr) & (out["lift"] >= min_lift)
    out["p"], out["q"] = out["p"].round(4), out["q"].round(4)
    out["_k"] = out["version"].map(_vkey)
    return out.sort_values(["app", "_k"]).drop(columns="_k").reset_index(drop=True)


def weekly_trend(df: pd.DataFrame, neg_col: str = "is_negative") -> pd.DataFrame:
    d = df.dropna(subset=["date"]).copy()
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W").dt.start_time
    t = d.groupby(["app", "week"]).agg(reviews=("review_id", "count"), neg_rate=(neg_col, "mean"),
                                       avg_rating=("rating", "mean")).reset_index()
    return t
