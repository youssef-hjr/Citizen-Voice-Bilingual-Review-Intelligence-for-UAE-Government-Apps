from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime

import pandas as pd

from . import config

COLUMNS = [
    "review_id", "app", "app_id", "lang_requested", "text", "rating",
    "thumbs_up", "version", "date", "reply_text", "reply_date", "user_hash",
    "collected_at",
]


def _gp():
    import google_play_scraper as gp
    return gp


def _title_matches(app: dict, title: str) -> bool:
    words = [w for w in app.get("match", app["name"]).lower().split() if w not in {"uae", "dubai", "app"}]
    title = (title or "").lower().replace(" ", "")
    return any(w.replace(" ", "") in title for w in words) if words else True


def resolve_app_id(app: dict, country: str = config.COUNTRY, used: set | None = None, log=print) -> str | None:
    gp = _gp()
    used = used or set()
    try:
        gp.app(app["id"], lang="en", country=country)
        if app["id"] not in used:
            return app["id"]
    except Exception:
        pass
    try:
        hits = gp.search(app["query"], n_hits=10, lang="en", country=country)
        for h in hits:
            if h.get("appId") and h["appId"] not in used and _title_matches(app, h.get("title")):
                log(f"  '{app['name']}': id {app['id']} not found, using {h['appId']} ({h.get('title')})")
                return h["appId"]
    except Exception as e:
        log(f"  search failed for {app['name']}: {e}")
    log(f"  could not find {app['name']} on the store, skipping it")
    return None


def _to_row(r: dict, app_name: str, app_id: str, lang: str) -> dict:
    user = r.get("userName") or ""
    return {
        "review_id": r.get("reviewId"),
        "app": app_name,
        "app_id": app_id,
        "lang_requested": lang,
        "text": (r.get("content") or "").strip(),
        "rating": int(r.get("score") or 0),
        "thumbs_up": int(r.get("thumbsUpCount") or 0),
        "version": r.get("reviewCreatedVersion") or r.get("appVersion"),
        "date": r.get("at"),
        "reply_text": r.get("replyContent"),
        "reply_date": r.get("repliedAt"),
        "user_hash": hashlib.sha256(user.encode()).hexdigest()[:10] if user else None,
        "collected_at": datetime.now(),
    }


def fetch_reviews(app_id: str, lang: str, count: int, seen: set[str],
                  country: str = config.COUNTRY, page_size: int = 200,
                  pause: float = 0.5, log=print) -> list[dict]:
    gp = _gp()
    out, token = [], None
    while len(out) < count:
        n = min(page_size, count - len(out))
        for attempt in range(4):
            try:
                batch, token = gp.reviews(app_id, lang=lang, country=country,
                                          sort=gp.Sort.NEWEST, count=n,
                                          continuation_token=token)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                log(f"    retry {attempt + 1} for {app_id}/{lang} in {wait}s ({e})")
                time.sleep(wait)
        else:
            break
        if not batch:
            break
        new = [r for r in batch if r.get("reviewId") not in seen]
        out.extend(new)
        if not new or token is None or getattr(token, "token", None) is None:
            break
        time.sleep(pause)
    return out


def load_reviews(path=None) -> pd.DataFrame:
    path = path or config.RAW_REVIEWS
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=COLUMNS)


def collect(apps=None, per_lang: int = config.REVIEWS_PER_LANG, langs=config.LANGS,
            country: str = config.COUNTRY, path=None, log=print) -> pd.DataFrame:
    path = path or config.RAW_REVIEWS
    apps = apps or config.APPS
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_reviews(path)
    seen = set(existing["review_id"].dropna())
    rows, used = [], set()
    for app in apps:
        log(f"[{app['name']}]")
        app_id = resolve_app_id(app, country, used=used, log=log)
        if not app_id:
            continue
        used.add(app_id)
        for lang in langs:
            got = fetch_reviews(app_id, lang, per_lang, seen, country, log=log)
            for r in got:
                seen.add(r.get("reviewId"))
            rows.extend(_to_row(r, app["name"], app_id, lang) for r in got)
            log(f"  {lang}: {len(got)} new reviews")
    new = pd.DataFrame(rows, columns=COLUMNS)
    if len(existing) and len(new):
        df = pd.concat([existing, new], ignore_index=True)
    else:
        df = new if len(new) else existing
    df = df.drop_duplicates("review_id").reset_index(drop=True)
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    df.to_parquet(path, index=False)
    log(f"Saved {len(df)} reviews ({len(new)} new) to {path}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-lang", type=int, default=config.REVIEWS_PER_LANG)
    ap.add_argument("--apps", nargs="*", help="app names from config.APPS (default: all)")
    args = ap.parse_args()
    apps = [a for a in config.APPS if not args.apps or a["name"] in args.apps]
    collect(apps=apps, per_lang=args.per_lang)


if __name__ == "__main__":
    main()
