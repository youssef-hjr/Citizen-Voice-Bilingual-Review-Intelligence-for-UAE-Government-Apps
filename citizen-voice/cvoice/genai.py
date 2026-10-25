from __future__ import annotations

import json

import pandas as pd

from . import config, llm

NER_PROMPT = """You are an information-extraction system for reviews of UAE government apps.
For EACH review extract:
- feature: the app feature involved (e.g. login, OTP, payment, Emirates ID, visa renewal, fines), or null
- issue_type: one of [login/auth, OTP/verification, payment, crash/bug, performance, UI/UX, service process, customer support, praise, other]
- entities: named entities as [{{"text": "...", "type": "ORG|GOV_SERVICE|LOCATION|DEVICE|MONEY|DATE|ERROR_MESSAGE|PERSON"}}]
  (translate Arabic entity text to English in brackets, e.g. "الهوية [Emirates ID]")
- relations: triples [{{"subject": "...", "relation": "...", "object": "..."}}] such as ["user", "cannot_access", "UAE PASS"]
- language: en / ar / mixed

Return ONLY a JSON list with one object per review, in order, each with "id" plus the fields above.

REVIEWS:
{reviews}"""


def extract_entities(df: pd.DataFrame, batch: int = 8) -> pd.DataFrame:
    rows = []
    for s in range(0, len(df), batch):
        chunk = df.iloc[s:s + batch]
        payload = "\n".join(f"{r.review_id}: {str(r.text)[:350]}" for r in chunk.itertuples())
        try:
            res = llm.ask_json(NER_PROMPT.format(reviews=payload), temperature=0.0)
        except llm.LLMError as e:
            print("LLM extraction failed for a batch:", e)
            continue
        if isinstance(res, dict):
            res = res.get("reviews") or [res]
        for item, rid in zip(res, chunk["review_id"]):
            if isinstance(item, dict):
                item["review_id"] = rid
                rows.append(item)
    return pd.DataFrame(rows)


STRATEGIES = {
    "zero_shot": """Classify the sentiment of each review as negative, neutral or positive.
Return ONLY a JSON list of labels in order.

{reviews}""",
    "role_definitions": """You are a senior customer-experience analyst at a UAE government digital services office.
Classify each app review (English or Arabic) by the reviewer's overall sentiment:
- negative: complaint, frustration, the app or service failed them
- neutral: mixed feelings, a question, a suggestion without clear emotion
- positive: satisfaction, praise or thanks
Focus on the overall experience, not individual words. Sarcasm counts as negative.
Return ONLY a JSON list of labels in the same order, e.g. ["negative","positive"].

{reviews}""",
    "few_shot": """Classify the sentiment of UAE government app reviews as negative, neutral or positive.

Examples:
"OTP never arrives, I tried 10 times. Useless." -> negative
"التطبيق ممتاز وسهل الاستخدام شكرا" -> positive
"Good idea but needs a dark mode and faster loading" -> neutral
"بعد التحديث الأخير ما يفتح أبداً" -> negative
"Works fine most of the time, sometimes slow." -> neutral

Return ONLY a JSON list of labels in order.

{reviews}""",
}


def classify_with_prompt(texts: list[str], strategy: str, batch: int = 15) -> list[str | None]:
    preds = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        listing = "\n".join(f"{i + 1}. {str(t)[:300]}" for i, t in enumerate(chunk))
        try:
            labels = llm.ask_json(STRATEGIES[strategy].format(reviews=listing), temperature=0.0)
        except llm.LLMError as e:
            print(f"{strategy}: batch failed ({str(e)[:300]})")
            labels = []
        if isinstance(labels, dict):
            labels = list(labels.values())[0] if len(labels) == 1 else list(labels.values())
        if not isinstance(labels, list):
            labels = []
        labels = [str(l).lower().strip() for l in labels][:len(chunk)]
        labels = [l if l in config.LABELS else None for l in labels]
        preds.extend(labels + [None] * (len(chunk) - len(labels)))
    return preds


REPLY_SYSTEM = ("You write replies on behalf of a UAE government digital service team. Tone: respectful, warm, "
                "concise, accountable. Never promise refunds or timelines, never invent phone numbers, emails, "
                "links or policies. Refer the user to the in-app support / official channels in general terms.")

REPLY_PROMPT = """Draft a public reply to this review of the "{app}" app.

Review ({rating} stars): {text}

Analysis: sentiment={sentiment}, emotion={emotion}, issue={issue}

Requirements:
1. Thank the user and acknowledge the specific problem in their own terms.
2. Give one practical next step they can take now.
3. Max 70 words per language.
Return ONLY JSON: {{"en": "English reply", "ar": "Arabic reply (Modern Standard Arabic)"}}"""


def draft_reply(row: dict) -> dict:
    return llm.ask_json(REPLY_PROMPT.format(app=row.get("app"), rating=row.get("rating"),
                                            text=str(row.get("text"))[:600],
                                            sentiment=row.get("sentiment"), emotion=row.get("emotion"),
                                            issue=row.get("issue_label", "unknown")),
                        system=REPLY_SYSTEM, temperature=0.4)


BRIEF_PROMPT = """You are the Chief Experience Officer's analyst for UAE government apps.
Using ONLY the statistics below, write an executive briefing with these sections:
1. Headline (one sentence)
2. Top 3 issues to fix first, with evidence (cluster sizes, ratings, apps)
3. Release alerts (versions that increased complaints), if any
4. Citizen emotion (what share is angry / sad and what it means)
5. Recommended actions for the next sprint (3 bullets)
Max 250 words. Do not invent numbers that are not in the data.

STATISTICS:
{stats}"""


def executive_briefing(stats: dict) -> str:
    return llm.ask(BRIEF_PROMPT.format(stats=json.dumps(stats, ensure_ascii=False, default=str, indent=1)),
                   temperature=0.3)
