from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cvoice import config

NEG_EN = ["The OTP never arrives, I tried ten times. Useless app!",
          "After the last update face recognition fails every time and I can't login.",
          "Payment failed but the money was deducted from my card. Terrible service.",
          "App keeps crashing when I open the fines page. Very slow and frustrating.",
          "I cannot renew my Emirates ID, the server shows an error. Customer support is not helpful.",
          "Login is broken since version 4.2, the verification page is stuck."]
POS_EN = ["Excellent app, very easy to pay my bills. Thank you DEWA!",
          "Great service, renewing my visa took two minutes.",
          "Smooth interface and fast login with UAE PASS. Love it.",
          "Very helpful app, all government services in one place."]
NEU_EN = ["Good idea but needs dark mode and faster loading.",
          "Works fine most of the time, sometimes slow."]
NEG_AR = ["رمز التحقق لا يصل ابداً، تطبيق سيء جداً",
          "بعد التحديث الأخير التطبيق لا يفتح والبصمة لا تعمل",
          "تم خصم المبلغ ولم يتم الدفع، خدمة سيئة",
          "لا أستطيع تجديد الهوية، يظهر خطأ في النظام"]
POS_AR = ["تطبيق ممتاز وسهل الاستخدام شكراً لكم",
          "خدمة رائعة وسريعة، تجديد الإقامة كان سهلاً"]
NEU_AR = ["التطبيق جيد لكن يحتاج تحسين السرعة"]


def make_reviews(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    apps = [a["name"] for a in config.APPS]
    rows = []
    start = datetime(2026, 8, 1)
    for i in range(n):
        app = apps[i % len(apps)]
        version = rnd.choice(["4.1.0", "4.2.0", "4.2.1", "4.3.0"])
        bad_release = app == "UAE PASS" and version == "4.2.0"
        r = rnd.random()
        p_neg = 0.75 if bad_release else 0.4
        ar = rnd.random() < 0.4
        if r < p_neg:
            text, rating = rnd.choice(NEG_AR if ar else NEG_EN), rnd.choice([1, 1, 2])
        elif r < p_neg + 0.1:
            text, rating = rnd.choice(NEU_AR if ar else NEU_EN), 3
        else:
            text, rating = rnd.choice(POS_AR if ar else POS_EN), rnd.choice([4, 5, 5])
        if rnd.random() < 0.3:
            text += rnd.choice([" 😡", " 👍", " Please fix.", "!!", " يرجى الحل"])
        rows.append({
            "review_id": f"r{i}", "app": app, "app_id": app.lower().replace(" ", "."),
            "lang_requested": "ar" if ar else "en", "text": text, "rating": rating,
            "thumbs_up": rnd.choice([0, 0, 1, 3, 12, 40]), "version": version if rnd.random() > 0.1 else None,
            "date": start + timedelta(days=rnd.randint(0, 80), hours=rnd.randint(0, 23)),
            "reply_text": "Thank you, please contact support." if rnd.random() < 0.2 else None,
            "reply_date": None, "user_hash": hashlib.sha256(str(i).encode()).hexdigest()[:10],
            "collected_at": datetime(2026, 10, 24),
        })
    return pd.DataFrame(rows)


NEG_WORDS = ["never", "fail", "crash", "useless", "terrible", "error", "can't", "cannot", "broken", "stuck",
             "slow", "لا", "سيء", "خطأ", "خصم"]
POS_WORDS = ["excellent", "great", "love", "helpful", "smooth", "ممتاز", "رائعة", "شكرا"]


def fake_embed(texts, **_):
    from cvoice.preprocess import tokens
    X = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokens(str(t)) or ["empty"]:
            X[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X


def _polarity(t: str) -> tuple[float, float, float]:
    t = t.lower()
    n = sum(w in t for w in NEG_WORDS)
    p = sum(w in t for w in POS_WORDS)
    if n > p:
        return 0.8, 0.15, 0.05
    if p > n:
        return 0.05, 0.15, 0.8
    return 0.2, 0.6, 0.2


def fake_sentiment_routed(texts, langs, **_):
    df = pd.DataFrame([_polarity(str(t)) for t in texts], columns=config.LABELS)
    df["label"] = df[config.LABELS].idxmax(axis=1)
    df["model"] = ["fake-ar" if l == "ar" else "fake-en" for l in langs]
    return df


def fake_emotion(texts, langs, **_):
    rows = []
    for t in texts:
        neg, _, pos = _polarity(str(t))
        rows.append({"anger": neg * 0.6, "disgust": neg * 0.1, "fear": 0.02, "joy": pos * 0.9,
                     "neutral": 0.1, "sadness": neg * 0.3, "surprise": 0.01})
    df = pd.DataFrame(rows)[config.EMOTIONS]
    df["emotion"] = df.idxmax(axis=1)
    return df


def fake_ner_ar(texts, **_):
    return [[("الهوية", "MISC")] if "الهوية" in str(t) else [] for t in texts]


def install(monkeypatch):
    from cvoice import models
    monkeypatch.setattr(models, "embed", fake_embed)
    monkeypatch.setattr(models, "sentiment_routed", fake_sentiment_routed)
    monkeypatch.setattr(models, "emotion", fake_emotion)
    monkeypatch.setattr(models, "ner_ar", fake_ner_ar)
    monkeypatch.setattr(models, "lora_available", lambda: False)
