import json
import types

import pandas as pd
import pytest

import fake_data
from cvoice import collect, config, genai, llm, pipeline, preprocess, regression, search, topics


def test_detect_lang():
    assert preprocess.detect_lang("The app is slow") == "en"
    assert preprocess.detect_lang("التطبيق بطيء جدا") == "ar"
    assert preprocess.detect_lang("التطبيق لا يعمل after update") == "mixed"
    assert preprocess.detect_lang("👍👍") == "other"


def test_arabic_normalization():
    s = preprocess.normalize_ar("إِقامة رااااائعة ٢٠٢٦ ـــ")
    assert "ِ" not in s and "ـ" not in s
    assert s.startswith("اقامه")
    assert "2026" in s and "راائعه" in s


def test_arabic_tokens_keep_negation():
    toks = preprocess.tokens_ar("التطبيق لا يعمل ابدا", mode="lemma")
    assert "لا" in toks
    assert "تطبيق" in toks


def test_english_tokens():
    assert "not" in preprocess.tokens_en("I can't login, it doesn't work", "lemma")
    assert preprocess.tokens_en("crashes crashing", "stem") == ["crash", "crash"]
    assert any(t.startswith("emoji_") for t in preprocess.tokens_en("bad app 😡"))


def test_explain_keys():
    info = preprocess.explain("رمز التحقق لا يصل")
    assert info["language"] == "ar" and info["stems"] and info["lemmas"]


def test_aspects_and_svo():
    from cvoice import linguistic
    doc = linguistic.nlp()("The login is slow and the OTP never arrives. Support ignored my complaint.")
    pairs = linguistic.aspects(doc)
    assert ("login", "slow") in pairs
    assert any(a == "otp" and o.startswith("not ") for a, o in pairs)
    assert ("support", "ignore", "complaint") in linguistic.svo(doc)
    assert any(lab == "ERROR_MSG" for lab, _ in linguistic.match_phrases(linguistic.nlp()("Server error again")))


def test_negated_adjective():
    from cvoice import linguistic
    pairs = linguistic.aspects(linguistic.nlp()("The new design is not good."))
    assert ("design", "not good") in pairs


def test_parse_json_variants():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Sure! Here: ["negative", "positive"] hope it helps') == ["negative", "positive"]
    with pytest.raises(llm.LLMError):
        llm.parse_json("no json here")


def test_ask_falls_back_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(llm, "_CACHE", None)
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    monkeypatch.setattr(llm, "candidate_models", lambda: ["bad/model:free", "good/model:free"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        if json["model"] == "bad/model:free":
            return types.SimpleNamespace(status_code=404, text="not found")
        return types.SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.ask("hello") == "hi"
    assert llm.last_model_used == "good/model:free"
    assert llm.ask("hello") == "hi"
    assert calls == ["bad/model:free", "good/model:free"]
    assert json.loads((tmp_path / "cache.json").read_text())


def test_collect_incremental(env, monkeypatch):
    pages = {"en": [{"reviewId": f"e{i}", "userName": "Ali", "content": f"review {i}", "score": 1 + i % 5,
                     "thumbsUpCount": 0, "reviewCreatedVersion": "1.0", "at": pd.Timestamp("2026-10-01"),
                     "replyContent": None, "repliedAt": None} for i in range(5)],
             "ar": [{"reviewId": "a1", "userName": "سارة", "content": "تطبيق ممتاز", "score": 5,
                     "thumbsUpCount": 2, "reviewCreatedVersion": None, "appVersion": None,
                     "at": pd.Timestamp("2026-10-02"), "replyContent": "شكرا", "repliedAt": None}]}
    Tok = types.SimpleNamespace
    gp = types.SimpleNamespace(
        Sort=types.SimpleNamespace(NEWEST=2),
        app=lambda app_id, **k: {"appId": app_id},
        search=lambda *a, **k: [],
        reviews=lambda app_id, lang, country, sort, count, continuation_token: (pages[lang][:count], Tok(token=None)))
    monkeypatch.setattr(collect, "_gp", lambda: gp)
    apps = [{"name": "X", "id": "x.app", "query": "x"}]
    df = collect.collect(apps=apps, per_lang=10, log=lambda *_: None)
    assert len(df) == 6 and "userName" not in df.columns and df["user_hash"].notna().all()
    df2 = collect.collect(apps=apps, per_lang=10, log=lambda *_: None)
    assert len(df2) == 6


def test_resolve_app_id_falls_back_to_search(monkeypatch):
    def bad_app(*a, **k):
        raise ValueError("404")
    gp = types.SimpleNamespace(app=bad_app, search=lambda q, **k: [{"appId": None}, {"appId": "other.id", "title": "Something else"}, {"appId": "real.id", "title": "Alpha App"}])
    monkeypatch.setattr(collect, "_gp", lambda: gp)
    assert collect.resolve_app_id({"name": "Alpha", "id": "wrong", "query": "A"}, log=lambda *_: None) == "real.id"
    assert collect.resolve_app_id({"name": "Alpha", "id": "wrong", "query": "A"}, used={"real.id"},
                                  log=lambda *_: None) is None


def test_pipeline_end_to_end(env):
    raw = fake_data.make_reviews(400)
    raw.to_parquet(config.RAW_REVIEWS, index=False)
    df, X = pipeline.run(log=lambda *_: None)
    assert len(df) == len(X) == 400
    assert set(df["lang"]) <= {"en", "ar", "mixed"}
    assert df["urgency"].between(0, 100).all()
    assert df["cluster"].notna().sum() > 0
    assert any(len(a) for a in df["aspects"])
    assert any(("الهوية", "MISC") in map(tuple, e) for e in df["entities"])

    cl = topics.load()
    assert cl["clusters"][0]["label_ar"].startswith("مشكلة")

    df2, X2 = pipeline.load()
    assert df2["aspects"].map(type).eq(list).all() and X2.shape == X.shape

    more = fake_data.make_reviews(420, seed=1).iloc[400:].assign(review_id=lambda d: "n" + d["review_id"])
    pd.concat([raw, more]).to_parquet(config.RAW_REVIEWS, index=False)
    df3, X3 = pipeline.run(log=lambda *_: None)
    assert len(df3) == len(X3) == 420


def test_regression_flags_bad_release(env):
    raw = fake_data.make_reviews(1200)
    df = raw.assign(is_negative=(raw["rating"] <= 2).astype(int))
    vt = regression.version_table(df)
    flagged = vt[vt["flag"]]
    assert ((flagged["app"] == "UAE PASS") & (flagged["version"] == "4.2.0")).any()
    assert not (flagged["app"] == "DEWA").any()
    assert not regression.weekly_trend(df).empty


def test_search_and_rag(env):
    raw = fake_data.make_reviews(200)
    raw.to_parquet(config.RAW_REVIEWS, index=False)
    df, X = pipeline.run(log=lambda *_: None)
    idx = search.ReviewIndex(df, X)
    hits = idx.search("OTP never arrives", k=5)
    assert len(hits) == 5 and hits["score"].is_monotonic_decreasing
    assert idx.search("x", k=5, apps=["nope"]).empty
    out = search.ask(idx, "Why is login failing?", k=6)
    assert out["cited"] == ["R1", "R2", "R3"] and not out["invalid_citations"]
    assert out["judge"]["faithfulness"] == 1.0


def test_genai_helpers(env):
    rep = genai.draft_reply({"app": "UAE PASS", "rating": 1, "text": "OTP fails", "sentiment": "negative",
                             "emotion": "anger"})
    assert rep["en"] and rep["ar"]
    labels = genai.classify_with_prompt(["bad", "good", "ok"] * 6, "few_shot", batch=7)
    assert len(labels) == 18 and set(labels) <= set(config.LABELS)
    ext = genai.extract_entities(pd.DataFrame({"review_id": ["a", "b", "c"], "text": ["x", "y", "z"]}), batch=2)
    assert list(ext["review_id"]) == ["a", "b", "c"]


def test_ctfidf_keywords_are_distinctive():
    toks = [["otp", "code"]] * 5 + [["payment", "card"]] * 5
    kw = topics.ctfidf_keywords(toks, [0] * 5 + [1] * 5, top_n=2)
    assert set(kw[0]) <= {"otp", "code", "otp code"} and "payment" not in kw[0]


def test_urgency_bounds():
    hi = pipeline.urgency({"negative": 1, "anger": 1, "thumbs_up": 500, "text": "money deducted"})
    lo = pipeline.urgency({"negative": 0, "anger": 0, "thumbs_up": 0, "text": "great"})
    assert hi == 100 and lo == 0


def test_judge_accepts_claims_list(env, monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda *a, **k: '[{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]')
    assert search.judge("answer", "ctx")["faithfulness"] == 0.5
    monkeypatch.setattr(llm, "ask", lambda *a, **k: '{"claims": [], "faithfulness": "0.8"}')
    assert search.judge("answer", "ctx")["faithfulness"] == 0.8


def test_benjamini_hochberg():
    p = pd.Series([0.001, 0.01, 0.03, 0.04, 0.5])
    q = regression.benjamini_hochberg(p)
    assert q.round(4).tolist() == [0.005, 0.025, 0.05, 0.05, 0.5]


def test_ask_json_moves_to_next_model_on_prose(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(llm, "_CACHE", None)
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    monkeypatch.setattr(llm, "candidate_models", lambda: ["thinker:free", "good:free"])
    answers = {"thinker:free": "The user wants me to extract...", "good:free": "<think>hmm</think>[\"negative\"]"}

    def fake_post(url, headers, json, timeout):
        return types.SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": answers[json["model"]]}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.ask_json("classify") == ["negative"]
    assert llm.last_model_used == "good:free"
    cached = [v["model"] for v in json.loads((tmp_path / "cache.json").read_text()).values()]
    assert cached == ["good:free"]


def test_api_key_is_remembered(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import streamlit as st
    monkeypatch.setattr(st, "secrets", {"OPENROUTER_API_KEY": " sk-test "}, raising=False)
    assert llm.api_key() == "sk-test"
    monkeypatch.setattr(st, "secrets", {}, raising=False)
    assert llm.api_key() == "sk-test"
    monkeypatch.delenv("OPENROUTER_API_KEY")


def test_prompt_failures_are_not_guessed(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "ask_json", boom)
    assert genai.classify_with_prompt(["a", "b"], "zero_shot") == [None, None]


def test_model_ranking_prefers_direct_answers():
    ms = [{"id": "a/think:free", "supported_parameters": ["reasoning"], "context_length": 900000},
          {"id": "b/vl:free", "supported_parameters": [], "context_length": 1000},
          {"id": "c/chat:free", "supported_parameters": ["response_format"], "context_length": 1000}]
    assert [m["id"] for m in sorted(ms, key=llm._rank)] == ["c/chat:free", "b/vl:free", "a/think:free"]


def test_clitics_dropped():
    assert "'s" not in preprocess.tokens_en("it's slow and the app's login fails")


def test_parse_json_repairs():
    assert llm.parse_json('[{"a": 1,}, {"b": 2},]') == [{"a": 1}, {"b": 2}]
    assert llm.parse_json('[{"a": 1}, {"b": 2}, {"c": ') == [{"a": 1}, {"b": 2}]


def test_naming_and_judge_plain_fallbacks(monkeypatch):
    def fake_ask_json(*a, **k):
        raise llm.LLMError("no JSON")
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask_json)
    monkeypatch.setattr(llm, "ask", lambda prompt, **k: (
        "**EN:** OTP not received\nAR: رمز التحقق لا يصل\nDESC: Codes never arrive." if "three lines" in prompt
        else "1. SUPPORTED | login fails\n2. UNSUPPORTED | servers were hacked"))
    res = {"clusters": [{"id": 0, "label_en": "otp, code", "keywords": ["otp"], "examples": ["no otp"]}]}
    c = topics.name_clusters(res, log=lambda *_: None)["clusters"][0]
    assert c["label_en"] == "OTP not received" and c["label_ar"] == "رمز التحقق لا يصل"
    assert search.judge("answer", "ctx")["faithfulness"] == 0.5


def test_daily_quota_stops_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(llm, "_CACHE", None)
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    monkeypatch.setattr(llm, "candidate_models", lambda: ["a:free", "b:free"])
    calls = []
    def fake_post(url, headers, json, timeout):
        calls.append(1)
        return types.SimpleNamespace(status_code=429, text='{"error":{"message":"Rate limit exceeded: free-models-per-day"}}')
    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError, match="daily"):
        llm.ask("hi")
    assert len(calls) == 1
