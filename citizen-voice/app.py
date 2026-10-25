from __future__ import annotations

import os
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from cvoice import collect, config, genai, llm, models, pipeline, preprocess, regression, search, topics

st.set_page_config(page_title="Citizen Voice", page_icon="🗣️", layout="wide")

SENT_COLORS = {"negative": "#d1495b", "neutral": "#edae49", "positive": "#00798c"}


@st.cache_data(show_spinner="Loading analysed reviews…")
def load_data(mtime: float):
    return pipeline.load()


@st.cache_resource(show_spinner="Loading the BGE-M3 search index…")
def get_index(mtime: float):
    df, X = pipeline.load()
    return search.ReviewIndex(df, X)


def data_mtime() -> float:
    return config.ENRICHED.stat().st_mtime if config.ENRICHED.exists() else 0.0


def refresh(per_lang: int):
    box = st.status("Refreshing data…", expanded=True)
    try:
        collect.collect(per_lang=per_lang, log=box.write)
        pipeline.run(log=box.write)
        box.update(label="Data refreshed", state="complete")
    except BaseException as e:
        box.update(label=f"Refresh failed: {e}", state="error")
    st.cache_data.clear()
    st.cache_resource.clear()


with st.sidebar:
    st.title("🗣️ Citizen Voice")
    st.caption("What residents say about UAE government apps, in English and Arabic.")
    key = st.text_input("OpenRouter API key", type="password",
                        help="Needed for the LLM features (Ask, replies, briefing). Free at openrouter.ai.")
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
    st.write("LLM:", "✅ connected" if llm.available() else "⚪ not configured")
    with st.expander("Refresh data"):
        per_lang = st.slider("New reviews per app and language", 50, 1000, 200, 50)
        if st.button("Collect + analyse new reviews"):
            refresh(per_lang)

df, X = load_data(data_mtime())
if df is None:
    st.info("No analysed data yet. Run the Colab notebook (or `python -m cvoice.collect` then "
            "`python -m cvoice.pipeline`), or use **Refresh data** in the sidebar.")
    st.stop()

with st.sidebar:
    st.divider()
    apps = st.multiselect("Apps", sorted(df["app"].unique()), default=sorted(df["app"].unique()))
    langs = st.multiselect("Languages", ["en", "ar", "mixed"], default=["en", "ar", "mixed"])
    dmin, dmax = pd.to_datetime(df["date"]).min().date(), pd.to_datetime(df["date"]).max().date()
    dr = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    st.caption(f"{len(df):,} reviews · last update {pd.to_datetime(df['collected_at']).max():%Y-%m-%d %H:%M}")

start, end = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax))
dates = pd.to_datetime(df["date"]).dt.date
mask = df["app"].isin(apps) & df["lang"].isin(langs) & (dates >= start) & (dates <= end)
fdf = df[mask]
clusters = topics.load() or {"clusters": []}
cl_by_id = {c["id"]: c for c in clusters["clusters"]}

tabs = st.tabs(["📊 Overview", "🧩 Issues", "🚀 Releases", "🔎 Search", "💬 Ask Citizen Voice",
                "🚨 Triage & Reply", "🧪 Analyze a review"])

with tabs[0]:
    if fdf.empty:
        st.warning("No reviews match the filters.")
    else:
        c = st.columns(5)
        c[0].metric("Reviews", f"{len(fdf):,}")
        c[1].metric("Average rating", f"{fdf['rating'].mean():.2f} ★")
        c[2].metric("Complaints (1-2★)", f"{fdf['is_negative'].mean():.0%}")
        neg = fdf[fdf["is_negative"] == 1]
        c[3].metric("Angry complaints", f"{(neg['emotion'] == 'anger').mean():.0%}" if len(neg) else "–")
        c[4].metric("Answered by the team", f"{fdf['has_reply'].mean():.0%}")

        a, b = st.columns(2)
        share = fdf.groupby(["app", "sentiment"]).size().reset_index(name="n")
        share["%"] = 100 * share["n"] / share.groupby("app")["n"].transform("sum")
        a.plotly_chart(px.bar(share, x="app", y="%", color="sentiment", color_discrete_map=SENT_COLORS,
                              title="Predicted sentiment by app (%)"))
        emo = neg["emotion"].value_counts().reset_index()
        emo.columns = ["emotion", "n"]
        b.plotly_chart(px.bar(emo, x="emotion", y="n", title="Emotions in complaints"))

        trend = regression.weekly_trend(fdf)
        trend = trend[trend["reviews"] >= 5]
        a.plotly_chart(px.line(trend, x="week", y="neg_rate", color="app", markers=True,
                               title="Weekly complaint rate (1-2★ share)"))
        lang = fdf["lang"].value_counts().reset_index()
        lang.columns = ["language", "n"]
        b.plotly_chart(px.pie(lang, names="language", values="n", title="Review language", hole=0.5))

        with st.expander("How good is the sentiment model on this data? (star rating used as ground truth)"):
            acc = (fdf["sentiment"] == fdf["label"]).mean()
            st.write(f"Agreement with star-rating labels: **{acc:.1%}**")
            st.dataframe(pd.crosstab(fdf["label"], fdf["sentiment"], rownames=["stars"], colnames=["model"]))

        if st.button("📝 Generate executive briefing", disabled=not llm.available()):
            vt = regression.version_table(fdf)
            stats = {
                "reviews": len(fdf), "avg_rating": round(fdf["rating"].mean(), 2),
                "complaint_rate": round(fdf["is_negative"].mean(), 3),
                "by_app": fdf.groupby("app").agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                                                 complaint_rate=("is_negative", "mean")).round(3).to_dict("index"),
                "emotions_in_complaints": neg["emotion"].value_counts(normalize=True).round(3).to_dict(),
                "top_issues": [{k: cl[k] for k in ("label_en", "size", "avg_rating", "apps")}
                               for cl in sorted(clusters["clusters"], key=lambda x: -x["size"])[:6]],
                "release_alerts": vt[vt["flag"]].to_dict("records") if len(vt) else [],
            }
            with st.spinner("Writing briefing…"):
                try:
                    st.markdown(genai.executive_briefing(stats))
                except llm.LLMError as e:
                    st.error(str(e))

with tabs[1]:
    if not clusters["clusters"]:
        st.info("No issue clusters yet: run the pipeline.")
    else:
        sub = fdf.dropna(subset=["cluster"])
        counts = sub["cluster"].astype(int).value_counts()
        table = pd.DataFrame([{
            "id": cid, "issue": cl_by_id[cid]["label_en"], "بالعربية": cl_by_id[cid].get("label_ar", ""),
            "complaints": int(n), "avg ★": round(sub[sub["cluster"] == cid]["rating"].mean(), 2),
            "% Arabic": f"{(sub[sub['cluster'] == cid]['lang'] == 'ar').mean():.0%}",
            "keywords": ", ".join(cl_by_id[cid]["keywords"][:6]),
        } for cid, n in counts.items() if cid in cl_by_id])
        st.subheader("Automatically discovered issues")
        st.caption("Complaints clustered by meaning with BGE-M3 embeddings (Arabic and English together), "
                   "keywords from class-based TF-IDF, names written by the LLM.")
        st.dataframe(table, hide_index=True)
        if len(table):
            fig = px.treemap(table, path=["issue"], values="complaints", color="avg ★",
                             color_continuous_scale="RdYlGn", range_color=(1, 3))
            st.plotly_chart(fig)
            pick = st.selectbox("Explore an issue", table["id"],
                                format_func=lambda i: f"{cl_by_id[i]['label_en']} ({counts.get(i, 0)})")
            cl = cl_by_id[pick]
            if cl.get("description"):
                st.write(cl["description"])
            rows = sub[sub["cluster"] == pick].sort_values("urgency", ascending=False)
            st.dataframe(rows[["app", "rating", "lang", "emotion", "urgency", "date", "text"]].head(50),
                         hide_index=True)

        st.subheader("What exactly is wrong? Aspect → opinion pairs (dependency parsing, English)")
        neg_en = fdf[(fdf["is_negative"] == 1) & (fdf["lang"] != "ar")]
        pairs = Counter(f"{a} → {o}" for lst in neg_en["aspects"] for a, o in lst)
        if pairs:
            ap = pd.DataFrame(pairs.most_common(20), columns=["aspect → opinion", "count"])
            st.plotly_chart(px.bar(ap[::-1], x="count", y="aspect → opinion", orientation="h"))
        st.subheader("Most frequent phrases (bigrams) in complaints vs praise")
        from cvoice.linguistic import top_ngrams
        g1, g2 = st.columns(2)
        for col, lab in [(g1, "negative"), (g2, "positive")]:
            grams = top_ngrams(fdf[fdf["label"] == lab]["tokens"], n=2, k=15)
            col.dataframe(pd.DataFrame(grams, columns=[f"{lab} bigram", "count"]), hide_index=True)

with tabs[2]:
    st.subheader("Release regression detector")
    st.caption("Each version's complaint rate vs the rest of the app's reviews (two-proportion z-test). "
               "Flagged = significantly worse after Benjamini-Hochberg correction for testing many versions "
               "(false discovery rate 5%) and at least 10 points higher.")
    min_n = st.slider("Minimum reviews per version", 10, 100, 20, 5)
    vt = regression.version_table(fdf, min_reviews=min_n)
    if vt.empty:
        st.info("Not enough reviews with version information for these filters.")
    else:
        vt["issue that grew most"] = vt.get("top_growing_issue", pd.Series(dtype=float)).map(
            lambda c: cl_by_id.get(int(c), {}).get("label_en") if pd.notna(c) else None)
        flagged = vt[vt["flag"]]
        if len(flagged):
            for _, r in flagged.iterrows():
                st.error(f"**{r['app']} {r['version']}**: complaint rate {r['neg_rate']:.0%} vs "
                         f"{r['baseline']:.0%} baseline (z = {r['z']}, q = {r['q']:.3f}, {r['reviews']} reviews). "
                         f"Issue that grew most: {r['issue that grew most'] or '–'}")
        else:
            st.success("No version is significantly worse than its app's baseline.")
        app_pick = st.selectbox("App", sorted(vt["app"].unique()))
        va = vt[vt["app"] == app_pick]
        fig = px.bar(va, x="version", y="neg_rate", color="flag", hover_data=["reviews", "z", "avg_rating"],
                     color_discrete_map={True: "#d1495b", False: "#8d99ae"},
                     title=f"{app_pick}: complaint rate by version")
        fig.add_hline(y=fdf[fdf["app"] == app_pick]["is_negative"].mean(), line_dash="dot",
                      annotation_text="app average")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig)
        st.dataframe(va.drop(columns=["top_growing_issue"], errors="ignore"), hide_index=True)

with tabs[3]:
    st.subheader("Semantic search (cross-lingual)")
    st.caption("Search by meaning with BGE-M3. Try an English query and see Arabic results, or the reverse: "
               "e.g. *face recognition does not work* / *رمز التحقق لا يصل*.")
    q = st.text_input("Query", "OTP code never arrives")
    k = st.slider("Results", 5, 50, 15)
    sent = st.multiselect("Sentiment", config.LABELS, default=[])
    if q:
        idx = get_index(data_mtime())
        res = idx.search(q, k=k, apps=apps, langs=langs, sentiment=sent or None)
        st.write(f"Language mix of results: {res['lang'].value_counts().to_dict()}")
        st.dataframe(res[["score", "app", "rating", "lang", "sentiment", "emotion", "text"]],
                     hide_index=True)

with tabs[4]:
    st.subheader("Ask Citizen Voice")
    st.caption("Retrieval-augmented answers: the LLM only sees the most relevant real reviews and must cite them. "
               "A second LLM call fact-checks every claim (faithfulness).")
    question = st.text_input("Question (English or Arabic)", "Why are people unhappy with UAE PASS login?")
    if st.button("Ask", disabled=not llm.available()):
        with st.spinner("Retrieving reviews and writing the answer…"):
            try:
                out = search.ask(get_index(data_mtime()), question, k=12, apps=apps, langs=langs)
            except llm.LLMError as e:
                st.error(str(e))
                out = None
        if out:
            st.markdown(out["answer"])
            j = out.get("judge", {})
            if "faithfulness" in j:
                f = float(j["faithfulness"])
                (st.success if f >= 0.8 else st.warning)(f"Faithfulness check: {f:.0%} of claims supported")
                st.dataframe(pd.DataFrame(j.get("claims", [])), hide_index=True)
            if out["invalid_citations"]:
                st.warning(f"Answer cites reviews that were not provided: {out['invalid_citations']}")
            ev = out["hits"].copy()
            ev.insert(0, "ref", [f"R{i}" for i in range(1, len(ev) + 1)])
            st.caption(f"Evidence (model: {out['model']})")
            st.dataframe(ev[["ref", "score", "app", "rating", "date", "text"]], hide_index=True)
    elif not llm.available():
        st.info("Add an OpenRouter key in the sidebar to use this tab.")

with tabs[5]:
    st.subheader("Who needs an answer today?")
    st.caption("Urgency = 45% negativity + 35% anger/disgust/fear + 20% agreement (thumbs-up), "
               "+10 if money, fines, visas or account lock-outs are mentioned.")
    only_open = st.checkbox("Only reviews without a reply from the team", True)
    tri = fdf[~fdf["has_reply"]] if only_open else fdf
    tri = tri.sort_values("urgency", ascending=False).head(30)
    st.dataframe(tri[["urgency", "app", "rating", "emotion", "issue_label", "date", "text"]],
                 hide_index=True)
    if len(tri):
        rid = st.selectbox("Draft a reply for", tri["review_id"],
                           format_func=lambda r: f"{tri.set_index('review_id').loc[r, 'urgency']} · "
                                                 f"{str(tri.set_index('review_id').loc[r, 'text'])[:90]}")
        row = tri.set_index("review_id").loc[rid].to_dict()
        st.write(f"> {row['text']}")
        if st.button("✍️ Draft bilingual reply", disabled=not llm.available()):
            try:
                rep = genai.draft_reply(row)
                c1, c2 = st.columns(2)
                c1.markdown("**English**")
                c1.write(rep.get("en", ""))
                c2.markdown("**العربية**")
                c2.markdown(f"<div dir='rtl'>{rep.get('ar', '')}</div>", unsafe_allow_html=True)
            except llm.LLMError as e:
                st.error(str(e))
        if row.get("reply_text"):
            st.caption("Actual reply from the team:")
            st.write(row["reply_text"])

with tabs[6]:
    st.subheader("Analyze any review, step by step")
    txt = st.text_area("Review text", "After the last update the face recognition fails every time and the "
                                      "OTP never arrives. I can't renew my Emirates ID!")
    if st.button("Analyze") and txt.strip():
        info = preprocess.explain(txt)
        st.markdown(f"**Language:** `{info['language']}`  ·  CAMeL Tools: {'yes' if info['camel_tools'] else 'fallback'}")
        st.dataframe(pd.DataFrame({k: [" | ".join(info[k]) if isinstance(info[k], list) else info[k]]
                                   for k in ["cleaned", "tokens", "without_stopwords", "stems", "lemmas"]}).T
                     .rename(columns={0: "result"}))
        lang = info["language"]
        with st.spinner("Running models…"):
            s = models.sentiment_routed([txt], [lang]).iloc[0]
            e = models.emotion([txt], [lang]).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", s["label"], help=f"model: {s['model']}")
        c1.bar_chart(s[config.LABELS].astype(float))
        c2.metric("Emotion", e["emotion"])
        c2.bar_chart(e[config.EMOTIONS].astype(float))
        row = {**s[config.LABELS].to_dict(), **e[config.EMOTIONS].to_dict(), "text": txt, "thumbs_up": 0}
        c3.metric("Urgency", pipeline.urgency(row))
        if lang != "ar":
            from spacy import displacy
            from cvoice import linguistic
            doc = linguistic.nlp()(txt)
            st.markdown("**Aspects (dependency rules):** " + ", ".join(f"`{a} → {o}`" for a, o in linguistic.aspects(doc)))
            st.markdown("**Relations (SVO):** " + ", ".join(f"`{' | '.join(t)}`" for t in linguistic.svo(doc)))
            st.markdown("**Entities (spaCy):** " + ", ".join(f"`{e_.text}` ({e_.label_})" for e_ in doc.ents))
            st.dataframe(pd.DataFrame([(t.text, t.pos_, t.dep_, t.head.text, t.lemma_) for t in doc],
                                      columns=["token", "POS", "dependency", "head", "lemma"]).T)
            svg = displacy.render(list(doc.sents)[0], style="dep", options={"compact": True, "distance": 90})
            st.components.v1.html(svg, height=320, scrolling=True)
        else:
            ents = models.ner_ar([txt])[0]
            st.markdown("**Entities (CAMeLBERT NER):** " + (", ".join(f"`{w}` ({t})" for w, t in ents) or "none"))
        if llm.available():
            with st.spinner("LLM extraction…"):
                try:
                    ext = genai.extract_entities(pd.DataFrame([{"review_id": "x", "text": txt}]))
                    st.markdown("**LLM structured extraction (entities + relations):**")
                    st.json(ext.drop(columns=["review_id"], errors="ignore").to_dict("records")[0] if len(ext) else {})
                except llm.LLMError as err:
                    st.error(str(err))
        st.markdown("**Most similar existing reviews:**")
        hits = get_index(data_mtime()).search(txt, k=5)
        st.dataframe(hits[["score", "app", "rating", "lang", "text"]], hide_index=True)
