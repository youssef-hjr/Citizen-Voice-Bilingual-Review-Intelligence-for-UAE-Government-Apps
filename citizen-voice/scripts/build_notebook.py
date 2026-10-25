import io
import re
import tokenize
from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def _no_comments(src):
    lines = src.split("\n")
    magic = {i for i, l in enumerate(lines) if l.lstrip().startswith(("!", "%"))}
    safe = "\n".join("pass" if i in magic else l for i, l in enumerate(lines))
    cut = {t.start[0] - 1: t.start[1] for t in tokenize.generate_tokens(io.StringIO(safe).readline)
           if t.type == tokenize.COMMENT}
    out = []
    for i, l in enumerate(lines):
        if i in cut and i not in magic:
            l = l[:cut[i]].rstrip()
            if not l.strip():
                continue
        out.append(l)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def code(s):
    cells.append(nbf.v4.new_code_cell(_no_comments(s.strip())))


md("""
# 🗣️ Citizen Voice: AI analyst for UAE government app reviews
**MAAI902 Natural Language Processing & LLMs · Individual project**

Millions of residents use UAE government apps (UAE PASS, DubaiNow, RTA, DEWA, ICP, TAMM) and leave reviews in
**English and Arabic**. Nobody can read them all. Citizen Voice collects reviews automatically and answers three
questions for a government digital-services team: **what is broken, how upset people are, and what to fix first.**

**Alignment:** UAE National AI Strategy 2031, *Smart Government* priority (lecture 3), with Arabic-first resources
(CAMeL Tools and CAMeLBERT from NYU Abu Dhabi, multilingual BGE-M3).

| # | Section | Course concept |
|---|---|---|
| 1 | Automated data collection | data science workflow |
| 2 | Exploratory analysis | |
| 3 | English + Arabic preprocessing | tokenization, stop words, stemming, lemmatization, CAMeL normalization |
| 4 | POS + dependency parsing | POS tags, dependency parsing, Matcher, aspect extraction, relation extraction |
| 5 | N-grams | n-grams |
| 6 | Full NLP pipeline | everything below, applied to every review |
| 7 | Text representations | BoW, TF-IDF, Word2Vec, sentence embeddings (BGE-M3), similarity search, cross-lingual retrieval |
| 8 | Sentiment, 5 approaches | lexicon (VADER), ML (TF-IDF + LR), pre-trained transformers |
| 9 | Fine-tuning with LoRA | pre-training vs fine-tuning, parameter-efficient adaptation |
| 10 | Emotion / mood | emotion models, valence-arousal |
| 11 | NER + relations | spaCy NER, CAMeLBERT NER, LLM-based NER, relation extraction |
| 12 | Issue discovery | embeddings + clustering + LLM naming |
| 13 | Release regression detector | statistics on NLP outputs |
| 14 | Prompt engineering | zero-shot vs role/definitions vs few-shot |
| 15 | RAG + faithfulness check | retrieval-augmented generation, hallucination control |
| 16 | Bilingual replies + executive briefing | generation |
| 17 | Results export + launch the app | |

**Before you start:** `Runtime → Change runtime type → T4 GPU`. Add your OpenRouter key as a Colab secret named
`OPENROUTER_API_KEY` (🔑 icon on the left, enable notebook access).
""")

md("## 0. Setup")
code("""
import os, sys, glob, zipfile, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO = "citizen-voice"

# Get the project code: upload citizen-voice.zip when asked (or put it in the Colab file panel first).
if os.path.basename(os.getcwd()) == REPO:
    os.chdir("..")
zips = glob.glob("citizen-voice*.zip") or glob.glob("/content/drive/MyDrive/citizen-voice*.zip")
if not zips and not os.path.exists(REPO) and IN_COLAB:
    from google.colab import files
    zips = list(files.upload().keys())
if zips:
    # always unpack the NEWEST zip over the folder: code is updated, collected data and the LLM cache are kept
    newest = max(zips, key=os.path.getmtime)
    zipfile.ZipFile(newest).extractall(".")
    print("Code from:", newest)
os.chdir(REPO)
# make sure Python imports our `cvoice` package from this folder (and not an older copy already in memory)
sys.path.insert(0, os.getcwd())
for m in [m for m in sys.modules if m == "cvoice" or m.startswith("cvoice.")]:
    sys.modules.pop(m)
print("Working directory:", os.getcwd(), "| package found:", os.path.exists("cvoice/__init__.py"))
""")
code("""
def sh(cmd):
    print(">", cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-600:] if r.returncode else "ok")
    return r.returncode

if IN_COLAB:
    sh(f"{sys.executable} -m pip install -q -r requirements.txt")
    # CAMeL Tools is installed separately: if it ever fails, our code falls back to equivalent rules.
    sh(f"{sys.executable} -m pip install -q camel-tools")
    sh(f"{sys.executable} -m spacy download en_core_web_sm -q")
    # Colab ships an old torchao that makes recent `peft` (LoRA) refuse to import; we don't need it.
    sh(f"{sys.executable} -m pip uninstall -y -q torchao")
""")
code("""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import nltk
warnings.filterwarnings("ignore")
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
    nltk.download(pkg, quiet=True)

from cvoice import (config, collect, preprocess, linguistic, models, pipeline, evaluate, topics,
                regression, search, genai, llm, finetune)

os.makedirs("figures", exist_ok=True)
def savefig(name):
    plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=160, bbox_inches="tight"); plt.show()

R = {}   # every number that goes into the report is stored here and exported at the end
sns.set_theme(style="whitegrid")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU: slower but works)")
print("LLM key found:", llm.available(), "| CAMeL Tools:", preprocess.CAMEL_AVAILABLE)
if not llm.available():
    print("⚠️  LLM sections (cluster names, LLM NER, prompting, RAG, replies) will be skipped.\\n   ", llm.key_problem())
""")

md("""
## 1. Automated data collection
`cvoice/collect.py` uses `google-play-scraper` to fetch the newest reviews of each app in English and Arabic from the
UAE store (`country="ae"`). It is **incremental**: reviews already on disk are skipped, so the same command (or the
**Refresh data** button in the app) keeps the dataset up to date. Wrong package ids are fixed automatically with a
Play Store search.

**Ethics:** reviews are public, but we never store reviewer names (only a short one-way hash), and we only report
aggregates.
""")
code("""
PER_LANG = config.REVIEWS_PER_LANG    # 600 per app per language; lower it (e.g. 200) for a quick run
raw = collect.collect(per_lang=PER_LANG)
R["n_raw"] = len(raw); R["n_apps"] = raw["app"].nunique()
print(raw.groupby(["app", "lang_requested"]).size().unstack(fill_value=0))
raw.sample(5, random_state=1)[["app", "rating", "version", "date", "text"]]
""")

md("## 2. Exploratory analysis")
code("""
raw["lang"] = raw["text"].map(preprocess.detect_lang)
raw["n_words"] = raw["text"].str.split().str.len()
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
raw.groupby("app")["rating"].value_counts(normalize=True).unstack().plot.bar(stacked=True, ax=ax[0], colormap="RdYlGn")
ax[0].set_title("Star ratings by app"); ax[0].legend(title="★", bbox_to_anchor=(1, 1))
raw["lang"].value_counts().plot.pie(autopct="%1.0f%%", ax=ax[1]); ax[1].set_title("Detected language"); ax[1].set_ylabel("")
sns.boxplot(data=raw[raw.lang.isin(["en", "ar"])], x="lang", y="n_words", ax=ax[2], showfliers=False)
ax[2].set_title("Review length (words)")
savefig("eda")
R["avg_rating"] = round(raw["rating"].mean(), 2)
R["share_negative"] = round((raw["rating"] <= 2).mean(), 3)
R["share_arabic"] = round((raw["lang"] == "ar").mean(), 3)
R["median_words"] = int(raw["n_words"].median())
print(R)
""")
md("""
**What to notice:** government app reviews are **polarised**, mostly 1★ or 5★, with few 3★. That is why we
report **macro-F1** (it weights the rare *neutral* class equally) and use class weights when training.
Reviews are short (a few dozen words), so we can use `max_length=128-256` tokens everywhere.
""")

md("""
## 3. Preprocessing: English and Arabic
Arabic needs different treatment from English:
* **Orthographic variation:** أ / إ / آ / ا are often written interchangeably, ى vs ي, ة vs ه. CAMeL Tools normalises these.
* **Diacritics (tashkeel) and tatweel (ـ)** are optional and must be removed.
* **Rich morphology and clitics:** "والتطبيق" = و + ال + تطبيق ("and the app"). Root stemmers (ISRI) are aggressive;
  light stemming only removes common prefixes and suffixes.
* **Negations must be kept** in both languages (لا، لم، لن، ما، ليس / not, never): standard stop-word lists remove
  them, which destroys sentiment.
""")
code("""
en_example = "After the latest update, the app keeps crashing and I couldn't log in with Face ID!! 😡"
ar_example = "بعد التحديث الأخير، التطبيقُ لا يعمل والبصمة لم تَعُد تعمل ابداااً 😡"
for t in (en_example, ar_example):
    info = preprocess.explain(t)
    print(f"\\nLANGUAGE: {info['language']}")
    for k in ["cleaned", "tokens", "without_stopwords", "stems", "lemmas"]:
        print(f"  {k:18}: {info[k]}")
""")
code("""
# Three NLTK stemmers vs lemmatization (English)
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
words = ["crashing", "crashes", "verification", "updated", "payments", "renewal", "services", "unable", "easily"]
wnl = WordNetLemmatizer()
pd.DataFrame({"word": words,
              "Porter": [PorterStemmer().stem(w) for w in words],
              "Snowball": [SnowballStemmer("english").stem(w) for w in words],
              "Lancaster": [LancasterStemmer().stem(w) for w in words],
              "WordNet lemma": [wnl.lemmatize(wnl.lemmatize(w), "v") for w in words],
              "spaCy lemma": [linguistic.nlp()(w)[0].lemma_ for w in words]})
""")
code("""
# Arabic: CAMeL normalization, ISRI root stemmer vs light stemmer
ar_words = ["التطبيقات", "والخدمة", "إقامتي", "بالبطاقة", "مستشفى", "رائعـــة", "تَحْدِيث"]
from nltk.stem.isri import ISRIStemmer
pd.DataFrame({"word": ar_words,
              "CAMeL normalised": [preprocess.normalize_ar(w) for w in ar_words],
              "ISRI root": [ISRIStemmer().stem(preprocess.normalize_ar(w)) for w in ar_words],
              "light stem (ours)": [preprocess.light_stem_ar(preprocess.normalize_ar(w)) for w in ar_words]})
""")
code("""
# How much does each step shrink the vocabulary?
rows = []
for lang in ["en", "ar"]:
    texts = raw[raw.lang == lang]["text"].tolist()
    for mode in ["raw", "nostop", "stem", "lemma"]:
        toks = [preprocess.tokens(t, lang, mode) for t in texts]
        rows.append({"language": lang, "step": mode, "tokens": sum(map(len, toks)),
                     "vocabulary": len(set(w for ts in toks for w in ts))})
vocab = pd.DataFrame(rows)
R["vocab"] = vocab.to_dict("records")
vocab.pivot(index="step", columns="language", values="vocabulary").loc[["raw", "nostop", "stem", "lemma"]]
""")
md("""
Stemming shrinks the vocabulary the most but produces non-words (ISRI turns many words into 3-letter roots,
merging unrelated meanings). For features we use **lemmas (English) and light stems (Arabic)**, a balance between
grouping variants and keeping meaning. Transformer models receive the **original text**, because their own
sub-word tokenizers were trained on raw text.
""")

md("""
## 4. Linguistic analysis: POS tagging and dependency parsing (spaCy)
We use the dependency tree to extract **what** is wrong (the aspect) and **how** (the opinion):
* `amod`: "slow **login**" → (login, slow)
* `nsubj` + `acomp`: "the **app** is useless" → (app, useless)
* a negated verb: "the **OTP** never arrives" → (otp, not arrive)
""")
code("""
from spacy import displacy
doc = linguistic.nlp()("The OTP never arrives and the new design is not good.")
displacy.render(doc, style="dep", jupyter=True, options={"compact": True, "distance": 95})
print("Aspects:", linguistic.aspects(doc))
print("SVO relations:", linguistic.svo(linguistic.nlp()("Support ignored my complaint about the payment.")))
""")
code("""
en = raw[raw.lang == "en"].copy()
en["label"] = en["rating"].map(config.rating_to_label)
docs = {lab: list(linguistic.nlp().pipe(en[en.label == lab]["text"].head(800))) for lab in ["negative", "positive"]}
pos = pd.DataFrame({lab: pd.Series(linguistic.pos_counts(d)) for lab, d in docs.items()}).fillna(0)
pos = (pos / pos.sum()).loc[["NOUN", "VERB", "ADJ", "ADV", "PRON", "PART", "PROPN", "AUX"]]
pos.plot.bar(figsize=(9, 3.5), title="POS distribution: complaints vs praise"); savefig("pos")
from collections import Counter
pairs = Counter(f"{a} → {o}" for d in docs["negative"] for a, o in linguistic.aspects(d))
matches = Counter(f"{lab}: {txt}" for d in docs["negative"] for lab, txt in linguistic.match_phrases(d) if lab != "ADV_ADJ_NOUN")
display(pd.DataFrame(pairs.most_common(15), columns=["aspect → opinion (complaints)", "count"]))
pd.DataFrame(matches.most_common(10), columns=["Matcher rule: phrase", "count"])
""")
md("""
Complaints use more **verbs and particles** ("can't", "not", "doesn't open"): people describe failed actions.
Praise uses more **adjectives** ("easy", "great"). This is why the aspect rules include the negated-verb pattern.
""")

md("## 5. N-grams: the phrases people repeat")
code("""
raw["label"] = raw["rating"].map(config.rating_to_label)
raw["tokens"] = [preprocess.tokens(t, l) for t, l in zip(raw.text, raw.lang)]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, lang in enumerate(["en", "ar"]):
    for j, lab in enumerate(["negative", "positive"]):
        sub = raw[(raw.lang == lang) & (raw.label == lab)]["tokens"]
        grams = linguistic.top_ngrams(sub, n=2, k=12)
        if grams:
            g = pd.DataFrame(grams, columns=["bigram", "n"])
            if lang == "ar":   # display Arabic right-to-left correctly in matplotlib
                try:
                    from bidi.algorithm import get_display; import arabic_reshaper
                    g["bigram"] = g["bigram"].map(lambda s: get_display(arabic_reshaper.reshape(s)))
                except ImportError:
                    pass
            axes[i, j].barh(g["bigram"][::-1], g["n"][::-1], color="#d1495b" if lab == "negative" else "#00798c")
        axes[i, j].set_title(f"Top bigrams: {lang} {lab}")
savefig("ngrams")
for n in (1, 3):
    print(f"Top {n}-grams in English complaints:", linguistic.top_ngrams(raw[(raw.lang == 'en') & (raw.label == 'negative')]['tokens'], n=n, k=8))
""")

md("""
## 6. Run the full NLP pipeline on every review
`cvoice/pipeline.py` applies, to each review: language detection → preprocessing → **sentiment** (routed: English
RoBERTa / Arabic CAMeLBERT / XLM-R for mixed) → **emotion** → **urgency score** → spaCy **aspects, relations, NER**
(English) and **CAMeLBERT NER** (Arabic) → **BGE-M3 embeddings** → **issue clustering** with LLM names.
Results are saved to `data/`, which is what the Streamlit app reads. On a T4 this takes a few minutes for ~7k reviews.
""")
code("""
t0 = time.time()
df, X = pipeline.run(full=True)
R["n_reviews"] = len(df); R["pipeline_minutes"] = round((time.time() - t0) / 60, 1)
print(f"{len(df)} reviews analysed in {R['pipeline_minutes']} min; embeddings: {X.shape}")
df[["app", "lang", "rating", "sentiment", "emotion", "urgency", "issue_label", "text"]].head(8)
""")

md("""
## 7. Text representations: BoW vs TF-IDF vs Word2Vec vs BGE-M3
A **fair comparison**: the same classifier (logistic regression) predicts the sentiment label (from stars) using
each representation; we also measure **precision@5**: for a review, how many of its 5 nearest neighbours share its
label. Train/test split is stratified by label **and** language, and it is reused for every model in the notebook.
""")
code("""
train_idx, test_idx = evaluate.split(df)
train, test = df.loc[train_idx], df.loc[test_idx]
R["n_train"], R["n_test"] = len(train), len(test)

# Word2Vec: try three settings, keep the best on the validation of the downstream task
w2v_rows = []
for vs, win in [(50, 3), (100, 5), (200, 8)]:
    m = evaluate.train_word2vec(train["tokens"], vector_size=vs, window=win)
    V = evaluate.w2v_doc_vectors(m, df["tokens"].tolist())
    pred, _ = evaluate.classify(V, df["label"].values, train_idx, test_idx)
    w2v_rows.append({"vector_size": vs, "window": win, **evaluate.scores(test["label"], pred), "model": m})
w2v = pd.DataFrame(w2v_rows)
best_w2v = w2v.loc[w2v["macro_f1"].idxmax(), "model"]
display(w2v.drop(columns="model"))
for w in ["otp", "login", "تطبيق"]:
    if w in best_w2v.wv:
        print(w, "→", [x for x, _ in best_w2v.wv.most_similar(w, topn=6)])
""")
code("""
rep_table, reps = evaluate.compare_representations(df, X, train_idx, test_idx, best_w2v)
R["representations"] = rep_table.to_dict("records")
rep_table
""")
code("""
from sklearn.decomposition import PCA, TruncatedSVD
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
sample = np.random.default_rng(0).choice(len(df), size=min(2000, len(df)), replace=False)
for ax, (name, M) in zip(axes, reps.items()):
    M = M[sample]
    P = (TruncatedSVD(2, random_state=0).fit_transform(M) if hasattr(M, "toarray") else PCA(2, random_state=0).fit_transform(M))
    for lab, col in [("negative", "#d1495b"), ("neutral", "#edae49"), ("positive", "#00798c")]:
        m = df["label"].values[sample] == lab
        ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, c=col, label=lab)
    ax.set_title(name)
axes[0].legend(markerscale=4); savefig("embeddings_pca")
""")
md("""
### Cross-lingual retrieval test
The same question asked in English and in Arabic should retrieve **the same reviews**. TF-IDF cannot do this
(Arabic and English share no words); a multilingual embedding model maps both into one semantic space.
""")
code("""
pairs = [("the OTP verification code does not arrive", "رمز التحقق لا يصل"),
         ("the app crashes after the update", "التطبيق يتوقف بعد التحديث"),
         ("face recognition does not work", "خاصية التعرف على الوجه لا تعمل"),
         ("payment failed but money was deducted", "فشل الدفع وتم خصم المبلغ"),
         ("excellent and easy to use app", "تطبيق ممتاز وسهل الاستخدام")]
tfv = evaluate.tfidf(evaluate.joined(train), evaluate.joined(df))[1]
X_tfidf = tfv.transform(evaluate.joined(df))
xl = evaluate.cross_lingual_test(df, pairs, models.embed, tfv, X_tfidf, X)
R["cross_lingual"] = xl.groupby("method")[["overlap@10", "arabic_share_for_en_query"]].mean().round(2).to_dict("index")
display(xl.groupby("method")[["overlap@10", "arabic_share_for_en_query"]].mean())
idx = search.ReviewIndex(df, X)
idx.search("the OTP verification code does not arrive", k=6)[["score", "app", "lang", "rating", "text"]]
""")

md("""
## 8. Sentiment analysis: five approaches on the same test set
| Approach | Family (lecture 3) | Languages |
|---|---|---|
| VADER | lexicon / rule-based | English only |
| TF-IDF + logistic regression | classical machine learning, trained on our data | both |
| Routed pre-trained: RoBERTa (EN) + CAMeLBERT-DA (AR) | deep learning, pre-trained, used as is | both |
| XLM-RoBERTa sentiment | one multilingual pre-trained model | both |
| **XLM-R + LoRA (ours)** | **fine-tuned on our reviews** (section 9) | both |

Ground truth comes from the stars (1-2 negative, 3 neutral, 4-5 positive). This is a **weak label**: some people
write a positive text and give 1★, so 100% is impossible.
""")
code("""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()
def vader_label(t):
    c = vader.polarity_scores(t)["compound"]
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"

test = test.copy()
test["pred_vader"] = test["text"].map(vader_label)   # only meaningful for English
pred_lr, _ = evaluate.classify(X_tfidf, df["label"].values, train_idx, test_idx)
test["pred_tfidf_lr"] = pred_lr
test["pred_routed"] = test["sentiment"]               # computed by the pipeline (no training on our data)
multi = models.sentiment_hf(test["text"].tolist(), config.SENTIMENT_MULTI)
test["pred_xlmr"] = multi[config.LABELS].idxmax(axis=1).values
""")

md("""
## 9. Fine-tuning: XLM-RoBERTa + LoRA (parameter-efficient)
*Pre-training* gave XLM-R general knowledge of 100 languages; *fine-tuning* adapts it to our task and domain
(government app complaints, Gulf dialect, OTP / Emirates ID vocabulary). With **LoRA** we freeze the 278M
pre-trained weights and train small low-rank matrices in the attention layers, under 1% of the parameters.
""")
code("""
ft = finetune.train_lora(train, test, epochs=3)
R["lora_trainable_pct"] = round(100 * ft["trainable_params"] / ft["total_params"], 2)
R["lora_minutes"] = round(ft["minutes"], 1)
test["pred_lora"] = ft["y_pred"]
display(ft["history"])
""")
code("""
from sklearn.metrics import ConfusionMatrixDisplay
methods = {"VADER (lexicon)": "pred_vader", "TF-IDF + LR": "pred_tfidf_lr",
           "RoBERTa / CAMeLBERT (pre-trained)": "pred_routed", "XLM-R (pre-trained)": "pred_xlmr",
           "XLM-R + LoRA (fine-tuned)": "pred_lora"}
rows = []
for name, col in methods.items():
    s = evaluate.scores_by_lang(test, col)
    if name.startswith("VADER"):
        s = {"en": s.get("en")}               # VADER is English-only
    rows.append({"method": name,
                 **{f"{k} {m}": v[m] for k, v in s.items() if v for m in ("accuracy", "macro_f1")}})
sent_table = pd.DataFrame(rows)
R["sentiment"] = sent_table.to_dict("records")
display(sent_table)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["pred_tfidf_lr", "pred_routed", "pred_lora"]):
    ConfusionMatrixDisplay.from_predictions(test["label"], test[col], labels=config.LABELS, ax=ax,
                                            colorbar=False, normalize="true", values_format=".2f")
    ax.set_title(col)
savefig("confusion")
""")
md("""
**How to read this:** the general pre-trained models already work without any training, but the **fine-tuned**
model learns the domain (and the star-label convention) and usually wins on macro-F1, especially for Arabic.
The neutral class is the hardest for every method: 3★ reviews mix praise and complaints.
""")

md("""
## 10. Emotion / mood detection
English: `j-hartmann/emotion-english-distilroberta-base` (Ekman's 6 emotions + neutral). Arabic: multilingual
zero-shot NLI (mDeBERTa) with the same labels. We also place each emotion on Russell's **valence-arousal**
circumplex: anger is negative + high arousal (urgent), sadness is negative + low arousal (disappointment).
""")
code("""
neg = df[df["is_negative"] == 1]
emo_app = pd.crosstab(neg["app"], neg["emotion"], normalize="index")
VA = {"anger": (-0.6, 0.8), "disgust": (-0.7, 0.3), "fear": (-0.6, 0.6), "joy": (0.8, 0.5),
      "neutral": (0.0, 0.0), "sadness": (-0.7, -0.4), "surprise": (0.3, 0.8)}
df["valence"] = sum(df[e] * VA[e][0] for e in config.EMOTIONS)
df["arousal"] = sum(df[e] * VA[e][1] for e in config.EMOTIONS)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
emo_app.plot.barh(stacked=True, ax=ax[0], colormap="tab10"); ax[0].set_title("Emotions in complaints, by app")
ax[0].legend(bbox_to_anchor=(1, 1))
va = df.groupby("app")[["valence", "arousal"]].mean()
ax[1].scatter(va["valence"], va["arousal"], s=120)
for app, (v, a) in va.iterrows(): ax[1].annotate(app, (v, a), xytext=(5, 5), textcoords="offset points")
ax[1].axhline(0, c="grey", lw=.5); ax[1].axvline(0, c="grey", lw=.5)
ax[1].set_xlabel("valence (negative → positive)"); ax[1].set_ylabel("arousal (calm → agitated)")
ax[1].set_title("Average mood per app (valence-arousal)")
savefig("emotions")
R["anger_share_complaints"] = round((neg["emotion"] == "anger").mean(), 3)
R["sadness_share_complaints"] = round((neg["emotion"] == "sadness").mean(), 3)
R["emotion_by_rating"] = pd.crosstab(df["rating"], df["emotion"], normalize="index").round(2).to_dict("index")
pd.crosstab(df["rating"], df["emotion"], normalize="index").round(2)
""")

md("""
## 11. Named Entity Recognition and relation extraction
Three NER systems: spaCy (English, statistical), CAMeLBERT-NER (Arabic, transformer) and an LLM with a structured
JSON prompt that also extracts **relations** (subject, relation, object) and app-specific types (feature,
error message, government service) that general NER models do not know.
""")
code("""
en_ents = Counter(lab for ents in df[df.lang == "en"]["entities"] for _, lab in ents)
ar_ents = Counter(f"{w} ({lab})" for ents in df[df.lang == "ar"]["entities"] for w, lab in ents)
print("spaCy entity types (English):", en_ents.most_common(8))
print("Top CAMeLBERT entities (Arabic):", ar_ents.most_common(10))
svo = Counter(" | ".join(t) for lst in df[df.is_negative == 1]["svo"] for t in lst)
pd.DataFrame(svo.most_common(12), columns=["subject | verb | object (spaCy, complaints)", "count"])
""")
code("""
sample_ner = pd.concat([df[(df.is_negative == 1) & (df.lang == "en")].nlargest(8, "urgency"),
                        df[(df.is_negative == 1) & (df.lang == "ar")].nlargest(8, "urgency")])
if llm.available():
    ext = genai.extract_entities(sample_ner, batch=4)
    if len(ext):
        for col in ["feature", "issue_type", "entities", "relations"]:
            if col not in ext:
                ext[col] = None
        ext = ext[["review_id", "feature", "issue_type", "entities", "relations"]]   # ignore extra keys the LLM adds
        classical = sample_ner[["review_id", "text", "entities"]].rename(columns={"entities": "classical_entities"})
        ext = ext.rename(columns={"entities": "llm_entities"}).merge(classical, on="review_id")
        ext["n classical"] = ext["classical_entities"].map(len)
        ext["n LLM"] = ext["llm_entities"].map(lambda e: len(e) if isinstance(e, list) else 0)
        R["ner_entities_per_review"] = {"classical": round(ext["n classical"].mean(), 2), "llm": round(ext["n LLM"].mean(), 2)}
        display(ext[["text", "feature", "issue_type", "llm_entities", "relations", "classical_entities"]].head(10))
""")

md("""
## 12. Automatic issue discovery
Complaints are clustered on their BGE-M3 embeddings (k-means; k chosen by **silhouette score**), each cluster is
described by **class-based TF-IDF** keywords, and the LLM writes a bilingual name. Because the embeddings are
multilingual, Arabic and English reports of the same problem fall in the **same** cluster.
""")
code("""
cl = topics.load()
R["n_clusters"] = cl["k"]
if cl["silhouette"]:
    pd.Series(cl["silhouette"]).plot(marker="o", figsize=(6, 3), title="Silhouette score vs k"); savefig("silhouette")
issues = pd.DataFrame([{k: c[k] for k in ["id", "label_en", "label_ar", "size", "avg_rating", "share_arabic", "keywords"]}
                       for c in cl["clusters"]]).sort_values("size", ascending=False)
R["issues"] = issues.head(8).drop(columns="keywords").to_dict("records")
R["mixed_language_clusters"] = int(((issues.share_arabic > 0.1) & (issues.share_arabic < 0.9)).sum())
issues
""")

md("""
## 13. Release regression detector
For each app version: complaint rate vs the rest of that app's reviews (two-proportion z-test). We test dozens of
versions, so p-values are corrected with **Benjamini-Hochberg** (false discovery rate 5%); otherwise some versions
would be flagged by pure chance. A flag means "this release made things significantly worse", and the cluster that
grew the most says *what* broke.
""")
code("""
vt = regression.version_table(df)
names = {c["id"]: c["label_en"] for c in cl["clusters"]}
R["n_versions_tested"] = len(vt)
R["n_versions_p05"] = int((vt["p"] < 0.05).sum()) if len(vt) else 0
print(f"{len(vt)} versions tested; {int((vt['p'] < 0.05).sum()) if len(vt) else 0} have p < 0.05 before correction, "
      f"{int(vt['flag'].sum()) if len(vt) else 0} remain flagged after Benjamini-Hochberg")
if len(vt):
    vt["issue_that_grew"] = vt["top_growing_issue"].map(lambda c: names.get(int(c)) if pd.notna(c) else None)
    R["release_flags"] = vt[vt.flag][["app", "version", "reviews", "neg_rate", "baseline", "z", "q", "issue_that_grew"]].to_dict("records")
    display(vt[vt.flag] if vt.flag.any() else "No significant release regressions in this data.")
    top_app = vt.groupby("app")["reviews"].sum().idxmax()
    v = vt[vt.app == top_app]
    plt.figure(figsize=(10, 3.5))
    plt.bar(v["version"].astype(str), v["neg_rate"], color=np.where(v["flag"], "#d1495b", "#8d99ae"))
    plt.axhline(df[df.app == top_app]["is_negative"].mean(), ls=":", c="k", label="app average")
    plt.title(f"{top_app}: complaint rate per version (red = flagged)"); plt.xticks(rotation=45); plt.legend()
    savefig("regression")
""")

md("""
## 14. Prompt engineering: three strategies
Same LLM, same reviews, only the **prompt** changes (lab: Prompt Engineering). Reviews are sent in batches of 15 to
respect the free-tier limits, and answers are cached on disk.
""")
code("""
if llm.available():
    # 8 reviews per (label, language) group = a small balanced evaluation set
    pe_sample = pd.concat([g.sample(min(len(g), 8), random_state=0) for _, g in test.groupby(["label", "lang"])])
    pe_rows = []
    for strat in genai.STRATEGIES:
        preds = pd.Series(genai.classify_with_prompt(pe_sample["text"].tolist(), strat), index=pe_sample.index)
        ok = preds.notna()          # reviews the LLM actually answered (failed batches are not guessed)
        if ok.sum() == 0:
            print(f"{strat}: no answers, left out of the table"); continue
        pe_rows.append({"strategy": strat, "coverage": round(ok.mean(), 2),
                        **evaluate.scores(pe_sample.loc[ok, "label"], preds[ok])})
    pe_rows.append({"strategy": "fine-tuned XLM-R + LoRA (same reviews)", "coverage": 1.0,
                    **evaluate.scores(pe_sample["label"], pe_sample["pred_lora"])})
    pe = pd.DataFrame(pe_rows); R["prompting"] = pe.to_dict("records"); R["llm_model"] = llm.last_model_used
    print("LLM:", llm.last_model_used, "| reviews:", len(pe_sample)); display(pe)
    print(genai.STRATEGIES["role_definitions"])
""")

md("""
## 15. "Ask Citizen Voice": RAG with a faithfulness check
1. **Retrieve** the 12 most relevant reviews with BGE-M3 (works for Arabic or English questions).
2. **Generate** an answer that may only use those reviews and must cite them ([R3]).
3. **Verify**: a second LLM call (LLM-as-a-judge) checks every claim against the cited reviews.

We compare against asking the LLM **without retrieval**, to show why grounding matters.
""")
code("""
questions = ["Why are people unhappy with UAE PASS login?", "What payment problems do users report?",
             "ما هي أكثر مشاكل تطبيق الهوية والإقامة؟"]
rag_rows = []
if llm.available():
    for q in questions:
        try:
            out = search.ask(idx, q, k=12)
            no_ctx = llm.ask(f"Answer briefly (max 120 words): {q}")
            judge_noctx = search.judge(no_ctx, search._context(out["hits"])[0])
        except llm.LLMError as e:          # free models busy: skip this question, keep the others
            print("Skipped:", q, "|", str(e)[:200]); continue
        rag_rows.append({"question": q, "faithfulness_rag": out["judge"].get("faithfulness"),
                         "faithfulness_no_retrieval": judge_noctx.get("faithfulness"),
                         "citations": len(out["cited"]), "invalid_citations": len(out["invalid_citations"])})
        if "error" in out["judge"]:
            print("Judge failed:", out["judge"]["error"][:200])
        print("Q:", q, "\\n", out["answer"], "\\n", "-" * 80)
    if rag_rows:
        rag = pd.DataFrame(rag_rows); R["rag"] = rag.to_dict("records"); display(rag)
""")

md("## 16. Bilingual replies and the executive briefing")
code("""
if llm.available():
    for _, r in df[~df.has_reply].nlargest(2, "urgency").iterrows():
        try:
            rep = genai.draft_reply(r.to_dict())
        except llm.LLMError as e:
            print("Reply skipped:", str(e)[:200]); continue
        print(f"REVIEW ({r.app}, {r.rating}★, urgency {r.urgency}): {r.text[:200]}\\nEN: {rep.get('en')}\\nAR: {rep.get('ar')}\\n")
    stats = {"reviews": len(df), "complaint_rate": R["share_negative"], "issues": R["issues"][:5],
             "release_alerts": R.get("release_flags", []), "anger_share_in_complaints": R["anger_share_complaints"]}
    try:
        print(genai.executive_briefing(stats))
    except llm.LLMError as e:
        print("Briefing skipped:", str(e)[:200])
""")

md("""
## 17. Export results for the report, then launch the app
`results.json` and `report/results.tex` (LaTeX macros) contain every number; the figures are in `figures/`.
Download `report_assets.zip` and upload its content to your Overleaf project: the report reads the numbers
automatically, so they always match your run.
""")
code("""
from cvoice import report_export
zpath = report_export.export(R)
print(open("report/results.tex").read()[:1200])
if IN_COLAB:
    from google.colab import files; files.download(str(zpath))
""")
code("""
# Launch the Streamlit app with a public link (Cloudflare quick tunnel). Stop with the ■ button.
if IN_COLAB:
    sh("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared")
    subprocess.Popen("streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1", shell=True)
    time.sleep(8)
    tunnel = subprocess.Popen("./cloudflared tunnel --url http://localhost:8501 --no-autoupdate",
                              shell=True, stderr=subprocess.PIPE, text=True)
    import re
    for line in tunnel.stderr:
        m = re.search(r"https://[\\w-]+\\.trycloudflare\\.com", line)
        if m:
            print("🌍 Open the app:", m.group(0)); break
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "notebooks" / "Citizen_Voice_Colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
