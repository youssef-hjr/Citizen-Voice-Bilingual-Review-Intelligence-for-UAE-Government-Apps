# Citizen Voice: Bilingual Review Intelligence for UAE Government Apps

Youssef Hajri, Student ID 1425158
MAAI902 Natural Language Processing and LLMs, University of Wollongong in Dubai

Citizen Voice collects Google Play reviews of UAE government apps (UAE PASS, RTA Dubai, DEWA, DubaiNow, Dubai Police)
in English and Arabic, and tells a digital services team what is broken, how upset people are, which release caused it,
and how to answer.

Collect → Clean → Represent → Analyse → Explain

## Main results (5,271 reviews, 41.5% Arabic)

| Experiment | Result |
|---|---|
| Best representation | BGE-M3: macro-F1 0.633, Precision@5 0.866 |
| Cross-lingual search overlap@10 | BGE-M3 0.36 vs TF-IDF 0 |
| Sentiment, pre-trained XLM-R | macro-F1 0.643 |
| Sentiment, XLM-R + LoRA (0.42% of weights trained) | accuracy 0.899, macro-F1 0.615 |
| Prompting on 50 reviews | zero-shot 0.569, role and definitions 0.594, few-shot 0.655 |
| RAG faithfulness | 1.00 with retrieval vs 0.57 without |
| Issue discovery | 13 clusters, 11 mix English and Arabic |
| Release regression detector | 65 versions tested, 10 flagged after Benjamini-Hochberg |

## What is in this repository

| Path | Content |
|---|---|
| `notebooks/Citizen_Voice_Colab_results.ipynb` | the executed notebook with all outputs |
| `notebooks/Citizen_Voice_Colab.ipynb` | clean notebook to run again |
| `app.py`, `cvoice/` | Streamlit app and the NLP package |
| `docs/dashboard.html` | static results dashboard (download and open in a browser) |
| `docs/Citizen_Voice_Presentation.pptx` | presentation slides |
| `tests/` | offline tests |

## Features

| Feature | NLP / LLM technique |
|---|---|
| Automated, incremental review collection | `google-play-scraper`, deduplication, no personal data stored |
| Bilingual preprocessing | NLTK (tokenize, stop words, Porter, WordNet), spaCy lemmas, **CAMeL Tools** Arabic normalization, ISRI + light Arabic stemming |
| What exactly is wrong | spaCy POS + **dependency parsing** aspect → opinion rules, Matcher rules, SVO relation extraction, n-grams |
| Sentiment | VADER, TF-IDF + logistic regression, RoBERTa (EN) + **CAMeLBERT** (AR), XLM-R, **XLM-R fine-tuned with LoRA** |
| Emotion and urgency | DistilRoBERTa emotions (EN), multilingual zero-shot NLI (AR), valence-arousal, urgency score |
| Entities and relations | spaCy NER, CAMeLBERT NER, LLM structured extraction (entities + relation triples) |
| Cross-lingual semantic search | **BGE-M3** multilingual embeddings (vs BoW, TF-IDF, Word2Vec) |
| Automatic issue discovery | k-means on embeddings + silhouette, class-based TF-IDF keywords, LLM bilingual names |
| Release regression detector | two-proportion z-test per app version + which issue grew |
| Ask Citizen Voice | **RAG** with citations + LLM-as-a-judge **faithfulness check** |
| Replies and briefing | prompt engineering (role, constraints, JSON output), English + Arabic |

## Run it (Google Colab, recommended)

1. Open `notebooks/Citizen_Voice_Colab.ipynb` in Colab and pick **Runtime → Change runtime type → T4 GPU**.
2. Add a Colab secret `OPENROUTER_API_KEY` (key icon on the left, then turn on notebook access). Get a free key at openrouter.ai.
3. **Runtime → Run all.** When asked, upload `citizen-voice.zip`.
   The notebook collects the data, runs every experiment, trains the LoRA model, downloads `report_assets.zip`
   (figures + numbers for the report), and prints a public link to the app at the end.

Total time on a T4 is about 20 to 30 minutes with the default 600 reviews per app per language.
Lower `PER_LANG` in section 1 (for example to 200) for a quick test run.

**Free LLM limits:** OpenRouter free models allow about 50 requests a day without credits. The notebook uses
about 30 and caches every answer in `data/llm_cache.json`, so re-running costs nothing.

## Run it on your computer

```bash
pip install torch -r requirements.txt camel-tools
python -m spacy download en_core_web_sm
python -m cvoice.collect          # collect reviews (incremental; schedule it with cron to keep data fresh)
python -m cvoice.pipeline         # analyse new reviews, rebuild issue clusters
streamlit run app.py
```
Put your key in `.streamlit/secrets.toml` (`OPENROUTER_API_KEY = "sk-or-..."`) or paste it in the app sidebar.
To use the LoRA model in the app, copy the `models/` folder from Colab.

## Project layout

```
cvoice/config.py        apps, model names, paths
cvoice/collect.py       automated Play Store collector
cvoice/preprocess.py    English + Arabic preprocessing (CAMeL Tools)
cvoice/linguistic.py    spaCy POS, dependency aspects, SVO relations, Matcher, n-grams
cvoice/models.py        BGE-M3, sentiment, emotion, Arabic NER, LoRA inference
cvoice/finetune.py      XLM-R + LoRA training loop
cvoice/topics.py        issue clustering + class-based TF-IDF + LLM naming
cvoice/regression.py    release regression detector
cvoice/search.py        semantic search + RAG + faithfulness judge
cvoice/genai.py         LLM NER/relations, prompt strategies, replies, briefing
cvoice/llm.py           OpenRouter client (free-model fallback, retries, cache)
cvoice/pipeline.py      end-to-end enrichment
cvoice/evaluate.py      experiment helpers (representations, metrics)
app.py              Streamlit app (7 tabs)
notebooks/          Colab notebook with all experiments
tests/              offline test suite (pytest) with synthetic data and tiny models
```

## Tests

```bash
pytest -q                          # 38 tests: preprocessing, rules, pipeline, LoRA, app (all offline)
python tests/run_notebook_offline.py   # executes every notebook cell with stand-in models
```
