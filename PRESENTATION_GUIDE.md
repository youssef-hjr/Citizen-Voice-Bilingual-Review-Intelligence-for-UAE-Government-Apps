# Citizen Voice: presentation and defense guide

## The 30-second pitch
Residents review UAE government apps every day, in English and in Arabic. Nobody can read thousands of reviews, so
problems are found late. **Citizen Voice** collects the reviews automatically and tells the app team what is broken,
how upset people are, which release caused it, and what to answer. It supports the *Smart Government* priority of the
UAE AI Strategy 2031, and it treats Arabic as a first-class language.

## Demo script (5 minutes, in the app)
1. **Overview:** the KPIs and the sentiment-by-app chart. Open the expander to show model agreement with the star ratings.
2. **Issues:** the automatically discovered clusters with English and Arabic names. Point at the *% Arabic* column:
   one cluster mixes both languages, because BGE-M3 understands both. Then show the aspect → opinion chart
   (dependency parsing).
3. **Releases:** a flagged version and "the issue that grew most".
4. **Search:** type *OTP code never arrives* and show the Arabic results. Then type *رمز التحقق لا يصل*.
5. **Ask Citizen Voice:** ask *Why are people unhappy with UAE PASS login?* Show the citations and the faithfulness check.
6. **Triage & Reply:** pick the most urgent review and draft the English and Arabic reply.
7. **Analyze a review:** paste a review and walk through every step: tokens, stems, lemmas, POS, dependency tree,
   entities, sentiment, emotion, urgency.

## Where each course concept lives
| Concept | Where | One-line explanation |
|---|---|---|
| Tokenization | `preprocess.tokens_en` (NLTK), `tokens_ar` (CAMeL) | split text into words; NLTK splits "can't" into "ca" + "n't", which we repair |
| Stop words | `preprocess` | removed, **except negations** (not, لا, لم, لن, ليس), otherwise "not good" becomes "good" |
| Stemming | Porter, Snowball, Lancaster (EN); ISRI (AR) | chops suffixes; fast but makes non-words (verif, servic) |
| Lemmatization | WordNet + spaCy (EN); light stemming (AR) | dictionary form, keeps meaning; used for our features |
| Arabic normalization | CAMeL Tools | أ/إ/آ → ا, ى → ي, ة → ه, remove diacritics and tatweel, collapse راااائع |
| POS tagging | spaCy | complaints contain more verbs and particles ("can't open"); praise more adjectives |
| Dependency parsing | `linguistic.aspects` | amod, nsubj + acomp, and negated-verb rules give aspect → opinion pairs |
| Matcher | `linguistic.matcher` | rule patterns such as CANT_DO + VERB, error messages |
| Relation extraction | `linguistic.svo` + LLM triples | subject, verb, object, e.g. (money, deduct, card) |
| N-grams | notebook §5, app Issues tab | frequent bigrams per language and class |
| BoW / TF-IDF | `evaluate.bow/tfidf` | counts / weighted counts; strong on keywords, zero cross-lingual ability |
| Word2Vec | `evaluate.train_word2vec` | skip-gram trained on our reviews, 3 settings compared |
| Sentence embeddings | BGE-M3 (`models.embed`) | one 1024-d vector per review; English and Arabic share the same space |
| Similarity search | `search.ReviewIndex` | cosine similarity = dot product of normalised vectors |
| Sentiment: lexicon | VADER | English-only rules, no training |
| Sentiment: ML | TF-IDF + logistic regression | trained on our data |
| Sentiment: pre-trained | RoBERTa (EN), CAMeLBERT (AR), XLM-R | transformers used as they are |
| Pre-training vs fine-tuning | `finetune.train_lora` | XLM-R learned 100 languages in pre-training; we adapt it to our task |
| Parameter-efficient tuning | LoRA r=16 on query/value | under 1% of the weights are trained; the adapter is a few MB |
| Emotion / mood | DistilRoBERTa (EN), zero-shot mDeBERTa (AR) | Ekman emotions, mapped onto Russell's valence-arousal plane |
| NER | spaCy, CAMeLBERT-NER, LLM | the LLM also knows domain types such as GOV_SERVICE and ERROR_MESSAGE |
| Prompt engineering | `genai.STRATEGIES` | zero-shot vs role + definitions vs few-shot, measured on the same reviews |
| LLM generation | `genai.draft_reply`, `executive_briefing` | role, constraints, JSON output, bilingual |
| RAG | `search.ask` | retrieve real reviews, answer only from them, cite [R#] |
| Hallucination control | `search.judge` | a second LLM checks every claim (LLM-as-a-judge) |

## Likely questions, and good answers
**Why star ratings as labels?** They are free and given by the users themselves, so we did not need manual
annotation. They are a *weak* label: some people write positive text and give 1★. We report macro-F1 and discuss this
limit; a human-labelled sample is the next step.

**Why macro-F1 and not accuracy?** Reviews are mostly 1★ or 5★. A model that never predicts *neutral* can still get
high accuracy; macro-F1 averages the F1 of each class, so the rare class counts equally.

**Why BGE-M3 and not TF-IDF for search?** TF-IDF needs the same words, so an English query never finds an Arabic
review (cross-lingual overlap 0). BGE-M3 is trained on more than 100 languages to place translations close together.

**Why fine-tune if pre-trained models already work?** Pre-trained sentiment models learned from tweets with their
own label definitions. Fine-tuning teaches the model our domain (OTP, Emirates ID, renewals) and our label
convention. LoRA makes that possible on a free GPU in minutes.

**What does LoRA do exactly?** It freezes the original weight matrix W and learns W + B·A, where B and A are thin
matrices of rank r = 16. Only A, B and the classifier head are trained.

**Why keep negations in stop-word lists?** Standard lists contain "not", "no" and لا. Removing them turns
"not working" into "working" and flips the sentiment.

**How does the release detector decide?** For each version it compares the complaint rate with the rest of the app's
reviews using a two-proportion z-test. Because we test dozens of versions at once, about 1 in 20 would look
"significant" at p < 0.05 by pure chance, so we correct the p-values with the **Benjamini-Hochberg** procedure
(false discovery rate 5%). We also require at least 10 percentage points more complaints, so tiny differences are
not flagged. In the final run 65 versions were tested: 15 had p < 0.05 on their own, 10 survive the correction.
Most are consecutive RTA Dubai releases, which points to a lasting problem rather than one bad update.

**How do you stop the LLM from inventing things?** Three ways. Retrieval: it only sees real reviews. Constraints: it
must cite review ids and is told never to invent numbers or contacts. Verification: a second call checks every
claim, and the app shows the faithfulness score and any invalid citations.

**What about privacy and ethics?** Reviews are public, but we never store user names (only a short one-way hash) and
only show aggregates. Replies are drafts for a human to approve.

**Why is LoRA best on accuracy but not on macro-F1?** It learns the big classes (1★ and 5★) and Arabic very well
(accuracy 0.88, Arabic 0.90), but three epochs on a few hundred 3★ reviews are not enough to learn *neutral*.
XLM-T was already pre-trained on a three-class sentiment task, so it keeps the best macro-F1 (0.635). That is the
pre-training vs fine-tuning trade-off in one table.

**Limitations?** Star labels are noisy; the dependency rules and spaCy NER are English-only; Arabic emotion uses
zero-shot NLI; dialect spelling and Arabizi are hard; free LLMs change over time (we cache answers for
reproducibility).

**Future work?** An Arabic dependency parser for Arabic aspects, a human-annotated test set, Falcon-Arabic (TII) for
replies, and automatic alerts to app teams when a regression is detected.
