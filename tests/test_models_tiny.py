import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cvoice import config, models, finetune
import fake_data

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz") + \
        ["the", "app", "otp", "login", "not", "good", "bad", "great", "never", "error", "##s", "##ing"] + \
        list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")


def _tiny(path, id2label, kind="seq"):
    from transformers import (BertConfig, BertForSequenceClassification, BertForTokenClassification,
                              BertTokenizerFast)
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(VOCAB))
    tok = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=300,
                     id2label=id2label, label2id={v: k for k, v in id2label.items()})
    from transformers import BertModel
    cls = {"seq": BertForSequenceClassification, "tok": BertForTokenClassification, "base": BertModel}[kind]
    cls(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    return {
        "sent": _tiny(d / "sent", {0: "negative", 1: "neutral", 2: "positive"}),
        "sent_ar": _tiny(d / "sent_ar", {0: "positive", 1: "negative", 2: "neutral"}),
        "sent_old": _tiny(d / "sent_old", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}),
        "emo": _tiny(d / "emo", dict(enumerate(config.EMOTIONS))),
        "nli": _tiny(d / "nli", {0: "entailment", 1: "neutral", 2: "contradiction"}),
        "ner": _tiny(d / "ner", {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-ORG", 4: "I-ORG"}, kind="tok"),
        "base": _tiny(d / "base", {0: "LABEL_0", 1: "LABEL_1"}, kind="base"),
        "dir": d,
    }


@pytest.fixture
def patched(tiny, monkeypatch):
    monkeypatch.setattr(config, "SENTIMENT_EN", tiny["sent"])
    monkeypatch.setattr(config, "SENTIMENT_AR", tiny["sent_ar"])
    monkeypatch.setattr(config, "SENTIMENT_MULTI", tiny["sent_old"])
    monkeypatch.setattr(config, "EMOTION_EN", tiny["emo"])
    monkeypatch.setattr(config, "EMOTION_ZS", tiny["nli"])
    monkeypatch.setattr(config, "NER_AR", tiny["ner"])
    monkeypatch.setattr(config, "EMBEDDING_MODEL", tiny["sent"])
    monkeypatch.setattr(config, "LORA_BASE", tiny["base"])
    monkeypatch.setattr(config, "LORA_DIR", tiny["dir"] / "lora")
    for f in [models.embedder, models.sentiment_pipe, models.emotion_en_pipe, models.emotion_zs_pipe,
              models.ner_ar_pipe, models.lora_pipe]:
        f.cache_clear()
    return tiny


TEXTS = ["the app is great", "otp never arrives error", "التطبيق سيء", "login not good لا يعمل"]
LANGS = ["en", "en", "ar", "mixed"]


def test_sentiment_routed_shapes(patched):
    s = models.sentiment_routed(TEXTS, LANGS)
    assert list(s.columns[:3]) == config.LABELS and len(s) == 4
    assert s[config.LABELS].sum(axis=1).round(3).eq(1).all()
    assert set(s["label"]) <= set(config.LABELS)
    assert s.loc[2, "model"] == "sent_ar" and s.loc[3, "model"] == "sent_old"


def test_label_normalisation():
    assert models._norm_label("LABEL_2") == "positive"
    assert models._norm_label("Negative") == "negative"


def test_emotion_both_paths(patched):
    e = models.emotion(TEXTS, LANGS)
    assert set(e["emotion"]) <= set(config.EMOTIONS)
    assert e.loc[2, config.EMOTIONS].sum() == pytest.approx(1, abs=1e-3)


def test_embed_normalised(patched):
    import numpy as np
    X = models.embed(TEXTS)
    assert X.shape[0] == 4 and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-3)


def test_ner_ar(patched):
    out = models.ner_ar(["زرت دبي امس"])
    assert isinstance(out, list) and len(out) == 1


def test_lora_train_and_reload(patched):
    df = fake_data.make_reviews(60)
    df["label"] = df["rating"].map(config.rating_to_label)
    res = finetune.train_lora(df.iloc[:45], df.iloc[45:], epochs=1, batch_size=8, max_len=32,
                              log=lambda *_: None)
    assert 0 < res["trainable_params"] < res["total_params"]
    assert len(res["y_pred"]) == 15 and set(res["y_pred"]) <= set(config.LABELS)
    assert models.lora_available()
    s = models.sentiment_lora(TEXTS)
    assert set(s["label"]) <= set(config.LABELS) and len(s) == 4
