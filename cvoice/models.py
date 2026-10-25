from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config


def device() -> int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cuda" if device() == 0 else "cpu")
    model.max_seq_length = 256
    if device() == 0:
        model.half()
    return model


def embed(texts, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    vecs = embedder().encode(list(map(str, texts)), batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.astype(np.float32)


def _pipe(task: str, model: str, **kw):
    from transformers import pipeline
    return pipeline(task, model=model, device=device(), **kw)


def _norm_label(label: str, id_order=config.LABELS) -> str:
    l = label.lower()
    if l.startswith("label_"):
        return id_order[int(l.split("_")[1])]
    for name in config.LABELS:
        if l.startswith(name[:3]):
            return name
    return l


def _scores_to_frame(outputs, label_names) -> pd.DataFrame:
    rows = []
    for out in outputs:
        out = out if isinstance(out, list) else [out]
        rows.append({_norm_label(d["label"]): float(d["score"]) for d in out})
    df = pd.DataFrame(rows).reindex(columns=label_names).fillna(0.0)
    return df


@lru_cache(maxsize=None)
def sentiment_pipe(name: str):
    return _pipe("text-classification", name, top_k=None)


def sentiment_hf(texts, model: str, batch_size: int = 32) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame(columns=config.LABELS)
    outs = sentiment_pipe(model)(list(map(str, texts)), batch_size=batch_size, truncation=True, max_length=256)
    return _scores_to_frame(outs, config.LABELS)


def sentiment_routed(texts, langs, batch_size: int = 32) -> pd.DataFrame:
    texts, langs = list(map(str, texts)), list(langs)
    res = pd.DataFrame(0.0, index=range(len(texts)), columns=config.LABELS)
    used = pd.Series("", index=range(len(texts)))
    for lang_set, model in [({"en", "other"}, config.SENTIMENT_EN), ({"ar"}, config.SENTIMENT_AR),
                            ({"mixed"}, config.SENTIMENT_MULTI)]:
        idx = [i for i, l in enumerate(langs) if l in lang_set]
        if idx:
            res.loc[idx, config.LABELS] = sentiment_hf([texts[i] for i in idx], model, batch_size).values
            used.loc[idx] = model.split("/")[-1]
    res["label"] = res[config.LABELS].idxmax(axis=1)
    res["model"] = used
    return res


@lru_cache(maxsize=1)
def lora_pipe():
    import json

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    base_name = json.loads((config.LORA_DIR / "adapter_config.json").read_text())["base_model_name_or_path"]
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=len(config.LABELS), id2label=dict(enumerate(config.LABELS)),
        label2id={l: i for i, l in enumerate(config.LABELS)})
    model = PeftModel.from_pretrained(base, str(config.LORA_DIR)).merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(config.LORA_DIR))
    return pipeline("text-classification", model=model, tokenizer=tok, device=device(), top_k=None)


def sentiment_lora(texts, batch_size: int = 32) -> pd.DataFrame:
    outs = lora_pipe()(list(map(str, texts)), batch_size=batch_size, truncation=True, max_length=256)
    df = _scores_to_frame(outs, config.LABELS)
    df["label"] = df[config.LABELS].idxmax(axis=1)
    return df


def lora_available() -> bool:
    return (config.LORA_DIR / "adapter_config.json").exists()


@lru_cache(maxsize=1)
def emotion_en_pipe():
    return _pipe("text-classification", config.EMOTION_EN, top_k=None)


@lru_cache(maxsize=1)
def emotion_zs_pipe():
    return _pipe("zero-shot-classification", config.EMOTION_ZS)


def emotion(texts, langs, batch_size: int = 32) -> pd.DataFrame:
    texts, langs = list(map(str, texts)), list(langs)
    res = pd.DataFrame(0.0, index=range(len(texts)), columns=config.EMOTIONS)
    en = [i for i, l in enumerate(langs) if l in ("en", "other")]
    other = [i for i, l in enumerate(langs) if l not in ("en", "other")]
    if en:
        outs = emotion_en_pipe()([texts[i] for i in en], batch_size=batch_size, truncation=True, max_length=256)
        for i, out in zip(en, outs):
            for d in out:
                res.loc[i, d["label"].lower()] = float(d["score"])
    if other:
        outs = emotion_zs_pipe()([texts[i] for i in other], candidate_labels=config.EMOTIONS,
                                 hypothesis_template="The emotion expressed in this text is {}.",
                                 batch_size=batch_size)
        outs = outs if isinstance(outs, list) else [outs]
        for i, out in zip(other, outs):
            for lab, sc in zip(out["labels"], out["scores"]):
                res.loc[i, lab] = float(sc)
    res["emotion"] = res[config.EMOTIONS].idxmax(axis=1)
    return res


@lru_cache(maxsize=1)
def ner_ar_pipe():
    return _pipe("token-classification", config.NER_AR, aggregation_strategy="simple")


def ner_ar(texts, batch_size: int = 32) -> list[list[tuple[str, str]]]:
    if not len(texts):
        return []
    outs = ner_ar_pipe()(list(map(str, texts)), batch_size=batch_size)
    return [[(e["word"], e["entity_group"]) for e in out] for out in outs]
