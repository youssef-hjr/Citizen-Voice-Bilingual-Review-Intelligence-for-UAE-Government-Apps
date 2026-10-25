from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

AR_CHARS = r"؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
_AR = re.compile(f"[{AR_CHARS}]")
_LAT = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    text = str(text)
    ar, lat = len(_AR.findall(text)), len(_LAT.findall(text))
    if ar + lat == 0:
        return "other"
    share = ar / (ar + lat)
    if share >= 0.7:
        return "ar"
    if share <= 0.3:
        return "en"
    return "mixed"


_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")


def _demojize(text: str) -> str:
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" emoji_", " "))
    except ImportError:
        return text


NEGATIONS_EN = {"not", "no", "nor", "never", "n't", "cannot", "dont", "don't", "cant", "can't",
                "won't", "wont", "isn't", "doesn't", "didn't", "wasn't", "aren't", "couldn't"}


@lru_cache(maxsize=1)
def _nltk():
    import nltk
    for pkg, probe in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords"),
                       ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(probe)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.stem.isri import ISRIStemmer
    from nltk.tokenize import word_tokenize
    return {
        "tokenize": word_tokenize,
        "stop_en": set(stopwords.words("english")) - NEGATIONS_EN,
        "stop_ar_raw": set(stopwords.words("arabic")),
        "porter": PorterStemmer(),
        "wnl": WordNetLemmatizer(),
        "isri": ISRIStemmer(),
    }


def clean_en(text: str) -> str:
    text = _URL.sub(" ", str(text))
    text = _demojize(text)
    text = text.replace("’", "'").lower()
    text = _REPEAT.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens_en(text: str, mode: str = "lemma") -> list[str]:
    t = _nltk()
    toks = [w for w in t["tokenize"](clean_en(text)) if re.search(r"[a-z]", w)]
    toks = [{"n't": "not", "ca": "can", "wo": "will"}.get(w, w) for w in toks]
    if mode == "raw":
        return toks
    toks = [w for w in toks if w not in t["stop_en"] and len(w) > 1 and not w.startswith("'")]
    if mode == "stem":
        return [t["porter"].stem(w) for w in toks]
    if mode == "lemma":
        return [t["wnl"].lemmatize(t["wnl"].lemmatize(w), pos="v") for w in toks]
    return toks


try:
    from camel_tools.utils.dediac import dediac_ar
    from camel_tools.utils.normalize import (normalize_alef_ar, normalize_alef_maksura_ar,
                                             normalize_teh_marbuta_ar, normalize_unicode)
    from camel_tools.tokenizers.word import simple_word_tokenize
    CAMEL_AVAILABLE = True
except Exception:
    CAMEL_AVAILABLE = False
    _DIAC = re.compile("[ً-ْٰـ]")

    def normalize_unicode(s):
        return unicodedata.normalize("NFKC", s)

    def dediac_ar(s):
        return _DIAC.sub("", s)

    def normalize_alef_ar(s):
        return re.sub("[إأآٱ]", "ا", s)

    def normalize_alef_maksura_ar(s):
        return s.replace("ى", "ي")

    def normalize_teh_marbuta_ar(s):
        return s.replace("ة", "ه")

    def simple_word_tokenize(s):
        return re.findall(r"\w+|[^\w\s]", s)

TATWEEL = "ـ"
NEGATIONS_AR = {"لا", "لم", "لن", "ما", "ليس", "غير", "مش", "مو", "ماهو", "بدون"}
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_ar(text: str, teh_marbuta: bool = True) -> str:
    s = normalize_unicode(str(text))
    s = dediac_ar(s).replace(TATWEEL, "")
    s = normalize_alef_ar(s)
    s = normalize_alef_maksura_ar(s)
    if teh_marbuta:
        s = normalize_teh_marbuta_ar(s)
    s = s.translate(AR_DIGITS)
    return _REPEAT.sub(r"\1\1", s)


def clean_ar(text: str) -> str:
    s = _URL.sub(" ", str(text))
    s = _demojize(s)
    s = normalize_ar(s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _stop_ar() -> set[str]:
    raw = _nltk()["stop_ar_raw"]
    return {normalize_ar(w) for w in raw} - {normalize_ar(w) for w in NEGATIONS_AR}


def tokens_ar(text: str, mode: str = "lemma") -> list[str]:
    toks = [w for w in simple_word_tokenize(clean_ar(text)) if _AR.search(w) or w.startswith("emoji_")]
    if mode == "raw":
        return toks
    stop = _stop_ar()
    toks = [w for w in toks if w not in stop and len(w) > 1]
    if mode == "stem":
        isri = _nltk()["isri"]
        return [w if w.startswith("emoji_") else isri.stem(w) for w in toks]
    if mode == "lemma":
        return [light_stem_ar(w) for w in toks]
    return toks


_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("ها", "هم", "كم", "نا", "ات", "ين", "ون")


def light_stem_ar(w: str) -> str:
    if w.startswith("emoji_"):
        return w
    for p in _PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    for s in _SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[: -len(s)]
            break
    return w


def clean(text: str, lang: str | None = None) -> str:
    lang = lang or detect_lang(text)
    return clean_ar(text) if lang == "ar" else clean_en(text)


def tokens(text: str, lang: str | None = None, mode: str = "lemma") -> list[str]:
    lang = lang or detect_lang(text)
    if lang == "ar":
        return tokens_ar(text, mode)
    if lang == "mixed":
        return tokens_ar(text, mode) + tokens_en(text, mode)
    return tokens_en(text, mode)


def explain(text: str) -> dict:
    lang = detect_lang(text)
    return {
        "language": lang,
        "cleaned": clean(text, lang),
        "tokens": tokens(text, lang, "raw"),
        "without_stopwords": tokens(text, lang, "nostop"),
        "stems": tokens(text, lang, "stem"),
        "lemmas": tokens(text, lang, "lemma"),
        "camel_tools": CAMEL_AVAILABLE,
    }
