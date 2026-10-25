from __future__ import annotations

from collections import Counter
from functools import lru_cache

from . import config

NEG_DEPS = {"neg"}
NEG_WORDS = {"not", "n't", "never", "no", "cannot"}
NON_OPINION = {"last", "latest", "new", "old", "other", "same", "first", "next", "previous", "recent", "current",
               "many", "few", "much", "more", "most", "several", "whole", "own", "different", "certain", "such",
               "only", "main", "single", "full", "official", "mobile", "digital", "personal", "online"}


@lru_cache(maxsize=1)
def nlp():
    import spacy
    try:
        return spacy.load(config.SPACY_MODEL)
    except OSError:
        from spacy.cli import download
        download(config.SPACY_MODEL)
        return spacy.load(config.SPACY_MODEL)


def _negated(tok) -> bool:
    return any(c.dep_ in NEG_DEPS or c.lower_ in NEG_WORDS for c in tok.children)


def _noun_phrase(tok) -> str:
    parts = [c.lemma_.lower() for c in tok.lefts if c.dep_ == "compound"]
    return " ".join(parts + [tok.lemma_.lower()])


def aspects(doc) -> list[tuple[str, str]]:
    pairs = []
    for tok in doc:
        if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN") and tok.lemma_.lower() not in NON_OPINION:
            op = ("not " if _negated(tok) else "") + tok.lemma_.lower()
            pairs.append((_noun_phrase(tok.head), op))
        elif tok.dep_ in ("acomp", "attr") and tok.pos_ == "ADJ":
            verb = tok.head
            subj = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass") and c.pos_ in ("NOUN", "PROPN")]
            neg = _negated(verb) or _negated(tok)
            for s in subj:
                pairs.append((_noun_phrase(s), ("not " if neg else "") + tok.lemma_.lower()))
        elif tok.pos_ == "VERB" and _negated(tok) and tok.lemma_.lower() not in ("be", "have", "do"):
            subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.pos_ in ("NOUN", "PROPN")]
            obj = [c for c in tok.children if c.dep_ in ("dobj", "obj") and c.pos_ in ("NOUN", "PROPN")]
            if subj:
                pairs.append((_noun_phrase(subj[0]), "not " + tok.lemma_.lower()))
            elif obj:
                pairs.append((_noun_phrase(obj[0]), "cannot " + tok.lemma_.lower()))
            else:
                pairs.append((tok.lemma_.lower(), "not working"))
    return pairs


def svo(doc) -> list[tuple[str, str, str]]:
    triples = []
    for tok in doc:
        if tok.pos_ != "VERB":
            continue
        subs = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
        objs = [c for c in tok.children if c.dep_ in ("dobj", "obj", "attr")]
        for p in (c for c in tok.children if c.dep_ == "prep"):
            objs += [g for g in p.children if g.dep_ == "pobj"]
        verb = ("not " if _negated(tok) else "") + tok.lemma_.lower()
        for s in subs:
            for o in objs:
                triples.append((_noun_phrase(s), verb, _noun_phrase(o)))
    return triples


@lru_cache(maxsize=1)
def matcher():
    from spacy.matcher import Matcher
    m = Matcher(nlp().vocab)
    m.add("CANT_DO", [[{"LOWER": {"IN": ["can't", "cant", "cannot", "unable", "couldn't", "ca"]}},
                       {"LOWER": {"IN": ["n't", "not", "to"]}, "OP": "?"},
                       {"POS": "VERB"}]])
    m.add("ERROR_MSG", [[{"LOWER": {"IN": ["error", "failed", "failure", "crash", "crashes", "crashing"]}}],
                        [{"LOWER": "something"}, {"LOWER": "went"}, {"LOWER": "wrong"}]])
    m.add("ADV_ADJ_NOUN", [[{"POS": "ADV", "OP": "?"}, {"POS": "ADJ"}, {"POS": "NOUN"}]])
    return m


def match_phrases(doc) -> list[tuple[str, str]]:
    m = matcher()
    return [(nlp().vocab.strings[mid], doc[s:e].text.lower()) for mid, s, e in m(doc)]


def pos_counts(docs) -> Counter:
    c = Counter()
    for d in docs:
        c.update(t.pos_ for t in d if not t.is_space)
    return c


def analyze_docs(texts, batch_size: int = 64):
    out = []
    for doc in nlp().pipe((str(t) for t in texts), batch_size=batch_size):
        out.append({
            "aspects": aspects(doc),
            "svo": svo(doc),
            "entities": [(e.text, e.label_) for e in doc.ents],
            "matches": match_phrases(doc),
        })
    return out


def top_ngrams(token_lists, n: int = 2, k: int = 15) -> list[tuple[str, int]]:
    c = Counter()
    for toks in token_lists:
        c.update(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return c.most_common(k)
