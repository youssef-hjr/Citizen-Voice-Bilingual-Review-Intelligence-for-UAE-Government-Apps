from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from functools import lru_cache

import requests

from . import config

_lock = threading.Lock()


class LLMError(RuntimeError):
    pass


def api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    for _ in range(3):
        try:
            from google.colab import userdata
        except ImportError:
            break
        try:
            key = userdata.get("OPENROUTER_API_KEY")
        except Exception as e:
            if "NotFound" in type(e).__name__ or "Access" in type(e).__name__:
                break
            time.sleep(1)
            continue
        break
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            key = None
    if key:
        os.environ["OPENROUTER_API_KEY"] = key.strip()
        return key.strip()
    return None


def available() -> bool:
    return bool(api_key())


def key_problem() -> str | None:
    if api_key():
        return None
    try:
        from google.colab import userdata
    except ImportError:
        return "Set the OPENROUTER_API_KEY environment variable, or paste the key in the app sidebar."
    try:
        userdata.get("OPENROUTER_API_KEY")
    except Exception as e:
        name = type(e).__name__
        if "Access" in name:
            return ("The secret OPENROUTER_API_KEY exists but this notebook has no access: open the key icon "
                    "and switch on 'Notebook access' next to it.")
        if "NotFound" in name:
            return ("No Colab secret named exactly OPENROUTER_API_KEY. Add it with the key icon on the left "
                    "(name without spaces, value = your sk-or-... key).")
        return f"Could not read the Colab secret: {name}: {e}"
    return "The Colab secret OPENROUTER_API_KEY is empty."


def _headers():
    return {"Authorization": f"Bearer {api_key()}", "HTTP-Referer": "https://github.com/citizen-voice",
            "X-Title": "Citizen Voice"}


def _rank(m: dict) -> tuple:
    params = set(m.get("supported_parameters") or [])
    mid = m["id"].lower()
    thinking = "reasoning" in params or any(w in mid for w in ("think", "reason", "-r1", "qwq"))
    vision = "-vl" in mid or "vision" in mid
    return (thinking, vision, "response_format" not in params and "structured_outputs" not in params,
            -(m.get("context_length") or 0))


@lru_cache(maxsize=1)
def candidate_models() -> list[str]:
    if config.LLM_MODEL:
        return [config.LLM_MODEL]
    try:
        r = requests.get(f"{config.LLM_BASE_URL}/models", headers=_headers(), timeout=30)
        models = r.json().get("data", [])
    except Exception:
        return list(config.LLM_PREFERRED)
    free = {m["id"]: m for m in models if str(m.get("id", "")).endswith(":free")}
    ordered = [m for m in config.LLM_PREFERRED if m in free]
    rest = [m["id"] for m in sorted((free[m] for m in free if m not in ordered), key=_rank)]
    return ordered + rest[:6] or list(config.LLM_PREFERRED)


def _load_cache() -> dict:
    try:
        return json.loads(config.LLM_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict):
    config.LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.LLM_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config.LLM_CACHE)


_CACHE: dict | None = None


def _cache() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    return _CACHE


last_model_used: str | None = None


_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)


def _cache_key(prompt, system, temperature) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return hashlib.sha256(json.dumps([messages, temperature]).encode()).hexdigest()


def ask(prompt: str, system: str | None = None, temperature: float = 0.2,
        max_tokens: int = 4000, use_cache: bool = True, skip_models: tuple = ()) -> str:
    global last_model_used
    if not available():
        raise LLMError("No OPENROUTER_API_KEY found (env var, Colab secret or Streamlit secret).")
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    key = _cache_key(prompt, system, temperature)
    if use_cache and key in _cache():
        last_model_used = _cache()[key]["model"]
        return _cache()[key]["text"]

    errors = []
    deadline = time.time() + config.LLM_DEADLINE
    for model in [m for m in candidate_models() if m not in skip_models]:
        if time.time() > deadline:
            errors.append(f"gave up after {config.LLM_DEADLINE:.0f}s")
            break
        for attempt in range(3):
            remaining = deadline - time.time()
            if remaining < 5:
                break
            try:
                r = requests.post(f"{config.LLM_BASE_URL}/chat/completions", headers=_headers(),
                                  timeout=min(60, remaining),
                                  json={"model": model, "messages": messages,
                                        "temperature": temperature, "max_tokens": max_tokens,
                                        "reasoning": {"exclude": True}})
            except requests.RequestException as e:
                errors.append(f"{model}: {e}")
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429 and "per-day" in r.text:
                raise LLMError("OpenRouter daily free-model limit reached (resets at 00:00 UTC). "
                               "Cached answers still work; adding $10 credit raises the limit to 1000/day.")
            if r.status_code == 429:
                time.sleep(min(5 * (attempt + 1), max(0, deadline - time.time())))
                continue
            if r.status_code != 200:
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:150]}")
                break
            data = r.json()
            try:
                text = _THINK.sub("", data["choices"][0]["message"].get("content") or "").strip()
            except (KeyError, IndexError):
                errors.append(f"{model}: unexpected response {str(data)[:150]}")
                break
            if not text:
                errors.append(f"{model}: empty answer")
                break
            last_model_used = model
            with _lock:
                _cache()[key] = {"model": model, "text": text}
                _save_cache(_cache())
            return text
        else:
            errors.append(f"{model}: still rate limited")
    raise LLMError("All free models failed:\n" + "\n".join(errors[-6:]))


def _complete_objects(text: str, dec) -> list:
    items, i = [], 1
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                break
            items.append(obj)
            i += end
            continue
        i += 1
    return items


def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r",\s*([\]}])", r"\1", text)
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                if ch == "[":
                    items = _complete_objects(text[i:], dec)
                    if items:
                        return items
                continue
    raise LLMError(f"No JSON found in answer: {text[:200]}")


def ask_json(prompt: str, system: str | None = None, retries: int = 2, **kw):
    skip = []
    for i in range(retries + 1):
        text = ask(prompt, system=system, skip_models=tuple(skip), **kw)
        try:
            return parse_json(text)
        except LLMError:
            with _lock:
                _cache().pop(_cache_key(prompt, system, kw.get("temperature", 0.2)), None)
                _save_cache(_cache())
            skip.append(last_model_used)
            if i == retries:
                raise
