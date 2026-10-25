import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvoice import config
import fake_data


def fake_llm_answer(prompt: str, system=None, **_):
    if "issue name" in prompt:
        ids = sorted({int(i) for i in re.findall(r'"id": (\d+)', prompt.split("CLUSTERS:")[1])})
        return json.dumps([{"id": i, "label_en": f"Issue {i}", "label_ar": f"مشكلة {i}",
                            "description": "desc"} for i in ids])
    if "fact-checker" in prompt:
        return '{"claims": [{"claim": "OTP fails", "supported": true, "evidence": ["R1"]}], "faithfulness": 1.0}'
    if "Question:" in prompt:
        return "- OTP codes do not arrive [R1][R2]\n- Login fails after update [R3]"
    if "public reply" in prompt:
        return '```json\n{"en": "Thank you for your feedback.", "ar": "شكرا لملاحظاتك"}\n```'
    if "information-extraction" in prompt:
        n = prompt.split("REVIEWS:")[1].strip().count("\n") + 1
        return json.dumps([{"id": i, "feature": "login", "issue_type": "login/auth", "entities": [],
                            "relations": [{"subject": "user", "relation": "cannot", "object": "login"}],
                            "language": "en"} for i in range(n)])
    if "Classify" in prompt:
        n = sum(1 for line in prompt.splitlines() if line[:3].strip().rstrip(".").isdigit())
        return "Here you go: " + json.dumps(["negative"] * n)
    return "Headline: things are fine."


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, rel in [("DATA_DIR", ""), ("RAW_REVIEWS", "reviews.parquet"), ("ENRICHED", "enriched.parquet"),
                      ("EMBEDDINGS", "embeddings.npy"), ("CLUSTERS", "clusters.json"),
                      ("LLM_CACHE", "llm_cache.json")]:
        monkeypatch.setattr(config, name, tmp_path / rel if rel else tmp_path)
    fake_data.install(monkeypatch)
    from cvoice import llm
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", fake_llm_answer)
    return tmp_path
