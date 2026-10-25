from __future__ import annotations

import glob
import json
import zipfile
from pathlib import Path

from . import config


def tex(s) -> str:
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}"}
    ascii_map = {"–": "-", "—": ", ", "‘": "`", "’": "'", "“": "``", "”": "''", "→": "->", "…": "..."}
    out = []
    for ch in str(s):
        ch = ascii_map.get(ch, ch)
        if len(ch) == 1 and ord(ch) > 0x24F:
            continue
        out.append(special.get(ch, ch))
    return "".join(out)


def _fmt(v):
    if isinstance(v, float) and v != v:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:.2f}"
    return v


def table_rows(rows, cols) -> str:
    lines = []
    for r in rows or []:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, str) and any("؀" <= ch <= "ۿ" for ch in v):
                v = "(Arabic)"
            cells.append(tex(_fmt(v)) if v is not None else "n/a")
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) or r"\multicolumn{" + str(len(cols)) + r"}{c}{not run} \\"


def export(R: dict, root: Path | None = None) -> Path:
    root = Path(root or config.ROOT)
    (root / "report").mkdir(exist_ok=True)
    (root / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    pct = lambda x: f"{100 * x:.0f}\\%" if isinstance(x, (int, float)) else "n/a"
    macros = {
        "NReviews": f"{R.get('n_reviews', 0):,}", "AvgRating": R.get("avg_rating", "n/a"),
        "ShareNeg": pct(R.get("share_negative")), "ShareAr": pct(R.get("share_arabic")),
        "MedianWords": R.get("median_words", "n/a"), "NClusters": R.get("n_clusters", "n/a"),
        "NTrain": R.get("n_train", "n/a"), "NTest": R.get("n_test", "n/a"),
        "LoraPct": R.get("lora_trainable_pct", "n/a"), "LoraMin": R.get("lora_minutes", "n/a"),
        "AngerShare": pct(R.get("anger_share_complaints")), "PipelineMin": R.get("pipeline_minutes", "n/a"),
        "LLMModel": tex(R.get("llm_model") or "an OpenRouter free model"),
        "MixedClusters": R.get("mixed_language_clusters", "n/a"),
        "NFlags": len(R.get("release_flags") or []),
    }
    def pick(rows, key, name, col):
        for r in rows or []:
            if str(r.get(key, "")).startswith(name):
                v = r.get(col)
                return _fmt(v) if v is not None else "n/a"
        return "n/a"
    reps, sent = R.get("representations"), R.get("sentiment")
    xl = R.get("cross_lingual") or {}
    macros.update({
        "BgeFone": pick(reps, "representation", "BGE", "macro_f1"), "BgePfive": pick(reps, "representation", "BGE", "precision@5"),
        "TfidfAcc": pick(reps, "representation", "TF-IDF", "accuracy"), "TfidfFone": pick(reps, "representation", "TF-IDF", "macro_f1"),
        "WvFone": pick(reps, "representation", "Word2Vec", "macro_f1"),
        "XLOverlap": _fmt((xl.get("BGE-M3") or {}).get("overlap@10", "n/a")),
        "XLArShare": pct((xl.get("BGE-M3") or {}).get("arabic_share_for_en_query")),
        "VaderFone": pick(sent, "method", "VADER", "en macro_f1"),
        "TfidfLrFone": pick(sent, "method", "TF-IDF", "all macro_f1"),
        "RoutedFone": pick(sent, "method", "RoBERTa", "all macro_f1"),
        "XlmrFone": pick(sent, "method", "XLM-R (pre", "all macro_f1"), "XlmrAcc": pick(sent, "method", "XLM-R (pre", "all accuracy"),
        "LoraFone": pick(sent, "method", "XLM-R + LoRA", "all macro_f1"), "LoraAcc": pick(sent, "method", "XLM-R + LoRA", "all accuracy"),
        "LoraArAcc": pick(sent, "method", "XLM-R + LoRA", "ar accuracy"), "LoraArFone": pick(sent, "method", "XLM-R + LoRA", "ar macro_f1"),
        "SadShare": pct(R.get("sadness_share_complaints")), "NVersions": R.get("n_versions_tested", "n/a"),
        "NSigRaw": R.get("n_versions_p05", "n/a"),
        "NApps": R.get("n_apps", "n/a"),
    })
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    tables = {
        "RepTable": (R.get("representations"), ["representation", "dimensions", "accuracy", "macro_f1", "precision@5"]),
        "SentTable": (R.get("sentiment"), ["method", "all accuracy", "all macro_f1", "en macro_f1", "ar macro_f1"]),
        "PromptTable": (R.get("prompting"), ["strategy", "accuracy", "macro_f1"]),
        "RagTable": (R.get("rag"), ["question", "faithfulness_rag", "faithfulness_no_retrieval", "citations"]),
        "IssueTable": (R.get("issues"), ["label_en", "size", "avg_rating", "share_arabic"]),
        "XLTable": ([{"method": m, **v} for m, v in (R.get("cross_lingual") or {}).items()],
                    ["method", "overlap@10", "arabic_share_for_en_query"]),
        "FlagTable": (R.get("release_flags"), ["app", "version", "reviews", "neg_rate", "baseline", "z", "q"]),
    }
    for name, (rows, cols) in tables.items():
        lines.append(f"\\newcommand{{\\{name}}}{{\n{table_rows(rows, cols)}\n}}")
    (root / "report" / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    from . import dashboard
    dashboard.build(json.loads((root / "results.json").read_text(encoding="utf-8")), root / "dashboard.html")

    zpath = root / "report_assets.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in glob.glob(str(root / "figures" / "*.png")):
            z.write(f, f"figures/{Path(f).name}")
        z.write(root / "report" / "results.tex", "results.tex")
        z.write(root / "results.json", "results.json")
        z.write(root / "dashboard.html", "dashboard.html")
    return zpath
