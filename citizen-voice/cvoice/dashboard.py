import json
import math
from pathlib import Path

from . import config

HEAD = """<title>Citizen Voice Dashboard</title>
<meta name="claude:viewport-fit-sides" content="cover">
"""

STYLE = """<style>
:root {
  --bg: #ffffff; --ink: #1d1d1f; --muted: #5f6368; --line: #e3e3e3; --soft: #f6f7f8;
  --accent: #1f4e79; --grey: #a3acb5; --neg: #b03a2e; --neu: #c08a1e;
  --e-anger: #b03a2e; --e-disgust: #8c6d4f; --e-fear: #6c5b8e; --e-joy: #4d8b57;
  --e-neutral: #c4cad0; --e-sadness: #1f4e79; --e-surprise: #d4a640;
  --body: Arial, "Helvetica Neue", Helvetica, "Segoe UI", sans-serif;
  --arabic: "Geeza Pro", "Segoe UI", Tahoma, Arial, sans-serif;
  color-scheme: light;
}
* { box-sizing: border-box }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body) }
.wrap { max-width: 1040px; margin: 0 auto; padding-block: 32px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)) }
header { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 8px }
h1 { font-size: 30px; line-height: 1.15; margin: 0 0 4px; font-weight: 700 }
.sub { margin: 0; color: var(--ink); font-size: 16px }
.by { margin: 6px 0 0; color: var(--muted); font-size: 13.5px }
.lede { color: var(--muted); max-width: 75ch; margin: 14px 0 0 }
.tiles { display: grid; grid-template-columns: repeat(6, 1fr); border-bottom: 1px solid var(--line); margin-bottom: 6px }
.tile { padding: 16px 12px 16px 0; display: grid; gap: 2px; min-width: 0 }
.tile + .tile { border-left: 1px solid var(--line); padding-left: 14px }
.tile b { font-size: 24px; line-height: 1.1; font-weight: 700; font-variant-numeric: tabular-nums }
.tile span { color: var(--muted); font-size: 13px }
.tile.alert b { color: var(--neg) }
@media (max-width: 760px) { .tiles { grid-template-columns: repeat(2, 1fr) } .tile + .tile { border-left: 0; padding-left: 0 } .tile { border-top: 1px solid var(--line) } }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); column-gap: 40px }
section { padding: 22px 0 18px; border-bottom: 1px solid var(--line); min-width: 0; display: grid; gap: 10px; align-content: start }
h2 { font-size: 17px; line-height: 1.3; margin: 0; font-weight: 700 }
.note { color: var(--muted); font-size: 13.5px; margin: 0 }
.note b { color: var(--ink) }
svg { display: block; width: 100%; height: auto; overflow: visible }
svg text { fill: var(--ink); font: 13px var(--body) }
svg .m { fill: var(--muted) }
svg .num { font: 12.5px var(--body); font-variant-numeric: tabular-nums }
svg .ar { font-family: var(--arabic) }
svg .grid { stroke: var(--line); stroke-width: 1 }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted) }
.legend i { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px }
.tbl { overflow-x: auto }
table { border-collapse: collapse; width: 100%; font-size: 14px }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); white-space: nowrap }
thead th { border-bottom: 1.5px solid var(--ink); font-weight: 700 }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums }
td.bad { color: var(--neg); font-weight: 700 }
footer { color: var(--muted); font-size: 13px; padding-top: 16px }
</style>
"""

BODY = """<div class="wrap">
<header>
  <h1>Citizen Voice</h1>
  <p class="sub">What residents say about UAE government apps, in English and Arabic</p>
  <p class="by">Youssef Hajri (1425158) &middot; MAAI902 Natural Language Processing and LLMs &middot; University of Wollongong in Dubai &middot; October 2026</p>
  <p class="lede" id="lede"></p>
</header>
<div class="tiles" id="tiles"></div>
<section>
  <h2>What citizens complain about</h2>
  <p class="note">Complaints clustered with BGE-M3 embeddings and named by the LLM in English and Arabic. Bar length is the number of complaints. On the right: average star rating and the share written in Arabic.</p>
  <div id="issues"></div>
</section>
<div class="grid2">
  <section>
    <h2>Sentiment: lexicon vs. ML vs. transformers</h2>
    <p class="note">Macro-F1 on the held-out test set (all languages; VADER is English only).</p>
    <div id="sent"></div>
    <p class="note" id="sentnote"></p>
  </section>
  <section>
    <h2>Text representations</h2>
    <p class="note">Same logistic regression on each representation, plus precision@5 for review search.</p>
    <div id="reps"></div>
    <div class="legend"><span><i style="background:var(--accent)"></i>Macro-F1</span><span><i style="background:var(--grey)"></i>Precision@5</span></div>
  </section>
</div>
<div class="grid2">
  <section>
    <h2>How complaints feel</h2>
    <p class="note">Emotion share by star rating. Unhappy reviews are mostly <b>sad</b>, rarely angry.</p>
    <div id="emo"></div>
    <div class="legend" id="emolegend"></div>
  </section>
  <section>
    <h2>One meaning, two languages</h2>
    <p class="note">For an English query, how many of the top 10 results match the Arabic query with the same meaning, and how many results are in Arabic.</p>
    <div id="xl"></div>
    <div class="legend"><span><i style="background:var(--accent)"></i>Overlap@10 with the Arabic query</span><span><i style="background:var(--grey)"></i>Arabic results for an English query</span></div>
  </section>
</div>
<section>
  <h2>Releases that increased complaints</h2>
  <p class="note" id="relnote"></p>
  <div class="tbl" id="rel"></div>
</section>
<div class="grid2">
  <section>
    <h2>Prompt engineering</h2>
    <p class="note">Same LLM and the same balanced reviews; only the prompt changes. Macro-F1.</p>
    <div id="prompt"></div>
  </section>
  <section>
    <h2>Grounding the assistant (RAG)</h2>
    <p class="note">Share of the answer's claims that a judge LLM finds supported by real reviews.</p>
    <div id="rag"></div>
    <div class="legend"><span><i style="background:var(--accent)"></i>With retrieval and citations</span><span><i style="background:var(--neg)"></i>Without retrieval</span></div>
  </section>
</div>
<section>
  <h2>Preprocessing: vocabulary size per step</h2>
  <p class="note">Distinct word forms after each step. Arabic shrinks much more with stemming because of its rich morphology.</p>
  <div id="vocab"></div>
  <div class="legend"><span><i style="background:var(--accent)"></i>English</span><span><i style="background:var(--neu)"></i>Arabic</span></div>
</section>
<footer id="foot"></footer>
</div>
"""

SCRIPT = """<script>
const R = __DATA__;
const $ = id => document.getElementById(id);
const pct = v => v == null ? "n/a" : Math.round(v * 100) + "%";
const f3 = v => v == null ? "n/a" : (+v).toFixed(3);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const css = n => "var(--" + n + ")";

function hbars(el, rows, o) {
  const W = o.W || 520, lw = o.labelW || 190, rh = o.rowH || 30, bh = o.barH || 16, vw = 64;
  const series = o.series || [{key: "v", color: css("accent")}];
  const max = o.max || Math.max(...rows.flatMap(r => series.map(s => r[s.key] || 0)), 1e-9);
  const plot = W - lw - vw, H = rows.length * rh + 8;
  let g = "";
  for (const t of (o.ticks || [])) {
    const x = lw + plot * t / max;
    g += `<line class="grid" x1="${x}" x2="${x}" y1="0" y2="${H - 4}"/>`;
  }
  rows.forEach((r, i) => {
    const y = i * rh + 4, n = series.length, sb = n > 1 ? bh / n + 2 : bh;
    g += `<text x="${lw - 10}" y="${y + rh / 2}" text-anchor="end" dominant-baseline="middle" class="${r.cls || ""}">${esc(r.label)}</text>`;
    series.forEach((s, j) => {
      const v = r[s.key];
      const yy = y + (rh - (n > 1 ? n * sb : bh)) / 2 + j * sb;
      if (v == null) { g += `<text x="${lw + 4}" y="${yy + sb / 2}" dominant-baseline="middle" class="m num">n/a</text>`; return; }
      const w = Math.max(2, plot * v / max);
      g += `<rect x="${lw}" y="${yy}" width="${w}" height="${n > 1 ? sb - 2 : bh}" rx="3" fill="${r.color || s.color}"/>`;
      g += `<text x="${lw + w + 6}" y="${yy + (n > 1 ? sb - 2 : bh) / 2}" dominant-baseline="middle" class="num">${(o.fmt || f3)(v)}</text>`;
    });
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "")}">${g}</svg>`;
}

const n = x => (x ?? 0).toLocaleString("en-US");
$("lede").textContent = `${n(R.n_reviews)} Google Play reviews of ${R.n_apps} UAE government apps, in English and Arabic, ` +
  `analysed with classical NLP, transformers, a LoRA fine-tune and an LLM. Star ratings are used as weak labels.`;

const flags = (R.release_flags || []).length;
const tiles = [
  [n(R.n_reviews), "reviews analysed"], [(R.avg_rating ?? 0).toFixed(1) + " / 5", "average rating"],
  [pct(R.share_negative), "are complaints (1 or 2 stars)", true], [pct(R.share_arabic), "written in Arabic"],
  [R.n_clusters, `issues found, ${R.mixed_language_clusters} bilingual`],
  [flags + " / " + R.n_versions_tested, "releases flagged after FDR", true],
];
$("tiles").innerHTML = tiles.map(([b, s, a]) => `<div class="tile${a ? " alert" : ""}"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");

(function issues() {
  const rows = (R.issues || []).slice().sort((a, b) => b.size - a.size);
  const W = 1000, lw = 300, rh = 44, H = rows.length * rh + 6, plot = W - lw - 190;
  const max = Math.max(...rows.map(r => r.size), 1);
  let g = "";
  rows.forEach((r, i) => {
    const y = i * rh + 4, w = Math.max(2, plot * r.size / max);
    g += `<text x="0" y="${y + 15}">${esc(r.label_en)}</text>`;
    if (r.label_ar) g += `<text x="0" y="${y + 33}" class="m ar" style="direction:ltr" >${esc(r.label_ar)}</text>`;
    g += `<rect x="${lw}" y="${y + 10}" width="${w}" height="16" rx="3" fill="${css("accent")}"/>`;
    g += `<text x="${lw + w + 6}" y="${y + 19}" dominant-baseline="middle" class="num">${r.size}</text>`;
    g += `<text x="${W}" y="${y + 19}" text-anchor="end" dominant-baseline="middle" class="m num">${(r.avg_rating ?? 0).toFixed(2)}★ · ${pct(r.share_arabic)} AR</text>`;
  });
  $("issues").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Issue clusters by size">${g}</svg>`;
})();

(function sentiment() {
  const rows = (R.sentiment || []).map(m => ({label: m.method, v: m["all macro_f1"] ?? m["en macro_f1"]}));
  hbars($("sent"), rows, {max: 1, ticks: [.25, .5, .75], labelW: 250, aria: "Sentiment macro-F1 by method"});
  const lo = (R.sentiment || []).find(m => /LoRA/.test(m.method));
  if (lo) $("sentnote").innerHTML = `The LoRA fine-tune trains only <b>${R.lora_trainable_pct}%</b> of XLM-R's weights in ${R.lora_minutes} min and reaches the best accuracy (<b>${f3(lo["all accuracy"])}</b>; Arabic ${f3(lo["ar accuracy"])}).`;
})();

hbars($("reps"), (R.representations || []).map(r => ({label: r.representation, f1: r.macro_f1, p5: r["precision@5"]})),
  {max: 1, ticks: [.25, .5, .75], series: [{key: "f1", color: css("accent")}, {key: "p5", color: css("grey")}], rowH: 40, barH: 26, aria: "Representations"});

(function emotions() {
  const E = ["sadness", "anger", "disgust", "fear", "surprise", "neutral", "joy"];
  const by = R.emotion_by_rating || {}, stars = Object.keys(by).sort();
  const W = 520, lw = 36, rh = 34, H = stars.length * rh + 4, plot = W - lw;
  let g = "";
  stars.forEach((s, i) => {
    const y = i * rh + 4; let x = lw;
    g += `<text x="0" y="${y + 13}" dominant-baseline="middle" class="num">${s}★</text>`;
    const tot = E.reduce((a, e) => a + (by[s][e] || 0), 0) || 1;
    E.forEach(e => {
      const w = plot * (by[s][e] || 0) / tot;
      if (w <= 0) return;
      g += `<rect x="${x}" y="${y}" width="${w}" height="26" fill="${css("e-" + e)}"><title>${e}: ${pct(by[s][e])}</title></rect>`;
      if (w > 34) g += `<text x="${x + w / 2}" y="${y + 13}" text-anchor="middle" dominant-baseline="middle" class="num" style="fill:${["neutral", "surprise"].includes(e) ? "#1d1d1f" : "#ffffff"}">${pct(by[s][e])}</text>`;
      x += w;
    });
  });
  $("emo").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Emotion by star rating">${g}</svg>`;
  $("emolegend").innerHTML = E.map(e => `<span><i style="background:${css("e-" + e)}"></i>${e}</span>`).join("");
})();

hbars($("xl"), Object.entries(R.cross_lingual || {}).map(([k, v]) => ({label: k, a: v["overlap@10"], b: v.arabic_share_for_en_query})),
  {max: 1, ticks: [.25, .5, .75], series: [{key: "a", color: css("accent")}, {key: "b", color: css("grey")}], rowH: 48, barH: 30, labelW: 90, fmt: pct, aria: "Cross-lingual retrieval"});

(function releases() {
  const rows = R.release_flags || [];
  $("relnote").innerHTML = `${R.n_versions_tested} app versions were tested with a two-proportion z-test. <b>${R.n_versions_p05}</b> have p&nbsp;&lt;&nbsp;0.05 on their own; <b>${rows.length}</b> remain after the Benjamini-Hochberg correction (false discovery rate 5%).`;
  $("rel").innerHTML = `<table><thead><tr><th>App</th><th>Version</th><th class="n">Reviews</th><th class="n">Complaints</th><th class="n">App baseline</th><th class="n">q</th><th>Issue that grew</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.app)}</td><td>${esc(r.version)}</td><td class="n">${r.reviews}</td><td class="n bad">${pct(r.neg_rate)}</td><td class="n">${pct(r.baseline)}</td><td class="n">${(+r.q).toFixed(3)}</td><td>${esc(r.issue_that_grew || "n/a")}</td></tr>`).join("") + `</tbody></table>`;
})();

hbars($("prompt"), (R.prompting || []).map(p => ({label: p.strategy.replace(/_/g, " ").replace(" (same reviews)", ""), v: p.macro_f1,
  color: /LoRA/.test(p.strategy) ? css("grey") : css("accent")})), {max: 1, ticks: [.25, .5, .75], labelW: 220, aria: "Prompt strategies"});

hbars($("rag"), (R.rag || []).map((q, i) => ({label: /[\\u0600-\\u06FF]/.test(q.question) ? "Arabic question (ICP)" : q.question.replace(/\\?$/, ""),
  a: q.faithfulness_rag, b: q.faithfulness_no_retrieval})),
  {max: 1, ticks: [.25, .5, .75], series: [{key: "a", color: css("accent")}, {key: "b", color: css("neg")}], rowH: 48, barH: 30, labelW: 250, fmt: pct, aria: "RAG faithfulness"});

(function vocab() {
  const steps = ["raw", "nostop", "stem", "lemma"], names = {raw: "raw tokens", nostop: "no stop words", stem: "stemmed", lemma: "lemmatised"};
  const get = (l, s) => ((R.vocab || []).find(v => v.language === l && v.step === s) || {}).vocabulary;
  hbars($("vocab"), steps.map(s => ({label: names[s], en: get("en", s), ar: get("ar", s)})),
    {series: [{key: "en", color: css("accent")}, {key: "ar", color: css("neu")}], W: 1000, rowH: 42, barH: 28, labelW: 130, fmt: v => n(v), aria: "Vocabulary per step"});
})();

$("foot").textContent = `Source: Citizen Voice pipeline (results.json). Evaluation on ${R.n_test} test reviews, models trained on ${R.n_train}.`;
</script>
"""


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def render(R: dict, standalone: bool = True) -> str:
    data = json.dumps(_clean(R), ensure_ascii=False, default=str).replace("</", "<\\/")
    page = HEAD + STYLE + BODY + SCRIPT.replace("__DATA__", data)
    if standalone:
        return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + HEAD + STYLE + "</head>\n<body>\n" + BODY + SCRIPT.replace("__DATA__", data) + "</body>\n</html>\n")
    return page


def build(R: dict | None = None, path=None, standalone: bool = True) -> Path:
    if R is None:
        R = json.loads((config.ROOT / "results.json").read_text(encoding="utf-8"))
    path = Path(path or config.ROOT / "dashboard.html")
    path.write_text(render(R, standalone), encoding="utf-8")
    return path
