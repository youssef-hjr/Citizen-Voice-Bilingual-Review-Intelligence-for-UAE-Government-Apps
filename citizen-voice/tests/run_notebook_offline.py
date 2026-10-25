import sys
import tempfile
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="cv_nb_"))

PRE = f"""
import os, sys
os.environ["CV_DATA_DIR"] = r"{TMP / 'data'}"
os.environ["CV_MODELS_DIR"] = r"{TMP / 'models'}"
os.makedirs(os.environ["CV_DATA_DIR"], exist_ok=True)
sys.path.insert(0, r"{ROOT / 'tests'}")
"""

POST = f"""
import fake_data, conftest
from cvoice import collect, models, llm, config
def _collect(per_lang=0, **k):
    df = fake_data.make_reviews(1500); df.to_parquet(config.RAW_REVIEWS, index=False); return df
collect.collect = _collect
models.embed = fake_data.fake_embed
models.sentiment_routed = fake_data.fake_sentiment_routed
models.emotion = fake_data.fake_emotion
models.ner_ar = fake_data.fake_ner_ar
models.sentiment_hf = lambda texts, model, **k: fake_data.fake_sentiment_routed(texts, ["en"] * len(texts))
llm.available = lambda: True
llm.ask = conftest.fake_llm_answer
# tiny real model for the LoRA section
import test_models_tiny as tmt
config.LORA_BASE = tmt._tiny(Path(r"{TMP}") / "tinybase", {{0: "LABEL_0", 1: "LABEL_1"}}, kind="base")
from pathlib import Path
"""


def main():
    nb = nbformat.read(ROOT / "notebooks" / "Citizen_Voice_Colab.ipynb", as_version=4)
    imp = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "from cvoice import (" in c.source)
    nb.cells.insert(imp, nbformat.v4.new_code_cell(PRE))
    nb.cells.insert(imp + 2, nbformat.v4.new_code_cell(POST.replace("import test_models_tiny as tmt",
                                                                    "from pathlib import Path\nimport test_models_tiny as tmt")))
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
        ok = True
    finally:
        out = TMP / "executed.ipynb"
        nbformat.write(nb, out)
        print("executed notebook:", out)
    errors = [c for c in nb.cells if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    print("cells with errors:", len(errors))
    return 0 if ok and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
