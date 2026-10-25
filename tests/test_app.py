from pathlib import Path

import pytest

import fake_data
from cvoice import config, pipeline

APP = str(Path(__file__).resolve().parent.parent / "app.py")
st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def app(env):
    fake_data.make_reviews(1200).to_parquet(config.RAW_REVIEWS, index=False)
    pipeline.run(log=lambda *_: None)
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_all_tabs_render(app):
    _ok(app)
    assert len(app.tabs) == 7
    assert any("Reviews" == m.label for m in app.metric)
    assert any("UAE PASS 4.2.0" in e.value for e in app.error)


def test_buttons(app):
    labels = {b.label: b for b in app.button}
    for name in ["📝 Generate executive briefing", "Ask", "✍️ Draft bilingual reply", "Analyze"]:
        assert name in labels, name
    for name in ["📝 Generate executive briefing", "Ask", "✍️ Draft bilingual reply"]:
        next(b for b in app.button if b.label == name).click().run()
        _ok(app)
    next(b for b in app.button if b.label == "Analyze").click().run()
    _ok(app)
    assert any(m.label == "Sentiment" for m in app.metric)


def test_analyze_arabic(app):
    app.text_area[0].set_value("رمز التحقق لا يصل ولا أستطيع تجديد الهوية").run()
    next(b for b in app.button if b.label == "Analyze").click().run()
    _ok(app)


def test_filters_empty(app):
    app.multiselect[0].set_value([]).run()
    _ok(app)


def test_no_data_message(env):
    at = st_testing.AppTest.from_file(APP, default_timeout=60)
    at.run()
    _ok(at)
    assert any("No analysed data yet" in i.value for i in at.info)
