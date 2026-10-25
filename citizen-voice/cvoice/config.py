from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CV_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("CV_MODELS_DIR", ROOT / "models"))
FIGURES_DIR = ROOT / "figures"

RAW_REVIEWS = DATA_DIR / "reviews.parquet"
ENRICHED = DATA_DIR / "enriched.parquet"
EMBEDDINGS = DATA_DIR / "embeddings.npy"
CLUSTERS = DATA_DIR / "clusters.json"
LLM_CACHE = DATA_DIR / "llm_cache.json"
LORA_DIR = MODELS_DIR / "xlmr-lora-sentiment"

APPS = [
    {"name": "UAE PASS", "id": "ae.uaepass.mainapp", "query": "UAE PASS", "match": "pass"},
    {"name": "DubaiNow", "id": "com.deg.mdubai", "query": "DubaiNow", "match": "dubainow"},
    {"name": "RTA Dubai", "id": "com.rta.rtadubai", "query": "RTA Dubai", "match": "rta"},
    {"name": "DEWA", "id": "com.dewa.application", "query": "DEWA Dubai Electricity Water", "match": "dewa"},
    {"name": "UAEICP", "id": "ae.gov.icp.smartservices", "query": "UAEICP identity citizenship", "match": "icp"},
    {"name": "TAMM", "id": "ae.abudhabi.tamm", "query": "TAMM Abu Dhabi government services", "match": "tamm"},
    {"name": "Dubai Police", "id": "com.dubaipolice.app", "query": "Dubai Police", "match": "police"},
    {"name": "MOHRE", "id": "ae.gov.mohre", "query": "MOHRE UAE human resources", "match": "mohre"},
]

COUNTRY = "ae"
LANGS = ("en", "ar")
REVIEWS_PER_LANG = 600

SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "BAAI/bge-m3"
SENTIMENT_EN = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_AR = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
SENTIMENT_MULTI = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
EMOTION_EN = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_ZS = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NER_AR = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner"
LORA_BASE = "xlm-roberta-base"

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABELS = ["negative", "neutral", "positive"]

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_DEADLINE = float(os.environ.get("LLM_DEADLINE", "120"))
LLM_PREFERRED = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
]


def rating_to_label(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"
