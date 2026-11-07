"""Central application configuration.

Values can be overridden with environment variables (see .env.example).
Later modules import their settings from here.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (real env vars win).

    Tiny stand-in for python-dotenv so no extra package is needed. Blank lines and
    lines starting with '#' are ignored; surrounding quotes on values are removed.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

# Database (Module 2)
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "feedback.db"))

# Sentiment model (Module 3)
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
SENTIMENT_MAX_TOKENS = int(os.getenv("SENTIMENT_MAX_TOKENS", "512"))  # model input limit
SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "8"))  # small batches keep memory low on free hosts
SENTIMENT_LOW_CONFIDENCE = float(os.getenv("SENTIMENT_LOW_CONFIDENCE", "0.6"))  # below this, flag "low_confidence"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# Experimental: shrink the model with 8-bit dynamic quantisation (less memory, may change accuracy slightly).
# Off by default. Re-run scripts/check_sentiment.py --sample before relying on it.
SENTIMENT_QUANTIZE = _env_flag("SENTIMENT_QUANTIZE")

# Public demo mode: smaller upload limits so a free host does not run out of memory.
DEMO_MODE = _env_flag("DEMO_MODE")
SHOW_ANSWER_DETAILS = _env_flag("SHOW_ANSWER_DETAILS")  # show the "Answered with: ..." line under each answer (debugging only)
DEMO_MAX_CSV_ROWS = int(os.getenv("DEMO_MAX_CSV_ROWS", "300"))
DEMO_SEED = DATA_DIR / "demo_seed.db"  # pre-analysed demo reviews every demo visitor starts from (scripts/build_demo_seed.py)

# Local LLM via Ollama (Module 5)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Validation limits (Module 2)
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "5000"))
MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(5 * 1024 * 1024)))  # 5 MB
MAX_REVIEW_LENGTH = int(os.getenv("MAX_REVIEW_LENGTH", "2000"))
MAX_PRODUCT_LENGTH = int(os.getenv("MAX_PRODUCT_LENGTH", "100"))
MAX_SOURCE_LENGTH = 50
MAX_EXTRA_COLUMNS = int(os.getenv("MAX_EXTRA_COLUMNS", "30"))  # extra CSV columns kept per review
MAX_EXTRA_VALUE_LENGTH = int(os.getenv("MAX_EXTRA_VALUE_LENGTH", "500"))
MIN_RATING = 1
MAX_RATING = 5

# Feedback table pagination (Module 2)
PAGE_SIZE_OPTIONS = (10, 25, 50, 100)
DEFAULT_PAGE_SIZE = 25

# Allowed sentiment labels (filled in by Module 3)
SENTIMENT_LABELS = ("positive", "neutral", "negative")

# Sample data used during development
SAMPLE_CSV = DATA_DIR / "sample_reviews.csv"

# Hosted LLM (Google Gemini free tier). Everything stays optional: without a key the rule-based chatbot is used.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()          # copy the exact name from Google AI Studio
LLM_DAILY_LIMIT = int(os.getenv("LLM_DAILY_LIMIT", "15"))       # whole app, per day. Keep it below your real quota
LLM_SESSION_LIMIT = int(os.getenv("LLM_SESSION_LIMIT", "4"))    # per visitor session
LLM_MINUTE_LIMIT = int(os.getenv("LLM_MINUTE_LIMIT", "10"))      # whole app, per minute (free tier allows about 15)
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
PLANNER_MODE = os.getenv("PLANNER_MODE", "hybrid").strip().lower()  # rules | hybrid | llm_first
# AI-written summaries ("summarise what customers say about X"). Sends up to 12 short review excerpts (e-mails, phone
# numbers and links removed) to Gemini for those questions only. Set to 0 to keep review text on this machine.
AI_SUMMARIES = os.getenv("AI_SUMMARIES", "1").strip() == "1"
