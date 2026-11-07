# Customer Feedback & Sentiment Analysis Chatbot

A zero-budget, local application that accepts customer reviews, classifies sentiment,
stores feedback, visualizes trends, and answers business questions grounded in database
evidence. Built with Streamlit, SQLite, Hugging Face Transformers, and (later) a local
LLM through Ollama.

> Portfolio MVP and learning project, not a production-certified system.

## Build status

| Module | Focus | Status |
|--------|-------|--------|
| 1 | Environment and project setup | Done |
| 2 | Database and feedback ingestion | Done |
| 3 | Sentiment analysis | Done |
| 4 | Topic tagging and analytics | Pending |
| 5 | Hybrid chatbot (router + SQL + local LLM/RAG) | Pending |
| 6 | Streamlit integration | Pending |
| 7 | Testing, reliability, documentation | Pending |

## Setup (Windows PowerShell)

```powershell
cd customer-feedback-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app has four tabs: Overview, Add review, Import CSV and Feedback. The SQLite file is
created automatically at `data/feedback.db` on first run and persists between restarts.

## Test

```bash
pytest -v
```

## Verify the environment without the browser

```bash
python check_env.py
```

## Project structure

```
customer-feedback-chatbot/
├── app.py            # Streamlit pages and UI (thin: calls the modules below)
├── database.py       # SQLite schema, parameterized queries, filtering, pagination
├── schemas.py        # Validation for the review form and CSV imports
├── check_env.py      # Command-line environment check
├── config.py         # App configuration and limits (also reads .env)
├── ui/styles.py      # Theme CSS and HTML components (cards, badges, score bars)
├── .streamlit/config.toml   # Theme colours and upload limit
├── services/
│   ├── sentiment.py          # Hugging Face model wrapper -> structured results
│   ├── feedback_service.py   # Analyse pending / re-analyse reviews in batches
│   └── ...                   # topics, analytics, chatbot (later modules)
├── scripts/check_sentiment.py   # Try the model on edge cases / the sample CSV
├── data/
│   └── sample_reviews.csv   # 30 synthetic development reviews (English)
├── tests/            # schemas, database, styles, config, sentiment, feedback service
├── requirements.txt
├── .env.example
└── .gitignore
```

## CSV format

| Column | Required | Notes |
|--------|----------|-------|
| `review_text` | Yes | Max 2,000 characters |
| `product` | No | Max 100 characters |
| `rating` | No | Whole number 1 to 5 |
| `source` | No | Defaults to `csv_import` |
| `created_at` | No | `YYYY-MM-DD` (blank means now); future dates are rejected |

Other columns (such as `expected_sentiment`) are ignored. Files must be UTF-8, at most 5 MB
and 5,000 rows. Invalid rows are reported with their row number and skipped; valid rows still import.

## Sentiment analysis (Module 3)

- Default model: `cardiffnlp/twitter-roberta-base-sentiment-latest` (positive / neutral / negative).
  The **first** analysis downloads it (about 500 MB) and caches it; later runs work offline.
- Reviews are saved first and analysed afterwards, so a model problem never loses feedback.
  Unanalysed reviews show "Pending analysis" and can be processed from the **Sentiment** tab.
- Scores are the model's confidence-like outputs, not guaranteed probabilities of being correct.
- Texts longer than 512 tokens are truncated (the app reports how many).
- **English only.** The default model was trained for English and this project does not support other
  languages. On a test, Spanish and Roman-Urdu reviews were labelled neutral, sometimes with high scores,
  so scores should not be trusted for non-English text.
- Quick checks (the first one downloads the model):

```bash
python scripts/check_sentiment.py --sample
```

```powershell
$env:RUN_MODEL_TESTS = "1"; pytest tests/test_sentiment_model.py -v
```

## Data and privacy

`data/sample_reviews.csv` contains synthetic reviews written for development. Do not commit
real customer feedback. Local database files and `.env` are git-ignored.

## Limitations

To be expanded in Module 7.

## Flexible CSV import (backend)

`services/csv_import.py` reads almost any review CSV: UTF-8 or Windows-1252, comma/semicolon/tab separated.
It suggests which column is the review text, rating, date and product; you confirm before anything is saved.

- Extra columns are kept as JSON in the `extra` column and can be filtered (`database.fetch_reviews(extra_key=..., extra_value=...)`).
- Email/phone-like columns are left out by default and never go into AI prompts.
- Ambiguous dates (01/02/2026) and 1-10 / 1-100 rating scales are detected and reported.
- Every upload gets a `batch_id` and a row in `import_batches`.
- Reviews with a low model confidence or truncated text get a `flags` value (`low_confidence`, `truncated`).
- Old databases are upgraded automatically (columns are only added, existing rows are untouched).
- `DEMO_MODE=1` caps uploads at `DEMO_MAX_CSV_ROWS` rows for free hosting.

## Topics and analytics

- `services/topics.py` tags each English review with one or more topics (delivery, product quality, price, usability, customer support, performance, other) using whole-word keyword rules. No model or network. Call `topics.tag_pending()` after an import.
- `services/analytics.py` answers numeric questions with SQL only: `overview`, `sentiment_trend`, `by_product`, `topic_breakdown`, `compare_periods`, `examples`. All accept the same filters as `database.fetch_reviews`, so the chatbot can quote exact numbers and use an LLM only for wording.

## Topics and analytics

- `services/topics.py` tags each review with one or more topics (delivery, product quality, price, usability,
  customer support, performance, other) using word lists. It is instant, free and needs no model.
  Reviews that match nothing get `other`; `topics IS NULL` means "not tagged yet".
- `services/analytics.py` answers numeric questions with SQL only: overview, sentiment trend (day/week/month),
  by product, topic breakdown, example reviews and period-over-period comparison. The chatbot will quote these
  numbers as facts; an LLM (when available) only words the summary.

## Running the app (chat-first UI)

```
streamlit run app.py
```

- **Chat** (default page): ask questions, attach a CSV with the attachment button, or say "add a review".
  Answers come from SQL over your data; the sidebar is open by default.
- **Data view**: filters (date, product, sentiment, topic, custom CSV column, quality flag, search),
  metric cards, charts and a paginated table with CSV download.
- `DEMO_MODE=1` gives each visitor a private temporary database pre-filled with the sample reviews (for public hosting).
- AI summaries are not configured yet; summaries are rule-based until the optional free LLM provider is added.
- `ui/styles.py` belongs to the old tabbed UI and is no longer used by `app.py`.

## AI question helper (optional) and evaluation

The chatbot always works without any API key: a rule-based parser turns a question into a structured query and SQL answers it.
Rules cannot cover every phrasing, so an optional **question planner** (Google Gemini free tier) takes over when the rules are unsure.

* The model only sees the **question** plus the list of valid products and topics, never your reviews.
* It returns a small JSON plan. `services/planner.py` validates every field (allowed intents, real products and topics, real dates computed by our own code) before it is used. It cannot write SQL or delete anything.
* `services/llm/quota.py` protects the shared free quota: a cap per visitor session, a per-minute cap and a daily cap for the whole app (set below your real limits), and a cache so a repeated question costs nothing. When Google answers "too many requests" the guard reads which limit was hit: a per-minute limit only pauses the AI for about a minute, a per-day limit blocks it until the next Pacific midnight. If the AI is unavailable the rules answer and the chat says so.

Setup (all in `.env`, never in the chat or on GitHub):

```
GEMINI_API_KEY=your key from Google AI Studio
GEMINI_MODEL=the exact model name shown in AI Studio
LLM_DAILY_LIMIT=15      # keep below your real daily limit (aistudio.google.com/rate-limit)
LLM_MINUTE_LIMIT=10     # keep below your real per-minute limit
PLANNER_MODE=hybrid     # rules | hybrid (AI only when rules are unsure) | llm_first
```

Check it with `python scripts/check_gemini.py` (uses one request).

### Speed, timeouts and what a review must look like

With `PLANNER_MODE=llm_first` every question that is not an action waits for Gemini (usually 2-4 seconds); `hybrid` answers clear
questions instantly from the rules and only asks Gemini about the unsure ones. `python scripts/check_gemini.py` prints how long a real
request takes on your connection. If Google does not answer within `LLM_TIMEOUT_SECONDS`, the rules answer instead and the reply says
why; after two timeouts in a row the AI is skipped for a minute, so later questions are not slowed down too.

A review typed into the chat or the form is refused (nothing is saved) when it cannot be a real review: a keyboard mash such as
"mmmmmmmm" or "asdfghjkl", only symbols or numbers, or a single word. CSV import keeps its own row checks.

### Follow-up questions

The chat keeps the **previous question's filters** (intent, sentiment, topic, product, period, topics left out; never review text) for the
current browser session. A question that refers back to it ("and for September?", "what about the desk lamp?", "show me those",
"after fixing delivery, what should we improve first?") borrows only what it leaves out, and says what it kept; anything you state now
replaces the old value, and "overall" / "all products" drops the period / product. "After fixing delivery", "excluding price" or
"apart from packaging" leave a topic out: for *why / what to improve* questions the topic is simply not ranked (reviews that also
mention something else still count), for counts and lists those reviews are filtered out. The AI planner is shown the previous plan
only for such follow-ups (and for very short questions the rules cannot read), and the context is part of its cache key. Importing a
new CSV clears the context.

### Which product drove a change?

"Which product caused the increase in positive reviews in October?" compares the period with the one before it (a whole month
against the previous month, a month so far against the same days of the last month, anything else against the equally long period
before) and shows, per product, positive and negative reviews and the negative share in both periods. The per-product changes add up
exactly to the overall change. Products with fewer than 5 reviews in either period are marked "few reviews" and never named as
drivers. The wording is "most of the rise comes from …", never "caused": reviews show what changed, not why.

### How the answers are worded

Answers are still built by code from SQL results (no AI writes them), but they are phrased like a person would say them: they lead with
the answer, use plain words and contractions ("About a third of the 34 reviews about delivery are negative (11, or 32%)"), say
"in October" rather than a date range, and add a short note about what the data covers when nothing matches. `services/wording.py`
holds the small helpers (period and filter phrases, shares in words). The technical description in the "understood as" line is
unchanged. An optional AI wording layer is a possible next step, not part of this version.

### AI-written summaries

"Summarise what customers say about packaging in two sentences" (also "what do customers say about…", "main themes", "overview")
asks Gemini to word a short summary of a **sample of up to 12 matching review excerpts** (balanced across positive/neutral/negative).
The calculated numbers always come first and the model cannot change them. Before the summary is shown, `services/summarizer.py`
checks it: it must be valid JSON; every cited review id must be one of the excerpts it was given (the cited reviews are shown under
Evidence); every number must appear in the calculated facts; anything in quotation marks must be copied exactly from a review; no
links, addresses or markdown; no more sentences than you asked for. If any check fails, or the quota is gone, you get the calculated
summary instead, with a note saying so. What it cannot check is tone and emphasis, so the answer is labelled as AI-written.

**Privacy:** only these questions send review text to Google, and only the sample (at most 300 characters each) with e-mail
addresses, phone-like numbers and links replaced by placeholders. Free-tier Gemini content may be used by Google to improve its
products, so do not summarise reviews you are not allowed to share. Set `AI_SUMMARIES=0` in `.env` to keep review text on your
machine (counts, lists, rankings and "why" answers never send review text).

### Measuring question understanding

```
python evaluation/run_eval.py                      # rules on the main set (no network)
python evaluation/run_eval.py --set heldout --show # rules on harder, untuned phrasings
python evaluation/run_eval.py --set heldout --planner hybrid --max-calls 15
python evaluation/run_eval.py --set fresh          # newest untuned set: report this number
```

`evaluation/planner_cases.py` holds the cases the rules were built against; `evaluation/heldout_cases.py` holds messier phrasings
the rules were **not** tuned to when it was written, but its failures were used to improve the rules and the prompt, so it is now a
regression test rather than an honest estimate. `evaluation/fresh_cases.py` is the set to report: do not tune against it. Gemini results
are cached in `evaluation/.llm_cache.json` (thrown away automatically when the prompt or model changes), so the free quota can be spread
over several days. With `--planner hybrid|gemini` the script spaces its calls out, waits out a per-minute rate limit and retries once, and
leaves cases Gemini could not answer for API reasons out of the percentage (printed as "NOT SCORED").

Complaint answers also name the specific problems (for example for customer support: rude or unhelpful staff, no reply to emails,
long waits) by reading the review text for common problem phrases (`services/issues.py`). Questions like "Are customers happy with
delivery?" start with a Yes / Mostly yes / Mixed / No before the numbers, and "Which product has the lowest average rating?" returns the
lowest, not the best. A message can hold up to five questions, and "How is A doing compared to B?" compares the two products.

## Deploying the public demo (Streamlit Community Cloud, free)

1. **Build the demo data once, on your computer** (venv active). It runs the real sentiment model over the 246 demo reviews and saves
   `data/demo_seed.db`, so visitors see data immediately and the app does not load the 500 MB model just to open:
   `python scripts/build_demo_seed.py`
2. **Put the project on GitHub** (public repository). Check that no secret goes up: `.env` and `.streamlit/secrets.toml` are git-ignored,
   and `data/demo_seed.db` must be included. `git ls-files` should list `data/demo_seed.db` and must not list `.env`.
3. **Create the app** at https://share.streamlit.io: New app, pick the repository, branch `main`, main file `app.py`.
   Under *Advanced settings* choose **Python 3.12** and paste the contents of `.streamlit/secrets.toml.example` into *Secrets*
   (replace the key with your own). The first build takes several minutes (PyTorch).
4. **Test the live link** in a private window: ask a few questions, open Data view, add a review (this is the first time the model loads),
   upload `data/sample_reviews.csv`.
5. Keep billing **off** on the Google project. When the daily AI limit is used up, the app keeps answering with its rules.

Notes: free apps sleep after a period without visitors and wake in about a minute; visitors never see each other's data (each session has
its own temporary database); `requirements-dev.txt` adds `pytest` for running the tests locally.
