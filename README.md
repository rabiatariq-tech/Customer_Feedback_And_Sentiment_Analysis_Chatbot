# Customer Feedback Assistant

A chat-first web app that lets an online-store owner **upload customer reviews and simply ask questions about them**, such as "What are the top complaints about delivery?" or "Are refund complaints getting worse?", and get answers backed by real numbers and real review quotes.

**Live demo:** https://customer-reviews-feedback-assistant.streamlit.app/

The demo opens with 246 synthetic reviews across 8 products. You can ask questions straight away, upload your own CSV, or paste a single review.

## 1. The problem this solves

Small and medium online sellers collect hundreds or thousands of reviews, but nobody has time to read them all. Today they either:

- skim a few reviews and guess, or
- export a spreadsheet and build pivot tables by hand, or
- pay for an analytics tool that is too heavy for a small shop.

The questions an owner really has are simple: *What do customers hate? Is it getting worse? Which product should I fix first? Which one should I promote?* This app answers those questions in plain English, from the owner's own data, without any dashboard training.

## 2. What it does

- **Upload reviews** from a CSV (any column layout; the review-text column is auto-detected and you confirm it), or type or paste a single review.
- **Automatic sentiment** (positive / neutral / negative) on every review, using a local machine-learning model.
- **Automatic topic tags** per review: delivery, packaging, product quality, price, usability, customer support, returns & refunds, performance.
- **Ask questions in plain English**, including several at once:
  - counts and breakdowns: "How many negative reviews do we have?"
  - rankings: "Which product has the lowest average rating?"
  - trends: "How have delivery complaints changed month by month?"
  - comparisons: "Compare the Smart Speaker with the Running Shoes", "Is it better than last month?"
  - reasons: "Why are customers returning the running shoes?"
  - decisions: "What should we fix first?", "Which product should we promote?"
  - examples: "Show me 3 reviews where customers complain about battery life"
- **Charts and a Data page** with filters, a review table, and delete with confirmation. The trend chart has a Day / Week / Month / Auto selector and a "Share %" view so it stays readable on large datasets.
- **Optional AI-written summaries** for "why" and "what should we fix" questions. These are checked automatically and replaced by a rule-based answer if they fail the checks.
- **Safe by design:** the chatbot refuses prompt-injection attempts, never lets the AI run SQL, and asks for confirmation before deleting anything.

## 3. How it works (short version)

```
Owner's question
      |
      v
 Question understanding  --->  rules first (fast, free, predictable)
      |                         + Gemini "planner" for questions the rules are unsure about
      v
 A validated Query object  (intent, sentiment, topic, product, period, limit ...)
      |
      v
 Handler runs parameterised SQL on SQLite   <-- all numbers come from here
      |
      v
 Template wording + insight text (plain English, with the real figures)
      |
      v
 Answer (+ optional chart / example reviews / checked AI summary)
```

The key design rule: **the AI never computes numbers and never writes SQL.** It only helps to *understand* the question (and, optionally, to word an explanation using facts that code has already verified). That is what makes the answers trustworthy.

## 4. Architecture

Layers: `ui/` -> `services/` -> `database.py` (SQLite). The UI never touches the database directly.

| Layer | Responsibility |
|---|---|
| `app.py`, `ui/` | Streamlit pages (Chat, Data), charts, theme, session handling |
| `services/chat_parser.py` | Rule-based question understanding -> `Query` object |
| `services/planner.py` | Gemini planner: returns a JSON plan that is strictly validated |
| `services/chatbot.py` | Router + answer handlers (count, breakdown, trend, list, ranking, average, compare, peak, mismatch, reasons, delete ...) |
| `services/owner_questions.py` | Decision questions: what to improve, what to promote, mixed and "both topics" questions |
| `services/insights.py` | Plain-English trend reading, significance check |
| `services/wording.py`, `issues.py`, `concepts.py` | Natural wording templates, specific-problem lexicon, synonym expansion |
| `services/summarizer.py`, `product_change.py` | Validated AI summaries, per-product change analysis |
| `services/llm/provider.py`, `quota.py` | Gemini REST client, rate-limit and daily-quota guard |
| `services/sentiment.py`, `topics.py` | Sentiment model wrapper, topic tagger |
| `services/csv_import.py`, `import_service.py` | Flexible CSV import |
| `database.py` | SQLite access with parameterised queries |
| `evaluation/`, `tests/`, `scripts/` | Evaluation sets, 365 automated tests, helper scripts |

## 5. Tech stack

- **Python 3.10+**, **Streamlit** (UI), **SQLite** (storage), **pandas** and **Altair** (data and charts)
- **Hugging Face Transformers + PyTorch (CPU)** with `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment
- **Google Gemini API (free tier)** through plain REST calls (no SDK), used only for question understanding and optional summaries
- **pytest** for tests
- Deployed on **Streamlit Community Cloud**

## 6. Design decisions worth knowing

1. **Local sentiment model, not an LLM API.** Runs offline, costs nothing, and keeps the project from being a thin API wrapper.
2. **Numbers come only from SQL.** Language models are unreliable at arithmetic and may invent figures. Code counts; wording is separate.
3. **Rules first, AI second.** Rules handle common questions instantly and for free. The AI planner handles unusual phrasing, and its output is validated (allow-listed intents, topics and products; limits capped; dates computed by our code).
4. **Works without the AI.** If the quota is used up or the API fails, the app falls back to rules and shows a notice. Import, sentiment, charts and most questions still work.
5. **Quota guard.** Per-session, per-minute and daily caps, a response cache, rate-limit (429) handling and a timeout circuit breaker keep the free tier from being exhausted.
6. **Honest statistics.** A trend is called "rising" or "falling" only when a Fisher exact test says the change is unlikely to be chance. Recommendations use a Wilson lower-bound score and a minimum review count so one angry review does not win.
7. **Validated AI summaries.** Every cited review ID must exist, every number must appear in the verified facts, and every quote must be an exact substring of a real review. Otherwise a rule-based answer is shown with a badge.
8. **No RAG / no AI wording layer (on purpose).** Both would add latency and memory use on a free host. Wording uses deterministic templates instead.
9. **Privacy.** Email/phone-like columns are excluded by default (including from AI prompts). In the public demo each visitor gets their own temporary database, which is discarded afterwards.

## 7. Evaluation and honest limits

- **Sentiment model:** about **80% accuracy on a 30-review labelled sample**; weakest on mixed and neutral reviews. This is a small sample, so treat it as indicative, not definitive.
- **Question understanding** was measured on separate question sets (92 tuned cases, then held-out sets). Rules alone scored about 67% on an untuned held-out set; hybrid (rules + Gemini) scored about 79-81%; Gemini-first scored about 95% on a later set. Note that every set was used for tuning once it was run, so these numbers are **optimistic** for brand-new phrasing. The most reliable evidence is live testing with real business-owner questions, which is how most bugs were found.
- **Automated tests:** 365 passing, 2 skipped (the skipped ones need the real model).

Known limitations:

- English reviews only.
- The specific-problem lexicon and synonym lists were tuned on the demo data's wording; unfamiliar vocabulary may fall back to a more general answer.
- Reviews carry no customer identity, so questions such as "which customer wrote the most reviews" are declined.
- Some rule-based follow-up phrasings (for example "only about price" after a list) are not understood.
- Gemini free-tier limits are variable and can be reached; data sent to the free tier may be used by Google to improve its products, so do not upload sensitive reviews in AI mode.
- Sentiment on sarcasm, mixed and very short reviews is imperfect.

## 8. Run it locally

```powershell
git clone https://github.com/rabiatariq-tech/<repo-name>.git
cd <repo-name>
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # then edit .env
streamlit run app.py
```

The first start downloads the sentiment model (a few hundred MB). To work without AI, set `PLANNER_MODE=rules`.

### Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | Your free key from Google AI Studio (never commit it) |
| `GEMINI_MODEL` | Model id, for example `gemini-3.5-flash-lite` |
| `PLANNER_MODE` | `rules`, `hybrid` (AI only when rules are unsure) or `llm_first` |
| `LLM_SESSION_LIMIT` / `LLM_MINUTE_LIMIT` / `LLM_DAILY_LIMIT` | AI call caps |
| `AI_SUMMARIES` | Turn validated AI summaries on or off |
| `DEMO_MODE` | `1` = per-visitor temporary database and a row cap (300) |
| `SHOW_ANSWER_DETAILS` | `1` shows how each answer was produced (useful while testing) |

### Run the tests

```powershell
python -m pytest
```

### Optional scripts

- `python scripts/check_gemini.py` checks your key and model
- `python scripts/recount.py` recounts answers directly from SQL to verify them
- `python scripts/build_demo_seed.py` rebuilds the pre-analysed demo database

## 9. Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (include `data/demo_seed.db`; do **not** commit `.env`).
2. Create the app at share.streamlit.io, main file `app.py`, Python 3.12.
3. Add secrets (see `.streamlit/secrets.toml.example`): `GEMINI_API_KEY`, `PLANNER_MODE`, `DEMO_MODE = "1"` and the limits.
4. `requirements.txt` uses the CPU-only PyTorch index so the app fits in the free memory limit. The demo starts instantly because sentiment is pre-computed in the seed database.

## 10. CSV format

Any CSV with a review-text column works. Helpful optional columns: product, rating (1-5), date. Extra columns are kept. The app detects the text column and asks you to confirm.

## 11. Roadmap

- Fresh evaluation set written before any tuning
- Better follow-up handling for rule-only mode
- Larger labelled set for sentiment evaluation
- Optional semantic search if keyword synonyms prove insufficient

## 12. Author

Built by Rabia as a portfolio project (BS Computer Science, GCU Lahore). GitHub: [rabiatariq-tech](https://github.com/rabiatariq-tech)
