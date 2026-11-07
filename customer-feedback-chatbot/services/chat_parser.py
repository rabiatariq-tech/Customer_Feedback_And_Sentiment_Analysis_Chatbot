"""Turns a question into a structured ``Query`` using fixed rules. No database, no network, no model.

The parser only *understands* the question. ``services/chatbot.py`` runs the Query with safe SQL.
Keeping the two apart means a language model can later produce the very same ``Query`` for questions the
rules cannot parse, and the rest of the code does not change.

Intents: help, add_review, analyze_text, import_help, count, breakdown, trend, list, ranking, average,
compare, reasons, unknown.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

DEFAULT_LIMIT = 5
MAX_LIMIT = 20
MAX_TERMS = 3
INTENTS = (
    "help", "add_review", "save_review", "delete_review", "refuse", "review_prompt", "peak", "mismatch", "analyze_text", "import_help", "count", "breakdown", "trend",
    "list", "ranking", "average", "compare", "reasons", "unknown",
)


@dataclass
class Period:
    date_from: str
    date_to: str
    label: str
    days: int | None = None  # set for rolling windows such as "last 7 days"


@dataclass
class Query:
    intent: str = "unknown"
    sentiment: str | None = None
    topic: str | None = None
    product: str | None = None
    search_terms: list[str] = field(default_factory=list)
    period: Period | None = None
    previous_period: Period | None = None
    limit: int | None = None
    group_by: str | None = None  # "product" for rankings and averages
    text: str | None = None  # the review to analyse or save
    rating: int | None = None  # rating given with a review to save
    review_id: int | None = None  # first review to delete ("delete review #12")
    review_ids: list[int] = field(default_factory=list)  # all ids named ("delete reviews 3, 5 and 8")
    delete_last: bool = False  # "delete the last review I added"
    extreme: str = "max"  # "max" or "min" for "which month had the most/fewest ..."
    granularity: str | None = None  # "day" | "week" | "month" when the user said weekly, daily, monthly
    source: str = "rules"  # "rules" or "ai" (who interpreted the question)
    confident: bool = True  # False when the rules had to guess; the AI planner may take over
    asked_for_text: bool = False  # asked for a sentiment analysis but no review text was found
    order: str | None = None  # "worst" or "best" when the owner asked for the worst/best reviews (not just the newest)
    exclude_topics: list[str] = field(default_factory=list)  # "after fixing delivery", "excluding price": topics to leave out
    follow_up: bool = False  # the question refers back to the previous one ("and for September?", "after fixing delivery ...")
    kept_from_context: list[str] = field(default_factory=list)  # what was carried over from the previous question

    def filters(self, sentiment: bool = True, period: bool = True, exclude: bool = True) -> dict[str, object]:
        out: dict[str, object] = {}
        if exclude and self.exclude_topics:
            out["exclude_topics"] = list(self.exclude_topics)
        if sentiment and self.sentiment:
            out["sentiment"] = self.sentiment
        if self.topic:
            out["topic"] = self.topic
        if self.product:
            out["product"] = self.product
        if self.search_terms:
            out["search"] = [stem(t) for t in self.search_terms]
        if period and self.period:
            out["date_from"], out["date_to"] = self.period.date_from, self.period.date_to
        return out

    def scope(self, sentiment: bool = True, exclusions: bool = True) -> str:
        """Plain-English description of the filters, e.g. 'negative reviews about delivery this month'."""
        noun = f"{self.sentiment} reviews" if (sentiment and self.sentiment) else "reviews"
        parts = [noun]
        if exclusions and self.exclude_topics:
            parts.append("leaving out " + " and ".join(self.exclude_topics))
        if self.topic:
            parts.append(f"about {self.topic}")
        if self.product:
            parts.append(f"for {self.product}")
        if self.search_terms:
            parts.append("mentioning " + " or ".join(f"“{t}”" for t in self.search_terms))
        parts.append(self.period.label if self.period else "overall")
        return " ".join(parts)

    def ui_filters(self) -> dict[str, object]:
        """Filters for the Data view button."""
        out: dict[str, object] = {}
        if self.sentiment:
            out["sentiment"] = self.sentiment
        if self.topic:
            out["topic"] = self.topic
        if self.product:
            out["product"] = self.product
        if self.search_terms:
            out["search"] = ", ".join(self.search_terms)
        if self.period:
            out["date_from"], out["date_to"] = self.period.date_from, self.period.date_to
        return out


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------
def stem(term: str) -> str:
    """Crude singular form so 'watches' also finds 'watch' (a LIKE search then matches both)."""
    t = term.lower().strip()
    if len(t) <= 4:
        return t
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith("es") and t[-3] in "chsxz":
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


_NEG = re.compile(r"negativ|complain|unhappy|dissatisf|disappoint|\bbad\b|\bpoor\b|angr(?:y|ier|iest)|upset|frustrat|\bworst\b|terrible|awful|horrible|problem|issue|\bhat(?:e|es|ed|ing)\b|annoy|furious|unsatisf")
_POS = re.compile(r"positiv|\bhappy\b|\bsatisf(?:ied|action)\b|\bgood\b|\bgreat\b|\blove[sd]?\b|praise|compliment|\bbest\b|\bexcellent\b|\blike[sd]? (?:most|best)\b|\bstrengths?\b|appreciate|\benjoy|\bpleased\b|\bimpressed\b")
_NEU = re.compile(r"neutral|\bmixed\b|\bmeh\b")

TOPIC_PATTERNS = {
    "delivery": r"deliver|shipping|shipment|courier|parcel|\bpackage\b|arriv",
    "packaging": r"packag|packing|\bpacked\b|wrapp|\bbox(?:es)?\b",
    "product quality": r"quality|defect|broke|broken|durab|material|faulty|damag",
    "price": r"\bpric|expens|\bcost|cheap|afford|overpriced|value for money",
    "usability": r"usab|\bapp\b|setup|set up|instructions|interface|easy to use|confusing",
    "customer support": r"support|service|warranty|helpdesk|replacement",
    "returns & refunds": r"refund|\breturn(?:s|ed|ing)?\b|exchange|money back",
    "performance": r"performance|battery|\bspeed|\bslow|crash|\blag|sound|noise",
}

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}

_STOP = set("""
a an the this that these those all any each every some our my your their its we you they it i me us
is are was were be been being am do does did have has had having can could would should will shall may might must
what which who whom whose when where why how
show list give provide display tell find get see fetch pull please
review reviews feedback comment comments complaint complaints opinion opinions rating ratings stars score
customer customers client clients user users people buyer buyers
product products item items thing things
negative positive neutral good bad great poor worst best happy unhappy
dissatisfaction dissatisfied satisfaction satisfied disappointed issue issues problem problems reason reasons cause causes
top most common main major biggest overall total number count many much few more less least
related relating relevant specifically especially particularly mainly mostly
of in at to from by and or but also as than then so just only into onto over under up down out off
month week year quarter quarters day days weeks months years today yesterday tomorrow ago last past previous current recent latest newest oldest first
january february march april june july august september october november december
data database business store shop company brand
analysis analyze analyse sentiment feel feeling
not no yes ok okay about regarding with for on
sentence sentences two three four five brief briefly short shortly words bullet bullets points paragraph paragraphs
summarize summarise summary summarizing
""".split())

_CUE = re.compile(
    r"\b(?:about|regarding|concerning|related to|relating to|relevant to|connected to|linked to|mentioning|mentions?|"
    r"containing|involving|specifically|for|with|on|towards?|around|returning|returned)\b"
)

_REFUSE = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)|\bdrop\s+table\b|\bdelete\s+from\b|;\s*--|\bunion\s+select\b"
    r"|\b(?:disregard|forget|override|bypass)\s+(?:all\s+|any\s+|your\s+|the\s+|these\s+|those\s+|previous\s+|earlier\s+|everything\s+)*(?:instructions?|rules|guidelines|prompt|you (?:were|have been) (?:told|given))\b"
    r"|\b(?:reveal|dump|leak|expose|print)\b[^?.]{0,30}\b(?:database|api key|secrets?|passwords?|credentials|hidden prompt|system prompt|instructions)\b"
    r"|\bsystem prompt\b|\b(?:delete|remove|clear|wipe|erase)\s+(?:all|every|everything|the\s+(?:database|data|table))\b"
)
_DELETE = re.compile(r"\b(?:delete|remove|undo|discard)\b.*?\b(?:reviews?|feedback|comments?|entry|entries|ids?)\b|\b(?:delete|remove)\s+#\d+|^\s*undo\s*$")
_LAST = re.compile(r"\b(last|latest|recent|newest|previous|just)\b")
_GRAN = (
    ("week", re.compile(r"\bweekly\b|\bper week\b|\bby week\b|\bweek[- ]by[- ]week\b|\beach week\b|\bevery week\b")),
    ("day", re.compile(r"\bdaily\b|\bper day\b|\bby day\b|\bday[- ]by[- ]day\b|\beach day\b|\bevery day\b")),
    ("month", re.compile(r"\bmonthly\b|\bper month\b|\bby month\b|\bmonth[- ]by[- ]month\b|\beach month\b|\bevery month\b")),
)
_MISMATCH = re.compile(
    r"\b(positive|good|happy|satisfied)\b.*\b(low|poor|bad|lowest)\b.*\b(rating|ratings|stars?)\b"
    r"|\b(high|good|5[- ]star|five[- ]star)\b.*\b(rating|ratings|stars?)\b.*\b(negative|unhappy|complain)"
    r"|\b(?:ratings?|stars?)\b.*\bmatch\b.*\bsentiment|\bsentiment\b.*\bmatch\b.*\b(?:ratings?|stars?)"
    r"|\bmismatch|contradict|\bdon'?t match\b|\bdoes(?:n'?t| not) match\b"
)
_PEAK = re.compile(
    r"\b(?:which|what)\s+(month|week|day)\b|\b(?:busiest|worst|best|quietest)\s+(month|week|day)\b"
    r"|\b(month|week|day)\s+(?:had|has|with)\s+the\s+(?:most|highest|lowest|fewest|least|worst|best)\b"
)
# "when did we get the most complaints": the unit is not named, so the chatbot falls back to months
_PEAK_WHEN = re.compile(r"\bwhen\b[^?.]{0,40}\b(?:most|fewest|least|highest|lowest|biggest|peak|spike|busiest|quietest)\b")
_HELP = re.compile(r"\b(help|what can you|what do you do|how do i use|examples?)\b")
_ADD = re.compile(r"\b(add|submit|save|enter|log)\b.*\breview\b")
_IMPORT = re.compile(r"\b(upload|import|csv|file|spreadsheet|excel|xlsx)\b")
_IMPORT_HOW = re.compile(r"\b(?:how|can i|could i|want to|would like to|need to|help me)\b[^?.]{0,40}\b(?:import|upload|load|bring in|add)\b[^?.]{0,40}\b(?:csv|file|spreadsheet|excel|reviews)\b")
_COMPARE = re.compile(r"\bcompar|\bversus\b|\bvs\.?\b|\bthan (last|previous|the previous)\b|\bchanged?\b|getting (better|worse)|\bimprov(?:ed|ing)\b|\bworse(?:ned|ning)?\b")
_AVERAGE = re.compile(r"\b(average|avg|mean)\b.*\b(rating|ratings|stars?|score)\b|\b(rating|ratings|stars?)\b.*\b(average|avg|mean)\b|\bhow (?:well )?rated\b")
_COUNT = re.compile(r"\bhow many\b|\bhow much\b[^?.]{0,30}\b(?:feedback|reviews?|complaints?|comments?)\b|\bnumber of\b|\bcount\b|\btotal\b|\bvolume\b")
_BY_PRODUCT = re.compile(r"\b(by|per|each|every|across) products?\b|\b(which|what|top|best|worst|most|least)\b[^?.]{0,25}\bproducts?\b|\bproducts?\b[^?.]{0,20}\b(most|least|highest|lowest|best|worst)\b|\brank(?:ed|ing)?\b[^?.]{0,20}\bproducts?\b|\bproducts?\b[^?.]{0,20}\brank")
_TREND = re.compile(r"\btrends?\b|over time|\bweekly\b|\bdaily\b|\bmonthly\b|\b(?:by|per) (?:week|day|month)\b|timeline|month by month|week by week|day by day")
_BREAKDOWN = re.compile(r"\bsentiments?\b|breakdown|distribution|percent|proportion|\bsplit\b|overall sentiment|sentiment overall|how do (customers|people) feel|how (are|is) (customers|people)|\bshare of\b|\bratio\b")
_LIST_VERB = re.compile(r"\b(show|list|give|provide|display|see|find|get|fetch|pull|print)\b")
_LIST_NOUN = re.compile(r"\b(reviews?|comments?|feedback|complaints?)\b")
_REASONS = re.compile(
    r"\breasons?\b|\bwhy\b|\bcaus(?:es?|ed|ing)\b|root cause|pain points?|summar|overview|insights?|\bimprov(?:e|ement)|prioriti|recommend|"
    r"\bthings? (?:to|we should|we need to) (?:fix|improve|change|work on)\b|\brecurring\b|\brepeat(?:ed|ing|edly)?\b|"
    r"should we|what should|fix first|\b(main|top|biggest|common|major) (issues?|problems?|complaints?|reasons?)\b|"
    r"most common|what are (customers|people) (unhappy|upset|complaining)|what (do|did) (customers|people) (like|love|dislike|hate)"
)
_LIMIT = re.compile(r"\b(?:top|first|latest|last|recent|newest|oldest)\s+(\d{1,3})\b(?!\s*(?:days?|weeks?|months?|years?|quarters?))|\b(\d{1,3})\s+(?:(?:most|recent|latest|newest|oldest|top|worst|best|bad|good|positive|negative|neutral)\s+)*(?:reviews|comments|complaints)\b")

# "Which product caused the increase in positive reviews?": a change between two periods, broken down by product
_CHANGE_WORDS = re.compile(
    r"\b(?:increas\w*|decreas\w*|ris(?:e|es|en|ing)|rose|drop(?:s|ped|ping)?|fall(?:s|ing|en)?|fell|grow(?:s|ing|th|n)?|grew|declin\w*|"
    r"improv\w*|worsen\w*|worse|(?:got|get|getting|gotten) better|chang\w*|jump\w*|spik\w*|surg\w*|shift\w*)\b")
_WHICH_PRODUCT = re.compile(
    r"\b(?:which|what)\b[^?.]{0,25}\bproducts?\b|\bproducts?\b[^?.]{0,30}\b(?:caus\w*|driv\w*|drove|behind|responsible|contribut\w*|account\w*|led to|explain\w*)\b"
    r"|\b(?:by|per|each|every) product\b")


_SINCE_LAST = re.compile(r"\b(?:since|compared (?:to|with)|than|versus|vs\.?)\s+(?:the\s+)?(?:last|previous)\b")


def is_product_change(ql: str) -> bool:
    return bool(_WHICH_PRODUCT.search(ql) and _CHANGE_WORDS.search(ql))


# "summarise what customers say about packaging in two sentences": wants written prose, not only counts
_SUMMARY = re.compile(
    r"\bsummar(?:y|i[sz]e|i[sz]ing)\b|\bwhat (?:do|are|did) (?:customers|people|buyers|shoppers|reviewers)\b[^?.]{0,20}\b(?:say|saying|think|write|wrote|mention|talk)\b"
    r"|\bmain (?:themes|points|complaints|feedback)\b|\bin (?:one|two|three|1|2|3|a few) (?:short )?(?:sentences?|lines?)\b|\boverview\b|\bdescribe\b")
_SENTENCES = re.compile(r"\bin (?:(one|two|three|1|2|3)|a few) (?:short )?(?:sentences?|lines?)\b")


def wants_summary(question: str) -> bool:
    return bool(_SUMMARY.search((question or "").lower()))


def summary_sentences(question: str) -> int:
    """How many sentences the owner asked for ("in two sentences"); 3 when not said."""
    m = _SENTENCES.search((question or "").lower())
    if not m or not m.group(1):
        return 3
    word = m.group(1)
    return int(_NUMBER_WORDS.get(word, word))


_REQUEST = re.compile(r"\b(sentiment|analy[sz]e|analysis|classify|classification)\b|\bpositive or negative\b|\bis (?:this|it)\b[^.?!]{0,20}\b(positive|negative|neutral)\b", re.I)
_DEICTIC = re.compile(r"\b(this|that|the following|following|above|below)\b(?!\s+(?:month|week|year|quarter|day))[^.?!]{0,30}\b(review|text|comment|feedback|sentence|message|statement|remark)\b", re.I)


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def _period(label: str, start: date, end: date, days: int | None = None) -> Period:
    return Period(start.isoformat(), end.isoformat(), f"{label} ({_fmt(start)} – {_fmt(end)})", days)


def _calendar_period(which: str, kind: str, today: date) -> Period:
    if kind == "week":
        monday = today - timedelta(days=today.weekday())
        if which == "this":
            return _period("this week", monday, today)
        return _period("last week", monday - timedelta(days=7), monday - timedelta(days=1))
    if kind == "month":
        first = today.replace(day=1)
        if which == "this":
            return _period("this month", first, today)
        last_day = first - timedelta(days=1)
        return _period("last month", last_day.replace(day=1), last_day)
    if which == "this":
        return _period("this year", today.replace(month=1, day=1), today)
    return _period("last year", date(today.year - 1, 1, 1), date(today.year - 1, 12, 31))


_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}
_NUMBER_WORDS["a couple of"] = "2"
_NUMBER_WORDS["couple of"] = "2"
_NUM_BEFORE_UNIT = re.compile(r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b(?=\s+(?:days?|weeks?|months?|quarters?)\b)")


def _quarter_period(which: str, today: date) -> Period:
    start_month = (today.month - 1) // 3 * 3 + 1
    start = date(today.year, start_month, 1)
    if which == "this":
        return _period("this quarter", start, today)
    end = start - timedelta(days=1)
    return _period("last quarter", date(end.year, (end.month - 1) // 3 * 3 + 1, 1), end)


def parse_periods(ql: str, today: date) -> list[Period]:
    """All time periods mentioned, in the order they appear."""
    found: list[tuple[int, Period]] = []
    ql = _NUM_BEFORE_UNIT.sub(lambda m: _NUMBER_WORDS[m.group(1)], ql)

    def add(match: re.Match, period: Period) -> None:
        found.append((match.start(), period))

    week_spans: list[tuple[int, int]] = []
    for m in re.finditer(
        r"\b(first|1st|second|2nd|third|3rd|fourth|4th|last|final)\s+week\s+(?:of|in)\s+(" + "|".join(_MONTHS) + r")(?:\s+(\d{4}))?\b", ql
    ):
        month = _MONTHS[m.group(2)]
        year = int(m.group(3)) if m.group(3) else (today.year if month <= today.month else today.year - 1)
        n_days = calendar.monthrange(year, month)[1]
        which = m.group(1)
        if which in ("last", "final"):
            start_day, end_day, name = max(1, n_days - 6), n_days, "last week"
        else:
            idx = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3}[which]
            start_day, end_day, name = idx * 7 + 1, min((idx + 1) * 7, n_days), f"{which} week"
        start, end = date(year, month, start_day), min(date(year, month, end_day), today)
        if start <= today:
            add(m, _period(f"in the {name} of {calendar.month_name[month]} {year}", start, end))
            week_spans.append((m.start(), m.end()))
    for m in re.finditer(r"\btoday\b", ql):
        add(m, Period(today.isoformat(), today.isoformat(), "today", 1))
    for m in re.finditer(r"\byesterday\b", ql):
        y = today - timedelta(days=1)
        add(m, Period(y.isoformat(), y.isoformat(), "yesterday", 1))
    for m in re.finditer(r"\b(?:last|past|previous|recent|latest|over the last|in the last)\s+(\d{1,3})\s+(day|week|month)s?\b", ql):
        n = int(m.group(1)) * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        add(m, _period(f"in the last {n} days", today - timedelta(days=n - 1), today, n))
    for m in re.finditer(r"\b(this|current|last|previous)\s+quarter\b", ql):
        add(m, _quarter_period("this" if m.group(1) in ("this", "current") else "last", today))
    for m in re.finditer(r"\bsince\s+(" + "|".join(_MONTHS) + r")(?:\s+(\d{4}))?\b", ql):
        month = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else (today.year if month <= today.month else today.year - 1)
        start = date(year, month, 1)
        if start <= today:
            add(m, _period(f"since {calendar.month_name[month]} {year}", start, today))
    for m in re.finditer(r"\bpast week\b", ql):
        add(m, _period("in the last 7 days", today - timedelta(days=6), today, 7))
    for m in re.finditer(r"\bpast month\b", ql):
        add(m, _period("in the last 30 days", today - timedelta(days=29), today, 30))
    for m in re.finditer(r"\b(this|current|last|previous)\s+(week|month|year)\b", ql):
        if any(a <= m.start() < b for a, b in week_spans):
            continue
        which = "this" if m.group(1) in ("this", "current") else "last"
        add(m, _calendar_period(which, m.group(2), today))
    for m in re.finditer(r"\b(?:in|during|for|of|from)\s+(" + "|".join(_MONTHS) + r")(?:\s+(\d{4}))?\b", ql):
        if any(a <= m.start() < b for a, b in week_spans):
            continue
        month = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else (today.year if month <= today.month else today.year - 1)
        last = calendar.monthrange(year, month)[1]
        start, end = date(year, month, 1), min(date(year, month, last), today)
        if start <= today:
            add(m, _period(f"in {calendar.month_name[month]} {year}", start, end))
    return [p for _, p in sorted(found, key=lambda item: item[0])]


# ---------------------------------------------------------------------------
# Pieces of a question
# ---------------------------------------------------------------------------
def detect_sentiment(ql: str) -> str | None:
    hits = [name for name, pat in (("negative", _NEG), ("positive", _POS), ("neutral", _NEU)) if pat.search(ql)]
    return hits[0] if len(hits) == 1 else None


def detect_topic(ql: str) -> str | None:
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, ql):
            return topic
    return None


_COUNT_WORDS = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b")


_WORST = re.compile(r"\b(?:worst|angriest|nastiest|harshest|most (?:negative|severe|critical))\b")
_BEST = re.compile(r"\b(?:best|happiest|most positive|top[- ]rated|finest)\b")


def detect_order(ql: str) -> str | None:
    """"the three worst reviews" asks for the most negative ones, not the newest negative ones."""
    ql = ql.lower()
    if _WORST.search(ql):
        return "worst"
    if _BEST.search(ql):
        return "best"
    return None


def detect_limit(ql: str) -> int | None:
    ql = _COUNT_WORDS.sub(lambda m: _NUMBER_WORDS[m.group(1)], ql)  # "top three" -> "top 3", "five reviews" -> "5 reviews"
    m = _LIMIT.search(ql)
    if not m:
        return None
    return max(1, min(int(m.group(1) or m.group(2)), MAX_LIMIT))


_EXCLUDE = re.compile(
    r"\b(?:after (?:we |you |i )?(?:fix(?:ing|ed)?|solv(?:ing|ed|e)|resolv(?:ing|ed|e)|address(?:ing|ed)?|handl(?:ing|ed|e)|improv(?:ing|ed|e))|"
    r"excluding|exclude|without|except(?: for)?|other than|apart from|aside from|besides|not counting|ignoring|ignore|"
    r"leaving out|leave out|setting aside|minus)\s+(?:the\s+|our\s+|all\s+|any\s+)?"
    r"([a-z&' -]{3,40}?)(?=\s*(?:,|\.|\?|$|\b(?:what|how|which|show|give|list|tell|who|why|where|then|and then|do|does|is|are|should)\b))"
)


def split_exclusions(ql: str) -> tuple[list[str], str]:
    """Topics the owner wants left out ("after fixing delivery", "excluding price") and the question without
    those phrases, so the left-out topic is not also read as a topic to filter FOR."""
    excluded: list[str] = []
    cleaned = ql
    for m in _EXCLUDE.finditer(ql):
        hits = [t for t, pattern in TOPIC_PATTERNS.items() if re.search(pattern, m.group(1))]
        if hits:
            excluded += [t for t in hits if t not in excluded]
            cleaned = cleaned.replace(m.group(0), " ")
    return excluded, cleaned


# ---------------------------------------------------------------------------
# Follow-up questions: "and for September?", "what about the lamp", "after fixing delivery ..."
# ---------------------------------------------------------------------------
_FOLLOW_START = re.compile(
    r"^\s*(?:and|also|then|now|so|but|ok(?:ay)?|what about|how about|what of|what if|"
    r"after(?: that| we| you| i)?(?: fix(?:ing|ed)?| solv(?:ing|ed|e)| resolv(?:ing|ed|e)| address(?:ing|ed)?)?|"
    r"once(?: we| that| it)|if we|same (?:question )?(?:for|but|with)|the same)\b", re.I)
_FOLLOW_REF = re.compile(r"\b(?:those|these|them|that one|those ones|the same|same period|same product|the ones)\b")
_ALL_TIME = re.compile(r"\b(?:overall|all time|ever|in total|entire|whole history)\b")
_ALL_PRODUCTS = re.compile(r"\b(?:all products|every product|each product|across products|all items)\b")
READ_INTENTS = _READ_INTENTS = ("count", "breakdown", "trend", "list", "ranking", "average", "compare", "reasons", "peak", "mismatch")


def is_follow_up(question: str) -> bool:
    ql = re.sub(r"\s+", " ", (question or "").lower().strip())
    if not ql:
        return False
    return bool(_FOLLOW_START.match(ql)) or (len(ql.split()) <= 10 and bool(_FOLLOW_REF.search(ql)))


def may_refer_back(question: str, q: "Query") -> bool:
    """Should the previous question be offered as context? Yes for an explicit follow-up cue, and for a short
    question the rules could not make sense of on its own ("why is that?")."""
    return is_follow_up(question) or (len((question or "").split()) <= 8 and (q.intent == "unknown" or not q.confident))


def describe_kept(q: "Query", prev: "Query") -> list[str]:
    """What the new question has in common with the previous one (shown to the owner so a wrong guess is visible)."""
    kept = []
    if q.sentiment and q.sentiment == prev.sentiment:
        kept.append(f"{q.sentiment} sentiment")
    if q.topic and q.topic == prev.topic:
        kept.append(f"topic {q.topic}")
    if q.product and q.product == prev.product:
        kept.append(f"product {q.product}")
    if q.period and prev.period and q.period.date_from == prev.period.date_from and q.period.date_to == prev.period.date_to:
        kept.append(f"period {q.period.label}")
    if q.search_terms and q.search_terms == prev.search_terms:
        kept.append("the words " + " or ".join(f"“{t}”" for t in q.search_terms))
    carried = [t for t in q.exclude_topics if t in prev.exclude_topics]
    if carried:
        kept.append("leaving out " + " and ".join(carried))
    return kept


def inherit_context(q: "Query", prev: "Query", question: str) -> "Query":
    """Fill what a follow-up leaves out from the previous question. Only EMPTY fields are filled, so anything the
    owner says now wins; "overall" / "all products" stop the period / product being carried over."""
    if prev is None or q.intent not in _READ_INTENTS + ("unknown",) or prev.intent not in _READ_INTENTS:
        return q
    ql = question.lower()
    q.follow_up = True
    if q.intent == "unknown":
        q.intent, q.confident = prev.intent, True
        q.granularity = q.granularity or prev.granularity
        q.group_by = q.group_by or prev.group_by
        q.extreme = prev.extreme if q.extreme == "max" else q.extreme
    if q.sentiment is None and q.intent != "mismatch":
        q.sentiment = prev.sentiment
    if q.topic is None and q.topic not in q.exclude_topics:
        topic = prev.topic
        q.topic = None if topic in q.exclude_topics else topic
    if q.product is None and not q.search_terms and not _ALL_PRODUCTS.search(ql):
        q.product = prev.product
        if q.product is None:
            q.search_terms = list(prev.search_terms)
    if q.period is None and q.intent != "compare" and not _ALL_TIME.search(ql):
        q.period = prev.period
    q.exclude_topics = list(dict.fromkeys([*q.exclude_topics, *[t for t in prev.exclude_topics if t != q.topic]]))
    q.kept_from_context = describe_kept(q, prev)
    return q


def _stems(text: str) -> str:
    return " ".join(stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()))


def _match_products(ql: str, products: list[str]) -> list[str]:
    """Products named in the question; plurals match ("watches" finds the product "Watch")."""
    question = " " + _stems(ql) + " "
    found = []
    for name in sorted(products, key=len, reverse=True):
        phrase = _stems(name)
        if phrase and f" {phrase} " in question:
            found.append(name)
    if found:
        return found
    # fallback: a distinctive single word ("lamps" -> "Desk Lamp"), only if it points to one product
    counts: dict[str, list[str]] = {}
    for name in products:
        for w in set(_stems(name).split()):
            if len(w) >= 4:
                counts.setdefault(w, []).append(name)
    for w, names in counts.items():
        if len(names) == 1 and f" {w} " in question and names[0] not in found:
            found.append(names[0])
    return found


def _is_topic_word(token: str) -> bool:
    return any(re.search(p, token) for p in TOPIC_PATTERNS.values())


def extract_terms(ql: str, product_names: list[str]) -> list[str]:
    """Search terms that follow cue words such as 'about', 'related to', 'with'.

    "negative reviews related to watches and perfumes" -> ["watches", "perfumes"]. Words we already
    understand (sentiment, topics, periods, generic words) are not search terms.
    """
    known = {stem(w) for name in product_names for w in re.findall(r"[a-z]+", name.lower())}
    pieces = _CUE.split(ql)[1:]  # text after each cue word
    terms: list[str] = []
    for piece in pieces:
        for chunk in re.split(r",|\band\b|\bor\b|&|\bplus\b", piece):
            words = []
            for token in re.findall(r"[a-z]+", chunk.replace("'s", "")):
                if token in _STOP or stem(token) in known or len(token) < 3 or _is_topic_word(token) or token in _MONTHS:
                    if words:
                        break  # a phrase ends at the first ordinary word
                    continue
                words.append(token)
            if words:
                phrase = " ".join(words)
                if phrase not in terms:
                    terms.append(phrase)
    return terms[:MAX_TERMS]


_PRODUCT_FIELD = re.compile(r"\bproduct\s*[:=]\s*(.+?)(?=\s+rating\s*[:=]|\n|$)", re.I)
_RATING_FIELD = re.compile(r"\brating\s*[:=]\s*([1-5])(?:\s*/\s*5)?(?![\d.]|\s*stars?\b)", re.I)
_SAVE_PHRASE = re.compile(
    r"\b(?:please\s+)?(?:add|submit|save|store|insert|enter|log)\s+(?:this|the|that|my|the following)\s+(?:review|feedback|comment)"
    r"(?:\s+(?:in|into|to)\s+(?:the\s+|my\s+)?(?:database|db|data|chatbot|system))?\s*[.!]?",
    re.I,
)


def split_review_fields(raw: str) -> tuple[str, str | None, int | None]:
    """Pull optional "Product: X" and "Rating: N" fields (own lines or inline) out of a review typed in the chat."""
    product = rating = None
    text = raw
    m = _PRODUCT_FIELD.search(text)
    if m:
        product = m.group(1).strip(" .,;")[:100] or None
        text = text[: m.start()] + text[m.end():]
    m = _RATING_FIELD.search(text)
    if m:
        rating = int(m.group(1))
        text = text[: m.start()] + text[m.end():]
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines()).strip(" \n,;:-")
    return text, product, rating


def extract_inline_review(raw: str, force: bool = False) -> tuple[str, str | None, int | None] | None:
    """"<review text> product: X rating: 4 add this review in the database" -> (text, product, rating).

    Returns None when there is not enough review text, so a plain "add a review" still asks for the text.
    """
    cleaned, n_phrases = _SAVE_PHRASE.subn(" ", raw)
    if not n_phrases and not force:
        return None  # no "add this review" instruction, so this is not a save request
    text, product, rating = split_review_fields(cleaned)
    if len(text.split()) >= 2:
        return text, product, rating
    return None


_LABEL_PREFIX = re.compile(r"^(?:this|that|the following|following|the below|below|my)\s+(?:review|text|comment|feedback|sentence|message|statement)\s*[:\-]\s*", re.I)


def extract_review_text(raw: str) -> tuple[bool, str | None]:
    """Detect "analyse this review" requests. Returns (is_request, review_text_or_None)."""
    m = re.match(r"^\s*(?:analy[sz]e|classify|sentiment of)\s*[:\-]?\s*(.+)$", raw, flags=re.I | re.S)
    if m and len(m.group(1).strip()) > 2:
        body = m.group(1).strip()
        labelled = _LABEL_PREFIX.sub("", body)  # "analyze this review: <text>": the label is not part of the text
        if labelled != body:
            body = labelled.strip()
            if not body:
                return True, None
        if len(body.split()) <= 3 and _DEICTIC.search(body):
            return True, None  # "analyze this review" with nothing pasted
        return True, body
    if not _REQUEST.search(raw):
        return False, None
    quotes = re.findall(r"[\"“”]([^\"“”]{12,})[\"“”]", raw)
    if quotes:
        return True, max(quotes, key=len).strip()
    m = re.match(r"^[^:]{0,80}\b(?:sentiment|analy[sz]e|analysis|classify)\b[^:]{0,60}:\s*(.{8,})$", raw, flags=re.I | re.S)
    if m:
        return True, m.group(1).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw.strip()) if s.strip()]
    asking = [s for s in sentences if _REQUEST.search(s) and _DEICTIC.search(s)]
    if asking:
        rest = " ".join(s for s in sentences if s not in asking).strip()
        if len(rest.split()) >= 3:
            return True, rest
    return False, None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
_QUESTION_START = re.compile(r"^\s*(do|does|did|is|are|was|were|will|would|can|could|should|have|has)\b", re.I)


_FILLER = re.compile(
    r"(weekly|daily|monthly|week|month|year|days?|weeks?|months?|quarter|lately|recent|recently|ago|since|over|time|first|last|next|"
    r"also|please|tell|just|some|those|that|with|have|from|they|them|were|been|does|what|when|where|which|there|their|would|could|"
    r"should|about|into)"
)


def unexplained_words(ql: str, products: list[str]) -> list[str]:
    """Content words the rules did not recognise. Two or more means the rules are probably misreading the question."""
    known = {stem(w) for p in products for w in re.findall(r"[a-z]+", p.lower())}
    cues = (_NEG, _POS, _NEU, _COUNT, _TREND, _BREAKDOWN, _COMPARE, _AVERAGE, _LIST_NOUN, _LIST_VERB, _REASONS, _BY_PRODUCT, _PEAK, _MISMATCH, _HELP, _CUE)
    out = []
    for token in re.findall(r"[a-z]+", ql):
        if len(token) < 4 or token in _STOP or token in _MONTHS or stem(token) in known or _is_topic_word(token) or _FILLER.fullmatch(token):
            continue
        if any(p.search(token) for p in cues):
            continue
        out.append(token)
    return out


def _looks_like_bare_review(raw: str, ql: str) -> bool:
    """A statement about a product (not a question or command): ask what to do with it instead of guessing."""
    if len(ql.split()) < 6 or "?" in raw or _COMMAND_START.match(raw) or _QUESTION_START.match(raw):
        return False
    if re.search(r"\b(customers?|people|users?|buyers?|shoppers?)\b", ql):
        return False  # a review is about the writer, not about "customers"
    if any(p.search(ql) for p in (_COUNT, _TREND, _COMPARE, _AVERAGE, _BREAKDOWN, _REASONS, _LIST_VERB, _LIST_NOUN)):
        return False
    return bool(re.search(r"\b(i|i'm|i've|my|we|our|it|this|the)\b", ql) and re.search(r"\b(was|is|are|were|bought|ordered|love|loved|hate|hated|arrived|broke|works?|stopped|got|received|like|liked)\b", ql))


def parse(question: str, products: list[str] | None = None, today: date | None = None) -> Query:
    today = today or date.today()
    products = [p for p in (products or []) if p]
    raw = (question or "").strip()
    ql = re.sub(r"\s+", " ", raw.lower())
    q = Query()
    if not ql:
        return q

    if _HELP.search(ql):
        q.intent = "help"
        return q
    if _REFUSE.search(ql):
        q.intent = "refuse"
        return q
    dm = _DELETE.search(ql)
    if dm:
        after = ql[dm.start():]
        ids = [] if _LAST.search(after) else [int(n) for n in re.findall(r"\d+", after)][:10]
        if ids or _LAST.search(after) or ql.strip() == "undo":
            q.intent = "delete_review"
            q.review_ids = ids
            q.review_id = ids[0] if ids else None
            q.delete_last = not ids
            return q
    if _ADD.search(ql) or ql.startswith("/review"):
        inline = extract_inline_review(raw[len("/review"):], force=True) if ql.startswith("/review") else extract_inline_review(raw)
        if inline:
            q.intent, (q.text, q.product, q.rating) = "save_review", inline
            return q
        q.intent = "add_review"
        return q
    is_request, text = extract_review_text(raw)
    if is_request:
        q.intent, q.text = "analyze_text", text
        return q
    if _IMPORT.search(ql) and (not _LIST_NOUN.search(ql) or _IMPORT_HOW.search(ql)):
        q.intent = "import_help"
        return q

    if _looks_like_bare_review(raw, ql):
        q.intent, q.text = "review_prompt", raw
        return q

    # ---- filters ----
    periods = parse_periods(ql, today)
    if periods:
        q.period = periods[0]
        if len(periods) > 1:
            q.previous_period = periods[1]
    q.sentiment = detect_sentiment(ql)
    q.exclude_topics, ql_topics = split_exclusions(ql)
    q.topic = detect_topic(ql_topics)
    matched = _match_products(ql, products)
    terms = extract_terms(ql, products)
    if len(matched) == 1 and not terms:
        q.product = matched[0]
    elif matched:
        # several things were named ("watches and perfumes"): match any of them in the text or product name
        terms = ([p.lower() for p in matched] + terms)[:MAX_TERMS]
    q.search_terms = terms
    q.limit = detect_limit(ql)
    for gran_name, gran_pat in _GRAN:
        if gran_pat.search(ql):
            q.granularity = gran_name
            break

    # ---- intent (first match wins) ----
    by_product = bool(_BY_PRODUCT.search(ql)) and bool(re.search(r"\bproducts?\b", ql))
    peak = _PEAK.search(ql)
    if _MISMATCH.search(ql):
        q.intent = "mismatch"
        q.sentiment = None
    elif (peak or _PEAK_WHEN.search(ql)) and not _COMPARE.search(ql):
        q.intent = "peak"
        if peak:
            q.granularity = next(g for g in peak.groups() if g)
        if re.search(r"\b(fewest|lowest|least|quietest)\b", ql):
            q.extreme = "min"
    elif (is_product_change(ql) or (by_product and len(periods) >= 2 and _COMPARE.search(ql))) and not _COUNT.search(ql):
        q.intent, q.group_by = "compare", "product"
        if q.previous_period and q.period and q.previous_period.date_from > q.period.date_from:
            q.period, q.previous_period = q.previous_period, q.period  # the newer period is the "current" one
        elif q.period and q.previous_period is None and _SINCE_LAST.search(ql):
            m = re.match(r"last (week|month|year)\b", q.period.label)
            if m:  # "since / compared to last month" means: this month so far against last month
                q.previous_period, q.period = q.period, _calendar_period("this", m.group(1), today)
    elif _COMPARE.search(ql) and not by_product and not (
        q.period is None and not re.search(r"compar|versus|\bvs\b|\bthan\b|changed?\b", ql)
    ) and not (q.granularity and not re.search(r"compar|versus|\bvs\b|\bthan\b", ql)):
        q.intent = "compare"
        if q.period and q.previous_period is None:
            m = re.match(r"last (week|month|year)\b", q.period.label)
            if m:  # "compared to last month" means: this month so far against last month
                q.previous_period = q.period
                q.period = _calendar_period("this", m.group(1), today)
        if q.previous_period and q.period and q.previous_period.date_from > q.period.date_from:
            q.period, q.previous_period = q.previous_period, q.period  # keep the newer period as "current"
    elif _AVERAGE.search(ql):
        q.intent = "average"
        if not re.search(r"positiv|negativ|neutral", ql):
            q.sentiment = None  # "best average rating" is not a request for positive reviews only
        q.group_by = "product" if (by_product or re.search(r"\bwhich\s+products?\b|\b(?:best|worst|highest|lowest)[- ]rated\b", ql)) else None
    elif _COUNT.search(ql) and not re.search(r"\b(by|per|each|every) product", ql) and not q.granularity and not (
        by_product and re.search(r"\b(?:rank\w*|sort\w*|order(?:ed)?)\b", ql)  # "rank products by number of complaints"
    ):
        q.intent = "count"  # (a count split into days/weeks/months is a trend, handled below)
    elif by_product:
        q.intent = "ranking"
        q.group_by = "product"
    elif _TREND.search(ql) or (_COMPARE.search(ql) and q.period is None):
        q.intent = "trend"  # "getting worse over time" with no period named is a trend question
    elif _BREAKDOWN.search(ql):
        q.intent = "count" if q.sentiment else "breakdown"  # "what percent are negative" is a count with a share
    elif q.limit and _LIST_NOUN.search(ql):
        q.intent = "list"
    elif _REASONS.search(ql):
        q.intent = "reasons"
    elif _LIST_NOUN.search(ql) and (_LIST_VERB.search(ql) or q.sentiment or q.period or re.search(r"\b(recent|latest|newest|oldest)\b", ql)):
        q.intent = "list"
        asked_to_list = _LIST_VERB.search(ql) or re.search(r"\b(recent|latest|newest|oldest)\b", ql)
        if not asked_to_list and (q.granularity or unexplained_words(ql, products) or re.search(r"\b(when|why|how)\b", ql)):
            q.confident = False  # a bare noun phrase plus leftovers ("volume of ...", "what is causing ..."): not clearly a list request
    elif (q.sentiment or q.topic or q.product or q.search_terms) and (
        _LIST_NOUN.search(ql) or re.search(r"\b(customers?|people|users?|buyers?)\b", ql)
    ):
        q.intent = "list" if (q.sentiment and _LIST_NOUN.search(ql)) else "reasons"
        q.confident = False  # no clear cue word, the rules are guessing
    if q.intent == "unknown" or len(unexplained_words(ql, products)) >= 2:
        q.confident = False
    if q.intent == "reasons" and not (q.product or q.topic or q.search_terms) and not _ANCHOR.search(ql) and not _DOMAIN.search(ql):
        q.confident = False  # "recommend a restaurant", "summarise this pdf": nothing ties it to the reviews, so it may be off-topic
    return q


# ---------------------------------------------------------------------------
# Several questions in one message
# ---------------------------------------------------------------------------
_SPLIT_AT = re.compile(
    r"\s*(?:\?+|;|,|\band\b|\balso\b|\bthen\b)\s+(?=(?:and\s+)?(?:how\s+(?:many|much|do|are|is|did|does)|what|which|why|who|"
    r"show|list|give|provide|compare|count|tell|rank|sort|plot|chart|summari[sz]e|find|display|describe|explain|identify|when|where|"
    r"are\s+there|is\s+there|negative|positive|neutral)\b)",
    re.I,
)
_QUALIFIER = re.compile(r"\b(about|related|regarding|mentioning|for|with|in|of|from|this|last|today|yesterday|week|month)\b", re.I)
_SENT_START = re.compile(r"^(?:and\s+)?(negative|positive|neutral)\b", re.I)
MAX_PARTS = 3
_LEAD_IN = re.compile(r"^\s*(?:and\s+)?(?:after|once|if|when|before|since|because|given|assuming|now that|with)\b", re.I)


def split_question(raw: str) -> list[str]:
    """Split "how many reviews today and how many are negative?" into separate questions."""
    raw = raw.strip()
    pieces: list[str] = []
    start = 0
    for m in _SPLIT_AT.finditer(raw):
        if _LEAD_IN.match(raw[start:m.start()]):
            continue  # "after fixing delivery, what should we improve?" is one question with a lead-in clause
        rest = raw[m.end():]
        head = _SENT_START.match(rest)
        if head and not (len(rest.split()) >= 4 and _QUALIFIER.search(rest)):
            continue  # "positive and negative reviews" is one question
        pieces.append(raw[start:m.start()])
        start = m.end()
    pieces.append(raw[start:])
    cleaned = [re.sub(r"^(?:and|also|then)\s+", "", p.strip(" ?.,;"), flags=re.I) for p in pieces]
    cleaned = [p for p in cleaned if len(p.split()) >= 2]
    return cleaned[:MAX_PARTS] if len(cleaned) > 1 else [raw]


_SINGLE_INTENTS = {"help", "add_review", "save_review", "delete_review", "refuse", "analyze_text", "import_help"}


def parse_many(question: str, products: list[str] | None = None, today: date | None = None) -> list[tuple[str, Query]]:
    """One (text, Query) per question in the message. Later parts inherit the period and product of earlier ones."""
    whole = parse(question, products, today)
    if whole.intent in _SINGLE_INTENTS:
        return [(question, whole)]
    parts = split_question(question)
    if len(parts) < 2:
        return [(question, whole)]
    out: list[tuple[str, Query]] = []
    for text in parts:
        q = parse(text, products, today)
        if out:
            prev = out[-1][1]
            if q.period is None and prev.period is not None and q.intent != "compare":
                q.period = prev.period
            if q.product is None and not q.search_terms and prev.product is not None:
                q.product = prev.product
            # "how many reviews about delivery and negative reviews about price": the second is a count too
            if prev.intent == "count" and q.intent == "list" and not q.limit and not _LIST_VERB.search(text.lower()):
                q.intent = "count"
        out.append((text, q))
    return out


_DOMAIN = re.compile(r"improv|\bfix|\bissues?\b|\bproblems?\b|complain|unhappy|pain point|negativ|dissatisf|disappoint")
_ANCHOR = re.compile(r"\b(reviews?|feedback|comments?|complaints?|customers?|people|shoppers?|buyers?|users?|products?)\b")
_COMMAND_START = re.compile(
    r"^\s*(how|what|which|why|who|show|list|give|provide|compare|tell|delete|remove|undo|help|add|save|i want to|i would like to|can you|could you|please)\b",
    re.I,
)


def looks_like_command(text: str) -> bool:
    """True for messages that are questions or commands rather than review text (used while waiting for a review)."""
    t = text.strip()
    return t.endswith("?") or bool(_COMMAND_START.match(t))
