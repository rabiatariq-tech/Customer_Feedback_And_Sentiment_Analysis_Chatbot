from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

import database
from services import chatbot, planner as planner_mod
from services.llm.provider import LLMResult
from services.llm.quota import QuotaGuard
from tests.test_chatbot import add, blocks, seed, text_of  # reuse the seeded shop

TODAY = date(2026, 10, 21)
PRODUCTS = ["Desk Lamp", "Blender Pro"]


class FakeProvider:
    name = "fake"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def generate_json(self, system, user, max_output_tokens=400):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else LLMResult(False, error="no reply")
        return reply if isinstance(reply, LLMResult) else LLMResult(True, data=reply)


def plan(**kw):
    base = {"intent": "count", "sentiment": None, "topic": None, "product": None, "search_terms": [], "period": None,
            "compare_with": None, "limit": None, "granularity": None, "group_by": None, "extreme": None, "clarification": None}
    base.update(kw)
    return base


def test_valid_plan_becomes_a_query_with_real_dates():
    q, clar, problems = planner_mod.validate_plan(
        plan(intent="reasons", sentiment="negative", product="desk lamp", period={"kind": "last_n_days", "n": 60}), PRODUCTS, TODAY)
    assert q.intent == "reasons" and q.product == "Desk Lamp" and q.source == "ai"
    assert (q.period.date_from, q.period.date_to) == ("2026-08-23", "2026-10-21") and not problems


def test_validation_rejects_or_neutralises_bad_plans():
    v = planner_mod.validate_plan
    assert v("DROP TABLE", PRODUCTS, TODAY)[0] is None
    assert v(plan(intent="delete_review"), PRODUCTS, TODAY)[0] is None          # no actions from the model
    assert v(plan(intent="run_sql"), PRODUCTS, TODAY)[0] is None
    q, _, problems = v(plan(sentiment="angry", topic="weather", limit=999, granularity="hour"), PRODUCTS, TODAY)
    assert q.sentiment is None and q.topic is None and q.limit == 20 and q.granularity is None and problems
    q, _, problems = v(plan(product="Running Shoes"), PRODUCTS, TODAY)               # unknown product -> searched as text
    assert q.product is None and q.search_terms == ["running shoes"]
    q, _, _ = v(plan(search_terms=["ok word", "x'; DROP TABLE feedback;--", "a"]), PRODUCTS, TODAY)
    assert q.search_terms == ["ok word"]
    q, _, _ = v(plan(period={"kind": "between", "from": "2026-12-01", "to": "2027-01-01"}), PRODUCTS, TODAY)
    assert q.period is None                                                         # the future is not a period
    q, _, _ = v(plan(intent="compare", period={"kind": "last_month"}, compare_with={"kind": "this_month"}), PRODUCTS, TODAY)
    assert q.period.label.startswith("this month") and q.previous_period.label.startswith("last month")


def test_quota_guard_limits_sessions_days_and_caches():
    now = [datetime(2026, 10, 21, 12, tzinfo=ZoneInfo("America/Los_Angeles"))]
    g = QuotaGuard(daily_limit=3, session_limit=2, clock=lambda: now[0])
    assert g.try_acquire("a")[0] and g.try_acquire("a")[0]
    assert g.try_acquire("a") == (False, "session_limit")
    assert g.try_acquire("b")[0]
    assert g.try_acquire("c") == (False, "daily_limit")
    now[0] = datetime(2026, 10, 22, 0, 1, tzinfo=ZoneInfo("America/Los_Angeles"))   # new Pacific day
    assert g.try_acquire("c")[0]
    g.provider_said_quota_gone()
    assert g.try_acquire("d") == (False, "daily_provider_quota")
    g.cache_put("k", {"x": 1})
    assert g.cache_get("k") == {"x": 1} and g.cache_get("missing") is None


def make_planner(*replies, daily=10, session=5):
    provider = FakeProvider(*replies)
    return planner_mod.QuestionPlanner(provider, QuotaGuard(daily, session)), provider


def test_planner_caches_and_counts_calls():
    p, provider = make_planner(plan(intent="count", sentiment="negative"))
    a = p.plan("how many unhappy people", PRODUCTS, TODAY, "s1")
    b = p.plan("How many   unhappy people", PRODUCTS, TODAY, "s2")
    assert a.status == "ok" and b.status == "cached" and provider.calls == 1


def test_planner_reports_quota_and_errors_without_raising():
    p, _ = make_planner(LLMResult(False, error="HTTP 429", quota_exhausted=True))
    assert p.plan("q one", PRODUCTS, TODAY, "s").status == "error"
    assert p.plan("q two", PRODUCTS, TODAY, "s").status == "quota"     # Google said the quota is gone
    p, _ = make_planner(plan(intent="delete_review"))
    assert p.plan("remove everything", PRODUCTS, TODAY, "s").status == "invalid"


# ---- routing: when the AI is used -----------------------------------------------------------------
def routed(db, question, planner, mode="hybrid"):
    return chatbot.route(question, db, TODAY, planner=planner, planner_mode=mode)


def test_confident_rules_do_not_spend_a_call(db):
    seed(db)
    p, provider = make_planner(plan(intent="list"))
    r = routed(db, "how many negative reviews this month", p)
    assert provider.calls == 0 and "**" in text_of(r)


def test_uncertain_rules_hand_over_to_the_planner(db):
    seed(db)
    p, provider = make_planner(plan(intent="count", sentiment="negative", period={"kind": "this_month"}))
    r = routed(db, "any idea how many folks were let down lately", p)
    assert provider.calls == 1
    assert "interpreted by AI planner" in blocks(r, "route")[0]["text"] and "negative review" in text_of(r)


def test_planner_failure_falls_back_to_rules_with_a_notice(db):
    seed(db)
    p, _ = make_planner(LLMResult(False, error="timeout"))
    r = routed(db, "where are we losing customers", p)
    assert "AI question helper is unavailable" in text_of(r)


def test_clarification_is_shown_instead_of_guessing(db):
    seed(db)
    p, _ = make_planner(plan(intent="unknown", clarification="Do you mean complaints or all reviews?"))
    r = routed(db, "tell me about stuff", p)
    assert "Do you mean complaints" in text_of(r)


def test_llm_first_mode_and_unknown_product_abstains(db):
    seed(db)
    p, provider = make_planner(plan(intent="reasons", product="Running Shoes"))
    r = routed(db, "why are customers returning the running shoes", p, mode="llm_first")
    assert provider.calls == 1 and "no reviews that mention" in text_of(r)


def test_actions_never_go_to_the_ai(db):
    seed(db)
    p, provider = make_planner(plan(intent="count"))
    routed(db, "delete review 999", p, mode="llm_first")
    routed(db, "help", p, mode="llm_first")
    assert provider.calls == 0 and database.count_reviews(db) == 7


def test_notice_names_the_real_reason_the_ai_was_skipped(db):
    seed(db)
    g = QuotaGuard(10, 5)
    g.provider_said_quota_gone("HTTP 429: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    p = planner_mod.QuestionPlanner(FakeProvider(), g)
    r = routed(db, "where are we losing customers", p)
    assert "rate limit" in text_of(r) and "daily limit" not in text_of(r)


def test_planner_passes_googles_message_to_the_guard_and_resets_after_success():
    g = QuotaGuard(10, 5)
    p = planner_mod.QuestionPlanner(FakeProvider(LLMResult(False, error="HTTP 429: ...PerDay...", quota_exhausted=True)), g)
    assert p.plan("a question", PRODUCTS, TODAY, "s").status == "error"
    assert g.try_acquire("s") == (False, "daily_provider_quota")           # the message said "per day"
    g2 = QuotaGuard(10, 5)
    g2.provider_said_quota_gone("HTTP 429")
    g2.clear_provider_pause()
    p2 = planner_mod.QuestionPlanner(FakeProvider(plan(intent="count")), g2)
    assert p2.plan("another question", PRODUCTS, TODAY, "s").status == "ok" and g2._consecutive_429 == 0


# ---- follow-up context for the AI planner ----------------------------------------------------------
class RecordingProvider(FakeProvider):
    def __init__(self, *replies):
        super().__init__(*replies)
        self.prompts = []

    def generate_json(self, system, user, max_output_tokens=400):
        self.prompts.append(user)
        return super().generate_json(system, user, max_output_tokens)


def recording_planner(*replies):
    provider = RecordingProvider(*replies)
    return planner_mod.QuestionPlanner(provider, QuotaGuard(10, 5)), provider


def test_excluded_topics_are_validated_like_every_other_field():
    q, _, problems = planner_mod.validate_plan(plan(intent="reasons", exclude_topics=["delivery", "weather", "other", "price"], topic="price"),
                                               PRODUCTS, TODAY)
    assert q.exclude_topics == ["delivery"] and problems                  # unknown, "other" and the topic being asked for are dropped
    q, _, _ = planner_mod.validate_plan(plan(intent="reasons", exclude_topics="delivery"), PRODUCTS, TODAY)
    assert q.exclude_topics == []                                           # not a list: ignored, never an error


def test_previous_plan_is_sent_to_the_model_and_is_part_of_the_cache_key():
    from services.chat_parser import Query
    prev = Query(intent="count", sentiment="negative", product="Desk Lamp",
                 period=planner_mod.to_period({"kind": "last_month"}, TODAY))
    p, provider = recording_planner(plan(intent="count", sentiment="positive", product="Desk Lamp"),
                                    plan(intent="count", sentiment="positive"))
    p.plan("and how many are positive", PRODUCTS, TODAY, "s", previous=prev)
    assert "Previous question's plan:" in provider.prompts[0] and '"product": "Desk Lamp"' in provider.prompts[0]
    assert '"from": "2026-09-01"' in provider.prompts[0]                    # the old period is handed over as real dates
    p.plan("and how many are positive", PRODUCTS, TODAY, "s")             # same words, no context: must not reuse that answer
    assert provider.calls == 2 and "Previous question's plan" not in provider.prompts[1]
    again = p.plan("and how many are positive", PRODUCTS, TODAY, "s", previous=prev)
    assert again.status == "cached" and provider.calls == 2


def test_route_offers_context_to_the_ai_only_for_follow_ups(db):
    seed(db)
    first = routed(db, "how many negative reviews this month", make_planner(plan())[0])
    p, provider = recording_planner(plan(intent="count", sentiment="negative", product="Blender"),
                                    plan(intent="count", sentiment="negative"))
    chatbot.route("and what about the blender", db, TODAY, planner=p, planner_mode="llm_first", previous=first["query"])
    assert "Previous question's plan" in provider.prompts[0]
    chatbot.route("how many reviews are there for the lamp this month", db, TODAY, planner=p, planner_mode="llm_first", previous=first["query"])
    assert "Previous question's plan" not in provider.prompts[1]            # a confident standalone question gets no context


def test_ai_follow_up_reports_what_it_kept(db):
    seed(db)
    first = routed(db, "how many negative reviews this month", make_planner(plan())[0])
    p, _ = recording_planner(plan(intent="count", sentiment="negative", product="Blender", period={"kind": "this_month"}))
    r = chatbot.route("and for the blender?", db, TODAY, planner=p, planner_mode="llm_first", previous=first["query"])
    assert "I kept: negative sentiment" in text_of(r)


# ---- several questions in one message ---------------------------------------------------------------
COMPOUND_Q = "average rating for the lamp plus the number of complaints about the blender"


def compound_plans(**second):
    return {"plans": [
        dict(plan(intent="average", product="Lamp"), question="average rating for the lamp"),
        dict(plan(intent="count", sentiment="negative", product="Blender", **second), question="the number of complaints about the blender"),
    ]}


def test_validate_plans_handles_one_many_and_broken_plans():
    one = planner_mod.validate_plans(plan(intent="count"), PRODUCTS, TODAY, "how many")
    assert len(one.queries) == 1 and one.titles == [None] and one.dropped == 0
    many = planner_mod.validate_plans(compound_plans(), ["Lamp", "Blender"], TODAY, COMPOUND_Q)
    assert [q.intent for q in many.queries] == ["average", "count"]
    assert many.titles == ["average rating for the lamp", "the number of complaints about the blender"]
    raw = compound_plans()
    raw["plans"][0]["question"] = "something the owner never wrote"
    raw["plans"].insert(1, plan(intent="delete_review"))                    # an action is never accepted
    raw["plans"].append(plan(intent="count"))                               # a fourth plan is over the limit
    out = planner_mod.validate_plans(raw, ["Lamp", "Blender"], TODAY, COMPOUND_Q)
    assert len(out.queries) == 2 and out.titles[0] is None and out.dropped == 2


def test_two_questions_in_one_message_are_both_answered_with_one_call(db):
    seed(db)
    p, provider = make_planner(compound_plans())
    r = routed(db, COMPOUND_Q, p, mode="llm_first")
    t = text_of(r)
    assert provider.calls == 1
    assert "**1. Average rating for the lamp?**" in t and "**2. The number of complaints about the blender?**" in t
    assert "negative review" in t and r["query"].intent == "count"


def test_an_unknown_part_shows_its_explanation_and_a_dropped_part_is_reported(db):
    seed(db)
    raw = compound_plans()
    raw["plans"][1] = dict(plan(intent="unknown", clarification="Reviews do not record customers, so I cannot say."),
                           question="the number of complaints about the blender")
    raw["plans"].append(plan(intent="run_sql"))
    p, _ = make_planner(raw)
    t = text_of(routed(db, COMPOUND_Q, p, mode="llm_first"))
    assert "cannot say" in t and "couldn't make sense of 1 part" in t


def test_when_the_ai_sees_one_question_but_the_rules_split_it_the_rules_win(db):
    seed(db)
    p, provider = make_planner(plan(intent="count"))
    t = text_of(routed(db, "how many reviews today and how many are negative", p, mode="llm_first"))
    assert provider.calls == 1 and "**1." in t and "**2." in t


def test_compound_falls_back_to_the_rules_with_a_notice_when_the_ai_is_down(db):
    seed(db)
    p, _ = make_planner(LLMResult(False, error="timeout"))
    t = text_of(routed(db, "how many reviews today and how many are negative", p, mode="llm_first"))
    assert "**1." in t and "**2." in t and "AI question helper is unavailable" in t


def test_confident_compound_questions_do_not_spend_a_call_in_hybrid_mode(db):
    seed(db)
    p, provider = make_planner(compound_plans())
    t = text_of(routed(db, "how many reviews today and how many are negative", p))
    assert provider.calls == 0 and "**2." in t
