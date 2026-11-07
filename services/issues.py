"""What people actually complain about inside a topic ("customer support" -> rude agents, no reply, long waits).

Topic tags only say a review is ABOUT customer support. This module reads the review text for the usual specific
problems, so "What are the biggest problems with customer service?" can answer with the problems themselves.
Plain word patterns, no model, no network. Every count comes from the reviews; nothing is invented.
"""
from __future__ import annotations

import re
from typing import Iterable

ISSUES: dict[str, tuple[tuple[str, str], ...]] = {
    "customer support": (
        ("rude or unhelpful staff", r"\b(rude|unhelpful|impolite|dismissive|disrespectful|arrogant|unprofessional|useless)\b|not (very )?helpful|wasn'?t helpful"),
        ("no reply to emails or messages", r"never (replied|responded|answered|got back|heard)|\bno (reply|response|answer)\b|did ?n[o']?t (reply|respond|answer)|didn'?t (reply|respond|answer)|ignored|unanswered|without (a )?(reply|response)"),
        ("long waits for a response", r"\bwait(ed|ing)?\b|took (days|weeks|ages|forever|too long)|slow (to )?(respon|repl)|\b(days|weeks) (for|to) (a )?(respon|repl|answer)"),
        ("the problem never got solved", r"not (been )?(resolved|solved|fixed)|unresolved|never (resolved|solved|fixed)|couldn'?t help|could not help|\bno help\b|didn'?t (solve|resolve|fix)"),
        ("warranty or replacement refused", r"(refused|denied|rejected|declined)[^.]{0,30}(warranty|replacement|claim)|warranty[^.]{0,30}(void|refused|denied)|no replacement"),
    ),
    "delivery": (
        ("late or delayed deliveries", r"\blate\b|delay|took (days|weeks|too long|forever|ages)|slow (delivery|shipping)|still (waiting|hasn'?t)|\bweeks?\b"),
        ("parcels that never arrived or got lost", r"never (arrived|came|showed|received)|didn'?t (arrive|come)|not (arrived|delivered)|\blost\b|\bmissing\b"),
        ("items damaged in transit", r"damaged|crushed|dented|arrived (broken|damaged|crushed)"),
        ("wrong item or wrong address", r"wrong (item|product|address|colou?r|size)|someone else"),
        ("no tracking or updates", r"tracking|no updates?|no notification"),
    ),
    "packaging": (
        ("damaged or crushed boxes", r"crushed|dented|\btorn\b|ripped|damaged (box|packag)|box (was )?(damaged|crushed|torn|open)|arrived (open|opened)"),
        ("not protected enough", r"flimsy|no (padding|protection|bubble)|(poorly|badly) (packed|packaged|wrapped)|not (well |properly )?(packed|packaged|protected|wrapped)|inadequate"),
        ("leaking or spilled contents", r"leak|spill"),
        ("opened or tampered packages", r"already (open|opened)|tamper|(seal|sealed) (was )?(broken|missing)|not sealed|unsealed"),
        ("too much wasteful packaging", r"excessive|wasteful|too much (packaging|plastic)|unnecessary"),
    ),
    "product quality": (
        ("things that broke or stopped working", r"broke|broken|stopped working|\bdied\b|fell apart|cracked|snapped|quit working"),
        ("defective or faulty items", r"defect|faulty|malfunction|doesn'?t work|does not work|not working|didn'?t work"),
        ("cheap or flimsy build", r"cheap|flimsy|(poorly|badly) (made|built)|fragile|low quality|wobbl|\bloose\b"),
        ("damaged on arrival", r"arrived (damaged|broken|cracked|scratched)|scratched|\bdent"),
        ("leaks", r"\bleak"),
    ),
    "price": (
        ("too expensive", r"expensive|overpriced|pricey|too much|costly|steep"),
        ("not worth the money", r"not worth|poor value|waste of money|rip[- ]?off|better value|\bcheaper\b"),
        ("hidden costs or price rises", r"price (went|increase|hike)|hidden|extra (fee|charge)|shipping (cost|fee)"),
    ),
    "usability": (
        ("confusing setup or instructions", r"confus|instructions?|manual|set ?up|hard to (set|install|assemble)|assembl"),
        ("app problems", r"\bapp\b"),
        ("hard to use", r"hard to use|difficult|complicated|not intuitive|unintuitive|clunky"),
        ("pairing or connection trouble", r"pair|connect|bluetooth|wifi"),
    ),
    "returns & refunds": (
        ("refunds that are late or never arrive", r"refund[^.]{0,40}(never|not|still|weeks|days|delay|wait)|(never|still|waiting)[^.]{0,30}refund|no refund|didn'?t (get|receive) (a |my |the )?refund"),
        ("returns that are refused or a hassle", r"return[^.]{0,30}(refused|denied|rejected|difficult|hard|hassle|complicated|policy)|refused (my |the )?return|no returns|hassle|restocking"),
        ("exchange problems", r"exchange|replacement|wrong (item|size)"),
    ),
    "performance": (
        ("battery problems", r"battery|charge|charging"),
        ("slow or laggy", r"\bslow\b|\blag|freez|sluggish"),
        ("crashes or disconnects", r"crash|disconnect|cuts? out|keeps? (stopping|restarting)"),
        ("sound or noise problems", r"noise|noisy|\bloud\b|static|buzz|\bhum\b|crackl|distort"),
        ("connection problems", r"connection|bluetooth|wifi|pair"),
    ),
}
_COMPILED = {t: [(label, re.compile(p, re.I)) for label, p in items] for t, items in ISSUES.items()}


def common_issues(topic: str, texts: Iterable[str], top: int = 3) -> tuple[list[tuple[str, int]], int]:
    """(issue label, number of reviews mentioning it) for the most common specific problems, plus how many reviews were read.
    A review counts once per problem; one review can show several."""
    patterns = _COMPILED.get(topic)
    texts = [t or "" for t in texts]
    if not patterns or not texts:
        return [], len(texts)
    counts = [(label, sum(1 for t in texts if rx.search(t))) for label, rx in patterns]
    counts = [(l, n) for l, n in counts if n > 0]
    counts.sort(key=lambda x: (-x[1], x[0]))
    return counts[:top], len(texts)
