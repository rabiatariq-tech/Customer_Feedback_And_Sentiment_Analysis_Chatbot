"""Check the chatbot's numbers yourself, with plain SQL that does not use the app's own analytics code.

    python scripts/recount.py --topic "returns & refunds"
    python scripts/recount.py --topic price
    python scripts/recount.py --topic "returns & refunds" --product "Running Shoes"
    python scripts/recount.py --db data/feedback.db --topic price

Prints the sentiment split for the reviews that carry that topic tag (and product, if given), then the recent-versus-earlier
comparison the chatbot uses for "getting worse over time" questions (recent = the last 28 days up to the newest matching review).
"""
import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from services.insights import chance_of_difference  # noqa: E402


def split(rows):
    out = {"positive": 0, "neutral": 0, "negative": 0, "pending": 0}
    for _, label in rows:
        out[label if label in out else "pending"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH) if hasattr(config, "DB_PATH") else "data/feedback.db")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--product")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    sql = "SELECT substr(created_at, 1, 10), sentiment_label, topics FROM feedback WHERE topics LIKE ?"
    params = [f"%{a.topic}%"]
    if a.product:
        sql += " AND product = ? COLLATE NOCASE"
        params.append(a.product)
    rows = [(d, lab) for d, lab, topics in con.execute(sql, params) if a.topic in [t.strip() for t in (topics or "").split(",")] or a.topic in (topics or "")]
    if not rows:
        print("No reviews with that topic (check the spelling: topics are e.g. price, delivery, returns & refunds).")
        return
    s = split(rows)
    total = len(rows)
    print(f"{total} reviews tagged '{a.topic}'" + (f" for {a.product}" if a.product else "") + ":")
    for k in ("positive", "neutral", "negative"):
        print(f"  {k:<9}{s[k]:>4}  ({s[k] / total:.0%})")
    if s["pending"]:
        print(f"  not analysed yet: {s['pending']}")
    days = sorted(date.fromisoformat(d) for d, _ in rows)
    first, last = days[0], days[-1]
    if (last - first).days + 1 < 14:
        print("Too short a period for a recent-versus-earlier comparison.")
        return
    window = 28 if (last - first).days + 1 >= 56 else max(7, ((last - first).days + 1) // 3)
    cut = last - timedelta(days=window - 1)
    recent = [r for r in rows if date.fromisoformat(r[0]) >= cut]
    earlier = [r for r in rows if date.fromisoformat(r[0]) < cut]
    rn, en = sum(1 for _, l in recent if l == "negative"), sum(1 for _, l in earlier if l == "negative")
    print(f"\nRecent ({cut} to {last}): {rn} of {len(recent)} negative" + (f" ({rn / len(recent):.0%})" if recent else ""))
    print(f"Earlier ({first} to {cut - timedelta(days=1)}): {en} of {len(earlier)} negative" + (f" ({en / len(earlier):.0%})" if earlier else ""))
    if recent and earlier:
        chance = chance_of_difference(rn, len(recent), en, len(earlier))
        print(f"Chance that a gap this big is luck: {chance:.0%} (under 10% is a real change; much higher means it could easily be chance)")
        if min(len(recent), len(earlier)) < 5:
            print("But one period has fewer than 5 reviews, so ignore that percentage: there is too little data to call any trend.")


if __name__ == "__main__":
    main()
