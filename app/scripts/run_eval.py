import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chatbot import create_chatbot  # noqa: E402

EVAL_FILE = Path("evaluation/test_queries.csv")
OUT_FILE = Path("evaluation/results.csv")


def main() -> None:
    bot = create_chatbot()
    bot.kb.load()
    rows = list(csv.DictReader(EVAL_FILE.open(encoding="utf-8")))
    results = []

    for row in rows:
        session = bot.new_session(user_id=None)  # a fresh session for each test case
        turns = [row["query"]] + ([row["followup"]] if row["followup"] else [])
        resp = None
        for text in turns:
            resp = bot.answer(session, text)

        results.append({
            "id": row["id"],
            "group": row["group"],
            "status": resp.status,
            "latency_ms": resp.latency_ms,
            "n_sources": len(resp.sources),
            "answer": resp.answer.replace("\n", " "),
            "manual_pass": "",      # fill in: 1 = pass, 0 = fail
            "clarity_1_5": "",
            "relevance_1_5": "",
            "helpfulness_1_5": "",
            "tone_1_5": "",
            "notes": "",
        })

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"{len(results)} casos evaluados. Resultados en {OUT_FILE}")


if __name__ == "__main__":
    main()