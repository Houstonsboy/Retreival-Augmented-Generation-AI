"""
Diagnostic runner for the five Phase 2 test queries.
Runs the full pipeline locally (no Flask, no auth) using the real Groq API.
Outputs results as JSON to stdout and to diagnostic_results.json.
"""
import json
import os
import sys
import time

# Load env vars from .env before importing anything that reads them.
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.isfile(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, os.path.dirname(__file__))

from groq_budget import MemoryLedger, research_budget
from groq_client import call_groq
from direct_retrieval import resolve, retrieve_resolved
from direct_answer import assemble, respond

QUERIES = [
    {
        "id": "Q03",
        "question": "Why was the CCTV evidence rejected in Republic v Barisa Wayu Mataguda?",
        "intent": "Secondary-source retrieval and graph integration",
    },
    {
        "id": "Q04",
        "question": "What was the court's decision regarding DVD evidence in Nonny Gathoni Njenga v Catherine Masitsa?",
        "intent": "Entity resolution and judicial outcomes",
    },
    {
        "id": "Q02",
        "question": "Who is qualified to sign a certificate of electronic evidence under section 106B(4) of the Evidence Act?",
        "intent": "Statutory clause precision",
    },
    {
        "id": "Q08",
        "question": "Does the Small Claims Court have jurisdiction over employment-related disputes?",
        "intent": "Jurisdiction and statutory retrieval",
    },
    {
        "id": "Q37",
        "question": "Why did Umoja Magharibi lose its claim for KES 553,000 despite proving that the money was advanced?",
        "intent": "Judicial reasoning and FIRAC utilisation",
    },
]


def run_query(query_text: str) -> dict:
    t0 = time.time()
    ledger = MemoryLedger()
    turn_key = f"diag-{int(t0)}"

    # Wrap MemoryLedger into the signature research_budget expects:
    # reserve(stage, estimated_tokens, timeout) -> attempt_count
    def reserve_fn(stage, estimated_tokens, timeout):
        return ledger.reserve(turn_key, stage)

    answer = ""
    claims = []
    verification = "no_passages"
    passages = []
    retrieval = {}
    resolution = {}
    groq_attempts = 0
    error = None

    try:
        resolution = resolve(query_text, graph_discovery=False)
        retrieval = retrieve_resolved(query_text, resolution) if resolution else {}
        passages = assemble(retrieval)

        with research_budget(reserve_fn):
            if passages:
                answer, claims, verification = respond(
                    query_text, retrieval, passages,
                    lambda msgs, *, stage, max_tokens=550: call_groq(
                        msgs, stage=stage, max_tokens=max_tokens
                    )
                )

        groq_attempts = len(ledger.used.get(turn_key, set()))
    except Exception as e:
        error = str(e)
        answer = f"[Error: {e}]"

    elapsed_ms = round((time.time() - t0) * 1000)

    passage_detail = [
        {
            "alias": p.alias,
            "label": p.label,
            "primary": p.primary,
            "score": round(p.score, 4),
            "text_preview": p.text[:300].replace("\n", " "),
        }
        for p in passages
    ]

    res = resolution or {}
    return {
        "resolution": {
            "mode": res.get("mode"),
            "case_identifier": res.get("case", {}).get("case_identifier"),
            "case_name": res.get("case", {}).get("name"),
            "provision": res.get("provision", {}).get("label"),
            "jurisdiction_hint": res.get("jurisdiction_hint", False),
            "ambiguous": res.get("ambiguous", False),
            "authority_status": retrieval.get("authority_status"),
        },
        "retrieval": {
            "cases_retrieved": len(retrieval.get("results", [])),
            "statutes_retrieved": len(retrieval.get("rules", [])),
            "case_names": [
                r.get("case_name", r.get("case_identifier", ""))
                for r in retrieval.get("results", [])
            ],
        },
        "passages_assembled": len(passages),
        "passage_detail": passage_detail,
        "answer": answer,
        "claims": claims,
        "verification": verification,
        "groq_attempts": groq_attempts,
        "latency_ms": elapsed_ms,
        **({"error": error} if error else {}),
    }


def main():
    results = []
    for q in QUERIES:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running {q['id']}: {q['question'][:70]}...", file=sys.stderr)
        result = run_query(q["question"])
        entry = {"id": q["id"], "question": q["question"], "intent": q["intent"], **result}
        results.append(entry)
        print(
            f"  mode={result['resolution']['mode']}  "
            f"case={result['resolution']['case_identifier'] or result['resolution']['provision'] or '-'}  "
            f"verification={result['verification']}  "
            f"groq={result['groq_attempts']}  "
            f"latency={result['latency_ms']}ms",
            file=sys.stderr,
        )
        if result.get("error"):
            print(f"  ERROR: {result['error']}", file=sys.stderr)

    out_path = os.path.join(os.path.dirname(__file__), "diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {out_path}", file=sys.stderr)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
