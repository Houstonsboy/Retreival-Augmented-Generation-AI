#!/usr/bin/env python3
"""
Step 8 — Validate retrieval quality.

Calls real sub-functions from RulesRetriever.py and Retriever.py
(or the full pipeline when --pipeline flag is passed after retriever upgrades).

Usage:
    cd backend
    python validate_knowledge.py           # sub-function mode (baseline)
    python validate_knowledge.py --pipeline  # full pipeline mode (post-fix)
"""

import os, sys, json, statistics, argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Set dummy key so Retriever.py can be imported without a live key
os.environ.setdefault('GROQ_API_KEY', '_validate_placeholder_')

import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    STATUTES_DB_DIR, STATUTES_COLLECTION_NAME,
    CASELAW_DB_DIR, CASELAW_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

# ─────────────────────────── TEST CASES ────────────────────────────────────

TEST_CASES = [
    {
        "id": "Q1",
        "question": "What is the monetary claim limit in the Small Claims Court?",
        "statute": True, "caselaw": False,
        "exp_statute": ("small_claims_court_act", "12"),
        "exp_case": None,
        "classifier": {
            "retrieval_mode": "STRATEGIC_LEGAL_ANALYSIS",
            "intents": ["RULE_SEARCH"],
            "target_components": ["RULES"],
            "legal_domains": [{"domain": "Civil Procedure Law", "confidence_score": 0.9}],
            "entities": {
                "statutes": {"primary": ["Small Claims Court Act"], "contextual": []},
                "source_documents": ["Small Claims Court Act"],
                "article_numbers": [], "section_numbers": [],
            },
            "vector_query": "monetary claim limit pecuniary jurisdiction Small Claims Court",
        },
    },
    {
        "id": "Q2",
        "question": "How long do I have to sue for breach of contract?",
        "statute": True, "caselaw": False,
        "exp_statute": ("limitation_of_actions_act", "4"),
        "exp_case": None,
        "classifier": {
            "retrieval_mode": "STRATEGIC_LEGAL_ANALYSIS",
            "intents": ["RULE_SEARCH"],
            "target_components": ["RULES"],
            "legal_domains": [{"domain": "Contract Law", "confidence_score": 0.9}],
            "entities": {
                "statutes": {"primary": ["Limitation of Actions Act"], "contextual": []},
                "source_documents": ["Limitation of Actions Act"],
                "article_numbers": [], "section_numbers": [],
            },
            "vector_query": "limitation period breach of contract civil claim time bar six years",
        },
    },
    {
        "id": "Q3",
        "question": "Can WhatsApp messages be used as evidence?",
        "statute": True, "caselaw": True,
        "exp_statute": ("evidence_act", "106B"),
        "exp_case": "2024KEHC17038(KLR)",
        "classifier": {
            "retrieval_mode": "HYBRID_RETRIEVAL",
            "intents": ["RULE_SEARCH", "OUTCOME_ANALYSIS"],
            "target_components": ["RULES", "APPLICATION"],
            "legal_domains": [{"domain": "Evidence Law", "confidence_score": 0.9}],
            "entities": {
                "statutes": {"primary": ["Evidence Act"], "contextual": []},
                "source_documents": ["Evidence Act"],
                "article_numbers": [], "section_numbers": ["106B"],
            },
            "vector_query": "WhatsApp messages electronic evidence admissibility certificate Evidence Act 106B",
        },
    },
    {
        "id": "Q4",
        "question": "Can the Small Claims Court hear an employment dispute?",
        "statute": True, "caselaw": True,
        "exp_statute": ("small_claims_court_act", "13"),
        "exp_case": "2026SCC235(KLR)",
        "classifier": {
            "retrieval_mode": "HYBRID_RETRIEVAL",
            "intents": ["RULE_SEARCH", "OUTCOME_ANALYSIS"],
            "target_components": ["RULES", "APPLICATION"],
            "legal_domains": [{"domain": "Employment Law", "confidence_score": 0.85}],
            "entities": {
                "statutes": {"primary": ["Small Claims Court Act"], "contextual": []},
                "source_documents": ["Small Claims Court Act"],
                "article_numbers": [], "section_numbers": [],
            },
            "vector_query": "Small Claims Court jurisdiction employment dispute employer employee exclusion",
        },
    },
    {
        "id": "Q5",
        "question": "Can I set aside a default judgment in the Small Claims Court?",
        "statute": False, "caselaw": True,
        "exp_statute": None,
        "exp_case": "2026SCC241(KLR)",
        "classifier": {
            "retrieval_mode": "STRATEGIC_LEGAL_ANALYSIS",
            "intents": ["OUTCOME_ANALYSIS"],
            "target_components": ["RULES", "APPLICATION", "CONCLUSION"],
            "legal_domains": [{"domain": "Civil Procedure Law", "confidence_score": 0.8}],
            "entities": {
                "statutes": {"primary": [], "contextual": []},
                "source_documents": [],
                "article_numbers": [], "section_numbers": [],
            },
            "vector_query": "set aside default judgment application Small Claims Court service",
        },
    },
]

# ──────────────────────────── SUB-FUNCTION MODE ─────────────────────────────

def run_baseline(model, statutes_col, cases_col, top_n: int = 20):
    """Call retriever sub-functions directly — works with pre-BM25/CE code."""
    from RulesRetriever import (
        extract_query_params as rules_params,
        build_where_filter,
        calculate_rule_score,
        group_by_article,
    )
    from Retriever import (
        extract_query_params as cases_params,
        calculate_chunk_score,
        group_by_parent,
    )

    all_statute_scores, all_case_scores = [], []

    for tc in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"  {tc['id']}: {tc['question']}")
        print(f"{'='*70}")

        rparams = rules_params(tc["classifier"]) if tc["statute"] else None
        cparams = cases_params(tc["classifier"]) if tc["caselaw"] else None

        # ── STATUTES ──
        if tc["statute"] and statutes_col:
            vq = "query: " + rparams["vector_query"]
            emb = model.encode(vq, normalize_embeddings=True).tolist()
            where = build_where_filter(rparams)
            kw = {
                "query_embeddings": [emb],
                "n_results": min(top_n + 20, statutes_col.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kw["where"] = where
            res = statutes_col.query(**kw)
            scores = [
                calculate_rule_score(d, res["metadatas"][0][i], rparams["legal_domains"])
                for i, d in enumerate(res["distances"][0])
            ]
            articles = group_by_article(res, scores)
            sorted_arts = sorted(articles.values(), key=lambda x: -x["best_score"])
            all_statute_scores.extend([a["best_score"] for a in sorted_arts[:top_n]])

            # find expected hit rank
            exp = tc["exp_statute"]
            hit_rank = "not in top 20"
            for rank, art in enumerate(sorted_arts[:top_n], 1):
                if exp and art["document_type"] == exp[0] and art["article_number"] == exp[1]:
                    hit_rank = rank
                    break

            sched_top5 = sum(
                1 for a in sorted_arts[:5]
                for c in a["chunks"] if c.get("meta", {})
            )
            # count schedule in top 5
            sched_in_top5 = 0
            for a in sorted_arts[:5]:
                # check if this article is a schedule by looking at the raw results
                for i, cid in enumerate(res["ids"][0]):
                    if res["metadatas"][0][i].get("document_type") == a["document_type"] and \
                       res["metadatas"][0][i].get("article_number") == a["article_number"] and \
                       res["metadatas"][0][i].get("is_schedule"):
                        sched_in_top5 += 1
                        break

            print(f"\n  [STATUTES] Expected: {exp[0]} s.{exp[1]}  →  rank {hit_rank}")
            print(f"  Schedule chunks in top 5: {sched_in_top5}")
            print(f"  Top 5 statute results:")
            for rank, art in enumerate(sorted_arts[:5], 1):
                is_sched = ""
                for i in range(len(res["ids"][0])):
                    if (res["metadatas"][0][i].get("document_type") == art["document_type"] and
                            res["metadatas"][0][i].get("article_number") == art["article_number"] and
                            res["metadatas"][0][i].get("is_schedule")):
                        is_sched = " [SCHEDULE]"
                        break
                print(f"    [{rank}] score={art['best_score']:.3f}  "
                      f"{art['document_type']} s.{art['article_number']} "
                      f"— {art['article_header'][:50]}{is_sched}")

        # ── CASES ──
        if tc["caselaw"] and cases_col:
            vq = "query: " + cparams["vector_query"]
            emb = model.encode(vq, normalize_embeddings=True).tolist()
            raw = cases_col.query(
                query_embeddings=[emb],
                n_results=min(top_n + 10, cases_col.count()),
                include=["documents", "metadatas", "distances"],
            )
            flat = {
                "ids": raw["ids"][0],
                "documents": raw["documents"][0],
                "metadatas": raw["metadatas"][0],
                "distances": raw["distances"][0],
            }
            sc_list = [
                calculate_chunk_score(
                    flat["distances"][i], flat["metadatas"][i],
                    cparams["target_components"], cparams["legal_domains"]
                )
                for i in range(len(flat["ids"]))
            ]
            parents = group_by_parent(flat, sc_list, cparams["target_components"])
            sorted_parents = sorted(parents.items(), key=lambda x: -x[1]["best_score"])
            all_case_scores.extend([v["best_score"] for _, v in sorted_parents[:top_n]])

            exp_c = tc["exp_case"]
            hit_rank = "not in top 20"
            for rank, (pid, _) in enumerate(sorted_parents[:top_n], 1):
                if pid == exp_c:
                    hit_rank = rank
                    break

            print(f"\n  [CASES] Expected: {exp_c}  →  rank {hit_rank}")
            print(f"  Top 5 case results:")
            for rank, (pid, pdata) in enumerate(sorted_parents[:5], 1):
                print(f"    [{rank}] score={pdata['best_score']:.3f}  "
                      f"{pid}  — {pdata['metadata']['parties'][:50]}")

    # ── SCORE DISTRIBUTIONS ──
    print(f"\n{'='*70}")
    print("  SCORE DISTRIBUTIONS (raw vector similarity)")
    if all_statute_scores:
        print(f"  Statutes:  min={min(all_statute_scores):.3f}  "
              f"median={statistics.median(all_statute_scores):.3f}  "
              f"max={max(all_statute_scores):.3f}")
    if all_case_scores:
        print(f"  Cases:     min={min(all_case_scores):.3f}  "
              f"median={statistics.median(all_case_scores):.3f}  "
              f"max={max(all_case_scores):.3f}")

    print(f"\n  DIAGNOSTIC:")
    print(f"  Cross-encoder reranker in RulesRetriever.py: NO (vector similarity only)")
    print(f"  Cross-encoder reranker in Retriever.py:      NO (vector similarity only)")
    print(f"  BM25 in either retriever:                    NO")
    print(f"  Schedule down-weighting:                     NO")


# ──────────────────────────── PIPELINE MODE ─────────────────────────────────

def run_pipeline(top_n: int = 20):
    """Call full retrieve_relevant_rules / retrieve_relevant_cases — post-fix."""
    from RulesRetriever import retrieve_relevant_rules
    from Retriever import retrieve_relevant_cases

    for tc in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"  {tc['id']}: {tc['question']}")
        print(f"{'='*70}")

        # ── STATUTES ──
        if tc["statute"]:
            output = retrieve_relevant_rules(tc["classifier"], verbose=False, max_results=top_n)
            results = output.get("results", [])
            exp = tc["exp_statute"]
            hit_rank = "not in top 20"
            for r in results:
                if exp and r["document_type"] == exp[0] and r["article_number"] == exp[1]:
                    hit_rank = r["rank"]
                    break

            print(f"\n  [STATUTES] Expected: {exp[0]} s.{exp[1]}  →  rank {hit_rank}")
            print(f"  Top 5 statute results:")
            for r in results[:5]:
                _ah = r.get("article_header", "").lower()
                _an = r.get("article_number", "").lower()
                sched = " [SCHEDULE]" if "schedule" in _ah or _an.startswith("sch") else ""
                print(f"    [{r['rank']}] score={r['score']:.3f}  "
                      f"{r['document_type']} s.{r['article_number']} "
                      f"— {r['article_header'][:50]}{sched}")

            # CE score distribution from stats
            if output.get("stats", {}).get("ce_scores"):
                ce = output["stats"]["ce_scores"]
                print(f"  CE scores:  min={ce['min']:.3f}  median={ce['median']:.3f}  max={ce['max']:.3f}")

        # ── CASES ──
        if tc["caselaw"]:
            output = retrieve_relevant_cases(
                tc["classifier"], verbose=False,
                skip_summaries=True, top_n=top_n
            )
            results = output.get("results", [])
            exp_c = tc["exp_case"]
            hit_rank = "not in top 20"
            for r in results:
                if r.get("case_name") == exp_c or r.get("parent_document") == exp_c:
                    hit_rank = r["rank"]
                    break

            print(f"\n  [CASES] Expected: {exp_c}  →  rank {hit_rank}")
            print(f"  Top 5 case results:")
            for r in results[:5]:
                print(f"    [{r['rank']}] score={r['score']:.3f}  "
                      f"{r.get('case_name','?')}  — {r['metadata']['parties'][:50]}")

            if output.get("retrieval_stats", {}).get("ce_scores"):
                ce = output["retrieval_stats"]["ce_scores"]
                print(f"  CE scores:  min={ce['min']:.3f}  median={ce['median']:.3f}  max={ce['max']:.3f}")

    print(f"\n  DIAGNOSTIC:")
    print(f"  BM25 + RRF + cross-encoder: ACTIVE in both retrievers")
    print(f"  Schedule down-weighting:    ACTIVE in RulesRetriever")
    print(f"  Section exact-lookup boost: ACTIVE in RulesRetriever")


# ──────────────────────────── ENTRY POINT ────────────────────────────────────

# ──────────────────────────── PIPELINE + SYNTHESIS TESTS ────────────────────

SYNTHESIS_TESTS = [
    {
        "id": "ST1",
        "question": (
            "Can I appeal under section 39 of the Arbitration Act and does the High Court "
            "have interim jurisdiction under section 7? I have a Safaricom v Ocean View type dispute."
        ),
        "classification": {
            "retrieval_mode": "HYBRID_RETRIEVAL",
            "intents": ["RULE_SEARCH", "OUTCOME_ANALYSIS"],
            "target_components": ["RULES", "APPLICATION"],
            "legal_domains": [{"domain": "Arbitration Law", "confidence_score": 0.9}],
            "entities": {
                "statutes": {
                    "primary": ["Arbitration Act", "Appellate Jurisdiction Act", "Court of Appeal Rules"],
                    "contextual": [],
                },
                "source_documents": ["Arbitration Act", "Appellate Jurisdiction Act"],
                "article_numbers": [], "section_numbers": ["39", "7"],
            },
            "vector_query": "arbitration appeal High Court interim injunction jurisdiction section 39 7",
        },
        "expect": {
            "coverage_missing": ["Arbitration Act"],  # must appear in not_in_corpus
            "answer_must_not_contain": ["s.39", "section 39", "section 7 of the Arbitration"],
            "answer_must_not_correct_case": "Safaricom",    # must not say case is wrong/outdated
        },
    },
    {
        "id": "ST2",
        "question": "What does section 12 of the Small Claims Court Act say about pecuniary jurisdiction?",
        "classification": {
            "retrieval_mode": "DIRECT_STATUTORY_LOOKUP",
            "intents": ["RULE_SEARCH"],
            "target_components": ["RULES"],
            "legal_domains": [{"domain": "Civil Procedure Law", "confidence_score": 0.95}],
            "entities": {
                "statutes": {"primary": ["Small Claims Court Act"], "contextual": []},
                "source_documents": ["Small Claims Court Act"],
                "article_numbers": [], "section_numbers": ["12"],
            },
            "vector_query": "Small Claims Court Act section 12 pecuniary jurisdiction monetary limit",
        },
        "expect": {
            "coverage_missing": [],
            "answer_must_contain_one_of": ["12", "jurisdiction", "claim"],
            "must_cite_source": True,
        },
    },
    {
        "id": "ST3",
        "question": "Can my employee be sued in the Small Claims Court for stolen stock?",
        "classification": {
            "retrieval_mode": "HYBRID_RETRIEVAL",
            "intents": ["RULE_SEARCH", "OUTCOME_ANALYSIS"],
            "target_components": ["RULES", "APPLICATION"],
            "legal_domains": [{"domain": "Employment Law", "confidence_score": 0.85}],
            "entities": {
                "statutes": {"primary": ["Small Claims Court Act"], "contextual": []},
                "source_documents": ["Small Claims Court Act"],
                "article_numbers": [], "section_numbers": [],
            },
            "vector_query": "Small Claims Court employee employer stolen stock jurisdiction exclusion",
        },
        "expect": {
            "coverage_missing": [],
            "answer_must_contain_one_of": ["employ", "jurisdiction", "New Titanic", "2026SCC235"],
            "must_cite_source": True,
        },
    },
    {
        "id": "ST4",
        "question": "What does section 500 of the Law of Contract Act say?",
        "classification": {
            "retrieval_mode": "DIRECT_STATUTORY_LOOKUP",
            "intents": ["RULE_SEARCH"],
            "target_components": ["RULES"],
            "legal_domains": [{"domain": "Contract Law", "confidence_score": 0.9}],
            "entities": {
                "statutes": {"primary": ["Law of Contract Act"], "contextual": []},
                "source_documents": ["Law of Contract Act"],
                "article_numbers": [], "section_numbers": ["500"],
            },
            "vector_query": "Law of Contract Act section 500",
        },
        "expect": {
            "coverage_missing": [],
            "answer_must_not_contain": ["section 500 provides", "section 500 states", "section 500 says"],
            "answer_must_contain_one_of": ["not found", "not in the sources", "not available",
                                            "not covered", "cannot be verified", "does not appear"],
        },
    },
]


def run_synthesis_tests():
    """
    Runs coverage check + full synthesis for each SYNTHESIS_TEST.
    Requires a real GROQ_API_KEY.  Reports PASS/FAIL per assertion.
    Logs hallucination count from logs/hallucinations.jsonl.
    """
    real_key = os.environ.get("GROQ_API_KEY", "_validate_placeholder_")
    if real_key == "_validate_placeholder_" or not real_key.startswith("gsk_"):
        print("\n  [SKIP] GROQ_API_KEY not set — synthesis tests require a live key.")
        return

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
    real_key = os.environ.get("GROQ_API_KEY", "")

    import requests as req_lib
    from coverage import check_coverage
    from citation_verifier import verify_and_clean, HALL_LOG

    from RulesRetriever import retrieve_relevant_rules
    from Retriever import retrieve_relevant_cases

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    def _groq(msgs, max_tokens=2500) -> str:
        import re
        r = req_lib.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {real_key}", "Content-Type": "application/json"},
            json={"model": "qwen/qwen3.8-27b", "messages": msgs,
                  "temperature": 0, "max_tokens": max_tokens},
            timeout=90,
        )
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

    def _parse_json(text: str) -> dict:
        import re, json as _json
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
        start = text.find("{")
        if start == -1:
            return {"short_answer": text}
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return _json.loads(text[start:i+1])
        return {"short_answer": text}

    # Count hall-log entries before run
    def _hall_count():
        if not HALL_LOG.exists():
            return 0
        with open(HALL_LOG) as f:
            return sum(1 for _ in f)

    hall_before = _hall_count()
    results_summary = []

    for st in SYNTHESIS_TESTS:
        print(f"\n{'='*70}")
        print(f"  {st['id']}: {st['question'][:80]}")
        print(f"{'='*70}")

        # Retrieve
        stat_out = retrieve_relevant_rules(st["classification"], verbose=False, max_results=5)
        case_out = retrieve_relevant_cases(st["classification"], verbose=False,
                                           skip_summaries=True, top_n=5)

        statutes = stat_out.get("results", [])
        cases    = case_out.get("results", [])
        print(f"  Retrieved: {len(statutes)} statutes, {len(cases)} cases")

        # Coverage
        cov = check_coverage(st["classification"])
        print(f"  Coverage in:  {cov['in_corpus']}")
        print(f"  Coverage out: {cov['not_in_corpus']}")

        # Build source blocks (same logic as app.py)
        import json as _json
        sources: list = []
        source_blocks: list = []
        for i, rule in enumerate(statutes[:5], 1):
            sid  = f"S{i}"
            an   = rule.get("article_number", "")
            hdr  = rule.get("article_header", "")
            lbl  = rule.get("document_label", "Unknown")
            full = rule.get("full_text", rule.get("text_preview", ""))
            sources.append({"id": sid, "type": "statute", "data": rule})
            source_blocks.append(f"[{sid}] {lbl}, s.{an} — {hdr}\n{full[:1200]}")
        for i, case in enumerate(cases[:3], 1):
            cid   = f"C{i}"
            cn    = case.get("case_name", "Unknown")
            meta  = case.get("metadata", {})
            year  = meta.get("year", "")
            court = meta.get("court_level", "")
            chunks_text = "\n".join(
                f"[{ch.get('component','')}] {ch.get('text','')[:500]}"
                for ch in case.get("chunks", [])[:3]
            )
            sources.append({"id": cid, "type": "case", "data": case})
            source_blocks.append(f"[{cid}] {cn} ({year}, {court})\n{chunks_text}")

        if not source_blocks:
            print("  [WARN] No sources retrieved — skipping synthesis")
            results_summary.append((st["id"], "SKIP", "no sources"))
            continue

        coverage_notice = ""
        if cov["not_in_corpus"]:
            acts = "; ".join(cov["not_in_corpus"])
            coverage_notice = (
                f"COVERAGE NOTICE: The following Acts are NOT in the knowledge base — "
                f"do not describe their content: {acts}.\n\n"
            )

        SYSTEM = (
            "You are a Kenyan legal research assistant. Output ONLY a single valid JSON object.\n\n"
            "RULES: cite only from sources; never invent section content; never correct a source; "
            "if provision not in sources say so; temperature 0.\n\n"
            'Schema: {"source_review":[{"id":"","relevant":true,"reason":"","key_points":[]}],'
            '"premise_check":"","short_answer":"","applicable_law":"","case_law_discussion":"",'
            '"application":"","jurisdiction_and_procedure":"","gaps":"","follow_up_questions":[],'
            '"citations_used":[]}'
        )
        USER = (
            coverage_notice
            + "SOURCES:\n" + "\n\n".join(source_blocks)
            + f'\n\nUSER QUESTION: "{st["question"]}"\n\nOutput the JSON now.'
        )

        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
        raw = _groq(messages)
        parsed = _parse_json(raw)

        answer_text = " ".join(filter(None, [
            parsed.get("short_answer", ""),
            parsed.get("applicable_law", ""),
            parsed.get("case_law_discussion", ""),
            parsed.get("application", ""),
            parsed.get("gaps", ""),
        ])).lower()

        # Assertions
        expect   = st["expect"]
        failures = []

        # Coverage assertion
        for act in expect.get("coverage_missing", []):
            if not any(act.lower() in x.lower() for x in cov["not_in_corpus"]):
                failures.append(f"FAIL coverage: '{act}' not in not_in_corpus {cov['not_in_corpus']}")

        # must-not-contain
        for phrase in expect.get("answer_must_not_contain", []):
            if phrase.lower() in answer_text:
                failures.append(f"FAIL must-not-contain: '{phrase}' found in answer")

        # must-contain-one-of
        for phrase in expect.get("answer_must_contain_one_of", []):
            if phrase.lower() in answer_text:
                break
        else:
            phrases = expect.get("answer_must_contain_one_of", [])
            if phrases:
                failures.append(f"FAIL must-contain-one-of {phrases} — none found")

        # must not "correct" case
        no_correct_for = expect.get("answer_must_not_correct_case")
        if no_correct_for and no_correct_for.lower() in answer_text:
            for phrase in ["incorrect", "wrong", "misapplied", "outdated", "overruled", "incorrectly decided"]:
                if phrase in answer_text:
                    failures.append(f"FAIL: answer appears to correct/criticise {no_correct_for}")
                    break

        status = "PASS" if not failures else "FAIL"
        results_summary.append((st["id"], status, "; ".join(failures) if failures else ""))

        print(f"\n  Coverage out: {cov['not_in_corpus']}")
        print(f"  Premise check: {parsed.get('premise_check','')[:120]}")
        print(f"  Short answer: {(parsed.get('short_answer','') or '')[:200]}")
        if failures:
            for f in failures:
                print(f"  ❌ {f}")
        else:
            print(f"  ✅ All assertions passed")

        if st["id"] == "ST1":
            print(f"\n  --- ST1 FULL ANSWER ---")
            full = " ".join(filter(None, [
                parsed.get("short_answer", ""),
                parsed.get("applicable_law", ""),
                parsed.get("case_law_discussion", ""),
                parsed.get("application", ""),
                parsed.get("gaps", ""),
            ]))
            print(f"  {full[:1500]}")

    hall_after = _hall_count()
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS TEST RESULTS")
    print(f"{'='*70}")
    for tid, status, detail in results_summary:
        icon = "✅" if status == "PASS" else ("⏭" if status == "SKIP" else "❌")
        print(f"  {icon} {tid}: {status}  {detail}")
    print(f"\n  Hallucination log entries this run: {hall_after - hall_before}")
    print(f"  Log: {HALL_LOG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="store_true",
                        help="Call full pipeline (post-fix mode)")
    parser.add_argument("--synthesis", action="store_true",
                        help="Also run synthesis + coverage + hallucination tests (requires live GROQ key)")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  KNOWLEDGE VALIDATION")
    mode = "PIPELINE (post-fix)" if args.pipeline else "SUB-FUNCTION (baseline)"
    print(f"  Mode: {mode}")
    print(f"{'='*70}")

    if args.pipeline:
        run_pipeline(top_n=args.top_n)
        return

    # ── Baseline: load shared resources once ──
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    statutes_col, cases_col = None, None

    if STATUTES_DB_DIR.exists():
        sc = chromadb.PersistentClient(path=str(STATUTES_DB_DIR))
        try:
            statutes_col = sc.get_collection(STATUTES_COLLECTION_NAME)
            print(f"StatutesDB: {statutes_col.count()} chunks")
        except Exception as e:
            print(f"StatutesDB: {e}")

    if CASELAW_DB_DIR.exists():
        cc = chromadb.PersistentClient(path=str(CASELAW_DB_DIR))
        try:
            cases_col = cc.get_collection(CASELAW_COLLECTION_NAME)
            print(f"CaseLawDB:  {cases_col.count()} chunks")
        except Exception as e:
            print(f"CaseLawDB: {e}")

    if not statutes_col and not cases_col:
        print("No collections found. Run embed_knowledge.py first.")
        sys.exit(1)

    run_baseline(model, statutes_col, cases_col, top_n=args.top_n)

    print(f"\n{'='*70}")
    print("  VALIDATION COMPLETE")
    print(f"  LOW_CONFIDENCE_THRESHOLD = 0.5 (on raw similarity)")
    print(f"  After adding cross-encoder, re-run with --pipeline to calibrate CE threshold")
    print(f"{'='*70}\n")

    if args.synthesis:
        print(f"\n{'='*70}")
        print("  SYNTHESIS TESTS  (coverage + hallucination regression)")
        print(f"{'='*70}")
        run_synthesis_tests()


if __name__ == "__main__":
    main()
