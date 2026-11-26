"""Deterministic authority resolution and primary-source lookup."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

from graph.keys import _CITATION_RE, case_key, provision_key

ROOT = Path(__file__).resolve().parent
CASE_MANIFEST = ROOT.parent / "data/processed/cases/cases_meta.json"
STATUTE_MANIFEST = ROOT.parent / "data/processed/statutes/manifest.json"
SEED = ROOT / "graph/seed/scc_caselaw_graph.json"
PROVISION = re.compile(
    r"\b(?P<kind>section|sec\.?|s\.|article|art\.?|rule)\s*(?P<section>\d+[a-z]?)(?P<subsections>\s*(?:\(\s*[a-z0-9]+\s*\)\s*)*)",
    re.I,
)
STRATEGIC = re.compile(r"\b(?:my case|can i|should i|my evidence|litigation strategy|compare|distinguish)\b", re.I)
DIRECT_MODES = frozenset({
    "DIRECT_CASE_LOOKUP", "CASE_FACT_LOOKUP", "CASE_OUTCOME_LOOKUP",
    "CASE_REASONING_LOOKUP", "DIRECT_STATUTE_LOOKUP", "LEGAL_PRINCIPLE_LOOKUP",
})

def normal(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\bv(?:s\.?|ersus)?\b", " v ", text)
    return " ".join(re.findall(r"[^\W_]+", text))

def _fingerprint(path):
    st = path.stat()
    return st.st_mtime_ns, st.st_size

@lru_cache(maxsize=1)
def _authority_index(revisions):
    corpus = json.loads(CASE_MANIFEST.read_text())
    seed = json.loads(SEED.read_text())
    authorities = {}
    aliases = defaultdict(set)
    for entry in seed["nodes"]:
        if entry.get("type") != "Case":
            continue
        key = case_key(entry.get("citation", ""), entry.get("name", ""))
        if key == "UNKNOWN-0000-case":
            continue
        authorities[key] = {
            "case_key": key, "name": entry["name"], "citation": entry.get("citation", ""),
            "source_available": False, "eligible": False,
        }
        for name in [entry["name"], *entry.get("aliases", [])]:
            normalized = normal(name)
            if len(normalized) >= 6:
                aliases[normalized].add(key)
    for row in corpus:
        ident = row["case_identifier"]
        key = case_key(ident, row["parties"])
        name = authorities.get(key, {}).get("name") or row["parties"]
        authorities[key] = {
            "case_key": key, "case_identifier": ident, "name": name,
            "citation": authorities.get(key, {}).get("citation") or ident,
            "source_available": (ROOT / "SCCCases" / row["file_name_on_disk"]).is_file(),
            "eligible": not row.get("exclude", False), "court": row.get("court_level"),
            "year": row.get("year"),
        }
        for alias in (name, row["parties"]):
            aliases[normal(alias)].add(key)
    manifest = json.loads(STATUTE_MANIFEST.read_text())
    acts = list(manifest["acts"])
    const = manifest.get("constitution_fallback", {})
    if const:
        acts.append({"act_slug": "constitution", "act_title": "Constitution of Kenya 2010", "aliases": const.get("aliases", [])})
    for act in acts:
        seed_names = {row["act"] for row in seed["nodes"] if row.get("act")
                      and normal(re.sub(r"\s*\(Cap\s+\d+\)\s*$", "", row["act"], flags=re.I)) == normal(act["act_title"])}
        if len(seed_names) == 1:
            act["graph_title"] = next(iter(seed_names))
    return authorities, aliases, acts

def authority_index():
    return _authority_index(tuple(_fingerprint(p) for p in (CASE_MANIFEST, STATUTE_MANIFEST, SEED)))

def excluded_case_ids():
    authorities, _, _ = authority_index()
    return {entry["case_identifier"] for entry in authorities.values()
            if entry.get("case_identifier") and not entry["eligible"]}

def _case_intent(query, name_only):
    if name_only or re.search(r"\b(?:give|find|show|identify|locate)\b.{0,20}\b(?:case|judgment|citation)\b", query, re.I):
        return "DIRECT_CASE_LOOKUP"
    if re.search(r"\b(?:why|reason(?:ing|s)?)\b", query, re.I):
        return "CASE_REASONING_LOOKUP"
    if re.search(r"\b(?:ruling|decid\w*|outcome|reject\w*|admit\w*|allow\w*|dismiss\w*|sustain\w*|order|hold|held)\b", query, re.I):
        return "CASE_OUTCOME_LOOKUP"
    if re.search(r"\b(?:principle|doctrine|legal test)\b", query, re.I):
        return "LEGAL_PRINCIPLE_LOOKUP"
    return "CASE_FACT_LOOKUP"

def resolve(query, *, graph_discovery=False):
    authorities, aliases, acts = authority_index()
    text = normal(query)
    neutral = list(_CITATION_RE.finditer(query))
    keys = {case_key(m.group()) for m in neutral}
    name_matches = [(len(alias), alias, names) for alias, names in aliases.items()
                    if f" {alias} " in f" {text} "]
    matched_alias = ""
    if name_matches:
        _, matched_alias, _ = max(name_matches)
        # Keep independent authority mentions, ignoring shorter aliases contained in longer ones.
        named = set().union(*(found for _, alias, found in name_matches
                            if not any(alias != longer and f" {alias} " in f" {longer} "
                                       for _, longer, _ in name_matches)))
        if keys and named - keys:
            return {"ambiguous": True, "case_keys": sorted(keys | named)}
        keys |= named
    elif not keys and (" v " in text) and not STRATEGIC.search(query):
        from rapidfuzz.distance import Levenshtein
        query_words = text.split()
        matches = []
        for alias, names in aliases.items():
            words = alias.split()
            if " v " not in alias or len(words) < 3:
                continue
            for i in range(len(query_words) - len(words) + 1):
                window = query_words[i:i + len(words)]
                different = [(a, b) for a, b in zip(words, window) if a != b]
                if len(different) == 1 and min(map(len, different[0])) >= 4:
                    if Levenshtein.distance(*different[0]) == 1:
                        matches.append((len(words), alias, names))
        if matches:
            longest, matched_alias, _ = max(matches)
            keys = set().union(*(names for n, _, names in matches if n == longest))
    graph_matches = {}
    if not keys and graph_discovery and " v " in text and not STRATEGIC.search(query):
        try:
            from graph.queries import find_case_names
            for row in find_case_names(query):
                names = [row["name"], *(row.get("aliases") or [])]
                if any(name and " v " in normal(name) and f" {normal(name)} " in f" {text} " for name in names):
                    graph_matches[row["case_key"]] = {**row, "eligible": False, "source_available": False}
            keys = set(graph_matches)
        except Exception:
            pass
    if len(keys) > 1:
        year = re.search(r"\b(?:19|20)\d{2}\b", query)
        if not neutral and year:
            narrowed = {key for key in keys if f"-{year[0]}-" in key}
            keys = narrowed or keys
        if len(keys) > 1:
            return {"ambiguous": True, "case_keys": sorted(keys)}
    if keys and not STRATEGIC.search(query):
        key = next(iter(keys))
        case = authorities.get(key) or graph_matches.get(key) or {"case_key": key, "citation": neutral[0].group(),
                                     "source_available": False, "eligible": False}
        historical_year = re.search(r"\[(\d{4})\]\s*eklr", query, re.I)
        if historical_year and case.get("year") and historical_year[1] != str(case["year"]):
            return {"ambiguous": True, "case_keys": [key], "reason": "conflicting_citation_year"}
        cleaned = _CITATION_RE.sub("", query)
        cleaned = re.sub(r"\(?\bk(?:e)?lr\b\)?|\[?\d{4}\]?\s*eklr", "", cleaned, flags=re.I)
        mode = _case_intent(query, normal(cleaned) in ("", matched_alias))
        return {"mode": mode, "case": case}
    matches = {(act["act_slug"], act["act_title"]) for act in acts
               if any(f" {normal(alias)} " in f" {text} "
                      for alias in [act["act_title"], *act.get("aliases", [])] if alias)}
    provisions = [(m["section"].upper(), tuple(re.findall(r"\w+", m["subsections"].lower())),
                   "art." if m["kind"].lower().startswith("a") else ("rule " if m["kind"].lower() == "rule" else "s."))
                  for m in PROVISION.finditer(query)]
    unique_provisions = list(dict.fromkeys(provisions))
    if not STRATEGIC.search(query) and len(matches) == 1 and len(unique_provisions) == 1 and not keys:
        slug, title = next(iter(matches))
        section, subs, prefix = unique_provisions[0]
        act = next(row for row in acts if row["act_slug"] == slug)
        label = prefix + section + "".join(f"({item})" for item in subs)
        return {"mode": "DIRECT_STATUTE_LOOKUP", "provision": {
            "act_slug": slug, "title": title, "section": section, "subsections": list(subs), "label": label,
            "provision_key": provision_key(act.get("graph_title", title), label),
        }}
    if not STRATEGIC.search(query) and not keys and re.search(r"\b(?:what (?:is|are)|explain|define)\b", query, re.I) and re.search(r"\b(?:principle|doctrine|legal test)\b", query, re.I):
        return {"mode": "LEGAL_PRINCIPLE_LOOKUP"}
    return {}

def _case_index_status(case):
    import chromadb
    import config
    if config.CASELAW_DB_DIR.exists():
        col = chromadb.PersistentClient(path=str(config.CASELAW_DB_DIR)).get_collection(config.CASELAW_COLLECTION_NAME)
        if col.get(where={"case_identifier": case["case_identifier"]}, include=[], limit=1)["ids"]:
            return "indexed_lookup"
    return "eligible_but_not_indexed"

def direct_case(query, case, mode):
    from Retriever import retrieve_relevant_cases
    ident = case.get("case_identifier")
    if not case["eligible"] or not ident:
        return {"results": [], "authority_status": "excluded" if case["source_available"] else "source_unavailable"}
    result = retrieve_relevant_cases(
        {"vector_query": query, "target_components": []}, verbose=False,
        skip_summaries=True, pinned_case_ids=[ident], top_n=1, direct=True,
    )
    if result["results"] and (
        mode in ("CASE_OUTCOME_LOOKUP", "CASE_REASONING_LOOKUP")
        or re.search(r"\b(?:costs|damages|award)\b", query, re.I)
    ):
        chunks = result["results"][0]["chunks"]
        if chunks:
            last = max(chunks, key=lambda c: c["chunk_index"])
            if len(chunks) > 1 and chunks[0] is not last:
                # Preserve a terminal disposition alongside the best semantic match.
                chunks.remove(last)
                chunks.insert(1, last)
    result["authority_status"] = "indexed" if result["results"] else "eligible_but_not_indexed"
    return result

def direct_statute(provision):
    import chromadb
    import config
    from RulesRetriever import format_rules_output, group_by_article
    is_const = provision["act_slug"] == "constitution"
    directory = config.MASTER_RULES_DB_DIR if is_const else config.STATUTES_DB_DIR
    collection = config.MASTER_RULES_COLLECTION_NAME if is_const else config.STATUTES_COLLECTION_NAME
    if not directory.exists():
        return []
    col = chromadb.PersistentClient(path=str(directory)).get_collection(collection)
    where = {"article_number": provision["section"]}
    if not is_const:
        where = {"$and": [where, {"source_document": provision["title"]}]}
    raw = col.get(where=where, include=["documents", "metadatas"])
    if not raw["ids"]:
        return []
    rows = {k: [raw[k]] for k in ("ids", "documents", "metadatas")}
    grouped = group_by_article(rows, [1.0] * len(raw["ids"]))
    params = {"vector_query": "", "entities": {}}
    return format_rules_output(grouped, params, {})["results"]

def _prioritize_verified_graph_passages(results, graph, mode):
    if not results or not graph.get("paragraphs"):
        return
    roles = {
        "CASE_OUTCOME_LOOKUP": {"conclusion", "orders"},
        "CASE_REASONING_LOOKUP": {"analysis", "conclusion"},
        "CASE_FACT_LOOKUP": {"facts"},
        "LEGAL_PRINCIPLE_LOOKUP": {"rule", "analysis"},
    }.get(mode, set())
    for chunk in results[0].get("chunks", []):
        text = " ".join(chunk["text"].split())
        matches = [
            p["para_key"] for p in graph["paragraphs"]
            if p.get("passage_role") in roles and p.get("source_document_hash")
            and p["source_document_hash"] == chunk.get("source_document_hash")
            and p.get("text") and " ".join(p["text"].split()) in text
        ]
        if matches:
            chunk["graph_passage_keys"] = list(dict.fromkeys(matches))
    results[0]["chunks"].sort(key=lambda ch: bool(ch.get("graph_passage_keys")), reverse=True)

def retrieve_resolved(query, resolution, *, graph_enabled=None, case_pins=None, statute_pins=None):
    """Use graph identity before choosing candidate documents."""
    from graph import config as graph_config
    from graph.queries import case_lookup
    from Retriever import retrieve_relevant_cases
    use_graph = graph_config.GRAPH_ENABLED if graph_enabled is None else graph_enabled
    graph = {}
    result = {"results": [], "rules": [], "graph_authorities": [], "resolution": resolution}
    case = resolution.get("case")
    provision = resolution.get("provision")
    if case:
        if use_graph:
            try:
                graph = case_lookup(case["case_key"])
            except Exception:
                result["graph_status"] = "unavailable"
        primary_allowed = not case_pins or case.get("case_identifier") in case_pins
        if primary_allowed:
            if resolution["mode"] == "DIRECT_CASE_LOOKUP":
                result["authority_status"] = _case_index_status(case) if case["eligible"] else (
                    "excluded" if case["source_available"] else "source_unavailable"
                )
            else:
                result.update(direct_case(query, case, resolution["mode"]))
                _prioritize_verified_graph_passages(result["results"], graph, resolution["mode"])
        else:
            result["authority_status"] = "outside_pinned_scope"
        if not result["results"] and resolution["mode"] != "DIRECT_CASE_LOOKUP" and primary_allowed:
            catalog, _, _ = authority_index()
            related = []
            for row in graph.get("incoming_citations", []):
                known = catalog.get(row["citing_case"], {})
                ident = known.get("case_identifier")
                if known.get("eligible") and ident and (not case_pins or ident in case_pins):
                    related.append(ident)
            if related:
                result["candidate_origin"] = "graph_incoming_cites"
            else:
                result["candidate_origin"] = "hybrid_secondary"
            secondary = retrieve_relevant_cases(
                {"vector_query": query, "target_components": []}, verbose=False, skip_summaries=True,
                pinned_case_ids=list(dict.fromkeys(related)) or case_pins, direct=True,
            )
            result["results"] = secondary["results"]
            result["retrieval_stats"] = secondary.get("retrieval_stats", {})
        if graph:
            # The manifest controls primary text eligibility independently of graph identity.
            result["graph_case"] = {
                **graph,
                "paragraphs": graph["paragraphs"] if case["eligible"] and primary_allowed else [],
            }
    elif provision:
        if use_graph:
            try:
                from graph.queries import provision_lookup
                result["graph_provision"] = provision_lookup(provision["provision_key"])
            except Exception:
                result["graph_status"] = "unavailable"
        if not statute_pins or provision["title"] in statute_pins:
            result["rules"] = direct_statute(provision)
    elif resolution.get("mode") == "LEGAL_PRINCIPLE_LOOKUP":
        result.update(retrieve_relevant_cases(
            {"vector_query": query, "target_components": []}, verbose=False, skip_summaries=True,
        ))
    return result
