"""
Rules Retrieval Script for MasterRulesDB

Searches Constitution and Evidence Act when "RULES" is in target_components.
Retrieves relevant statutory provisions based on vector query and filters.
"""

import json
import math
import re
import statistics as _stats
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import Dict, List, Any, Optional
import urllib.parse

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).resolve().parent

from config import (
    STATUTES_DB_DIR as CHROMA_DB_DIR,
    STATUTES_COLLECTION_NAME as CHROMA_COLLECTION_NAME,
    MASTER_RULES_DB_DIR,
    MASTER_RULES_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    STATUTE_MANIFEST_PATH,
)


def _load_manifest():
    """Load STATUTE_MAP, DOC_TYPE_LABELS, and Constitution aliases from manifest.json."""
    with open(STATUTE_MANIFEST_PATH) as _f:
        _manifest = json.load(_f)
    _statute_map = {}
    _doc_labels = {"constitution": "Constitution of Kenya"}  # for master_rules fallback chunks
    for _entry in _manifest["acts"]:
        _doc_labels[_entry["act_slug"]] = _entry["act_title"]
        for _alias in _entry["aliases"]:
            _statute_map[_alias] = _entry["act_title"]
    _const_aliases = set(_manifest["constitution_fallback"]["aliases"])
    return _statute_map, _doc_labels, _const_aliases


STATUTE_MAP, DOC_TYPE_LABELS, _CONSTITUTION_ALIASES = _load_manifest()

# ── Hybrid-retrieval module-level state (lazy-initialised once per process) ──
_SCHEDULE_TERMS = frozenset({"schedule", "form", "fee", "tariff", "appendix", "annex", "prescribed"})
_SECTION_RE = re.compile(r'\b(?:section|s\.|sec\.?|article|art\.?)\s*(\d+\w*)', re.IGNORECASE)
_STATUTES_BM25: dict = {"index": None, "ids": [], "docs": [], "metas": []}
_STATUTES_CE: dict = {"model": None}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _ensure_statutes_bm25() -> None:
    if _STATUTES_BM25["index"] is not None:
        return
    if not CHROMA_DB_DIR.exists():
        return
    try:
        _c = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _col = _c.get_collection(name=CHROMA_COLLECTION_NAME)
        _d = _col.get(include=["documents", "metadatas"])
        _STATUTES_BM25["ids"] = _d["ids"]
        _STATUTES_BM25["docs"] = _d["documents"]
        _STATUTES_BM25["metas"] = _d["metadatas"]
        _STATUTES_BM25["index"] = BM25Okapi([doc.split() for doc in _d["documents"]])
    except Exception:
        pass  # BM25 unavailable; fall back to vector-only


def _ensure_statutes_ce() -> CrossEncoder:
    if _STATUTES_CE["model"] is None:
        _STATUTES_CE["model"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _STATUTES_CE["model"]


def _matches_where(meta: dict, f: dict) -> bool:
    """Evaluate a ChromaDB where-filter dict against a metadata dict."""
    if "$and" in f:
        return all(_matches_where(meta, sub) for sub in f["$and"])
    for field, cond in f.items():
        if field.startswith("$"):
            continue
        mv = str(meta.get(field, ""))
        if isinstance(cond, dict):
            op, val = next(iter(cond.items()))
            if op == "$eq" and mv != str(val):
                return False
            if op == "$in" and mv not in [str(v) for v in val]:
                return False
        elif mv != str(cond):
            return False
    return True


def _rrf_fuse(ranked_a: list, ranked_b: list, k: int = 60, n: int = 30) -> list:
    """Reciprocal rank fusion of two ranked ID lists."""
    rrf: dict = {}
    for rank, cid in enumerate(ranked_a, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(ranked_b, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(rrf, key=lambda x: -rrf[x])[:n]


def _schedule_mult(meta: dict, query: str) -> float:
    """Return 0.3 penalty for schedule chunks unless the query mentions a schedule/form/fee."""
    if not meta.get("is_schedule", False):
        return 1.0
    return 1.0 if any(t in query.lower() for t in _SCHEDULE_TERMS) else 0.3


def _extract_section_nums(text: str) -> list:
    return list({m.group(1) for m in _SECTION_RE.finditer(text)})


def _is_constitution_query(source_docs: list, all_statutes: list) -> bool:
    """Return True if any named source looks like the Constitution of Kenya."""
    terms = [t.lower() for t in (source_docs + all_statutes)]
    for alias in _CONSTITUTION_ALIASES:
        al = alias.lower()
        if any(al in t or t in al for t in terms):
            return True
    return False


def _find_overlap(text_a: str, text_b: str, max_check: int = 400) -> int:
    """
    Return the number of leading characters of text_b that are a verbatim
    suffix of text_a (i.e. the overlap region from the sliding-window split).
    Only reports overlaps >= 10 chars to avoid false positives on short matches.
    """
    check = min(len(text_a), len(text_b), max_check)
    for size in range(check, 9, -1):
        if text_a[-size:] == text_b[:size]:
            return size
    return 0


def _stitch_chunks(chunks: list) -> str:
    """
    Join a list of chunk dicts (already sorted by chunk_index) into a single
    string, stripping the duplicated overlap text between consecutive chunks.
    Each dict must have a 'text' key.
    """
    if not chunks:
        return ""
    result = chunks[0]["text"]
    for chunk in chunks[1:]:
        overlap = _find_overlap(result, chunk["text"])
        result += chunk["text"][overlap:]
    return result


def _enrich_article_chunks(collection, articles: Dict) -> None:
    """
    For each article group, fetch any adjacent chunks referenced by
    prev_chunk_id / next_chunk_id that are not already in the group.
    Mutates `articles` in place.
    """
    for article in articles.values():
        known_ids = {c["chunk_id"] for c in article["chunks"]}
        to_fetch = set()
        for c in article["chunks"]:
            if c.get("prev_chunk_id") and c["prev_chunk_id"] not in known_ids:
                to_fetch.add(c["prev_chunk_id"])
            if c.get("next_chunk_id") and c["next_chunk_id"] not in known_ids:
                to_fetch.add(c["next_chunk_id"])

        if not to_fetch:
            continue

        try:
            fetched = collection.get(
                ids=list(to_fetch),
                include=["documents", "metadatas"],
            )
            for i, fid in enumerate(fetched["ids"]):
                if fid in known_ids:
                    continue
                article["chunks"].append({
                    "text": fetched["documents"][i],
                    "score": 0.0,  # fetched for context, not by similarity
                    "chunk_id": fid,
                    "chunk_index": int(fetched["metadatas"][i].get("chunk_index", 0)),
                    "prev_chunk_id": str(fetched["metadatas"][i].get("prev_chunk_id", "")),
                    "next_chunk_id": str(fetched["metadatas"][i].get("next_chunk_id", "")),
                })
                known_ids.add(fid)
        except Exception:
            pass  # adjacent chunks may not exist if collection was partially populated


def _query_constitution_fallback(query_embedding: list, n_results: int = 10) -> Dict:
    """Query master_rules for Constitution chunks and return raw ChromaDB result."""
    import chromadb as _chromadb
    empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    if not MASTER_RULES_DB_DIR.exists():
        return empty
    try:
        _client = _chromadb.PersistentClient(path=str(MASTER_RULES_DB_DIR))
        _col = _client.get_collection(name=MASTER_RULES_COLLECTION_NAME)
        return _col.query(
            query_embeddings=[query_embedding],
            where={"document_type": {"$eq": "constitution"}},
            n_results=min(n_results, _col.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return empty


def load_classifier_output(input_data: Any) -> Dict:
    """Load and parse classifier output from various input types."""
    if isinstance(input_data, dict):
        return input_data
    
    if isinstance(input_data, str):
        if input_data.strip().startswith('{'):
            try:
                return json.loads(input_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        
        # Try to load from file path
        file_path = Path(input_data)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"File not found: {input_data}")
    
    raise ValueError(f"Unsupported input type: {type(input_data)}")


def extract_query_params(classifier_output: Dict) -> Dict[str, Any]:
    """Extract key parameters from classifier output."""
    entities = classifier_output.get('entities', {})
    
    # Extract article/section numbers for direct lookup
    article_numbers = entities.get('article_numbers', [])
    section_numbers = entities.get('section_numbers', [])
    
    # Extract source documents directly (new schema)
    source_documents = entities.get('source_documents', [])
    
    # Support both old and new statute schema
    statutes_data = entities.get('statutes', {})
    if isinstance(statutes_data, dict):
        # New schema: {primary: [...], contextual: [...]}
        all_statutes = statutes_data.get('primary', []) + statutes_data.get('contextual', [])
    else:
        # Old schema: ["statute1", "statute2"]
        all_statutes = statutes_data if isinstance(statutes_data, list) else []
    
    # Extract mode-specific query if available (for hybrid mode)
    vector_query = classifier_output.get('vector_query_statutory') or classifier_output.get('vector_query', '')
    
    return {
        'vector_query': vector_query,
        'target_components': classifier_output.get('target_components', []),
        'legal_domains': classifier_output.get('legal_domains', []),
        'entities': entities,
        'intents': classifier_output.get('intents', []),
        # NEW FIELDS for direct article lookup
        'article_numbers': article_numbers,
        'section_numbers': section_numbers,
        'source_documents': source_documents,
        'all_statutes': all_statutes,
        'retrieval_mode': classifier_output.get('retrieval_mode', ''),
        'retrieval_strategy': classifier_output.get('retrieval_strategy', ''),
        'citation_analysis': classifier_output.get('citation_analysis', {}),
        'needs_constitution': _is_constitution_query(source_documents, all_statutes),
    }


def build_where_filter(query_params: Dict) -> Dict:
    """
    Build ChromaDB where filter based on query parameters.
    Returns {} (no filter) when no constraints apply so the caller can
    omit the `where` argument entirely and search all acts.
    """
    conditions = []
    # (hardcoded document_type filter removed — StatutesDB holds all 14 acts)
    
    # PRIORITY 1: Filter by article numbers if provided (for direct lookup)
    article_numbers = query_params.get('article_numbers', [])
    if article_numbers:
        # Convert to strings in case they're integers
        article_numbers_str = [str(num) for num in article_numbers]
        conditions.append({"article_number": {"$in": article_numbers_str}})
        print(f"  🎯 Filtering by article_numbers: {article_numbers_str}")
    
    # PRIORITY 2: Filter by section numbers if provided (for Evidence Act sections)
    section_numbers = query_params.get('section_numbers', [])
    if section_numbers:
        section_numbers_str = [str(num) for num in section_numbers]
        conditions.append({"article_number": {"$in": section_numbers_str}})  # Evidence Act uses same field
        print(f"  🎯 Filtering by section_numbers: {section_numbers_str}")
    
    # PRIORITY 3: Filter by source documents if provided
    source_documents = query_params.get('source_documents', [])
    if source_documents:
        # Map document names to actual PDF filenames
        mapped_docs = []
        for doc_name in source_documents:
            # Try exact match first
            if doc_name in STATUTE_MAP:
                mapped_docs.append(STATUTE_MAP[doc_name])
            else:
                # Try fuzzy match (e.g., "Constitution of Kenya 2010" -> "Constitution of Kenya")
                for key, value in STATUTE_MAP.items():
                    if key.lower() in doc_name.lower() or doc_name.lower() in key.lower():
                        mapped_docs.append(value)
                        break
        
        if mapped_docs:
            conditions.append({"source_document": {"$in": mapped_docs}})
            print(f"  📄 Filtering by source_documents: {mapped_docs}")
    else:
        # FALLBACK: Try old statute mapping (backward compatibility)
        all_statutes = query_params.get('all_statutes', [])
        if all_statutes:
            source_docs = []
            for statute in all_statutes:
                mapped_doc = STATUTE_MAP.get(statute)
                if mapped_doc:
                    source_docs.append(mapped_doc)
            
            if source_docs:
                conditions.append({"source_document": {"$in": source_docs}})
                print(f"  📄 Filtering by statutes (legacy): {source_docs}")
    
    # Return single condition, $and, or {} (= no filter → search all acts)
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def calculate_rule_score(distance: float, chunk_metadata: Dict, legal_domains: List[Dict]) -> float:
    """Calculate relevance score for a rule chunk."""
    base_similarity = 1 / (1 + distance)
    
    # Optional: Boost based on legal domain match
    # For now, keep it simple - just use similarity
    return base_similarity


def group_by_article(results: Dict, scores: List[float]) -> Dict:
    """Group chunks by article (multiple chunks can belong to same article)."""
    articles = {}
    
    for idx, chunk_id in enumerate(results['ids'][0]):
        metadata = results['metadatas'][0][idx]
        doc_text = results['documents'][0][idx]
        score = scores[idx]
        
        # Create unique article key: document_type + article_number
        doc_type = metadata.get('document_type', 'unknown')
        article_num = metadata.get('article_number', 'N/A')
        article_key = f"{doc_type}_{article_num}"
        
        if article_key not in articles:
            articles[article_key] = {
                'document_type': doc_type,
                'article_number': article_num,
                'article_header': metadata.get('article_header', 'N/A'),
                'chapter': metadata.get('chapter', 'N/A'),
                'chapter_title': metadata.get('chapter_title', 'N/A'),
                'part': metadata.get('part', ''),
                'part_title': metadata.get('part_title', ''),
                'source_document': metadata.get('source_document', ''),
                'chunks': [],
                'best_score': 0.0,
                'chunk_ids': []
            }
        
        articles[article_key]['chunks'].append({
            'text': doc_text,
            'score': score,
            'chunk_id': chunk_id,
            'chunk_index': int(metadata.get('chunk_index', 0)),
            'prev_chunk_id': str(metadata.get('prev_chunk_id', '')),
            'next_chunk_id': str(metadata.get('next_chunk_id', '')),
        })
        
        articles[article_key]['chunk_ids'].append(chunk_id)
        
        if score > articles[article_key]['best_score']:
            articles[article_key]['best_score'] = score
    
    return articles


def format_rules_output(articles: Dict, query_params: Dict, stats: Dict, max_results: int = 5) -> Dict:
    """Format rules results in frontend-compatible structure."""

    sorted_articles = sorted(
        articles.values(),
        key=lambda x: x['best_score'],
        reverse=True
    )[:max_results]
    
    results = []
    for rank, article in enumerate(sorted_articles, 1):
        # Combine all chunks for full article text
        # Sort chunks by score to show most relevant first
        sorted_chunks = sorted(article['chunks'], key=lambda x: x.get('chunk_index', 0))
        full_text = _stitch_chunks(sorted_chunks)
        
        # Get preview (first 500 chars)
        text_preview = full_text[:500]
        if len(full_text) > 500:
            text_preview += "..."
        
        # Build PDF URL
        pdf_url = f"/api/rule-pdf?doc_type={article['document_type']}&source={urllib.parse.quote(article['source_document'])}"
        
        results.append({
            'rank': rank,
            'document_type': article['document_type'],
            'document_label': DOC_TYPE_LABELS.get(article['document_type'], article['document_type']),
            'article_number': article['article_number'],
            'article_header': article['article_header'],
            'chapter': article['chapter'],
            'chapter_title': article['chapter_title'],
            'part': article['part'] if article['part'] else None,
            'part_title': article['part_title'] if article['part_title'] else None,
            'source_document': article['source_document'],
            'score': round(article['best_score'], 3),
            'text_preview': text_preview,
            'full_text': full_text,
            'chunk_count': len(article['chunks']),
            'pdf_url': pdf_url
        })
    
    return {
        'results': results,
        'count': len(results),
        'query_summary': {
            'vector_query': query_params['vector_query'],
            'filtered_statutes': query_params['entities'].get('statutes', []),
            'search_type': 'statutory_provisions'
        },
        'stats': stats
    }


def retrieve_relevant_rules(classifier_output: Any, verbose: bool = True, max_results: int = 5) -> Dict:
    """
    Main rules retrieval function.
    Searches MasterRulesDB for relevant Constitution and Evidence Act provisions.
    
    Args:
        classifier_output: Qclassifier output (dict, JSON string, or file path)
        verbose: Whether to print progress messages
        
    Returns:
        Dict with results in frontend-compatible format
    """
    
    if verbose:
        print("\n" + "="*80)
        print("📜 STATUTORY RULES RETRIEVAL")
        print("="*80)
    
    # ---- STEP 1: Parse Input ----
    if verbose:
        print("\n[1/6] Parsing classifier output...")
    
    classifier_data = load_classifier_output(classifier_output)
    query_params = extract_query_params(classifier_data)
    
    if verbose:
        print(f"  ✓ Vector Query: {query_params['vector_query'][:60]}...")
        print(f"  ✓ Intents: {', '.join(query_params['intents'])}")
        if query_params['entities'].get('statutes'):
            print(f"  ✓ Target Statutes: {', '.join(query_params['entities']['statutes'])}")
    
    # ---- STEP 2: Load Embedding Model ----
    if verbose:
        print("\n[2/6] Loading embedding model...")
    
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if verbose:
            print(f"  ✓ Loaded: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
    
    # ---- STEP 3: Connect to StatutesDB ----
    if verbose:
        print("\n[3/6] Connecting to StatutesDB...")

    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(f"StatutesDB not found at {CHROMA_DB_DIR}. Run embed_knowledge.py first.")

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        collection_count = collection.count()
        if verbose:
            print(f"  ✓ Connected to collection: {CHROMA_COLLECTION_NAME}")
            print(f"  ✓ Total documents: {collection_count}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to ChromaDB: {e}")
    
    # ---- STEP 4: Build Filters ----
    if verbose:
        print("\n[4/6] Building search filters...")
        retrieval_mode = query_params.get('retrieval_mode', '')
        if retrieval_mode:
            print(f"  🎯 Retrieval Mode: {retrieval_mode}")
        retrieval_strategy = query_params.get('retrieval_strategy', '')
        if retrieval_strategy:
            print(f"  🎯 Retrieval Strategy: {retrieval_strategy}")
    
    where_filter = build_where_filter(query_params)
    if verbose:
        print(f"  ✓ Filters: {json.dumps(where_filter, indent=2)}")
    
    # ---- STEP 5: Embed Query & Hybrid Search (BM25 + Vector → RRF → Cross-encoder) ----
    if verbose:
        print("\n[5/6] Embedding query and hybrid search (BM25 + vector + cross-encoder)...")

    query_str = query_params['vector_query']
    query_embedding = embedding_model.encode("query: " + query_str).tolist()

    if verbose:
        print(f"  ✓ Embedding dimension: {len(query_embedding)}")

    # 5a. BM25 search over entire StatutesDB corpus, filtered by where_filter scope
    _ensure_statutes_bm25()
    bm25_ranked_ids: list = []
    if _STATUTES_BM25["index"] is not None:
        bm25_raw = _STATUTES_BM25["index"].get_scores(query_str.split())
        if where_filter:
            eligible = [
                i for i, m in enumerate(_STATUTES_BM25["metas"])
                if _matches_where(m, where_filter)
            ]
        else:
            eligible = list(range(len(_STATUTES_BM25["ids"])))
        bm25_order = sorted(eligible, key=lambda i: -bm25_raw[i])[:50]
        bm25_ranked_ids = [_STATUTES_BM25["ids"][i] for i in bm25_order]

    # 5b. Vector search (top 50) — apply where_filter when present
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(50, collection_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter
    vec_results = collection.query(**query_kwargs)
    vec_ranked_ids = vec_results["ids"][0]

    if verbose:
        print(f"  ✓ Vector: {len(vec_ranked_ids)} chunks | BM25: {len(bm25_ranked_ids)} chunks")

    # 5c. RRF fusion → top 30 candidates
    fused_ids = _rrf_fuse(vec_ranked_ids, bm25_ranked_ids, k=60, n=30)

    # 5d. Section exact-lookup: inject any explicitly named sections not yet in candidates.
    #     Only inject when the query ALSO names an Act; bare section numbers without an Act
    #     would pull the same section number from every Act in the corpus.  When no Act is
    #     named, the BM25 + vector candidates already contain the relevant chunks — let the
    #     cross-encoder rank them.
    explicit_secs = set(
        _extract_section_nums(query_str) +
        [str(n) for n in query_params.get("section_numbers", [])] +
        [str(n) for n in query_params.get("article_numbers", [])]
    )
    # Resolve named Acts → act_title strings stored in chunk source_document metadata
    _named_act_titles: set = set()
    for _doc in query_params.get("source_documents", []):
        _mapped = STATUTE_MAP.get(_doc)
        if _mapped:
            _named_act_titles.add(_mapped)
        else:
            for _k, _v in STATUTE_MAP.items():
                if _k.lower() in _doc.lower() or _doc.lower() in _k.lower():
                    _named_act_titles.add(_v)
                    break
    for _s in query_params.get("all_statutes", []):
        _mapped = STATUTE_MAP.get(_s)
        if _mapped:
            _named_act_titles.add(_mapped)

    if explicit_secs and _named_act_titles and _STATUTES_BM25["index"] is not None:
        fused_set = set(fused_ids)
        inject = [
            _STATUTES_BM25["ids"][i]
            for i, m in enumerate(_STATUTES_BM25["metas"])
            if str(m.get("article_number", "")) in explicit_secs
            and m.get("source_document", "") in _named_act_titles
            and _STATUTES_BM25["ids"][i] not in fused_set
        ]
        fused_ids = inject + fused_ids
        fused_ids = fused_ids[:35]
        if inject and verbose:
            print(f"  ✓ Injected {len(inject)} exact-section chunks "
                  f"(acts: {', '.join(_named_act_titles)}, sections: {explicit_secs})")
    elif explicit_secs and verbose:
        print(f"  ✓ Section {explicit_secs} detected — no Act named, skipping injection")

    # 5e. Build id → (doc, meta) lookup from both sources
    id_to_data: dict = {}
    for i, cid in enumerate(vec_ranked_ids):
        id_to_data[cid] = (vec_results["documents"][0][i], vec_results["metadatas"][0][i])
    if _STATUTES_BM25["index"] is not None:
        for i, cid in enumerate(_STATUTES_BM25["ids"]):
            if cid not in id_to_data:
                id_to_data[cid] = (_STATUTES_BM25["docs"][i], _STATUTES_BM25["metas"][i])

    # 5f. Constitution fallback: add master_rules chunks to CE candidate pool
    if query_params.get('needs_constitution'):
        if verbose:
            print("  🏛️  Constitution query — also fetching master_rules chunks...")
        const_res = _query_constitution_fallback(query_embedding, n_results=10)
        const_added = 0
        fused_set = set(fused_ids)
        for i, cid in enumerate(const_res["ids"][0]):
            if cid not in id_to_data:
                id_to_data[cid] = (const_res["documents"][0][i], const_res["metadatas"][0][i])
            if cid not in fused_set:
                fused_ids.append(cid)
                fused_set.add(cid)
                const_added += 1
        if verbose and const_added:
            print(f"  ✓ Added {const_added} Constitution chunks from master_rules")

    # 5g. Cross-encoder rerank
    cand_ids = [cid for cid in fused_ids if cid in id_to_data]
    cand_docs = [id_to_data[cid][0] for cid in cand_ids]
    cand_metas = [id_to_data[cid][1] for cid in cand_ids]

    ce_model = _ensure_statutes_ce()
    ce_raw = ce_model.predict([(query_str, doc) for doc in cand_docs])

    # 5h. Sigmoid-normalise + schedule penalty → effective score
    effective = [
        _sigmoid(float(ce_raw[i])) * _schedule_mult(cand_metas[i], query_str)
        for i in range(len(cand_ids))
    ]

    # Sort descending by effective score
    order = sorted(range(len(cand_ids)), key=lambda i: -effective[i])
    cand_ids = [cand_ids[i] for i in order]
    cand_docs = [cand_docs[i] for i in order]
    cand_metas = [cand_metas[i] for i in order]
    effective = [effective[i] for i in order]

    if verbose and effective:
        print(f"  ✓ Cross-encoder done. CE-sigmoid scores: "
              f"min={min(effective):.3f}  median={_stats.median(effective):.3f}  max={max(effective):.3f}")

    # Rebuild results dict in the format expected by group_by_article
    results = {
        "ids": [cand_ids],
        "documents": [cand_docs],
        "metadatas": [cand_metas],
        "distances": [[0.0] * len(cand_ids)],
    }

    # ---- STEP 6: Group, enrich, and format ----
    if verbose:
        print("\n[6/6] Grouping by article, enriching, and formatting...")

    # Pass CE-derived effective scores directly (skip calculate_rule_score)
    articles = group_by_article(results, effective)

    # Fetch prev/next chunks for any partial hits (uses real collection)
    _enrich_article_chunks(collection, articles)

    if verbose:
        print(f"  ✓ Grouped into {len(articles)} unique articles")

    stats = {
        'total_chunks_retrieved': len(cand_ids),
        'unique_articles': len(articles),
        'top_articles_returned': min(max_results, len(articles)),
        'database': 'StatutesDB',
        'collection': CHROMA_COLLECTION_NAME,
        'ce_scores': {
            'min': round(min(effective), 3) if effective else 0,
            'median': round(_stats.median(effective), 3) if effective else 0,
            'max': round(max(effective), 3) if effective else 0,
        },
    }

    output = format_rules_output(articles, query_params, stats, max_results=max_results)

    if verbose:
        print(f"\n✓ Retrieval complete! Found {output['count']} relevant articles")
        print("="*80 + "\n")

    return output


def display_rules_results(output: Dict):
    """Pretty print rules retrieval results."""
    print("\n" + "="*80)
    print("📊 STATUTORY RULES RESULTS")
    print("="*80)
    
    # Query Summary
    print("\n🔎 Query Summary:")
    print(f"  Vector Query: {output['query_summary']['vector_query'][:70]}...")
    if output['query_summary']['filtered_statutes']:
        print(f"  Filtered Statutes: {', '.join(output['query_summary']['filtered_statutes'])}")
    
    # Stats
    print(f"\n📈 Statistics:")
    print(f"  Total Chunks Retrieved: {output['stats']['total_chunks_retrieved']}")
    print(f"  Unique Articles: {output['stats']['unique_articles']}")
    print(f"  Top Articles Returned: {output['stats']['top_articles_returned']}")
    
    # Results
    print(f"\n📜 Top {len(output['results'])} Articles:\n")
    
    for result in output['results']:
        print(f"{'─'*80}")
        print(f"[{result['rank']}] {result['document_label']} - Article {result['article_number']}")
        print(f"    {result['article_header']}")
        print(f"\n    📍 Location:")
        print(f"       Chapter: {result['chapter']} - {result['chapter_title']}")
        if result['part']:
            print(f"       Part: {result['part']} - {result['part_title']}")
        print(f"\n    📊 Relevance Score: {result['score']:.3f}")
        print(f"    📝 Chunks: {result['chunk_count']}")
        print(f"\n    📄 Text Preview:")
        print(f"       {result['text_preview'][:300]}...")
        print()
    
    print("="*80 + "\n")


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    # Sample test with a rule-focused query
    sample_classifier_output = {
        "intents": ["RULE_SEARCH"],
        "target_components": ["RULES"],
        "legal_domains": [
            {"domain": "Constitutional Law", "confidence_score": 0.9}
        ],
        "entities": {
            "statutes": ["Constitution of Kenya"],
            "cases": [],
            "judges": []
        },
        "vector_query": "fair trial rights, access to justice, due process, Article 47",
        "reasoning_summary": "Testing statutory retrieval for constitutional rights"
    }
    
    try:
        print("Testing MasterRulesDB Retrieval...")
        results = retrieve_relevant_rules(sample_classifier_output, verbose=True)
        display_rules_results(results)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
