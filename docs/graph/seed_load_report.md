# Seed Load Report — scc_caselaw_graph.json @ seed-0.1

Generated: 2026-11-24  
Branch: feat/graph-fastpath  
Both dev (port 7687) and test (port 7688) containers loaded cleanly.

## Counts

### Seed nodes (source = scc_caselaw_graph.json@seed-0.1)

| Label | Count |
|---|---|
| Case | 54 |
| Concept | 10 |
| Court | 9 |
| EvidenceType | 9 |
| Issue | 18 |
| LowerCourtDecision | 2 |
| Principle | 19 |
| Provision | 28 |
| **Total** | **149** |

### Pre-existing nodes from migrations (0001 schema + 0002 reference data)

| Label | Count |
|---|---|
| ClaimType | 9 |
| ContractElement | 11 |
| Court | +1 (KEMC from migration) |
| EvidenceType | +2 (from migration) |
| GateCondition | 8 |
| Outcome | 10 |
| PartyType | 5 |
| Provision | +9 (from migration) |
| Statute | 2 |

### Total graph nodes (excluding Migration nodes)

| Label | Count |
|---|---|
| Case | 54 |
| ClaimType | 9 |
| Concept | 10 |
| ContractElement | 11 |
| Court | 10 |
| EvidenceType | 11 |
| GateCondition | 8 |
| Issue | 18 |
| LowerCourtDecision | 2 |
| Outcome | 10 |
| PartyType | 5 |
| Principle | 19 |
| Provision | 37 |
| Statute | 2 |

### Edges from seed

| Total edges loaded | 269 |
|---|---|
| CITES | 49 |
| Other | 220 |

## Chroma case_identifier mapping (9 cases)

| case_key | chroma_case_identifier | Name |
|---|---|---|
| KECA-2001-154 | 2001KECA154(KLR) | Express (Kenya) Limited v Manju Patel |
| KECA-2010-346 | 2010KECA346(KLR) | Safaricom Limited v Ocean View Beach Hotel Limited & 2 others |
| KEHC-2014-6468 | 2014KEHC6468(KLR) | Nonny Gathoni Njenga & another v Catherine Masitsa & another |
| KEHC-2020-8905 | 2020KEHC8905(KLR) | Peter Ngethe Ngari t/a PNN Funeral Services v Standard Group Limited PLC & another |
| KEHC-2024-15763 | 2024KEHC15763(KLR) | Ogembo v Yongo |
| KEHC-2024-17038 | 2024KEHC17038(KLR) | Obado v Anayo |
| SCC-2026-235 | 2026SCC235(KLR) | New Titanic Jacaranda General Spares v Omil & another |
| SCC-2026-241 | 2026SCC241(KLR) | Ndeti v Kaguchwa |
| SCC-2026-242 | 2026SCC242(KLR) | Umoja Magharibi (K) Ltd v Lubano |

## In-corpus, not in Chroma (2 cases)

| case_key | seed_id | Name |
|---|---|---|
| KEHC-2011-1481 | case:2011KEHC1481 | Republic v Barisa Wayu Mataguda |
| KEELC-2021-220 | case:2021KEELC220 | Republic v Director Land Administration & 2 others; Ex parte Roysa Community Development Society Ltd |

These cases have `in_corpus=true` in the seed but `in_vector_index=false`, so no `chroma_case_identifier` is set. They participate in the graph (CITES, HOLDS, etc.) but cannot be sourced for Chroma chunks in the fast path.

## Spot-check validation results

All 4 spot checks passed on both containers:

- [x] Seed node count = 149
- [x] Ogembo (`KEHC-2024-15763`) CITES Nonny (`KEHC-2014-6468`) DISTINGUISHED para:44
- [x] `pr:scc-not-strict-evidence` QUALIFIES `pr:cert-mandatory`
- [x] `pr:cert-late-not-after-trial` CONFLICTS_WITH `pr:cert-late-allowed`
- [x] Barisa (`KEHC-2011-1481`) has ≥ 3 incoming CITES
