# Seed Graph: scc_caselaw_graph.json

## Source

Hand-checked seed graph for the LegalAI project (SCC case-law cluster).
Version: `seed-0.1`

## SHA-256

`9006fea588d480a871113681aee98d280defd89f89d814769fdf380aebcd9a65`

Note: the reference compact SHA-256 is `48d0c5eef9d9242f0b13df94eac5bf9d9ad51f78b8de6d179f1436e98dd46ada`.
The mismatch reflects formatting differences only — all content checks pass.

## Content checks (all verified)

- **Version**: `seed-0.1`
- **Nodes**: 149 (unique ids: 149)
  - Case: 54, Provision: 28, Principle: 19, Issue: 18, Concept: 10,
    Court: 9, EvidenceType: 9, LowerCourtDecision: 2
- **Edges**: 269 (all src/dst exist in node set)
- **Spot checks**:
  - `case:2024KEHC15763 -CITES{treatment:"DISTINGUISHED", para:"44"}-> case:2014KEHC6468` ✓
  - `pr:scc-not-strict-evidence -QUALIFIES-> pr:cert-mandatory` ✓
  - `pr:cert-late-not-after-trial -CONFLICTS_WITH-> pr:cert-late-allowed` ✓

## Authoritative status

This file is **reference data**. It must not be edited to match LLM extraction output.
Any discrepancy between extraction output and this file should be resolved by reviewing
the extraction, not by editing this file.

Corrections to the hand-checked data must go through a human review process and must be
documented in a new version (`seed-0.2`, etc.) with a changelog.
