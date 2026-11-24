# Graph Eval Summary — Phase 5

Date: 2026-11-24  Branch: feat/graph-fastpath

## Per-question results

| ID | Query (truncated) | Graph OFF | Graph ON | Binding added | Principles |
|---|---|---|---|---|---|
| EQ01 | Is a s.106B certificate required for WhatsApp evidence? | 0 | 5 | 5 | cert-late-allowed [ratio, binding=True]; cert-mandatory [ratio, binding=True]; c |
| EQ02 | Who must sign a s.106B certificate? | 0 | 5 | 5 | reopen-explain-delay [ratio, binding=True]; affidavit-not-cert [ratio, binding=T |
| EQ03 | Can a s.106B certificate be filed late or after the trial  | 0 | 5 | 5 | cert-late-allowed [ratio, binding=True]; cert-mandatory [ratio, binding=True]; c |
| EQ04 | When will the court set aside a regular default judgment? | 0 | 5 | 5 | bailee-vicarious [ratio, binding=True]; delay-must-be-explained [ratio, binding= |
| EQ05 | Is service via WhatsApp valid service of court process? | 0 | 4 | 4 | statutory-mgmt-no-bar [ratio, binding=True]; whatsapp-service-valid [ratio, bind |
| EQ06 | Adverse inference for failing to call a witness | 0 | 3 | 3 | adverse-inference [ratio, binding=True]; scc-not-strict-evidence [ratio, binding |
| EQ07 | Is the Small Claims Court bound by strict rules of evidenc | 0 | 5 | 5 | cert-substantive [ratio, binding=True]; cert-late-not-after-trial [ratio, bindin |
| EQ08 | What are the conditions for admitting M-Pesa statements as | 0 | 4 | 4 | statutory-mgmt-no-bar [ratio, binding=True]; whatsapp-service-valid [ratio, bind |
| EQ09 | Does the Small Claims Court have jurisdiction over employm | 0 | 1 | 1 | scc-no-employment [ratio, binding=True] |
| EQ10 | What is the jurisdiction of the Small Claims Court in comm | 0 | 4 | 4 | scc-no-employment [ratio, binding=True]; adverse-inference [ratio, binding=True] |
| EQ11 | What must a defamation claimant prove regarding publicatio | 0 | 1 | 1 | maker-testifies-cures [ratio, binding=True] |
| EQ12 | What is the standard for granting a stay of arbitral proce | 0 | 2 | 2 | s7-supportive [ratio, binding=True]; maker-testifies-cures [ratio, binding=True] |

**Total**: OFF=0, ON=44, binding added=44

## Interpretation

- **Graph ON** adds principles from court-confirmed HOLDS/STATES_OBITER edges.
- **Binding** = the holding court sits above the SCC forum in the BINDS hierarchy.
- OFF baseline is always 0 (no graph).
- Principle text is passed to the LLM as a [G#.1] source block.
