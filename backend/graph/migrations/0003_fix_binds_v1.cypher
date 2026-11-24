// Migration 0003 — Fix BINDS hierarchy.
// Migration 0002 had MERGE (variable)-[:BINDS]->(variable) statements where the
// variables were not bound in-query (each statement ran separately).  This created
// anonymous labelless nodes and useless BINDS edges instead of the intended court
// hierarchy.  This migration:
//   1. Removes all anonymous (label-free) nodes and their relationships.
//   2. Removes any stale BINDS edges between Court nodes (safe to rebuild).
//   3. Re-creates the full BINDS hierarchy using MATCH.
// Never edit this file after it has been applied — write 0004 instead.

// Step 1: remove orphan anonymous nodes and their relationships
MATCH (n) WHERE size(labels(n)) = 0 DETACH DELETE n;

// Step 2: remove any existing Court-BINDS-Court edges (safe to rebuild)
MATCH (:Court)-[r:BINDS]->(:Court) DELETE r;

// Step 3: rebuild BINDS hierarchy using MATCH
MATCH (sc:Court {code:"KESC"}),  (coa:Court {code:"KECA"})  MERGE (sc)-[:BINDS]->(coa);
MATCH (sc:Court {code:"KESC"}),  (hc:Court {code:"KEHC"})   MERGE (sc)-[:BINDS]->(hc);
MATCH (sc:Court {code:"KESC"}),  (elc:Court {code:"KEELC"}) MERGE (sc)-[:BINDS]->(elc);
MATCH (sc:Court {code:"KESC"}),  (elrc:Court {code:"KEELRC"}) MERGE (sc)-[:BINDS]->(elrc);
MATCH (sc:Court {code:"KESC"}),  (scc:Court {code:"SCC"})   MERGE (sc)-[:BINDS]->(scc);
MATCH (sc:Court {code:"KESC"}),  (mc:Court {code:"KEMC"})   MERGE (sc)-[:BINDS]->(mc);
MATCH (coa:Court {code:"KECA"}), (hc:Court {code:"KEHC"})   MERGE (coa)-[:BINDS]->(hc);
MATCH (coa:Court {code:"KECA"}), (elc:Court {code:"KEELC"}) MERGE (coa)-[:BINDS]->(elc);
MATCH (coa:Court {code:"KECA"}), (elrc:Court {code:"KEELRC"}) MERGE (coa)-[:BINDS]->(elrc);
MATCH (coa:Court {code:"KECA"}), (scc:Court {code:"SCC"})   MERGE (coa)-[:BINDS]->(scc);
MATCH (coa:Court {code:"KECA"}), (mc:Court {code:"KEMC"})   MERGE (coa)-[:BINDS]->(mc);
MATCH (hc:Court {code:"KEHC"}),  (scc:Court {code:"SCC"})   MERGE (hc)-[:BINDS]->(scc);
MATCH (hc:Court {code:"KEHC"}),  (mc:Court {code:"KEMC"})   MERGE (hc)-[:BINDS]->(mc);
MATCH (elc:Court {code:"KEELC"}), (scc:Court {code:"SCC"})  MERGE (elc)-[:BINDS]->(scc);
MATCH (elc:Court {code:"KEELC"}), (mc:Court {code:"KEMC"})  MERGE (elc)-[:BINDS]->(mc);
MATCH (elrc:Court {code:"KEELRC"}), (scc:Court {code:"SCC"}) MERGE (elrc)-[:BINDS]->(scc);
MATCH (elrc:Court {code:"KEELRC"}), (mc:Court {code:"KEMC"}) MERGE (elrc)-[:BINDS]->(mc);
