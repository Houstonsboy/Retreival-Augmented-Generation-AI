# Knowledge Graph Structure

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                    DICTIONARY KNOWLEDGE GRAPH                    │
└─────────────────────────────────────────────────────────────────┘

NODE TYPE: Term
┌────────────────────────┐
│      [Term Node]       │
│                        │
│  Properties:           │
│  • name: string        │
│  • definition: string  │
│  • page: integer       │
└────────────────────────┘


RELATIONSHIP TYPES:

1. SEE (Reference)
   
   [abaction] ──SEE──> [ABIGEATUS]
   
   Meaning: "See ABIGEATUS" - directs to the main definition

2. CF (Compare)
   
   [abandonment] ──CF──> [DESERTION]
   
   Meaning: "Compare with DESERTION" - related concept

3. ALSO_TERMED (Synonym/Alternative)
   
   [abandum] ──ALSO_TERMED──> [abandun]
   
   Meaning: Also known as "abandun" - alternative spelling/term


EXAMPLE SUBGRAPH:

                    ┌─────────────┐
                    │ abandonment │
                    │ (Term)      │
                    └──────┬──────┘
                           │
                          CF
                           │
                           ↓
                    ┌─────────────┐
                    │  DESERTION  │
                    │  (Term)     │
                    └─────────────┘


                    ┌─────────────┐
                    │  abandum    │
                    │  (Term)     │
                    └──────┬──────┘
                           │
                    ALSO_TERMED
                           │
                           ↓
                    ┌─────────────┐
                    │  abandun    │
                    │  (Term)     │
                    └─────────────┘


                    ┌─────────────┐
                    │  abaction   │
                    │  (Term)     │
                    └──────┬──────┘
                           │
                          SEE
                           │
                           ↓
                    ┌─────────────┐
                    │  ABIGEATUS  │
                    │  (Term)     │
                    └─────────────┘


FULL GRAPH TOPOLOGY:

         [Term 1] ──SEE──> [Term 2]
            │                  │
           CF                 CF
            │                  │
            ↓                  ↓
         [Term 3] ──ALSO_TERMED──> [Term 4]
            │
          SEE
            │
            ↓
         [Term 5] ──CF──> [Term 6]


```

## Graph Statistics (from sample data)

```
Total Nodes:        270 terms
Total Edges:        35 relationships
  
Breakdown:
  SEE:              15 relationships (42.9%)
  CF:               8 relationships (22.9%)
  ALSO_TERMED:      12 relationships (34.3%)

Average Connections: ~0.26 per node
Max Connections:     (varies by term)
```

## Node Properties in Detail

### Term Node
```json
{
  "name": "abandonment",
  "definition": "abandonment, n. 1. The relinquishing of a right or interest with the intention of never again claiming it...",
  "page": 19
}
```

## Cypher Schema

```cypher
// Create a Term node
CREATE (t:Term {
  name: "abandonment",
  definition: "abandonment, n. 1. The relinquishing...",
  page: 19
})

// Create SEE relationship
MATCH (a:Term {name: "abaction"})
MATCH (b:Term {name: "ABIGEATUS"})
CREATE (a)-[:SEE]->(b)

// Create CF relationship
MATCH (a:Term {name: "abandonment"})
MATCH (b:Term {name: "DESERTION"})
CREATE (a)-[:CF]->(b)

// Create ALSO_TERMED relationship
MATCH (a:Term {name: "abandum"})
MATCH (b:Term {name: "abandun"})
CREATE (a)-[:ALSO_TERMED]->(b)
```

## Index Strategy

```cypher
// Primary index for term lookup
CREATE INDEX term_name IF NOT EXISTS 
FOR (t:Term) ON (t.name)

// Optional: Full-text search index
CALL db.index.fulltext.createNodeIndex(
  "termDefinitionSearch",
  ["Term"],
  ["name", "definition"]
)
```

## Query Patterns

### Pattern 1: Direct Lookup
```cypher
MATCH (t:Term {name: $termName})
RETURN t
```

### Pattern 2: Relationship Traversal
```cypher
MATCH (t:Term {name: $termName})-[r]->(related)
RETURN t, type(r) as relationship, related
```

### Pattern 3: Multi-hop Navigation
```cypher
MATCH path = (start:Term {name: $termName})-[*1..3]-(end:Term)
RETURN path
```

### Pattern 4: Relationship Filtering
```cypher
MATCH (t:Term)-[r:SEE]->(target)
WHERE t.name CONTAINS $searchText
RETURN t, target
```

## Use Cases

1. **Term Resolution**: Follow SEE relationships to main definitions
2. **Concept Exploration**: Use CF to find related legal concepts
3. **Synonym Finding**: Navigate ALSO_TERMED for alternative terms
4. **Legal Research**: Build query chains across relationships
5. **RAG Enhancement**: Use graph structure for context-aware retrieval

## Extending the Graph

Future enhancements could include:

```
NEW NODE TYPES:
• Category (e.g., "Criminal Law", "Property Law")
• Source (e.g., books, statutes)
• Case (legal case citations)

NEW RELATIONSHIPS:
• DEFINED_IN → links to source documents
• CATEGORIZED_AS → links to legal categories
• CITED_IN → links to cases where term appears
• RELATED_TO → general semantic similarity
• PART_OF → hierarchical relationships

EXAMPLE ENHANCED STRUCTURE:

  [Category: Criminal Law]
         ↑
    CATEGORIZED_AS
         │
   [Term: abandonment] ──CITED_IN──> [Case: Smith v. Jones]
         │
       SEE
         │
         ↓
   [Term: desertion] ──DEFINED_IN──> [Source: Black's Law Dictionary]
```

---

*This knowledge graph enables efficient navigation, exploration, and retrieval of legal terminology and their interconnections.*
