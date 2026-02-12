# Neo4j Knowledge Graph Writer (Java)

Java client that writes nodes and relationships to your local Neo4j instance (Bolt at `localhost:7687`).

## Prerequisites

- **Java 17+** (`java -version`)
- **Maven** – if missing: `sudo apt install maven` (or [install from maven.apache.org](https://maven.apache.org/download.cgi))
- **Neo4j** running (e.g. `../neo4j-community-5.18.0/bin/neo4j start`)

## Configuration

By default the app uses:

- **URI:** `neo4j://localhost:7687`
- **User:** `neo4j`
- **Password:** `neo4j` (change in Neo4j after first login, then set `NEO4J_PASSWORD`)

Override with environment variables:

- `NEO4J_URI` – e.g. `neo4j://localhost:7687`
- `NEO4J_USER` – e.g. `neo4j`
- `NEO4J_PASSWORD` – your Neo4j password

## Build and run

```bash
cd neo4j-java
mvn compile
mvn exec:java
```

The default `main` creates two example nodes and a relationship. Check in Neo4j Browser (`http://localhost:7474`) with:

```cypher
MATCH (n) RETURN n
```

## Use in your code

```java
try (Neo4jWriter writer = new Neo4jWriter("neo4j://localhost:7687", "neo4j", "your-password")) {
    writer.createNode("Article", java.util.Map.of("id", "art1", "title", "Sovereignty"));
    writer.createNode("Chapter", java.util.Map.of("id", "ch1", "title", "Chapter One"));
    writer.createRelationship("Article", "id", "art1", "BELONGS_TO", "Chapter", "id", "ch1");
}
```

## API summary

- **createNode(label, properties)** – create a node with label and map of properties
- **createNode(label, key, value)** – create a node with one property
- **createRelationship(fromLabel, fromKey, fromVal, relType, toLabel, toKey, toVal)** – MERGE two nodes and create a relationship
- **createRelationship(..., relProperties)** – same with relationship properties
- **runWrite(cypher, params)** / **runRead(cypher, params)** – run custom Cypher

Labels and relationship types are sanitized (spaces → `_`, non-alphanumeric removed).
