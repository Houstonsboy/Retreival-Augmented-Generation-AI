# Dictionary Knowledge Graph Setup Guide

This guide walks you through building a Neo4j knowledge graph from your legal dictionary data.

## Graph Structure

**Nodes:**
- `Term` - Each dictionary term/entry
  - Properties: `name` (term text), `definition`, `page` (page number)

**Relationships:**
- `SEE` - Reference to another term (e.g., "See PROPERTY")
- `CF` - Compare with another term (e.g., "Cf. DESERTION")
- `ALSO_TERMED` - Alternative name for the term

## Prerequisites

### 1. Install Java 17+
```bash
java -version
# If not installed: sudo apt install openjdk-17-jdk
```

### 2. Install Maven
```bash
mvn -version
# If not installed: sudo apt install maven
```

### 3. Start Neo4j

First, make sure Neo4j is installed:
```bash
cd ~/Retreival-Augmented-Generation-AI
ls neo4j-community-5.18.0
```

Start Neo4j server:
```bash
cd neo4j-community-5.18.0
./bin/neo4j start
```

Wait a few seconds, then verify it's running:
```bash
./bin/neo4j status
```

### 4. Set Neo4j Password (First Time Only)

Open your browser to: **http://localhost:7474**

- Default username: `neo4j`
- Default password: `neo4j`
- You'll be prompted to change the password. Remember it!

## Build the Knowledge Graph

### Step 1: Navigate to the project
```bash
cd ~/Retreival-Augmented-Generation-AI/neo4j-java
```

### Step 2: Set your Neo4j password
```bash
export NEO4J_PASSWORD="your-new-password"
```

### Step 3: Compile the project
```bash
mvn clean compile
```

### Step 4: Run the knowledge graph builder
```bash
mvn exec:java -Dexec.mainClass="com.legal.rag.neo4j.DictionaryKnowledgeGraph"
```

This will:
1. Read the JSON dictionary file
2. Create a `Term` node for each dictionary entry
3. Create relationships based on metadata (`SEE`, `CF`, `ALSO_TERMED`)
4. Print statistics and sample queries

Expected output:
```
=== Dictionary Knowledge Graph Builder ===
Neo4j URI: neo4j://localhost:7687
JSON File: ../backend/Const_IntepretationLaws/extracted_dictionary_pages_18_to_30.json

Reading JSON file...
Found 270 dictionary entries.
Creating nodes and relationships...
Processed 50/270 terms...
Processed 100/270 terms...
...

=== Knowledge Graph Build Complete ===
Nodes created: 270
SEE relationships: 15
CF relationships: 8
ALSO_TERMED relationships: 12
Total relationships: 35
```

## Explore the Graph in Neo4j Browser

Open **http://localhost:7474** in your browser.

### Example Queries

#### 1. View all terms (limit to 100)
```cypher
MATCH (n:Term) RETURN n LIMIT 100
```

#### 2. Find a specific term (e.g., "abatement")
```cypher
MATCH (t:Term {name: 'abatement'}) RETURN t
```

#### 3. Find all terms connected to a specific term
```cypher
MATCH (t:Term {name: 'abandonment'})-[r]-(related) 
RETURN t, r, related
```

#### 4. View all SEE relationships
```cypher
MATCH (a:Term)-[r:SEE]->(b:Term) 
RETURN a, r, b 
LIMIT 25
```

#### 5. View all ALSO_TERMED relationships
```cypher
MATCH (a:Term)-[r:ALSO_TERMED]->(b:Term) 
RETURN a, r, b 
LIMIT 25
```

#### 6. Count relationships by type
```cypher
MATCH ()-[r]->() 
RETURN type(r) as RelationType, count(r) as Count
```

#### 7. Find most connected terms
```cypher
MATCH (t:Term)-[r]-() 
WITH t, count(r) as rels 
WHERE rels > 2 
RETURN t.name, rels 
ORDER BY rels DESC
```

#### 8. Search terms by name pattern
```cypher
MATCH (t:Term) 
WHERE t.name CONTAINS 'abandon' 
RETURN t.name, t.definition 
LIMIT 10
```

#### 9. Find paths between two terms
```cypher
MATCH path = shortestPath((a:Term {name: 'abaction'})-[*]-(b:Term {name: 'ABIGEATUS'}))
RETURN path
```

#### 10. Show term with all its relationships
```cypher
MATCH (t:Term {name: 'abandum'})-[r]-(other)
RETURN t, r, other
```

## Visualizing the Graph

When you run queries in Neo4j Browser:

1. **Graph View** (default) - Shows nodes and relationships visually
2. **Table View** - Shows results in tabular format
3. **Text View** - Shows raw JSON/text results

**Tips for better visualization:**
- Click and drag nodes to rearrange them
- Double-click a node to expand its relationships
- Use the style panel (bottom right) to customize colors and sizes
- Use `LIMIT` to avoid overwhelming the browser with too many nodes

## Clearing the Database

If you want to start fresh:

### Option 1: Using the code (recommended)
Uncomment this line in `DictionaryKnowledgeGraph.java`:
```java
// builder.clearDatabase();
```

Then run again:
```bash
mvn exec:java -Dexec.mainClass="com.legal.rag.neo4j.DictionaryKnowledgeGraph"
```

### Option 2: Using Cypher query
In Neo4j Browser:
```cypher
MATCH (n) DETACH DELETE n
```

## Troubleshooting

### "Connection refused" error
- Make sure Neo4j is running: `./bin/neo4j status`
- Check if it's listening on port 7687: `netstat -tulpn | grep 7687`

### "Authentication failed" error
- Verify your password is correct
- Set environment variable: `export NEO4J_PASSWORD="your-password"`

### Maven build errors
- Make sure you're in the `neo4j-java` directory
- Try: `mvn clean install`

### JSON file not found
- Check the path in the Java code (line with `jsonFile = ...`)
- You can pass a custom path: `mvn exec:java -Dexec.mainClass="com.legal.rag.neo4j.DictionaryKnowledgeGraph" -Dexec.args="/full/path/to/file.json"`

## Stopping Neo4j

When done:
```bash
cd ~/Retreival-Augmented-Generation-AI/neo4j-community-5.18.0
./bin/neo4j stop
```

## Next Steps

- Add more dictionary pages to expand the graph
- Create more complex queries to find term relationships
- Export the graph for use in your RAG system
- Add more node properties (synonyms, categories, etc.)
- Create more relationship types based on your needs
