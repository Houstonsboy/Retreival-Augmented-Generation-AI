# Legal Dictionary Knowledge Graph - Quick Start Guide

Build a Neo4j knowledge graph from your legal dictionary data in 3 simple steps!

## 🎯 What You're Building

A knowledge graph where:
- **Nodes**: Dictionary terms (e.g., "abatement", "abandonment")
- **Edges**: Relationships between terms:
  - `SEE` → Direct reference (e.g., "abaction" SEE "ABIGEATUS")
  - `CF` → Compare with (e.g., "abandonment" CF "DESERTION")
  - `ALSO_TERMED` → Alternative names (e.g., "abandum" ALSO_TERMED "abandun")

```
    [Term: abaction]
           |
         (SEE)
           |
           ↓
    [Term: ABIGEATUS]
    
    [Term: abandonment]
           |
         (CF)
           |
           ↓
    [Term: DESERTION]
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start Neo4j

```bash
cd ~/Retreival-Augmented-Generation-AI
./START_NEO4J.sh
```

**First time?** 
1. Open http://localhost:7474 in your browser
2. Login with username `neo4j`, password `neo4j`
3. Set a new password when prompted
4. Remember this password!

### Step 2: Set your password

```bash
export NEO4J_PASSWORD="your-new-password"
```

### Step 3: Build the knowledge graph

```bash
cd neo4j-java
./run_dictionary_graph.sh
```

That's it! ✨

---

## 📊 View Your Knowledge Graph

Open **http://localhost:7474** in your browser.

### Try These Queries:

**1. See all terms (first 50)**
```cypher
MATCH (n:Term) RETURN n LIMIT 50
```

**2. Find "abandonment" and its connections**
```cypher
MATCH (t:Term {name: 'abandonment'})-[r]-(related) 
RETURN t, r, related
```

**3. View all SEE relationships**
```cypher
MATCH (a:Term)-[r:SEE]->(b:Term) 
RETURN a, r, b
```

**4. Find most connected terms**
```cypher
MATCH (t:Term)-[r]-() 
WITH t, count(r) as connections 
RETURN t.name, connections 
ORDER BY connections DESC 
LIMIT 10
```

**5. Search for terms containing "abandon"**
```cypher
MATCH (t:Term) 
WHERE t.name CONTAINS 'abandon' 
RETURN t.name, t.page
```

---

## 🎨 Visualizing in Neo4j Browser

### Graph View Tips:
- **Drag nodes** to rearrange
- **Double-click** a node to expand its connections
- **Click relationships** to see their type
- **Right sidebar** shows node/relationship properties
- **Style panel** (bottom right) to customize colors

### View Modes:
- **Graph** 📊 - Visual network (default)
- **Table** 📋 - Spreadsheet view
- **Text** 📄 - JSON/raw data

---

## 📈 What Was Created?

After running the builder, you'll see:

```
=== Knowledge Graph Build Complete ===
Nodes created: 270
SEE relationships: 15
CF relationships: 8
ALSO_TERMED relationships: 12
Total relationships: 35
```

Each **Term node** has:
- `name` - The term text
- `definition` - Full definition from the dictionary
- `page` - Page number in source document

---

## 🔍 Example Use Cases

### 1. Find Alternative Terms
```cypher
MATCH (t:Term)-[:ALSO_TERMED]->(alt:Term)
RETURN t.name as original, alt.name as alternative
```

### 2. Trace Reference Chain
```cypher
MATCH path = (start:Term {name: 'abaction'})-[:SEE*1..3]->(end:Term)
RETURN path
```

### 3. Find Unconnected Terms
```cypher
MATCH (t:Term)
WHERE NOT (t)-[]-()
RETURN t.name, t.definition
LIMIT 10
```

### 4. Export Term List
```cypher
MATCH (t:Term)
RETURN t.name, t.page, t.definition
ORDER BY t.name
```

---

## 🛠️ Manual Build Instructions

If you prefer to run commands separately:

```bash
# 1. Start Neo4j
cd ~/Retreival-Augmented-Generation-AI/neo4j-community-5.18.0
./bin/neo4j start

# 2. Build Java project
cd ~/Retreival-Augmented-Generation-AI/neo4j-java
mvn clean compile

# 3. Run knowledge graph builder
export NEO4J_PASSWORD="your-password"
mvn exec:java -Dexec.mainClass="com.legal.rag.neo4j.DictionaryKnowledgeGraph"
```

---

## 🧹 Clearing the Database

**Option 1: In code**

Edit `DictionaryKnowledgeGraph.java` and uncomment:
```java
builder.clearDatabase();
```

**Option 2: In Neo4j Browser**
```cypher
MATCH (n) DETACH DELETE n
```

---

## 🐛 Troubleshooting

### "Connection refused"
```bash
# Check if Neo4j is running
cd ~/Retreival-Augmented-Generation-AI/neo4j-community-5.18.0
./bin/neo4j status

# If not running, start it
./bin/neo4j start
```

### "Authentication failed"
```bash
# Make sure password is set
export NEO4J_PASSWORD="your-actual-password"

# Or edit the Java code to hardcode it (not recommended)
```

### "JSON file not found"
The code looks for:
```
../backend/Const_IntepretationLaws/extracted_dictionary_pages_18_to_30.json
```

Make sure this file exists relative to the `neo4j-java` directory.

### Maven build fails
```bash
cd neo4j-java
mvn clean install
```

---

## 🛑 Stopping Neo4j

When you're done:
```bash
cd ~/Retreival-Augmented-Generation-AI/neo4j-community-5.18.0
./bin/neo4j stop
```

---

## 📚 Next Steps

1. **Add more data**: Process additional dictionary pages
2. **Enrich relationships**: Add more metadata types
3. **Integrate with RAG**: Query the graph from your Python backend
4. **Add full-text search**: Create indexes for faster searches
5. **Export for analysis**: Use Neo4j's export features

---

## 📖 More Resources

- **Detailed guide**: `neo4j-java/DICTIONARY_GRAPH_GUIDE.md`
- **Neo4j docs**: https://neo4j.com/docs/
- **Cypher guide**: https://neo4j.com/developer/cypher/

---

**Happy graphing! 🎉**
