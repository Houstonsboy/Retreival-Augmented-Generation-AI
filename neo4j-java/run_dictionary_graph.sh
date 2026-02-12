#!/bin/bash

# Dictionary Knowledge Graph Builder - Quick Run Script

set -e

echo "=== Dictionary Knowledge Graph Builder ==="
echo ""

# Check if Neo4j password is set
if [ -z "$NEO4J_PASSWORD" ]; then
    echo "⚠️  NEO4J_PASSWORD not set. Using default 'neo4j'"
    echo "   Set your password with: export NEO4J_PASSWORD='your-password'"
    echo ""
fi

# Check if Neo4j is running
echo "Checking if Neo4j is running..."
if ! nc -z localhost 7687 2>/dev/null; then
    echo "❌ Neo4j is not running on port 7687"
    echo ""
    echo "Start Neo4j with:"
    echo "  cd ../neo4j-community-5.18.0"
    echo "  ./bin/neo4j start"
    echo ""
    exit 1
fi
echo "✓ Neo4j is running"
echo ""

# Compile and run
echo "Building and running knowledge graph builder..."
mvn clean compile exec:java -Dexec.mainClass="com.legal.rag.neo4j.DictionaryKnowledgeGraph"

echo ""
echo "=== Done! ==="
echo "Open Neo4j Browser at: http://localhost:7474"
