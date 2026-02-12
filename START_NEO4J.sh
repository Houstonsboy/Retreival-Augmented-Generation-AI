#!/bin/bash

# Quick start script for Neo4j

echo "=== Starting Neo4j Database ==="
echo ""

cd neo4j-community-5.18.0

# Start Neo4j
./bin/neo4j start

# Wait a moment
sleep 3

# Check status
./bin/neo4j status

echo ""
echo "✓ Neo4j started!"
echo ""
echo "📊 Neo4j Browser: http://localhost:7474"
echo "🔌 Bolt Port: 7687"
echo ""
echo "Default credentials (first login):"
echo "  Username: neo4j"
echo "  Password: neo4j"
echo "  (You'll be prompted to change the password)"
echo ""
echo "To stop Neo4j later:"
echo "  ./neo4j-community-5.18.0/bin/neo4j stop"
