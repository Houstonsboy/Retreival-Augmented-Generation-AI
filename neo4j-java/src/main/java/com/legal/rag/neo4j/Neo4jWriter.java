package com.legal.rag.neo4j;

import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Session;
import org.neo4j.driver.Result;
import org.neo4j.driver.Values;

/**
 * Writes nodes and relationships to a Neo4j knowledge graph.
 * Connects to neo4j://localhost:7687 by default (set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD to override).
 */
public class Neo4jWriter implements AutoCloseable {

    private final Driver driver;

    public Neo4jWriter(String uri, String user, String password) {
        this.driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password));
    }

    @Override
    public void close() {
        driver.close();
    }

    /**
     * Creates a single node with the given label and properties.
     * Properties map keys become Cypher property names; values must be serializable (String, Number, Boolean).
     */
    public void createNode(String label, java.util.Map<String, Object> properties) {
        try (Session session = driver.session()) {
            StringBuilder cypher = new StringBuilder("CREATE (n:");
            cypher.append(sanitizeLabel(label));
            cypher.append(" $props)");
            session.run(cypher.toString(), Values.parameters("props", properties));
        }
    }

    /**
     * Creates a node with a single property (convenience method).
     */
    public void createNode(String label, String key, Object value) {
        createNode(label, java.util.Map.of(key, value));
    }

    /**
     * Creates a relationship between two nodes identified by label and key property.
     * Creates fromNode and toNode if they don't exist (MERGE), then creates the relationship.
     */
    public void createRelationship(
            String fromLabel, String fromKey, Object fromValue,
            String relType,
            String toLabel, String toKey, Object toValue,
            java.util.Map<String, Object> relProperties
    ) {
        try (Session session = driver.session()) {
            // Neo4j doesn't allow parameter maps in MERGE patterns for relationships
            // So we MERGE the relationship first, then SET properties
            String cypher = String.format(
                    "MERGE (a:%s {%s: $fromVal }) " +
                    "MERGE (b:%s {%s: $toVal }) " +
                    "MERGE (a)-[r:%s]->(b) " +
                    "SET r += $props " +
                    "RETURN r",
                    sanitizeLabel(fromLabel), fromKey,
                    sanitizeLabel(toLabel), toKey,
                    sanitizeRelType(relType)
            );
            session.run(cypher, Values.parameters(
                    "fromVal", fromValue,
                    "toVal", toValue,
                    "props", relProperties != null ? relProperties : java.util.Map.of()
            ));
        }
    }

    /**
     * Creates a relationship with no properties.
     */
    public void createRelationship(
            String fromLabel, String fromKey, Object fromValue,
            String relType,
            String toLabel, String toKey, Object toValue
    ) {
        createRelationship(fromLabel, fromKey, fromValue, relType, toLabel, toKey, toValue, null);
    }

    /**
     * Runs an arbitrary Cypher write query (use for batch or custom writes).
     */
    public Result runWrite(String cypher, java.util.Map<String, Object> parameters) {
        Session session = driver.session();
        return session.run(cypher, parameters != null ? parameters : java.util.Map.of());
    }

    /**
     * Runs a read query (e.g. to verify data).
     */
    public Result runRead(String cypher, java.util.Map<String, Object> parameters) {
        try (Session session = driver.session()) {
            return session.run(cypher, parameters != null ? parameters : java.util.Map.of());
        }
    }

    private static String sanitizeLabel(String label) {
        if (label == null || label.isBlank()) return "Node";
        return label.trim().replaceAll("\\s+", "_").replaceAll("[^a-zA-Z0-9_]", "");
    }

    private static String sanitizeRelType(String type) {
        if (type == null || type.isBlank()) return "RELATES_TO";
        return type.trim().toUpperCase().replaceAll("\\s+", "_").replaceAll("[^a-zA-Z0-9_]", "");
    }

    // --- Example usage when run as main ---

    public static void main(String[] args) {
        String uri = System.getenv("NEO4J_URI");
        if (uri == null || uri.isBlank()) uri = "neo4j://localhost:7687";
        String user = System.getenv("NEO4J_USER");
        if (user == null || user.isBlank()) user = "neo4j";
        String password = System.getenv("NEO4J_PASSWORD");
        if (password == null || password.isBlank()) password = "neo4j"; // change after first login

        try (Neo4jWriter writer = new Neo4jWriter(uri, user, password)) {
            // Example: create two nodes and a relationship
            writer.createNode("Concept", "name", "Sovereignty");
            writer.createNode("Concept", "name", "Constitution");
            writer.createRelationship("Concept", "name", "Sovereignty", "RELATES_TO", "Concept", "name", "Constitution");

            System.out.println("Done. Check Neo4j Browser (localhost:7474) with: MATCH (n) RETURN n");
        }
    }
}
