package com.legal.rag.neo4j;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Session;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Builds a knowledge graph from the legal dictionary JSON file.
 * 
 * Graph structure:
 * - Nodes: Term (with properties: name, definition, page)
 * - Relationships: SEE, CF (compare), ALSO_TERMED
 */
public class DictionaryKnowledgeGraph implements AutoCloseable {

    private final Driver driver;
    private final Gson gson;

    public DictionaryKnowledgeGraph(String uri, String user, String password) {
        this.driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password));
        this.gson = new Gson();
    }

    @Override
    public void close() {
        driver.close();
    }

    /**
     * Clears all existing data from the database (optional - use with caution!)
     */
    public void clearDatabase() {
        try (Session session = driver.session()) {
            session.run("MATCH (n) DETACH DELETE n");
            System.out.println("Database cleared.");
        }
    }

    /**
     * Creates indexes for better query performance
     */
    public void createIndexes() {
        try (Session session = driver.session()) {
            // Create index on Term name for faster lookups
            session.run("CREATE INDEX term_name IF NOT EXISTS FOR (t:Term) ON (t.name)");
            System.out.println("Indexes created.");
        }
    }

    /**
     * Loads the dictionary JSON and creates the knowledge graph
     */
    public void buildGraphFromJson(String jsonFilePath) throws IOException {
        System.out.println("Reading JSON file: " + jsonFilePath);
        
        JsonObject root = gson.fromJson(new FileReader(jsonFilePath), JsonObject.class);
        JsonArray entries = root.getAsJsonArray("entries");
        
        int totalEntries = entries.size();
        System.out.println("Found " + totalEntries + " dictionary entries.");
        System.out.println("Creating nodes and relationships...");

        int nodeCount = 0;
        int seeCount = 0;
        int cfCount = 0;
        int alsoTermedCount = 0;

        try (Session session = driver.session()) {
            for (JsonElement entryElement : entries) {
                JsonObject entry = entryElement.getAsJsonObject();
                
                String term = entry.get("term").getAsString();
                String definition = entry.get("definition").getAsString();
                int page = entry.get("page").getAsInt();
                JsonObject metadata = entry.getAsJsonObject("metadata");

                // Create the term node
                Map<String, Object> props = new HashMap<>();
                props.put("name", term);
                props.put("definition", definition);
                props.put("page", page);
                
                session.run(
                    "MERGE (t:Term {name: $name}) " +
                    "SET t.definition = $definition, t.page = $page",
                    props
                );
                nodeCount++;

                // Create SEE relationships
                JsonArray seeArray = metadata.getAsJsonArray("see");
                for (JsonElement seeElement : seeArray) {
                    String targetTerm = seeElement.getAsString();
                    createRelationship(session, term, targetTerm, "SEE");
                    seeCount++;
                }

                // Create CF (compare) relationships
                JsonArray cfArray = metadata.getAsJsonArray("cf");
                for (JsonElement cfElement : cfArray) {
                    String targetTerm = cfElement.getAsString();
                    createRelationship(session, term, targetTerm, "CF");
                    cfCount++;
                }

                // Create ALSO_TERMED relationships
                JsonArray alsoTermedArray = metadata.getAsJsonArray("also_termed");
                for (JsonElement alsoElement : alsoTermedArray) {
                    String targetTerm = alsoElement.getAsString();
                    createRelationship(session, term, targetTerm, "ALSO_TERMED");
                    alsoTermedCount++;
                }

                // Progress indicator
                if (nodeCount % 50 == 0) {
                    System.out.printf("Processed %d/%d terms...%n", nodeCount, totalEntries);
                }
            }
        }

        System.out.println("\n=== Knowledge Graph Build Complete ===");
        System.out.println("Nodes created: " + nodeCount);
        System.out.println("SEE relationships: " + seeCount);
        System.out.println("CF relationships: " + cfCount);
        System.out.println("ALSO_TERMED relationships: " + alsoTermedCount);
        System.out.println("Total relationships: " + (seeCount + cfCount + alsoTermedCount));
    }

    /**
     * Creates a relationship between two terms
     */
    private void createRelationship(Session session, String fromTerm, String toTerm, String relType) {
        Map<String, Object> params = new HashMap<>();
        params.put("from", fromTerm);
        params.put("to", toTerm);
        
        session.run(
            String.format(
                "MERGE (a:Term {name: $from}) " +
                "MERGE (b:Term {name: $to}) " +
                "MERGE (a)-[:%s]->(b)",
                sanitizeRelType(relType)
            ),
            params
        );
    }

    /**
     * Sanitizes relationship type names
     */
    private String sanitizeRelType(String type) {
        return type.trim().toUpperCase().replaceAll("\\s+", "_");
    }

    /**
     * Prints some sample queries to explore the graph
     */
    public void printSampleQueries() {
        System.out.println("\n=== Sample Neo4j Browser Queries ===");
        System.out.println("Open Neo4j Browser at: http://localhost:7474");
        System.out.println();
        System.out.println("1. View all terms and relationships:");
        System.out.println("   MATCH (n:Term) RETURN n LIMIT 100");
        System.out.println();
        System.out.println("2. Find a specific term (e.g., 'abatement'):");
        System.out.println("   MATCH (t:Term {name: 'abatement'}) RETURN t");
        System.out.println();
        System.out.println("3. Find all terms related to 'abatement':");
        System.out.println("   MATCH (t:Term {name: 'abatement'})-[r]-(related) RETURN t, r, related");
        System.out.println();
        System.out.println("4. Find terms with SEE relationships:");
        System.out.println("   MATCH (a:Term)-[r:SEE]->(b:Term) RETURN a, r, b LIMIT 25");
        System.out.println();
        System.out.println("5. Find terms with ALSO_TERMED relationships:");
        System.out.println("   MATCH (a:Term)-[r:ALSO_TERMED]->(b:Term) RETURN a, r, b LIMIT 25");
        System.out.println();
        System.out.println("6. Count all relationships by type:");
        System.out.println("   MATCH ()-[r]->() RETURN type(r) as RelationType, count(r) as Count");
        System.out.println();
        System.out.println("7. Find terms with multiple relationships:");
        System.out.println("   MATCH (t:Term)-[r]-() WITH t, count(r) as rels WHERE rels > 2 RETURN t.name, rels ORDER BY rels DESC");
    }

    public static void main(String[] args) {
        // Configuration
        String uri = System.getenv().getOrDefault("NEO4J_URI", "neo4j://localhost:7687");
        String user = System.getenv().getOrDefault("NEO4J_USER", "neo4j");
        String password = System.getenv().getOrDefault("NEO4J_PASSWORD", "neo4j");
        
        // Path to your JSON file
        String jsonFile = "../backend/Const_IntepretationLaws/extracted_dictionary_pages_18_to_30.json";
        
        // Allow command line argument for JSON file path
        if (args.length > 0) {
            jsonFile = args[0];
        }

        System.out.println("=== Dictionary Knowledge Graph Builder ===");
        System.out.println("Neo4j URI: " + uri);
        System.out.println("JSON File: " + jsonFile);
        System.out.println();

        try (DictionaryKnowledgeGraph builder = new DictionaryKnowledgeGraph(uri, user, password)) {
            // Optional: Clear existing data (comment out if you want to keep existing data)
            // builder.clearDatabase();
            
            // Create indexes
            builder.createIndexes();
            
            // Build the graph
            builder.buildGraphFromJson(jsonFile);
            
            // Print sample queries
            builder.printSampleQueries();
            
        } catch (IOException e) {
            System.err.println("Error reading JSON file: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Error connecting to Neo4j: " + e.getMessage());
            e.printStackTrace();
            System.err.println("\nMake sure Neo4j is running and check your credentials.");
        }
    }
}
