COMPOSE = docker compose -f infra/graph/docker-compose.yml --env-file infra/graph/.env

.PHONY: graph-up graph-down graph-test-up graph-logs graph-status

graph-up:
	@if [ ! -f infra/graph/.env ]; then \
		echo "ERROR: infra/graph/.env not found. Copy infra/graph/.env.example and set NEO4J_PASSWORD."; \
		exit 1; \
	fi
	$(COMPOSE) up -d
	@echo "Neo4j browser:  http://localhost:7474"
	@echo "Bolt:           bolt://localhost:7687"

graph-down:
	$(COMPOSE) down

graph-test-up:
	$(COMPOSE) --profile test up -d neo4j-test
	@echo "Test instance bolt: bolt://localhost:7688"

graph-logs:
	$(COMPOSE) logs -f neo4j

graph-status:
	$(COMPOSE) ps
