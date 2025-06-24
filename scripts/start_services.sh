#!/bin/bash
# Start Redis and Neo4j using docker-compose
DIR="$(dirname "$0")/.."
cd "$DIR" || exit 1
docker compose up -d redis neo4j
