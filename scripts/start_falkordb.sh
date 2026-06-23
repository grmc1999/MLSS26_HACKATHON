#!/bin/bash
# Start a local FalkorDB instance (knowledge graph backend for the Flu
# Literature Context RAG). Requires Docker.
# Usage: bash scripts/start_falkordb.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

FALKORDB_PORT="${FALKORDB_PORT:-6379}"
CONTAINER_NAME="mlss26-falkordb"

if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} already exists. Starting it..."
    docker start "${CONTAINER_NAME}"
else
    echo "Creating and starting ${CONTAINER_NAME} on port ${FALKORDB_PORT} (browser UI on 3001)..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${FALKORDB_PORT}:6379" \
        -p 3001:3001 \
        -v falkordb_data:/data \
        falkordb/falkordb:latest
fi

echo ""
echo "FalkorDB is running."
echo "  Graph (Redis protocol): localhost:${FALKORDB_PORT}"
echo "  Browser UI:             http://localhost:3001"
echo ""
echo "Stop with: docker stop ${CONTAINER_NAME}"
