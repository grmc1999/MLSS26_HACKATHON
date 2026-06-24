#!/bin/bash
# Start a local Ollama instance (free local LLM backend for the graph_summary
# narration in the Flu Literature Context RAG). Requires Docker.
# Usage: bash scripts/start_ollama.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
CONTAINER_NAME="mlss26-ollama"

if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} already exists. Starting it..."
    docker start "${CONTAINER_NAME}"
else
    echo "Creating and starting ${CONTAINER_NAME} on port ${OLLAMA_PORT}..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${OLLAMA_PORT}:11434" \
        -v ollama_data:/root/.ollama \
        ollama/ollama
fi

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -fs "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "Pulling model ${OLLAMA_MODEL} (skips if already present)..."
docker exec "${CONTAINER_NAME}" ollama pull "${OLLAMA_MODEL}"

echo ""
echo "Ollama is running."
echo "  API: http://localhost:${OLLAMA_PORT}"
echo "  Model: ${OLLAMA_MODEL}"
echo ""
echo "Stop with: docker stop ${CONTAINER_NAME}"
