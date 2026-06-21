#!/bin/bash
# MLSS26_HACKATHON Main Launch Script
# Usage: bash scripts/run_hackathon.sh [agent_role] [task_name]

set -e

AGENT_ROLE=${1:-cv_expert}
TASK_NAME=${2:-identify-contrails}
DEVICE=${3:-0}

echo "====================================="
echo "MLSS26_HACKATHON Launcher"
echo "Agent: $AGENT_ROLE"
echo "Task: $TASK_NAME"
echo "Device: $DEVICE"
echo "====================================="

# Activate venv
source "$(dirname "$0")/../.venv/bin/activate"

# Load environment variables
if [ -f "$(dirname "$0")/../.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/../.env" | xargs)
fi

# Set defaults (override with .env or environment variables)
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-""}
export KAGGLE_CONFIG_DIR=${KAGGLE_CONFIG_DIR:-".kaggle"}

# Create log directory
LOG_DIR="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"
echo "Starting agent..."

# Run the agent
python -u -m MLAgentBench.runner \
    --task "$TASK_NAME" \
    --device "$DEVICE" \
    --log-dir "$LOG_DIR" \
    --work-dir workspace \
    --agent-role "$AGENT_ROLE" \
    --llm-name "google/gemma-4-26b-a4b-it:free" \
    --fast-llm-name "openai/gpt-oss-20b:free" \
    --edit-script-llm-name "qwen/qwen3-coder:free" \
    --agent-max-steps 50 \
    --max-time 18000 \
    > "$LOG_DIR/log" 2>&1

echo "Agent finished. Check logs in $LOG_DIR"
