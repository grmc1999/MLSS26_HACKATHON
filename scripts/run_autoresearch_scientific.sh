#!/bin/bash
# ============================================================================
# Scientific AI AutoResearch Launcher
# Merges the autoresearch loop with 8 specialized scientific AI agents.
#
# Usage:
#   bash scripts/run_autoresearch_scientific.sh [agent_role] [iterations]
#
# Examples:
#   bash scripts/run_autoresearch_scientific.sh                    # default, 25 iters
#   bash scripts/run_autoresearch_scientific.sh cv_expert 10      # CV expert, 10 iters
#   bash scripts/run_autoresearch_scientific.sh medical_expert 5  # Medical expert
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

AGENT_ROLE="${1:-autoresearch}"
ITERATIONS="${2:-25}"
LOG_DIR="experiments/loop-$(date +%y%m%d-%H%M)"

# Source environment
source .venv/bin/activate
if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     Scientific AI AutoResearch                           ║"
echo "║     Agent: $AGENT_ROLE                                   "
echo "║     Iterations: $ITERATIONS                               "
echo "║     Log: $LOG_DIR                                        "
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

python -m MLAgentBench.agents.orchestrator \
    --agent "$AGENT_ROLE" \
    --iterations "$ITERATIONS" \
    --log-dir "$LOG_DIR" \
    --verify "python scripts/run_medmnist.py --epochs 50" 2>&1 | tee "$LOG_DIR/loop.log"

# Print summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Loop complete. Results in: $LOG_DIR"
echo "  View dashboard: cd dashboard/backend && uvicorn main:app --port 8000"
echo "═══════════════════════════════════════════════════════════════"
