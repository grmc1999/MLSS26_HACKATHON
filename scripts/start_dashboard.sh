#!/bin/bash
# Start the MLSS26_HACKATHON dashboard (backend + frontend)
# Usage: bash scripts/start_dashboard.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Add local Node.js to PATH if installed
export PATH="/home/exx/.local/node/bin:$PATH"

# Activate venv
source .venv/bin/activate

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Starting MLSS26_HACKATHON Dashboard..."
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""

# Start backend in background
echo "Starting FastAPI backend..."
uvicorn dashboard.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting Next.js frontend..."
cd dashboard/frontend
npm run dev &
FRONTEND_PID=$!

# Trap to kill both on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

echo ""
echo "Dashboard is running!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop."

wait
