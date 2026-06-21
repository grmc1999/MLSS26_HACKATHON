#!/bin/bash
# MLSS26_HACKATHON Full Setup Script
# For new team members

set -e

echo "====================================="
echo "MLSS26_HACKATHON Setup"
echo "====================================="

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m virtualenv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install MLAgentBench
echo "Installing MLAgentBench package..."
pip install -e .

# Install dependencies
echo "Installing core dependencies..."
pip install scikit-learn pandas matplotlib seaborn tqdm kaggle openai anthropic \
    transformers sentencepiece pydantic dacite pyyaml requests python-dotenv \
    h5py netcdf4 cdsapi fastapi uvicorn websockets sqlalchemy

# Set up Kaggle
echo "Setting up Kaggle API..."
mkdir -p .kaggle
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > .kaggle/kaggle.json
    chmod 600 .kaggle/kaggle.json
    echo "Kaggle API configured."
else
    echo "WARNING: KAGGLE_USERNAME and KAGGLE_KEY not set in .env"
fi

# Set up frontend
echo "Setting up dashboard frontend..."
cd dashboard/frontend
npm install
cd ../../

echo "====================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and fill in API keys"
echo "2. Run: bash scripts/run_hackathon.sh cv_expert identify-contrails"
echo "3. Start dashboard: cd dashboard/backend && uvicorn main:app --reload --port 8000"
echo "4. Start frontend: cd dashboard/frontend && npm run dev"
echo "====================================="
