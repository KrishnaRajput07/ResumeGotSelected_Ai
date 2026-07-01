#!/bin/bash
set -e

echo "=========================================="
echo "Starting Sandbox LLM Environment (Ollama)"
echo "=========================================="
ollama serve &
sleep 5

# We use sample_candidates.json because Section 10.5 says:
# "Accept a small candidate sample (≤100 candidates) as input"
SAMPLE_FILE="India_runs_data_and_ai_challenge/sample_candidates.json"

echo ""
echo "=========================================="
echo "1. Running Pre-computation"
echo "=========================================="
# We run on the small sample. We limit LLM deep scoring to 10 to keep it very fast.
python precompute.py --candidates "$SAMPLE_FILE" --llm-top-n 10 --embedding-top-n 10 --force

echo ""
echo "=========================================="
echo "2. Running Rank.py (The 5-minute timed step)"
echo "=========================================="
python rank.py --candidates "$SAMPLE_FILE" --out submission_sandbox.csv --top-n 10 --no-validate

echo ""
echo "=========================================="
echo "Sandbox successfully produced submission_sandbox.csv"
echo "=========================================="
