#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/elli/europatipset-optimizer"
cd "$PROJECT_DIR"

source ".venv/bin/activate"
python europatipset.py sync-free-api-history --out data/raw/history_api.csv --days-back 120
