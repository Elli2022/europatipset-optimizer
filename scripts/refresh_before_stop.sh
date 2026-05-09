#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/elli/europatipset-optimizer"
cd "$PROJECT_DIR"

source ".venv/bin/activate"
python europatipset.py sync-official-smart \
  --coupon-out data/input/official_coupon.csv \
  --meta-out data/input/official_meta.json \
  --min-interval-minutes 15
