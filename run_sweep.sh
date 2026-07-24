#!/usr/bin/env bash
set -euo pipefail

DIMS=(32 64 128 256 512 768)
MODELS=(default optuna)

for m  in "${MODELS[@]}"; do
    for d in "${DIMS[@]}"; do
        echo "==================== OUT_DIM=$d ===================="
        .venv/bin/python dim_reducer.py "$m" "$d"
    done
done
