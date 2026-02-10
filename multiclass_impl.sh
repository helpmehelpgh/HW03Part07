#!/bin/bash
set -e

keyword="hw02"
for i in 1 2 3 4 5; do
  uv run python scripts/multiclass_impl.py --keyword "${keyword}" --standardize --epoch 50 --eta 0.001
done

uv run python scripts/multiclass_eval.py --keyword "${keyword}"