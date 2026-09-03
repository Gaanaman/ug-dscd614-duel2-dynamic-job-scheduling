#!/usr/bin/env bash
# Single entry point. Reproduces the headline result from a clean environment.
#
#   bash scripts/run_all.sh
#
# Trains across all seeds, evaluates the agent and all three baselines through
# the same harness on the same held-out instances, and regenerates every figure
# in the report from the committed logs.
set -euo pipefail

SEEDS=(0 1 2)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs/train logs/eval figures models

echo "==> Environment check"
python -c "import gymnasium, torch, numpy; print(gymnasium.__version__, torch.__version__, numpy.__version__)"

echo "==> Load check: does the instance distribution have headroom?"
python scripts/check_load.py --config configs/env_default.yaml

echo "==> Training"
for seed in "${SEEDS[@]}"; do
  echo "--- seed $seed"
  python scripts/train.py --config configs/dueling_dqn.yaml --env-config configs/env_default.yaml --seed "$seed"
done

echo "==> Evaluation: agent, eight dispatching rules, required baselines"
python scripts/compare_all.py --out logs/eval/comparison.json

echo "==> Figures"
python scripts/make_figures.py --logs logs/ --out figures/

echo "==> Done. Figures in figures/, raw logs in logs/."
