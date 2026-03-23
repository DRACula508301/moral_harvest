#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

for reward_type in utilitarian virtue; do
  echo "Running 300-iteration baseline for reward_type=${reward_type}"
  .venv-linux/bin/python -m moral_harvest.cli.train \
    --mode multi-agent-reward-shaped \
    --reward-type "${reward_type}" \
    # --shaping-begin 1600000 \
    # --rew-shaping-horizon 2400000 \
    --stop-iters 300
  echo "Completed reward_type=${reward_type}"
  echo
 done
