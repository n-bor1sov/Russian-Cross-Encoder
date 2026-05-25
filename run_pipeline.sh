#!/usr/bin/env bash
# Full thesis pipeline: stages 1–6 plus evaluation.
set -euo pipefail
cd "$(dirname "$0")"

bash pipeline/01_split/run.sh
bash pipeline/02_embed/run.sh
bash pipeline/03_filter_consistency/run.sh
bash pipeline/04_hard_negative_mining/run.sh
bash pipeline/05_compile/run.sh
bash pipeline/06_train/run.sh
bash evaluation/run.sh
