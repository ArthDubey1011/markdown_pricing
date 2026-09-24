#!/usr/bin/env bash
# Runs the full pipeline from the repository root.
set -euo pipefail
cd "$(dirname "$0")"
python src/01_build_tables.py
python src/02_analysis.py
python src/03_figures.py
echo "Done. Results in outputs/, report in report/."
