#!/usr/bin/env bash
set -euo pipefail

echo "Running Experiment 2 visuals..."
mkdir -p experiments/experiment2/analysis/plots

python scripts/11_plot_experiment2_core_figures.py
python scripts/12_plot_experiment2_episode_heatmaps.py

echo
echo "Done. Figures saved in:"
echo "experiments/experiment2/analysis/plots/"
