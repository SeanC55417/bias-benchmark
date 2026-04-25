# Experiment 2 Visuals Bundle
## AdmitShift Task 1 on MIMIC-IV-ED

This bundle provides ready-to-run scripts for producing more useful figures from Experiment 2.

## What this bundle creates

### Core summary figures
1. Raw-valid ranking rate by model
2. Top-K overlap by model and persona
3. Bounded drift by model and persona
4. Kendall's tau by model and persona
5. High-acuity miss rate by model and persona
6. Acuity boundary-violation rate by model and persona
7. Fairness admission-gap summary by model and persona
8. Fairness high-acuity miss-gap summary by model and persona

### Episode-level visuals
9. Bounded-drift heatmap by episode and persona, one per model
10. Kendall-tau heatmap by episode and persona, one per model

## Expected inputs

These scripts expect the Experiment 2 files under:

experiments/experiment2/

and in particular these analysis outputs:
- experiments/experiment2/validated/llama31_8b_persona18_validated.csv
- experiments/experiment2/validated/qwen25_7b_persona18_validated.csv
- experiments/experiment2/validated/llama32_persona18_validated.csv
- experiments/experiment2/analysis/experiment2_drift_episode_level.csv
- experiments/experiment2/analysis/experiment2_rank_drift_episode_level.csv
- experiments/experiment2/analysis/experiment2_clinical_soundness_episode_level.csv
- experiments/experiment2/analysis/experiment2_fairness_gap_summary.csv

## How to use

Copy these files into your project under `scripts/`, activate your conda environment, then run:

```bash
bash scripts/run_experiment2_visuals.sh
```

All figures will be saved to:

```bash
experiments/experiment2/analysis/plots/
```

## Best figures for the paper

Keep:
- fig02_topk_overlap_by_model_persona.png
- fig04_kendall_tau_by_model_persona.png
- fig05_high_acuity_miss_by_model_persona.png
- fig07_fairness_admission_gap_by_model_persona.png
- fig08_fairness_high_acuity_gap_by_model_persona.png

## Best figures for slides

Use:
- fig01_raw_valid_rate_by_model.png
- fig03_bounded_drift_by_model_persona.png
- fig04_kendall_tau_by_model_persona.png
- one drift heatmap
- one fairness gap figure