# Experiment 2 README
## AdmitShift Task 1 on MIMIC-IV-ED

This README documents the **official Experiment ** setup for **Task 1: capacity-constrained admission selection on MIMIC-IV-ED**.

It is meant for collaborators who are receiving the code and folder directly and need:
- the logic of the experiment
- the exact file structure
- the order of scripts to run
- the outputs to expect
- the meaning of each analysis script
- the main troubleshooting notes

---

## 1. What Experiment 2 is

Experiment 2 is the **cleaner, primary analysis set** for Task 1.

### Task 1
For each episode, the model sees:
- **12 patient summaries**
- **4 available beds**

and must:
- choose exactly **4 patients for admission**
- rank all **12 patients** from highest to lowest admission priority
- return machine-readable JSON outputs

### Why Experiment 2 exists
Experiment 1 established the pipeline, but it had an important limitation:
- patients were reused across episodes

Experiment 2 fixes that by using:
- **18 disjoint episodes**
- **12 unique patients per episode**
- **0 patient reuse across episodes**

This makes Experiment 2 easier to defend and better suited for the main paper results.

---

## 2. Core benchmark logic

For each episode:
- the **same 12 patients**
- the **same 4-bed capacity**
- the **same model**

are evaluated under **4 prompt conditions**:

1. `none` (baseline / no persona)
2. `social_worker`
3. `revenue`
4. `psychiatry`

The benchmark question is:

> When the patient cases and bed limit stay fixed, how much does the model's admission set or ranking change when only the persona prompt changes?

---

## 3. Experiment 2 size

### Per model
- 18 episodes
- 4 prompt conditions

So:

`18 × 4 = 72 runs per model`

### Across all 3 models
- `llama3.1:8b`
- `qwen2.5:7b`
- `llama3.2:latest`

So:

`18 × 4 × 3 = 216 total runs`

---

## 4. Folder structure

Recommended folder structure:

```bash
experiments/
└── experiment2/
    ├── frozen_inputs/
    ├── scripts_snapshot/
    ├── raw_outputs/
    ├── flat_outputs/
    ├── validated/
    ├── analysis/
    └── logs/
```

### What each folder contains

- `frozen_inputs/`
  - disjoint episode file
  - prompt files
  - manifest for Experiment 2

- `scripts_snapshot/`
  - snapshot of the scripts used to produce Experiment 2

- `raw_outputs/`
  - raw JSONL model outputs from `03_run_task1.py`

- `flat_outputs/`
  - flattened CSV outputs from `03_run_task1.py`

- `validated/`
  - repaired / validated JSONL and CSV outputs from `05_validate_and_repair_run.py`

- `analysis/`
  - structural summaries
  - drift outputs
  - rank drift outputs
  - clinical soundness outputs
  - fairness subgroup outputs
  - plots

- `logs/`
  - shell logs
  - Ollama server logs
  - validation logs

---

## 5. Scripts used in Experiment 2

### Data / construction scripts
- `scripts/01_prepare_mimic_task1.py`
  - builds the case table from MIMIC-IV-ED inputs

- `scripts/02_build_task1_episodes.py`
  - builds benchmark episodes from the case table
  - for Experiment 2 this was used to produce an episode file

### Inference and validation
- `scripts/03_run_task1.py`
  - runs a model on episodes under one or more persona conditions
  - writes raw JSONL and flat CSV outputs

- `scripts/05_validate_and_repair_run.py`
  - validates raw rankings
  - detects duplicates and missing IDs
  - produces repaired rankings and repaired admitted sets

### Analysis scripts
- `scripts/06_experiment2_drift.py`
  - computes:
    - top-K overlap
    - bounded top-K drift
    - Jaccard drift

- `scripts/07_experiment2_rank_drift.py`
  - computes:
    - Kendall's tau between baseline and persona rankings

- `scripts/08_experiment2_clinical_soundness.py`
  - computes:
    - high-acuity miss rate
    - mean admitted severity proxy
    - mean admitted acuity
    - boundary violation by severity proxy
    - boundary violation by acuity

- `scripts/09_experiment2_fairness_slices.py`
  - computes subgroup-level fairness-oriented sensitivity analyses for:
    - caregiver_present
    - coverage_proxy
    - housing_stability_proxy

- `scripts/10_plot_experiment2_fairness.py`
  - produces fairness plots from the fairness-slice outputs

---

## 6. Inputs required

### MIMIC-IV-ED source tables
The Task 1 case table is built from:
- `edstays.csv`
- `triage.csv`
- `vitalsign.csv`

### Main Experiment 2 frozen input
The key file is:

`experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl`

This is the official Experiment 2 episode file.

---

## 7. Prompting setup

All models are run under:
- a shared **acuity-first system prompt**
- one persona condition per run

### Persona conditions
- `none`
- `social_worker`
- `revenue`
- `psychiatry`

### Important rule
For a given episode:
- the 12 patients do not change
- the capacity does not change
- only the persona prompt changes

This is what allows us to interpret any output shift as **persona-induced drift**.

---

## 8. Exact environment setup

Activate the conda environment in every fresh shell:

```bash
cd ~/EvalTask1/admitshift_task1_mimic_unity/admitshift_task1_mimic_unity

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/conda-envs/admitshift
```

Optional sanity checks:

```bash
which python
python -V
python -c "import pandas; print(pandas.__version__)"
```

---

## 9. Starting Ollama

Before any Experiment 2 inference runs, start the Ollama server:

```bash
export PATH="$(dirname "$(find "$HOME/.local/ollama-root" -type f -name ollama 2>/dev/null | head -n 1)"):$PATH"
hash -r

pkill -f "ollama serve" || true
sleep 2

mkdir -p experiments/experiment2/logs
nohup ollama serve > experiments/experiment2/logs/ollama_server_experiment2.log 2>&1 &
sleep 5

ps -ef | grep "[o]llama serve"
curl -s http://127.0.0.1:11434/api/tags
```

### Models used
- `llama3.1:8b`
- `qwen2.5:7b`
- `llama3.2:latest`

If they are not already pulled:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:latest
```

---

## 10. Quick smoke test before the full run

Always do one quick smoke test first:

```bash
python scripts/03_run_task1.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --backend ollama   --model llama3.1:8b   --persona-set none   --limit 1   --out-jsonl experiments/experiment2/raw_outputs/smoke_llama31_exp2_raw.jsonl   --out-csv experiments/experiment2/flat_outputs/smoke_llama31_exp2_flat.csv
```

Inspect it:

```bash
python - <<'PY'
import json
with open("experiments/experiment2/raw_outputs/smoke_llama31_exp2_raw.jsonl","r",encoding="utf-8") as f:
    rec = json.loads(next(f))
print("status:", rec.get("status"))
print("parsed_json:", rec.get("parsed_json"))
print("response_text:")
print(rec.get("response_text"))
PY
```

If `status: ok` and `parsed_json` is populated, proceed.

---

## 11. Full Experiment 2 run order

### 11.1 Run Llama 3.1 8B

```bash
python scripts/03_run_task1.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --backend ollama   --model llama3.1:8b   --persona-set default   --limit 18   --out-jsonl experiments/experiment2/raw_outputs/llama31_8b_persona18_raw.jsonl   --out-csv experiments/experiment2/flat_outputs/llama31_8b_persona18_flat.csv   | tee experiments/experiment2/logs/llama31_8b_run.log
```

Validate:

```bash
python scripts/05_validate_and_repair_run.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --raw-jsonl experiments/experiment2/raw_outputs/llama31_8b_persona18_raw.jsonl   --out-jsonl experiments/experiment2/validated/llama31_8b_persona18_validated.jsonl   --out-csv experiments/experiment2/validated/llama31_8b_persona18_validated.csv   | tee experiments/experiment2/logs/llama31_8b_validate.log
```

---

### 11.2 Run Qwen 2.5 7B

```bash
python scripts/03_run_task1.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --backend ollama   --model qwen2.5:7b   --persona-set default   --limit 18   --out-jsonl experiments/experiment2/raw_outputs/qwen25_7b_persona18_raw.jsonl   --out-csv experiments/experiment2/flat_outputs/qwen25_7b_persona18_flat.csv   | tee experiments/experiment2/logs/qwen25_7b_run.log
```

Validate:

```bash
python scripts/05_validate_and_repair_run.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --raw-jsonl experiments/experiment2/raw_outputs/qwen25_7b_persona18_raw.jsonl   --out-jsonl experiments/experiment2/validated/qwen25_7b_persona18_validated.jsonl   --out-csv experiments/experiment2/validated/qwen25_7b_persona18_validated.csv   | tee experiments/experiment2/logs/qwen25_7b_validate.log
```

---

### 11.3 Run Llama 3.2

```bash
python scripts/03_run_task1.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --backend ollama   --model llama3.2:latest   --persona-set default   --limit 18   --out-jsonl experiments/experiment2/raw_outputs/llama32_persona18_raw.jsonl   --out-csv experiments/experiment2/flat_outputs/llama32_persona18_flat.csv   | tee experiments/experiment2/logs/llama32_run.log
```

Validate:

```bash
python scripts/05_validate_and_repair_run.py   --episodes experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl   --raw-jsonl experiments/experiment2/raw_outputs/llama32_persona18_raw.jsonl   --out-jsonl experiments/experiment2/validated/llama32_persona18_validated.jsonl   --out-csv experiments/experiment2/validated/llama32_persona18_validated.csv   | tee experiments/experiment2/logs/llama32_validate.log
```

---

## 12. Analysis scripts: what to run next

### Structural reliability summary

```bash
python - <<'PY' | tee experiments/experiment2/analysis/structural_reliability_summary.txt
import pandas as pd

files = {
    "llama3.1:8b": "experiments/experiment2/validated/llama31_8b_persona18_validated.csv",
    "qwen2.5:7b": "experiments/experiment2/validated/qwen25_7b_persona18_validated.csv",
    "llama3.2:latest": "experiments/experiment2/validated/llama32_persona18_validated.csv",
}

dfs = []
for model, path in files.items():
    df = pd.read_csv(path)
    df["model_name"] = model
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

print("BY MODEL AND PERSONA")
print(
    all_df.groupby(["model_name","persona"])[
        ["valid_raw_ranked","valid_repaired_ranked","n_duplicates","n_missing"]
    ].sum()
)

print("\nBY MODEL")
print(
    all_df.groupby("model_name")[
        ["valid_raw_ranked","valid_repaired_ranked","n_duplicates","n_missing"]
    ].sum()
)

print("\nN_ROWS:", len(all_df))
PY
```

---

### Decision drift

```bash
python scripts/06_experiment2_drift.py | tee experiments/experiment2/analysis/experiment2_drift_summary.txt
```

Produces:
- top-K overlap
- bounded top-K drift
- Jaccard drift

---

### Rank drift

```bash
python scripts/07_experiment2_rank_drift.py | tee experiments/experiment2/analysis/experiment2_rank_drift_summary.txt
```

Produces:
- Kendall's tau

---

### Clinical soundness

```bash
python scripts/08_experiment2_clinical_soundness.py | tee experiments/experiment2/analysis/experiment2_clinical_soundness_summary.txt
```

Produces:
- high-acuity miss rate
- mean admitted severity proxy
- mean admitted acuity
- boundary violation by severity proxy
- boundary violation by acuity

---

### Fairness-oriented subgroup slices

```bash
python scripts/09_experiment2_fairness_slices.py | tee experiments/experiment2/analysis/experiment2_fairness_slices_summary.txt
```

Produces:
- subgroup admission-rate deltas
- subgroup high-acuity miss-rate deltas
- gap summaries

---

### Fairness plots

```bash
python scripts/10_plot_experiment2_fairness.py
```

Produces fairness figures in:
- `experiments/experiment2/analysis/plots/`

---

## 13. Main metrics reported in Experiment 2

### Structural reliability
- raw-valid ranking rate
- duplicate IDs
- missing IDs

### Decision drift
- top-K overlap
- bounded top-K drift
- Jaccard drift

### Rank drift
- Kendall's tau

### Clinical soundness
- high-acuity miss rate
- mean admitted severity proxy
- mean admitted acuity
- boundary violation by severity proxy
- boundary violation by acuity

### Fairness-oriented subgroup sensitivity
- subgroup admission-rate delta
- subgroup high-acuity miss-rate delta
- subgroup gap summaries

---

## 14. Main Experiment 2 findings

### Structural reliability
- **Llama 3.1:8b** was perfect on raw-valid rankings in Experiment 2
- **Qwen2.5:7b** and **Llama3.2** required repair more often

### Decision drift
- Llama 3.1 had the highest top-K overlap and lowest drift
- Qwen was close behind
- Llama 3.2 drifted the most

### Rank drift
- Llama 3.1 had the highest Kendall's tau
- Qwen was second
- Llama 3.2 was lowest

### Clinical soundness
- Llama 3.1 was best on acuity-based soundness
- Qwen was strongest on the severity-proxy criterion
- Llama 3.2 was weakest overall

### Fairness-oriented subgroup sensitivity
- some persona conditions widened subgroup gaps
- the largest subgroup-gap increases appeared in weaker or more persona-sensitive settings, especially Llama 3.2

---

## 15. Important interpretation notes

### Baseline
Baseline means:
- same model
- same episode
- same 12 patients
- same 4-bed capacity
- **no persona**

### Drift
Drift is always measured relative to the same model's own baseline on the same episode.

### Experiment 2 is the main analysis set
Experiment 1 is the pilot.
Experiment 2 is the cleaner version:
- disjoint episodes
- no patient reuse across episodes
- use these results in the main paper text

---

## 16. Troubleshooting

### Problem: `ConnectionError` / `RetryError`
This usually means the Ollama server is not running or not reachable.

Fix:
- restart `ollama serve`
- verify with:

```bash
ps -ef | grep "[o]llama serve"
curl -s http://127.0.0.1:11434/api/tags
```

### Problem: `ModuleNotFoundError: No module named 'pandas'`
This usually means the conda environment is not active in that shell.

Fix:
```bash
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/conda-envs/admitshift
```

### Problem: all validated outputs show missing IDs everywhere
This usually means the raw runs failed and the validator repaired empty outputs.
Check one raw JSONL record before trusting any summary.

---

## 17. What to send to collaborators

Recommended handoff package:
- the full `experiments/experiment2/` folder
- the `scripts/` folder
- the `prompts/` folder
- this README
- the MIMIC-derived case table or instructions to rebuild it
- the frozen disjoint episode file

The most important files for reproduction are:
- `experiments/experiment2/frozen_inputs/task1_episodes_disjoint.jsonl`
- `scripts/03_run_task1.py`
- `scripts/05_validate_and_repair_run.py`
- `scripts/06_experiment2_drift.py`
- `scripts/07_experiment2_rank_drift.py`
- `scripts/08_experiment2_clinical_soundness.py`
- `scripts/09_experiment2_fairness_slices.py`
- `scripts/10_plot_experiment2_fairness.py`

---

## 18. Short summary for GitHub / review handoff

Experiment 2 is the official Task 1 benchmark evaluation on MIMIC-IV-ED using 18 disjoint episodes of 12 patients each with a fixed capacity of 4 beds. Each episode is evaluated under a no-persona baseline and three persona conditions. The main outputs are structural reliability, decision drift, rank drift, clinical soundness, and fairness-oriented subgroup sensitivity.
