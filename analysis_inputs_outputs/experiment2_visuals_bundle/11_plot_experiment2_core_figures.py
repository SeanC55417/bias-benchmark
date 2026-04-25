from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = Path("experiments/experiment2/analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATED = {
    "llama3.1:8b": "experiments/experiment2/validated/llama31_8b_persona18_validated.csv",
    "qwen2.5:7b": "experiments/experiment2/validated/qwen25_7b_persona18_validated.csv",
    "llama3.2:latest": "experiments/experiment2/validated/llama32_persona18_validated.csv",
}

DRIFT_FILE = "experiments/experiment2/analysis/experiment2_drift_episode_level.csv"
RANK_FILE = "experiments/experiment2/analysis/experiment2_rank_drift_episode_level.csv"
CLINICAL_FILE = "experiments/experiment2/analysis/experiment2_clinical_soundness_episode_level.csv"
FAIRNESS_GAP_FILE = "experiments/experiment2/analysis/experiment2_fairness_gap_summary.csv"


def save_bar(df, title, ylabel, filename):
    ax = df.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main():
    # Structural
    dfs = []
    for model, path in VALIDATED.items():
        df = pd.read_csv(path)
        df["model_name"] = model
        dfs.append(df)
    validated = pd.concat(dfs, ignore_index=True)
    structural = validated.groupby("model_name")[["valid_raw_ranked"]].mean()
    structural = structural.rename(columns={"valid_raw_ranked": "raw_valid_rate"})
    save_bar(structural, "Raw-valid ranking rate by model", "Rate", "fig01_raw_valid_rate_by_model.png")

    # Drift
    drift = pd.read_csv(DRIFT_FILE)
    drift_summary = drift.groupby(["model_name", "persona"])[["topk_overlap", "bounded_topk_drift"]].mean().reset_index()
    save_bar(
        drift_summary.pivot(index="model_name", columns="persona", values="topk_overlap"),
        "Top-K overlap by model and persona",
        "Top-K overlap",
        "fig02_topk_overlap_by_model_persona.png"
    )
    save_bar(
        drift_summary.pivot(index="model_name", columns="persona", values="bounded_topk_drift"),
        "Bounded drift by model and persona",
        "Bounded drift",
        "fig03_bounded_drift_by_model_persona.png"
    )

    # Rank drift
    rank = pd.read_csv(RANK_FILE)
    rank_summary = rank.groupby(["model_name", "persona"])[["kendall_tau"]].mean().reset_index()
    save_bar(
        rank_summary.pivot(index="model_name", columns="persona", values="kendall_tau"),
        "Kendall's tau by model and persona",
        "Kendall's tau",
        "fig04_kendall_tau_by_model_persona.png"
    )

    # Clinical
    clinical = pd.read_csv(CLINICAL_FILE)
    clinical_summary = clinical.groupby(["model_name", "persona"])[["high_acuity_miss_rate", "boundary_violation_acuity"]].mean().reset_index()
    save_bar(
        clinical_summary.pivot(index="model_name", columns="persona", values="high_acuity_miss_rate"),
        "High-acuity miss rate by model and persona",
        "High-acuity miss rate",
        "fig05_high_acuity_miss_by_model_persona.png"
    )
    save_bar(
        clinical_summary.pivot(index="model_name", columns="persona", values="boundary_violation_acuity"),
        "Acuity boundary-violation rate by model and persona",
        "Boundary-violation rate",
        "fig06_boundary_violation_acuity_by_model_persona.png"
    )

    # Fairness
    fairness = pd.read_csv(FAIRNESS_GAP_FILE)
    fairness = fairness[fairness["persona"] != "none"].copy()
    save_bar(
        fairness.groupby(["model_name", "persona"])[["admission_rate_gap"]].mean().reset_index().pivot(index="model_name", columns="persona", values="admission_rate_gap"),
        "Fairness admission-gap summary by model and persona",
        "Admission-rate gap",
        "fig07_fairness_admission_gap_by_model_persona.png"
    )
    save_bar(
        fairness.groupby(["model_name", "persona"])[["high_acuity_miss_gap"]].mean().reset_index().pivot(index="model_name", columns="persona", values="high_acuity_miss_gap"),
        "Fairness high-acuity miss-gap summary by model and persona",
        "High-acuity miss gap",
        "fig08_fairness_high_acuity_gap_by_model_persona.png"
    )


if __name__ == "__main__":
    main()