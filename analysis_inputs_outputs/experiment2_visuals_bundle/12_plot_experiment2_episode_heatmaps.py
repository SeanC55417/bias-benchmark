from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = Path("experiments/experiment2/analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_FILE = "experiments/experiment2/analysis/experiment2_drift_episode_level.csv"
RANK_FILE = "experiments/experiment2/analysis/experiment2_rank_drift_episode_level.csv"


def save_heatmap(pivot_df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pivot_df.values, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_ylabel("Episode")
    ax.set_xlabel("Persona")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main():
    drift = pd.read_csv(DRIFT_FILE)
    drift = drift[drift["persona"] != "none"].copy()
    for model in sorted(drift["model_name"].unique()):
        sub = drift[drift["model_name"] == model].sort_values(["episode_id", "persona"])
        pivot = sub.pivot(index="episode_id", columns="persona", values="bounded_topk_drift")
        save_heatmap(pivot, f"Bounded drift by episode and persona: {model}", f"heatmap_bounded_drift_{model.replace(':','_')}.png")

    rank = pd.read_csv(RANK_FILE)
    rank = rank[rank["persona"] != "none"].copy()
    for model in sorted(rank["model_name"].unique()):
        sub = rank[rank["model_name"] == model].sort_values(["episode_id", "persona"])
        pivot = sub.pivot(index="episode_id", columns="persona", values="kendall_tau")
        save_heatmap(pivot, f"Kendall's tau by episode and persona: {model}", f"heatmap_kendall_tau_{model.replace(':','_')}.png")


if __name__ == "__main__":
    main()