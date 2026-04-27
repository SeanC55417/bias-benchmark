import pandas as pd
import numpy as np

from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    statistical_parity_difference,
    equal_opportunity_difference,
    average_odds_difference,
    between_group_generalized_entropy_error,
    consistency_score,
)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("aif/mimic_cases (1).csv")

# ----------------------------
# Required columns
# ----------------------------
# Ground-truth binary outcome
y_true = df["observed_admit"]

# -------------------------------------------------
# OPTIONAL:
# Replace this with your model predictions if you have them
# For now, this uses observed_admit as a placeholder so the
# code runs, but fairness-on-predictions should use real preds.
# -------------------------------------------------
if "pred_admit" in df.columns:
    y_pred = df["pred_admit"]
else:
    y_pred = df["observed_admit"].copy()  # placeholder only

# ----------------------------
# Build binary protected groups
# unprivileged group is the group NOT equal to priv_group
# ----------------------------
comparisons = {
    "gender_F_vs_M": {
        "prot_attr": df["gender"],
        "priv_group": "M"
    },
    "age_65plus_vs_under65": {
        "prot_attr": (df["age_group"] == "65+").astype(int),
        "priv_group": 0
    },
    "white_vs_other": {
        "prot_attr": (df["race"] == "WHITE").astype(int),
        "priv_group": 1
    },
    "nonprivate_vs_private_insurance": {
        "prot_attr": df["coverage_proxy"].isin(["public", "uninsured"]).astype(int),
        "priv_group": 0
    },
    "unstable_vs_stable_housing": {
        "prot_attr": (df["housing_stability_proxy"] == "unstable").astype(int),
        "priv_group": 0
    },
    "caregiver_yes_vs_no": {
        "prot_attr": (df["caregiver_present"] == "yes").astype(int),
        "priv_group": 0
    },
}

# ----------------------------
# Group fairness metrics
# ----------------------------
results = []

for name, info in comparisons.items():
    prot = info["prot_attr"]
    priv_group = info["priv_group"]

    temp = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prot": prot
    }).dropna()

    # Make sure labels are ints if needed
    temp["y_true"] = temp["y_true"].astype(int)
    temp["y_pred"] = temp["y_pred"].astype(int)

    row = {
        "comparison": name,
        "n": len(temp),
    }

    try:
        row["disparate_impact_ratio"] = disparate_impact_ratio(
            y_true=temp["y_true"],
            y_pred=temp["y_pred"],
            prot_attr=temp["prot"],
            priv_group=priv_group,
            pos_label=1
        )
    except Exception as e:
        row["disparate_impact_ratio"] = f"ERROR: {e}"

    try:
        row["statistical_parity_difference"] = statistical_parity_difference(
            y_true=temp["y_true"],
            y_pred=temp["y_pred"],
            prot_attr=temp["prot"],
            priv_group=priv_group,
            pos_label=1
        )
    except Exception as e:
        row["statistical_parity_difference"] = f"ERROR: {e}"

    try:
        row["equal_opportunity_difference"] = equal_opportunity_difference(
            y_true=temp["y_true"],
            y_pred=temp["y_pred"],
            prot_attr=temp["prot"],
            priv_group=priv_group,
            pos_label=1
        )
    except Exception as e:
        row["equal_opportunity_difference"] = f"ERROR: {e}"

    try:
        row["average_odds_difference"] = average_odds_difference(
            y_true=temp["y_true"],
            y_pred=temp["y_pred"],
            prot_attr=temp["prot"],
            priv_group=priv_group,
            pos_label=1
        )
    except Exception as e:
        row["average_odds_difference"] = f"ERROR: {e}"

    try:
        row["between_group_generalized_entropy_error"] = (
            between_group_generalized_entropy_error(
                y_true=temp["y_true"],
                y_pred=temp["y_pred"],
                prot_attr=temp["prot"]
            )
        )
    except Exception as e:
        row["between_group_generalized_entropy_error"] = f"ERROR: {e}"

    results.append(row)

results_df = pd.DataFrame(results)
print("\n=== GROUP FAIRNESS METRICS ===")
print(results_df.to_string(index=False))

# ----------------------------
# Individual fairness: consistency score
# ----------------------------
# consistency_score needs feature columns X and predictions y_pred
# You should only include columns that are meaningful model inputs.
# Remove labels / obviously post-outcome columns.
# Also convert categoricals to numeric.
# ----------------------------

feature_cols = [
    "anchor_age",
    "gender",
    "race",
    "coverage_proxy",
    "housing_stability_proxy",
    "caregiver_present",
    "n_ed_visits_30d",
    "n_hosp_discharges_365d",
    "triage_temp",
    "triage_heartrate",
    "triage_resprate",
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    "pain_score",
    "chief_complaint_category",
    "acuity",
    "arrival_time_bucket",
    "severity_proxy_score",
    "worst_o2sat",
    "worst_sbp",
    "max_heartrate",
    "abnormal_labs_count",
    "ed_los_hours"
]

existing_feature_cols = [c for c in feature_cols if c in df.columns]

X = df[existing_feature_cols].copy()
y_pred_consistency = y_pred.copy()

# Drop rows with missing y_pred
valid_idx = y_pred_consistency.dropna().index
X = X.loc[valid_idx]
y_pred_consistency = y_pred_consistency.loc[valid_idx]

# One-hot encode categoricals
X_encoded = pd.get_dummies(X, drop_first=False)

# Fill any missing values after encoding
X_encoded = X_encoded.fillna(0)

try:
    cons = consistency_score(
        X_encoded.to_numpy(),
        y_pred_consistency.astype(int).to_numpy(),
        n_neighbors=5
    )
    print("\n=== INDIVIDUAL FAIRNESS ===")
    print(f"consistency_score: {cons}")
except Exception as e:
    print("\n=== INDIVIDUAL FAIRNESS ===")
    print(f"consistency_score ERROR: {e}")