#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ensure_dir,
    severity_from_acuity,
    trend_score,
    vitals_abnormality_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ed-dir", required=True, help="Directory containing edstays, triage, vitalsign CSVs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--decision-window-hours", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv_auto(path: Path, usecols=None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, usecols=usecols)
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return pd.read_csv(gz, usecols=usecols)
    raise FileNotFoundError(path)


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("unknown")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", "unknown")
    )


def age_group_from_subject(subject_id: pd.Series) -> pd.Series:
    # Placeholder age grouping when patient age is not linked from hosp/patients.
    # This is deterministic and ONLY a temporary field for prompt consistency.
    # Replace with real age from linked MIMIC-IV patients if you add that table later.
    bins = subject_id.astype(str).str[-2:].astype(int, errors="ignore")
    grp = pd.Series(index=subject_id.index, dtype="object")
    grp[(bins % 3) == 0] = "18-39"
    grp[(bins % 3) == 1] = "40-64"
    grp[(bins % 3) == 2] = "65+"
    return grp.fillna("unknown")


def synth_context(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    probs = np.where(out["age_group"].eq("65+"), 0.45, np.where(out["age_group"].eq("40-64"), 0.22, 0.18))
    out["caregiver_present"] = np.where(rng.random(len(out)) < probs, "yes", "no")

    coverage = rng.choice(["private", "public", "uninsured"], size=len(out), p=[0.45, 0.45, 0.10])
    out["coverage_proxy"] = coverage

    housing = rng.choice(["stable", "unstable"], size=len(out), p=[0.82, 0.18])
    out["housing_stability_proxy"] = housing
    return out


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ed_dir = Path(args.ed_dir)

    edstays = read_csv_auto(
        ed_dir / "edstays.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "gender",
            "race",
            "arrival_transport",
            "disposition",
        ],
    )
    triage = read_csv_auto(
        ed_dir / "triage.csv",
        usecols=[
            "subject_id",
            "stay_id",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "pain",
            "acuity",
            "chiefcomplaint",
        ],
    )
    vitals = read_csv_auto(
        ed_dir / "vitalsign.csv",
        usecols=[
            "subject_id",
            "stay_id",
            "charttime",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "rhythm",
            "pain",
        ],
    )

    edstays["intime"] = pd.to_datetime(edstays["intime"], errors="coerce")
    edstays["outtime"] = pd.to_datetime(edstays["outtime"], errors="coerce")
    vitals["charttime"] = pd.to_datetime(vitals["charttime"], errors="coerce")

    triage["chiefcomplaint_clean"] = normalize_text(triage["chiefcomplaint"])
    edstays["gender"] = normalize_text(edstays["gender"])
    edstays["race"] = normalize_text(edstays["race"])
    edstays["arrival_transport"] = normalize_text(edstays["arrival_transport"])
    edstays["disposition"] = normalize_text(edstays["disposition"])

    vitals = vitals.merge(edstays[["stay_id", "intime"]], on="stay_id", how="left")
    window_hours = float(args.decision_window_hours)
    vitals["hours_from_intime"] = (vitals["charttime"] - vitals["intime"]).dt.total_seconds() / 3600.0
    vitals = vitals[(vitals["hours_from_intime"] >= 0) & (vitals["hours_from_intime"] <= window_hours)].copy()
    vitals.sort_values(["stay_id", "charttime"], inplace=True)

    def agg_stay(group: pd.DataFrame) -> pd.Series:
        first = group.iloc[0]
        last = group.iloc[-1]
        return pd.Series(
            {
                "first_temperature": first.get("temperature"),
                "first_heartrate": first.get("heartrate"),
                "first_resprate": first.get("resprate"),
                "first_o2sat": first.get("o2sat"),
                "first_sbp": first.get("sbp"),
                "last_temperature": last.get("temperature"),
                "last_heartrate": last.get("heartrate"),
                "last_resprate": last.get("resprate"),
                "last_o2sat": last.get("o2sat"),
                "last_sbp": last.get("sbp"),
                "worst_o2sat": pd.to_numeric(group["o2sat"], errors="coerce").min(),
                "worst_sbp": pd.to_numeric(group["sbp"], errors="coerce").min(),
                "max_heartrate": pd.to_numeric(group["heartrate"], errors="coerce").max(),
                "n_vitals_in_window": len(group),
                "decision_time": last.get("charttime"),
            }
        )

    vitals_agg = vitals.groupby("stay_id", as_index=False).apply(agg_stay).reset_index(drop=True)

    cases = edstays.merge(triage.drop(columns=["subject_id"]), on="stay_id", how="inner")
    cases = cases.merge(vitals_agg, on="stay_id", how="left")

    cases["patient_id"] = [f"P{i:06d}" for i in range(1, len(cases) + 1)]
    cases["age_group"] = age_group_from_subject(cases["subject_id"])

    cases["wait_minutes"] = ((cases["decision_time"] - cases["intime"]).dt.total_seconds() / 60.0).round(1)
    cases["wait_minutes"] = cases["wait_minutes"].fillna(args.decision_window_hours * 60.0)

    cases["severity_proxy_score"] = (
        cases["acuity"].map(severity_from_acuity)
        + cases.apply(
            lambda r: vitals_abnormality_score(
                r.get("first_heartrate"),
                r.get("first_resprate"),
                r.get("first_o2sat"),
                r.get("first_sbp"),
                r.get("first_temperature"),
            ),
            axis=1,
        )
        + cases.apply(
            lambda r: trend_score(
                r.get("first_heartrate"),
                r.get("last_heartrate"),
                r.get("first_o2sat"),
                r.get("last_o2sat"),
                r.get("first_sbp"),
                r.get("last_sbp"),
            ),
            axis=1,
        )
    )

    cases["observed_admit"] = cases["disposition"].str.contains("admit", case=False, na=False).astype(int)
    cases = synth_context(cases, seed=args.seed)

    keep_cols = [
        "patient_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "decision_time",
        "gender",
        "race",
        "arrival_transport",
        "disposition",
        "observed_admit",
        "age_group",
        "acuity",
        "chiefcomplaint_clean",
        "pain",
        "first_temperature",
        "first_heartrate",
        "first_resprate",
        "first_o2sat",
        "first_sbp",
        "last_temperature",
        "last_heartrate",
        "last_resprate",
        "last_o2sat",
        "last_sbp",
        "worst_o2sat",
        "worst_sbp",
        "max_heartrate",
        "n_vitals_in_window",
        "wait_minutes",
        "severity_proxy_score",
        "caregiver_present",
        "coverage_proxy",
        "housing_stability_proxy",
    ]
    cases = cases[keep_cols].copy()
    cases.sort_values(["severity_proxy_score", "wait_minutes"], ascending=[False, False], inplace=True)

    csv_path = out_dir / "mimic_cases.csv"
    pq_path = out_dir / "mimic_cases.parquet"
    cases.to_csv(csv_path, index=False)
    cases.to_parquet(pq_path, index=False)
    print(f"Saved {len(cases):,} cases to {csv_path} and {pq_path}")


if __name__ == "__main__":
    main()
