#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

def load_cases(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--n-episodes", type=int, default=18)
    ap.add_argument("--episode-size", type=int, default=12)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--occupancy-label", type=str, default="high_occupancy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cases = load_cases(args.cases).copy()
    cases = cases.dropna(subset=["patient_id", "severity_proxy_score"]).copy()

    # Build severity bins
    cases["sev_bin"] = pd.qcut(
        cases["severity_proxy_score"],
        q=min(4, cases["severity_proxy_score"].nunique()),
        duplicates="drop"
    )

    # Shuffle within each bin
    bin_dict = {}
    for sev_bin, g in cases.groupby("sev_bin"):
        g = g.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        bin_dict[sev_bin] = g

    bin_keys = list(bin_dict.keys())

    used_patient_ids = set()
    episodes = []

    for ep_idx in range(args.n_episodes):
        selected = []

        attempts = 0
        max_attempts = 10000

        while len(selected) < args.episode_size and attempts < max_attempts:
            bin_key = bin_keys[len(selected) % len(bin_keys)]
            bin_df = bin_dict[bin_key]

            # only candidates not already used globally and not already in this episode
            available = bin_df[
                (~bin_df["patient_id"].isin(used_patient_ids)) &
                (~bin_df["patient_id"].isin([r["patient_id"] for r in selected]))
            ]

            if len(available) == 0:
                # try other bins
                attempts += 1
                # rotate through bins until something is found
                found = False
                for alt_key in bin_keys:
                    alt_df = bin_dict[alt_key]
                    alt_available = alt_df[
                        (~alt_df["patient_id"].isin(used_patient_ids)) &
                        (~alt_df["patient_id"].isin([r["patient_id"] for r in selected]))
                    ]
                    if len(alt_available) > 0:
                        row = alt_available.iloc[rng.integers(0, len(alt_available))].to_dict()
                        row.pop("sev_bin", None)
                        selected.append(row)
                        used_patient_ids.add(row["patient_id"])
                        found = True
                        break
                if not found:
                    break
            else:
                row = available.iloc[rng.integers(0, len(available))].to_dict()
                row.pop("sev_bin", None)
                selected.append(row)
                used_patient_ids.add(row["patient_id"])

            attempts += 1

        if len(selected) < args.episode_size:
            print(f"Stopped early at episode {ep_idx+1}: not enough unused patients left.")
            break

        # Shuffle final prompt order
        order = rng.permutation(len(selected))
        selected = [selected[i] for i in order]

        # Reference top-K by severity proxy
        ref_sorted = sorted(
            selected,
            key=lambda x: (-float(x["severity_proxy_score"]), float(x["acuity"]) if pd.notna(x["acuity"]) else 99)
        )
        reference_topk = [r["patient_id"] for r in ref_sorted[:args.capacity]]

        ep = {
            "episode_id": f"E{ep_idx+1:05d}",
            "capacity": args.capacity,
            "occupancy_label": args.occupancy_label,
            "reference_topk_by_proxy": reference_topk,
            "patients": selected,
        }
        episodes.append(ep)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    print(f"Saved {len(episodes)} episodes to {args.out_jsonl}")
    print(f"Unique patients used: {len(used_patient_ids)}")

if __name__ == "__main__":
    main()