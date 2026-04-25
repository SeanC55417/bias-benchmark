import json
import csv
import argparse
from pathlib import Path

def load_episodes(path):
    episodes = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            episodes[ep["episode_id"]] = ep
    return episodes

def repair_ranked_ids(parsed, episode):
    expected_ids = [p["patient_id"] for p in episode["patients"]]
    expected_set = set(expected_ids)

    ranked = parsed.get("ranked_patient_ids", []) or []
    admitted = parsed.get("admitted_patient_ids", []) or []
    capacity = int(parsed.get("capacity", episode["capacity"]))

    seen = set()
    ranked_unique = []
    duplicates = []
    unexpected = []

    for pid in ranked:
        if pid not in expected_set:
            unexpected.append(pid)
            continue
        if pid in seen:
            duplicates.append(pid)
            continue
        seen.add(pid)
        ranked_unique.append(pid)

    missing = [pid for pid in expected_ids if pid not in seen]
    repaired_ranked = ranked_unique + missing
    repaired_ranked = repaired_ranked[:len(expected_ids)]
    repaired_admitted = repaired_ranked[:capacity]

    valid_raw_ranked = (
        len(ranked) == len(expected_ids)
        and len(set(ranked)) == len(expected_ids)
        and set(ranked) == expected_set
    )

    valid_repaired_ranked = (
        len(repaired_ranked) == len(expected_ids)
        and len(set(repaired_ranked)) == len(expected_ids)
        and set(repaired_ranked) == expected_set
    )

    return {
        "expected_ids": expected_ids,
        "raw_ranked_ids": ranked,
        "raw_admitted_ids": admitted,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "missing": missing,
        "repaired_ranked_ids": repaired_ranked,
        "repaired_admitted_ids": repaired_admitted,
        "valid_raw_ranked": valid_raw_ranked,
        "valid_repaired_ranked": valid_repaired_ranked,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--raw-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)

    validated_records = []
    with open(args.raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ep = episodes[rec["episode_id"]]
            parsed = rec.get("parsed_json") or {}
            validation = repair_ranked_ids(parsed, ep)
            rec["validation"] = validation
            validated_records.append(rec)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for rec in validated_records:
            f.write(json.dumps(rec) + "\n")

    rows = []
    for rec in validated_records:
        v = rec["validation"]
        rows.append({
            "episode_id": rec.get("episode_id"),
            "backend": rec.get("backend"),
            "model": rec.get("model"),
            "persona": rec.get("persona"),
            "status": rec.get("status"),
            "parse_ok": int(bool(rec.get("parse_ok"))),
            "valid_raw_ranked": int(v["valid_raw_ranked"]),
            "valid_repaired_ranked": int(v["valid_repaired_ranked"]),
            "n_expected": len(v["expected_ids"]),
            "n_raw_ranked": len(v["raw_ranked_ids"]),
            "n_repaired_ranked": len(v["repaired_ranked_ids"]),
            "n_duplicates": len(v["duplicates"]),
            "n_missing": len(v["missing"]),
            "duplicates": json.dumps(v["duplicates"]),
            "missing": json.dumps(v["missing"]),
            "repaired_admitted_ids": json.dumps(v["repaired_admitted_ids"]),
            "repaired_ranked_ids": json.dumps(v["repaired_ranked_ids"]),
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "episode_id","backend","model","persona","status","parse_ok",
            "valid_raw_ranked","valid_repaired_ranked",
            "n_expected","n_raw_ranked","n_repaired_ranked",
            "n_duplicates","n_missing","duplicates","missing",
            "repaired_admitted_ids","repaired_ranked_ids"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_jsonl={args.out_jsonl}")
    print(f"saved_csv={args.out_csv}")
    print(f"n_records={len(validated_records)}")
    if rows:
        print("first_row_summary=")
        print(rows[0])

if __name__ == "__main__":
    main()
