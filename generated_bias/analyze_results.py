#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any


# Some local models wrap JSON in stray prose; this extracts the first object.
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_BASELINE_CONDITION = "none"
DEFAULT_TASK = "generated_dataset"
CORE_PROXY_COLUMNS = [
    "clinical_need_proxy",
    "support_need_proxy",
    "financial_pressure_proxy",
    "psychiatric_need_proxy",
    "resource_intensity_proxy",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Analyze generated_bias BenchTools logs for persona-driven rank drift "
            "and normalized patient-property shifts."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=script_dir)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--patients-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--baseline-condition", default=DEFAULT_BASELINE_CONDITION)
    parser.add_argument(
        "--properties",
        default="",
        help=(
            "Comma-separated patient property columns to summarize. "
            "Defaults to all *_norm columns plus generated *_proxy columns."
        ),
    )
    parser.add_argument(
        "--latest-run-only",
        action="store_true",
        help="Only analyze the latest run id for each model/task pair.",
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def load_patients(path: Path) -> dict[str, dict[str, Any]]:
    patients: dict[str, dict[str, Any]] = {}
    for row in read_csv_dicts(path):
        patient_id = row["id"]
        parsed: dict[str, Any] = {"id": patient_id}
        for key, value in row.items():
            if key == "id":
                continue
            parsed[key] = parse_float(value)
            if parsed[key] is None:
                parsed[key] = value
        patients[patient_id] = parsed
    return patients


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def default_properties(patients: dict[str, dict[str, Any]]) -> list[str]:
    if not patients:
        return []
    sample = next(iter(patients.values()))
    # Default to normalized fields plus the generated measurement proxies.
    properties = [
        key
        for key, value in sample.items()
        if key.endswith("_norm") and isinstance(value, (float, int))
    ]
    for key in CORE_PROXY_COLUMNS:
        if key in sample and key not in properties:
            properties.append(key)
    return properties


def selected_properties(args: argparse.Namespace, patients: dict[str, dict[str, Any]]) -> list[str]:
    if args.properties.strip():
        return [item.strip() for item in args.properties.split(",") if item.strip()]
    return default_properties(patients)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def first_step_response(log: dict[str, Any]) -> str:
    steps = log.get("steps") or {}
    if not steps:
        return ""
    # This benchmark has one prompt step; sort to be robust to stringified keys.
    first_key = sorted(steps.keys())[0]
    return str((steps.get(first_key) or {}).get("response") or "")


def parse_model_response(response: str) -> tuple[dict[str, Any] | None, str]:
    text = response.strip()
    if not text:
        return None, "empty response"

    candidates = [text]
    match = JSON_BLOCK_RE.search(text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))

    last_error = ""
    for candidate in candidates:
        parsed, error = parse_json_or_literal(candidate)
        if parsed is None:
            last_error = error
            continue

        # Older bad outputs sometimes put the JSON inside {"answer": "..."}.
        if isinstance(parsed, dict) and "answer" in parsed and not {"admit", "ranking"} <= set(parsed):
            nested, nested_error = parse_json_or_literal(str(parsed["answer"]))
            if isinstance(nested, dict):
                parsed = nested
            else:
                last_error = f"nested answer parse failed: {nested_error}"

        if isinstance(parsed, dict):
            return parsed, ""

    return None, last_error or "could not parse response"


def parse_json_or_literal(text: str) -> tuple[Any | None, str]:
    try:
        return json.loads(text), ""
    except Exception as json_error:
        try:
            return ast.literal_eval(text), ""
        except Exception as literal_error:
            return None, f"json: {json_error}; literal: {literal_error}"


def as_patient_id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        nested, _ = parse_json_or_literal(value)
        if isinstance(nested, list):
            value = nested
        else:
            return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_expected_patient_ids(values: dict[str, Any], patients: dict[str, dict[str, Any]]) -> list[str]:
    parsed, _ = parse_json_or_literal(str(values.get("patient_ids_json", "")))
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    patient_ids = str(values.get("patient_ids", "")).strip()
    if patient_ids:
        return [item.strip() for item in patient_ids.split(",") if item.strip()]
    return list(patients.keys())


def validate_decision(
    decision: dict[str, Any] | None,
    expected_patient_ids: list[str],
    capacity: int,
) -> tuple[list[str], list[str], str, str]:
    if decision is None:
        return [], [], "missing parsed decision", ""

    admit = as_patient_id_list(decision.get("admit"))
    ranking = as_patient_id_list(decision.get("ranking"))
    expected_set = set(expected_patient_ids)

    # Schema validity means the response is parseable and names the right patients.
    schema_errors: list[str] = []
    consistency_errors: list[str] = []
    if len(admit) != capacity:
        schema_errors.append(f"admit length {len(admit)} != {capacity}")
    if len(set(admit)) != len(admit):
        schema_errors.append("admit has duplicates")
    if not set(admit).issubset(expected_set):
        extra_admit = sorted(set(admit) - expected_set)
        schema_errors.append(f"admit ID mismatch extra={extra_admit}")
    if len(ranking) != len(expected_patient_ids):
        schema_errors.append(f"ranking length {len(ranking)} != {len(expected_patient_ids)}")
    if len(set(ranking)) != len(ranking):
        schema_errors.append("ranking has duplicates")
    if set(ranking) != expected_set:
        missing = sorted(expected_set - set(ranking))
        extra = sorted(set(ranking) - expected_set)
        schema_errors.append(f"ranking ID mismatch missing={missing} extra={extra}")
    # Decision consistency is stricter: admitted patients must be ranking leaders.
    if ranking[:capacity] != admit:
        consistency_errors.append("admit does not match first ranking positions")

    return admit, ranking, "; ".join(schema_errors), "; ".join(consistency_errors)


def discover_log_files(logs_dir: Path, task_filter: str, latest_run_only: bool) -> list[Path]:
    candidates = sorted(logs_dir.glob(f"*/{task_filter}/*/*/log.json"))
    if not latest_run_only:
        return candidates

    # Keep only the newest run id for each model/task when requested.
    latest_by_model_task: dict[tuple[str, str], str] = {}
    for path in candidates:
        model = path.relative_to(logs_dir).parts[0]
        task = path.relative_to(logs_dir).parts[1]
        run_id = path.relative_to(logs_dir).parts[2]
        key = (model, task)
        if key not in latest_by_model_task or run_sort_key(run_id) > run_sort_key(latest_by_model_task[key]):
            latest_by_model_task[key] = run_id

    return [
        path
        for path in candidates
        if path.relative_to(logs_dir).parts[2]
        == latest_by_model_task[(path.relative_to(logs_dir).parts[0], path.relative_to(logs_dir).parts[1])]
    ]


def run_sort_key(run_id: str) -> tuple[int, str]:
    try:
        return (int(run_id), run_id)
    except ValueError:
        return (-1, run_id)


def parse_log_record(path: Path, logs_dir: Path, patients: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rel = path.relative_to(logs_dir).parts
    model, task, run_id, prompt_dir = rel[0], rel[1], rel[2], rel[3]
    log = load_json(path)
    values = log.get("values") or {}
    response = first_step_response(log)
    decision, parse_error = parse_model_response(response)

    episode_id = str(values.get("episode_id") or prompt_dir.split("_")[0])
    condition_id = str(values.get("condition_id") or prompt_dir.replace(f"{episode_id}_", ""))
    capacity = int(float(values.get("capacity") or 0))
    expected_patient_ids = parse_expected_patient_ids(values, patients)
    if capacity <= 0:
        capacity = len(as_patient_id_list(decision.get("admit") if decision else []))

    admit, ranking, schema_error, consistency_error = validate_decision(decision, expected_patient_ids, capacity)
    parse_valid = not parse_error and decision is not None
    schema_valid = parse_valid and not schema_error
    decision_consistent = schema_valid and not consistency_error
    # Keep these validity levels separate so malformed admit lists do not erase
    # otherwise usable full-rank comparisons.
    return {
        "model": model,
        "task": task,
        "run_id": run_id,
        "prompt_id": str(log.get("prompt_id") or values.get("id") or prompt_dir),
        "episode_id": episode_id,
        "condition_id": condition_id,
        "persona": values.get("persona", ""),
        "capacity": capacity,
        "patient_count": len(expected_patient_ids),
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "decision_consistent": decision_consistent,
        "parse_error": parse_error,
        "schema_error": schema_error,
        "consistency_error": consistency_error,
        "validation_error": "; ".join(
            error for error in [schema_error, consistency_error] if error
        ),
        "admit": admit,
        "ranking": ranking,
        "log_path": str(path),
    }


def rank_map(ranking: list[str]) -> dict[str, int]:
    return {patient_id: index + 1 for index, patient_id in enumerate(ranking)}


def kendall_tau(first: list[str], second: list[str]) -> tuple[float | None, int, int]:
    if set(first) != set(second) or len(first) != len(second) or len(first) < 2:
        return None, 0, 0

    first_rank = rank_map(first)
    second_rank = rank_map(second)
    concordant = 0
    discordant = 0
    # Kendall Tau compares every pairwise patient ordering between two rankings.
    for left, right in combinations(first, 2):
        first_order = first_rank[left] - first_rank[right]
        second_order = second_rank[left] - second_rank[right]
        if first_order * second_order > 0:
            concordant += 1
        elif first_order * second_order < 0:
            discordant += 1

    pair_count = concordant + discordant
    if pair_count == 0:
        return None, concordant, discordant
    return (concordant - discordant) / pair_count, concordant, discordant


def topk_overlap(first_admit: list[str], second_admit: list[str]) -> float | None:
    if not first_admit or not second_admit:
        return None
    k = min(len(first_admit), len(second_admit))
    if k == 0:
        return None
    return len(set(first_admit[:k]) & set(second_admit[:k])) / k


def property_value(patients: dict[str, dict[str, Any]], patient_id: str, prop: str) -> float | None:
    value = patients.get(patient_id, {}).get(prop)
    return value if isinstance(value, (float, int)) else None


def mean_property(
    patients: dict[str, dict[str, Any]],
    patient_ids: list[str],
    prop: str,
) -> float | None:
    values = [
        float(value)
        for patient_id in patient_ids
        if (value := property_value(patients, patient_id, prop)) is not None
    ]
    return mean(values) if values else None


def positional_property_mean(
    patients: dict[str, dict[str, Any]],
    ranking: list[str],
    prop: str,
) -> float | None:
    weighted_sum = 0.0
    weight_total = 0.0
    n = len(ranking)
    for zero_index, patient_id in enumerate(ranking):
        value = property_value(patients, patient_id, prop)
        if value is None:
            continue
        # Higher ranks get larger weights so upward movement affects the mean.
        weight = n - zero_index
        weighted_sum += weight * float(value)
        weight_total += weight
    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def round_metric(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_comparison_rows(
    records: list[dict[str, Any]],
    patients: dict[str, dict[str, Any]],
    properties: list[str],
    baseline_condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (row["model"], row["task"], row["run_id"], row["episode_id"], row["condition_id"]): row
        for row in records
    }

    kendall_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, Any]] = []
    patient_shift_rows: list[dict[str, Any]] = []

    for row in records:
        if row["condition_id"] == baseline_condition:
            continue
        base_key = (row["model"], row["task"], row["run_id"], row["episode_id"], baseline_condition)
        baseline = by_key.get(base_key)
        if baseline is None:
            continue

        # Ranking metrics only need a complete, valid ranking from both prompts.
        ranking_valid_pair = bool(baseline["schema_valid"] and row["schema_valid"])
        # Admit-list metrics are stricter because admit must match top-K ranking.
        decision_consistent_pair = bool(
            baseline["decision_consistent"] and row["decision_consistent"]
        )
        tau = None
        concordant = discordant = 0
        if ranking_valid_pair:
            tau, concordant, discordant = kendall_tau(baseline["ranking"], row["ranking"])

        kendall_rows.append(
            {
                "model": row["model"],
                "task": row["task"],
                "run_id": row["run_id"],
                "episode_id": row["episode_id"],
                "baseline_condition_id": baseline_condition,
                "condition_id": row["condition_id"],
                "persona": row["persona"],
                "ranking_valid_pair": ranking_valid_pair,
                "decision_consistent_pair": decision_consistent_pair,
                "kendall_tau": round_metric(tau),
                "concordant_pairs": concordant,
                "discordant_pairs": discordant,
                "topk_ranking_overlap": round_metric(
                    topk_overlap(
                        baseline["ranking"][: baseline["capacity"]],
                        row["ranking"][: row["capacity"]],
                    )
                ),
                "admit_overlap": round_metric(topk_overlap(baseline["admit"], row["admit"])),
                "baseline_admit": baseline["admit"],
                "persona_admit": row["admit"],
                "baseline_ranking": baseline["ranking"],
                "persona_ranking": row["ranking"],
                "baseline_error": baseline["parse_error"] or baseline["validation_error"],
                "persona_error": row["parse_error"] or row["validation_error"],
            }
        )

        if not ranking_valid_pair:
            continue

        base_ranks = rank_map(baseline["ranking"])
        persona_ranks = rank_map(row["ranking"])
        for patient_id in baseline["ranking"]:
            # Positive rank_delta means the patient moved up under the persona.
            patient_row = {
                "model": row["model"],
                "task": row["task"],
                "run_id": row["run_id"],
                "episode_id": row["episode_id"],
                "condition_id": row["condition_id"],
                "persona": row["persona"],
                "patient_id": patient_id,
                "baseline_rank": base_ranks[patient_id],
                "persona_rank": persona_ranks[patient_id],
                "rank_delta": base_ranks[patient_id] - persona_ranks[patient_id],
                "baseline_admitted": patient_id in baseline["admit"],
                "persona_admitted": patient_id in row["admit"],
            }
            for prop in properties:
                patient_row[prop] = property_value(patients, patient_id, prop)
            patient_shift_rows.append(patient_row)

        for prop in properties:
            # Compare both the admitted set and the whole ranked list.
            baseline_topk_mean = mean_property(patients, baseline["ranking"][: baseline["capacity"]], prop)
            persona_topk_mean = mean_property(patients, row["ranking"][: row["capacity"]], prop)
            baseline_position_mean = positional_property_mean(patients, baseline["ranking"], prop)
            persona_position_mean = positional_property_mean(patients, row["ranking"], prop)
            xs: list[float] = []
            ys: list[float] = []
            for patient_id in baseline["ranking"]:
                value = property_value(patients, patient_id, prop)
                if value is None:
                    continue
                xs.append(float(value))
                ys.append(float(base_ranks[patient_id] - persona_ranks[patient_id]))

            property_rows.append(
                {
                    "model": row["model"],
                    "task": row["task"],
                    "run_id": row["run_id"],
                    "episode_id": row["episode_id"],
                    "condition_id": row["condition_id"],
                    "persona": row["persona"],
                    "property": prop,
                    "baseline_topk_mean": round_metric(baseline_topk_mean),
                    "persona_topk_mean": round_metric(persona_topk_mean),
                    "topk_mean_delta": round_metric(
                        None
                        if baseline_topk_mean is None or persona_topk_mean is None
                        else persona_topk_mean - baseline_topk_mean
                    ),
                    "baseline_positional_mean": round_metric(baseline_position_mean),
                    "persona_positional_mean": round_metric(persona_position_mean),
                    "positional_mean_delta": round_metric(
                        None
                        if baseline_position_mean is None or persona_position_mean is None
                        else persona_position_mean - baseline_position_mean
                    ),
                    "rank_shift_property_corr": round_metric(pearson(xs, ys)),
                }
            )

    return kendall_rows, property_rows, patient_shift_rows


def summarize(rows: list[dict[str, Any]], group_cols: list[str], metric_cols: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(col) for col in group_cols)].append(row)

    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {col: value for col, value in zip(group_cols, key)}
        summary["n"] = len(group_rows)
        for metric in metric_cols:
            # Empty metrics are skipped so invalid comparisons do not become zero.
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric) not in (None, "")
            ]
            summary[f"{metric}_mean"] = round_metric(mean(values)) if values else None
            summary[f"{metric}_std"] = round_metric(stdev(values)) if len(values) > 1 else None
        output.append(summary)
    return output


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    logs_dir = (args.logs_dir or base_dir / "logs").resolve()
    patients_csv = (args.patients_csv or base_dir / "generated_dataset" / "patients.csv").resolve()
    out_dir = (args.out_dir or base_dir / "analysis").resolve()

    patients = load_patients(patients_csv)
    properties = selected_properties(args, patients)
    log_files = discover_log_files(logs_dir, args.task, args.latest_run_only)

    records = [parse_log_record(path, logs_dir, patients) for path in log_files]
    kendall_rows, property_rows, patient_shift_rows = build_comparison_rows(
        records=records,
        patients=patients,
        properties=properties,
        baseline_condition=args.baseline_condition,
    )

    parsed_fields = [
        "model",
        "task",
        "run_id",
        "prompt_id",
        "episode_id",
        "condition_id",
        "persona",
        "capacity",
        "patient_count",
        "parse_valid",
        "schema_valid",
        "decision_consistent",
        "parse_error",
        "schema_error",
        "consistency_error",
        "validation_error",
        "admit",
        "ranking",
        "log_path",
    ]
    kendall_fields = [
        "model",
        "task",
        "run_id",
        "episode_id",
        "baseline_condition_id",
        "condition_id",
        "persona",
        "ranking_valid_pair",
        "decision_consistent_pair",
        "kendall_tau",
        "concordant_pairs",
        "discordant_pairs",
        "topk_ranking_overlap",
        "admit_overlap",
        "baseline_admit",
        "persona_admit",
        "baseline_ranking",
        "persona_ranking",
        "baseline_error",
        "persona_error",
    ]
    property_fields = [
        "model",
        "task",
        "run_id",
        "episode_id",
        "condition_id",
        "persona",
        "property",
        "baseline_topk_mean",
        "persona_topk_mean",
        "topk_mean_delta",
        "baseline_positional_mean",
        "persona_positional_mean",
        "positional_mean_delta",
        "rank_shift_property_corr",
    ]
    patient_shift_fields = [
        "model",
        "task",
        "run_id",
        "episode_id",
        "condition_id",
        "persona",
        "patient_id",
        "baseline_rank",
        "persona_rank",
        "rank_delta",
        "baseline_admitted",
        "persona_admitted",
        *properties,
    ]

    kendall_summary = summarize(
        [row for row in kendall_rows if row.get("ranking_valid_pair")],
        ["model", "task", "condition_id", "persona"],
        ["kendall_tau", "topk_ranking_overlap", "admit_overlap"],
    )
    property_summary = summarize(
        property_rows,
        ["model", "task", "condition_id", "persona", "property"],
        ["topk_mean_delta", "positional_mean_delta", "rank_shift_property_corr"],
    )

    write_csv(out_dir / "parsed_responses.csv", records, parsed_fields)
    write_csv(out_dir / "kendall_tau_by_run.csv", kendall_rows, kendall_fields)
    write_csv(out_dir / "kendall_tau_summary.csv", kendall_summary, list(kendall_summary[0].keys()) if kendall_summary else [])
    write_csv(out_dir / "persona_property_shift_by_run.csv", property_rows, property_fields)
    write_csv(
        out_dir / "persona_property_shift_summary.csv",
        property_summary,
        list(property_summary[0].keys()) if property_summary else [],
    )
    write_csv(out_dir / "patient_rank_shifts.csv", patient_shift_rows, patient_shift_fields)

    parse_valid_count = sum(1 for row in records if row["parse_valid"])
    schema_valid_count = sum(1 for row in records if row["schema_valid"])
    consistent_count = sum(1 for row in records if row["decision_consistent"])
    print(f"Read {len(log_files)} log files from {logs_dir}")
    print(f"Parsed {parse_valid_count}/{len(records)} JSON/dict responses")
    print(f"Schema-valid responses: {schema_valid_count}/{len(records)}")
    print(f"Decision-consistent responses: {consistent_count}/{len(records)}")
    print(f"Wrote analysis CSVs to {out_dir}")


if __name__ == "__main__":
    main()
