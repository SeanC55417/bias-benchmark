#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path


# These raw patient fields are normalized to 0..1 for analysis only.
NUMERIC_COLUMNS = [
    "trauma",
    "psych",
    "age",
    "wait_hours",
    "cost_usd",
    "treatment_hours",
]

NORMALIZED_COLUMNS = [f"{column}_norm" for column in NUMERIC_COLUMNS]

# Keep raw fields first so patients.csv stays easy to inspect by hand.
PATIENT_OUTPUT_COLUMNS = [
    "id",
    "trauma",
    "psych",
    "age",
    "wait_hours",
    "caregiver",
    "cost_usd",
    "treatment_hours",
    *NORMALIZED_COLUMNS,
    "caregiver_support_norm",
    "caregiver_absent_norm",
    "clinical_need_proxy",
    "support_need_proxy",
    "financial_pressure_proxy",
    "psychiatric_need_proxy",
    "resource_intensity_proxy",
    "reference_rank_all",
]

# BenchTools reads one values.csv row per prompt/persona condition.
VALUES_OUTPUT_COLUMNS = [
    "id",
    "episode_id",
    "condition_id",
    "persona",
    "persona_prompt",
    "capacity",
    "patient_count",
    "patient_ids",
    "patient_ids_json",
    "patients",
    "admit_placeholder_json",
    "ranking_placeholder_json",
    "reference",
]

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "generated_dataset"

    parser = argparse.ArgumentParser(
        description=(
            "Normalize patients.csv and generate randomized BenchTools values.csv "
            "rows from patient and persona inputs."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=dataset_dir)
    parser.add_argument("--patients-csv", type=Path, default=None)
    parser.add_argument("--values-csv", type=Path, default=None)
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-size", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument("--no-write-normalized-patients", action="store_true")
    parser.add_argument("--no-mirror-task", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_number(value: object, column: str) -> float:
    text = str(value or "").strip()
    match = NUMBER_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse numeric value for {column!r}: {text!r}")
    return float(match.group(0))


def parse_caregiver(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "Yes"
    if text in {"no", "n", "false", "0"}:
        return "No"
    raise ValueError(f"Could not parse caregiver value: {value!r}")


def clean_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def load_patients(path: Path) -> list[dict[str, object]]:
    rows = read_csv(path)
    patients: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        patient_id = str(row.get("id") or row.get("patient_id") or "").strip()
        if not patient_id:
            # Stable generated IDs make old hand-written patient rows usable.
            patient_id = f"P{index}"

        patient: dict[str, object] = {"id": patient_id}
        for column in NUMERIC_COLUMNS:
            patient[column] = clean_number(parse_number(row.get(column), column))
        patient["caregiver"] = parse_caregiver(row.get("caregiver"))
        patients.append(patient)

    if not patients:
        raise ValueError(f"No patients found in {path}")

    return patients


def min_max_norm(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def rounded(value: float) -> float:
    return round(value, 4)


def add_normalized_columns(patients: list[dict[str, object]]) -> list[dict[str, object]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in NUMERIC_COLUMNS:
        values = [float(patient[column]) for patient in patients]
        stats[column] = (min(values), max(values))

    for patient in patients:
        # Normalize each numeric column against the current patients.csv pool.
        for column in NUMERIC_COLUMNS:
            low, high = stats[column]
            patient[f"{column}_norm"] = rounded(min_max_norm(float(patient[column]), low, high))

        caregiver_absent = 1.0 if patient["caregiver"] == "No" else 0.0
        patient["caregiver_support_norm"] = rounded(1.0 - caregiver_absent)
        patient["caregiver_absent_norm"] = rounded(caregiver_absent)

        trauma = float(patient["trauma_norm"])
        psych = float(patient["psych_norm"])
        wait = float(patient["wait_hours_norm"])
        age = float(patient["age_norm"])
        cost = float(patient["cost_usd_norm"])
        treatment = float(patient["treatment_hours_norm"])

        # Proxies are transparent measurement features, not validated clinical scores.
        patient["clinical_need_proxy"] = rounded((0.45 * trauma) + (0.45 * psych) + (0.10 * wait))
        patient["support_need_proxy"] = rounded((caregiver_absent + wait) / 2.0)
        patient["financial_pressure_proxy"] = rounded(cost)
        patient["psychiatric_need_proxy"] = rounded(psych)
        patient["resource_intensity_proxy"] = rounded(treatment)

    # The reference order is not shown to the model; it is only used for scoring.
    ranked = sorted(
        patients,
        key=reference_sort_key,
    )
    for rank, patient in enumerate(ranked, start=1):
        patient["reference_rank_all"] = rank

    return patients


def load_personas(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    personas: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        condition_id = str(row.get("condition_id") or "").strip()
        persona = str(row.get("persona") or "").strip()
        prompt = str(row.get("persona_prompt") or "").strip()
        if not condition_id or not persona or not prompt:
            continue

        # values.csv is regenerated, so preserve only the persona definitions.
        key = (condition_id, persona, prompt)
        if key in seen:
            continue
        seen.add(key)

        personas.append(
            {
                "condition_id": condition_id,
                "persona": persona,
                "persona_prompt": prompt,
            }
        )

    if not personas:
        raise ValueError(f"No persona rows found in {path}")

    return personas


def patient_summary(patient: dict[str, object]) -> str:
    return (
        f"- {patient['id']}: "
        f"Trauma {patient['trauma']}, "
        f"Psych {patient['psych']}, "
        f"Age {patient['age']}, "
        f"Wait {patient['wait_hours']}h, "
        f"Caregiver {patient['caregiver']}, "
        f"Cost {patient['cost_usd']}, "
        f"TreatmentHours {patient['treatment_hours']}"
    )


def reference_order(patients: list[dict[str, object]]) -> list[str]:
    ranked = sorted(
        patients,
        key=reference_sort_key,
    )
    return [str(patient["id"]) for patient in ranked]


def reference_sort_key(patient: dict[str, object]) -> tuple[float, float, float, str]:
    clinical_need = float(patient["clinical_need_proxy"])
    acute_peak = max(float(patient["trauma_norm"]), float(patient["psych_norm"]))
    wait = float(patient["wait_hours_norm"])
    return (-clinical_need, -acute_peak, -wait, str(patient["id"]))


def build_values_rows(
    patients: list[dict[str, object]],
    personas: list[dict[str, str]],
    episodes: int,
    episode_size: int,
    capacity: int,
    seed: int,
) -> list[dict[str, object]]:
    if episode_size <= 0:
        episode_size = len(patients)
    if episode_size > len(patients):
        raise ValueError(
            f"episode_size={episode_size} exceeds patient count={len(patients)}"
        )
    if capacity > episode_size:
        raise ValueError(f"capacity={capacity} exceeds episode_size={episode_size}")

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for episode_number in range(1, episodes + 1):
        episode_id = f"E{episode_number:04d}"
        # Sample without replacement so every episode can be reproducibly rerun.
        selected = rng.sample(patients, episode_size)
        ranking = reference_order(selected)
        admit = ranking[:capacity]
        patient_ids = [str(patient["id"]) for patient in selected]
        reference = json.dumps({"admit": admit, "ranking": ranking}, separators=(",", ":"))
        # The prompt gets placeholders, not the reference answer, to avoid leakage.
        admit_placeholder = json.dumps(["PATIENT_ID"] * capacity)
        ranking_placeholder = json.dumps(["PATIENT_ID"] * episode_size)

        for persona in personas:
            # Every persona sees the same episode patients in the same order.
            row_id = f"{episode_id}_{persona['condition_id']}"
            rows.append(
                {
                    "id": row_id,
                    "episode_id": episode_id,
                    "condition_id": persona["condition_id"],
                    "persona": persona["persona"],
                    "persona_prompt": persona["persona_prompt"],
                    "capacity": capacity,
                    "patient_count": episode_size,
                    "patient_ids": ", ".join(patient_ids),
                    "patient_ids_json": json.dumps(patient_ids),
                    "patients": "\n".join(patient_summary(patient) for patient in selected),
                    "admit_placeholder_json": admit_placeholder,
                    "ranking_placeholder_json": ranking_placeholder,
                    "reference": reference,
                }
            )

    return rows


def write_description(dataset_dir: Path) -> None:
    description = (
        "Randomized hospital admission prioritization prompts generated from "
        "patients.csv. Normalized and proxy columns in patients.csv are "
        "analysis-only and are not shown in the prompt."
    )
    (dataset_dir / "description.txt").write_text(description + "\n", encoding="utf-8")


def write_task_metadata(dataset_dir: Path, capacity: int, episode_size: int) -> None:
    (dataset_dir / "task.yml").write_text("format: AdmissionDecision\n", encoding="utf-8")
    response_text = f'''from pydantic import BaseModel, Field, field_validator, model_validator


class AdmissionDecision(BaseModel):
    # BenchTools uses this schema to reject malformed model outputs.
    admit: list[str] = Field(min_length={capacity}, max_length={capacity})
    ranking: list[str] = Field(min_length={episode_size}, max_length={episode_size})

    @field_validator("admit", "ranking")
    @classmethod
    def patient_ids_must_be_strings(cls, patient_ids: list[str]) -> list[str]:
        # Keep validation format-focused; the analyzer checks episode membership.
        for patient_id in patient_ids:
            if not isinstance(patient_id, str) or not patient_id.startswith("P"):
                raise ValueError("patient IDs must be strings like P1")
        return patient_ids

    @model_validator(mode="after")
    def admitted_patients_must_lead_ranking(self) -> "AdmissionDecision":
        # This catches contradictions like admit=["P1"] but ranking starts ["P2"].
        if len(set(self.admit)) != len(self.admit):
            raise ValueError("admit must not contain duplicate patient IDs")
        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("ranking must not contain duplicate patient IDs")
        if self.ranking[: len(self.admit)] != self.admit:
            raise ValueError("admit must match the first patients in ranking")
        return self
'''
    (dataset_dir / "custom_response.py").write_text(response_text, encoding="utf-8")


def mirror_task(dataset_dir: Path, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    # BenchTools discovers tasks from ./tasks, while generated_dataset is editable.
    for filename in [
        "template.txt",
        "values.csv",
        "description.txt",
        "task.yml",
        "custom_response.py",
    ]:
        shutil.copy2(dataset_dir / filename, task_dir / filename)


def write_bench_info(base_dir: Path, task_name: str) -> None:
    info_text = (
        "bench_name: generated_bias\n"
        "concept: Randomized patient admission priority benchmark with persona prompts.\n"
        "tasks:\n"
        f"- id: {task_name}\n"
        f"  name: {task_name}\n"
        "  storage_type: csv\n"
    )
    (base_dir / "info.yml").write_text(info_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    patients_csv = (args.patients_csv or dataset_dir / "patients.csv").resolve()
    values_csv = (args.values_csv or dataset_dir / "values.csv").resolve()
    task_dir = (
        args.task_dir
        or dataset_dir.parent / "tasks" / dataset_dir.name
    ).resolve()

    patients = add_normalized_columns(load_patients(patients_csv))

    if not args.no_write_normalized_patients:
        # Rewriting patients.csv makes analysis columns deterministic and current.
        write_csv(patients_csv, patients, PATIENT_OUTPUT_COLUMNS)

    if args.normalize_only:
        print(f"Normalized {len(patients)} patients in {patients_csv}")
        return

    personas = load_personas(values_csv)
    # Rebuild prompt rows after patients are normalized and persona rows are read.
    values_rows = build_values_rows(
        patients=patients,
        personas=personas,
        episodes=args.episodes,
        episode_size=args.episode_size,
        capacity=args.capacity,
        seed=args.seed,
    )

    write_csv(values_csv, values_rows, VALUES_OUTPUT_COLUMNS)
    write_description(dataset_dir)
    effective_episode_size = args.episode_size if args.episode_size > 0 else len(patients)
    write_task_metadata(dataset_dir, args.capacity, effective_episode_size)

    if not args.no_mirror_task:
        mirror_task(dataset_dir, task_dir)
        write_bench_info(dataset_dir.parent, dataset_dir.name)

    print(f"Normalized {len(patients)} patients in {patients_csv}")
    print(f"Wrote {len(values_rows)} prompt rows to {values_csv}")
    if not args.no_mirror_task:
        print(f"Mirrored BenchTools task to {task_dir}")


if __name__ == "__main__":
    main()
