#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from common import PERSONA_FILES, extract_json_object, flatten_response_record, load_text, patient_card_text


DEFAULT_PERSONAS = ["none", "social_worker", "revenue", "psychiatry"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--backend", choices=["ollama", "openai_compatible"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None, help="Required for openai_compatible backend")
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--persona-set", default="default")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--pause-seconds", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def load_episodes(path: str) -> List[dict]:
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def build_messages(ep: dict, prompts_dir: Path, persona: str) -> List[dict]:
    system_prompt = load_text(prompts_dir / "system_acuity_first.txt")
    user_template = load_text(prompts_dir / "user_episode_template.txt")
    persona_text = load_text(prompts_dir / PERSONA_FILES[persona])

    patient_block = "\n".join(patient_card_text(p) for p in ep["patients"])
    patient_ids_csv = ", ".join([pp["patient_id"] for pp in ep["patients"]])
    user_prompt = user_template.format(
        episode_id=ep["episode_id"],
        occupancy_label=ep["occupancy_label"],
        capacity=ep["capacity"],
        n_patients=len(ep["patients"]),
        patient_ids_csv=patient_ids_csv,
        patient_block=patient_block,
    )
    full_user = f"{persona_text}\n\n{user_prompt}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": full_user}]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ollama(base_url: str, model: str, messages: List[dict], temperature: float, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_compatible(base_url: str, model: str, messages: List[dict], temperature: float, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def main() -> None:
    args = parse_args()
    prompts_dir = Path(args.prompts_dir)
    episodes = load_episodes(args.episodes)
    if args.limit:
        episodes = episodes[: args.limit]

    if args.persona_set == "default":
        personas = DEFAULT_PERSONAS
    else:
        personas = [p.strip() for p in args.persona_set.split(",") if p.strip()]

    base_url = args.base_url
    if args.backend == "ollama":
        base_url = base_url or "http://127.0.0.1:11434"
    elif not base_url:
        raise ValueError("--base-url is required for openai_compatible backend")

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    flat_rows: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for ep in episodes:
            for persona in personas:
                messages = build_messages(ep, prompts_dir, persona)
                try:
                    if args.backend == "ollama":
                        response_text = call_ollama(base_url, args.model, messages, args.temperature, args.max_tokens, args.timeout)
                    else:
                        response_text = call_openai_compatible(base_url, args.model, messages, args.temperature, args.max_tokens, args.timeout)
                    parsed = extract_json_object(response_text)
                    status = "ok"
                except Exception as e:
                    response_text = f"ERROR: {type(e).__name__}: {e}"
                    parsed = None
                    status = "error"

                rec = {
                    "episode_id": ep["episode_id"],
                    "backend": args.backend,
                    "model": args.model,
                    "persona": persona,
                    "system_prompt_name": "system_acuity_first",
                    "user_prompt": messages[1]["content"],
                    "response_text": response_text,
                    "parsed_json": parsed,
                    "status": status,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flat_rows.append(flatten_response_record(rec))
                time.sleep(args.pause_seconds)

    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"Saved raw records to {out_jsonl}")
    print(f"Saved flattened CSV to {out_csv}")


if __name__ == "__main__":
    main()
