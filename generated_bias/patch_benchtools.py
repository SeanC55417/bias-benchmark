#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import benchtools.task


def main() -> None:
    task_path = Path(benchtools.task.__file__).resolve()
    text = task_path.read_text(encoding="utf-8")
    # BenchTools sometimes loses the CSV task path when loading custom_response.py.
    old = "source_path=source_path)"
    new = "source_path=source_path or task_path)"

    if new in text:
        # Make the patch idempotent so run_benchmark.sh can call it every time.
        print(f"BenchTools CSV custom response patch already applied: {task_path}")
        return

    if old not in text:
        raise RuntimeError(
            f"Could not find expected BenchTools source_path line in {task_path}"
        )

    task_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Applied BenchTools CSV custom response patch: {task_path}")


if __name__ == "__main__":
    main()
