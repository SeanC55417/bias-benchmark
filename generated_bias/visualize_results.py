#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


# Keep the default dashboard focused on interpretable patient attributes.
CORE_PROPERTIES = [
    "clinical_priority_score",
    "social_vulnerability_score",
    "financial_pressure_score",
    "psychiatric_need_score",
    "resource_intensity_score",
    "cost_usd_norm",
    "psych_norm",
    "caregiver_absent_norm",
]

# Shared palette for every SVG so the dashboard reads consistently.
COLORS = {
    "ink": "#1d2433",
    "muted": "#697386",
    "grid": "#d9dee8",
    "panel": "#f7f8fb",
    "valid": "#2b8a6e",
    "schema_only": "#4b7bec",
    "parse_only": "#f4d06f",
    "invalid": "#cf4c45",
    "neutral": "#f1f4f8",
    "positive": "#2b8a6e",
    "negative": "#cf4c45",
    "tau_low": "#cf4c45",
    "tau_mid": "#f4d06f",
    "tau_high": "#2b8a6e",
}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create SVG visualizations from generated_bias analysis CSV files."
    )
    parser.add_argument("--analysis-dir", type=Path, default=base_dir / "analysis")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--properties",
        default=",".join(CORE_PROPERTIES),
        help="Comma-separated properties to show in the property-shift heatmap.",
    )
    parser.add_argument(
        "--shift-metric",
        choices=["topk_mean_delta_mean", "positional_mean_delta_mean"],
        default="positional_mean_delta_mean",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        # Preserve CSV order but remove repeats for chart axes.
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def svg_text(
    x: float,
    y: float,
    text: Any,
    size: int = 12,
    fill: str = COLORS["ink"],
    anchor: str = "start",
    weight: str = "400",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-family="Inter, Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}"{transform}>{esc(text)}</text>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str | None = None,
    radius: int = 0,
) -> str:
    stroke_text = f' stroke="{stroke}"' if stroke else ""
    radius_text = f' rx="{radius}" ry="{radius}"' if radius else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}"{stroke_text}{radius_text}/>'
    )


def svg_base(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f'<rect width="{width}" height="{height}" fill="white"/>\n'
        f"{body}\n</svg>\n"
    )


def blend(hex_a: str, hex_b: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    a = tuple(int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(a_i + (b_i - a_i) * amount) for a_i, b_i in zip(a, b))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def tau_color(value: float | None) -> str:
    if value is None:
        return COLORS["neutral"]
    # Kendall Tau ranges from -1 to 1, so shift it onto a 0..1 color scale.
    amount = (value + 1.0) / 2.0
    if amount < 0.5:
        return blend(COLORS["tau_low"], COLORS["tau_mid"], amount / 0.5)
    return blend(COLORS["tau_mid"], COLORS["tau_high"], (amount - 0.5) / 0.5)


def positive_color(value: float | None) -> str:
    if value is None:
        return COLORS["neutral"]
    amount = max(0.0, min(1.0, value))
    return blend("#f5f8fb", COLORS["positive"], amount)


def diverging_color(value: float | None, max_abs: float) -> str:
    if value is None:
        return COLORS["neutral"]
    if max_abs <= 0:
        return "#f5f8fb"
    amount = min(1.0, abs(value) / max_abs)
    target = COLORS["positive"] if value >= 0 else COLORS["negative"]
    return blend("#f5f8fb", target, amount)


def valid_response_rate_chart(rows: list[dict[str, str]], out_path: Path) -> None:
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        model = row.get("model", "")
        if not model:
            continue
        if row.get("decision_consistent") == "True":
            by_model[model]["decision_consistent"] += 1
        elif row.get("schema_valid") == "True":
            by_model[model]["schema_only"] += 1
        elif row.get("parse_valid") == "True":
            by_model[model]["parse_only"] += 1
        else:
            by_model[model]["invalid"] += 1

    models = sorted(model for model in by_model if model)
    width = 920
    row_h = 38
    top = 88
    left = 190
    chart_w = 580
    height = top + len(models) * row_h + 80
    parts = [
        svg_text(32, 38, "Response Validity By Model", 22, weight="700"),
        svg_text(32, 62, "Decision-consistent, schema-valid, parse-only, and invalid responses.", 13, COLORS["muted"]),
    ]
    segments = [
        ("decision_consistent", "consistent", COLORS["valid"]),
        ("schema_only", "schema only", COLORS["schema_only"]),
        ("parse_only", "parse only", COLORS["parse_only"]),
        ("invalid", "invalid", COLORS["invalid"]),
    ]

    for idx, model in enumerate(models):
        y = top + idx * row_h
        counts = by_model[model]
        total = sum(counts.values())
        consistent = counts["decision_consistent"]
        parts.append(svg_text(left - 16, y + 22, model, 12, anchor="end"))
        x = left
        for key, _, color in segments:
            segment_w = chart_w * (counts[key] / total) if total else 0
            if segment_w > 0:
                parts.append(svg_rect(x, y, segment_w, 22, color, radius=3))
            x += segment_w
        parts.append(svg_rect(left, y, chart_w, 22, "none", COLORS["grid"], radius=3))
        parts.append(svg_text(left + chart_w + 14, y + 16, f"{consistent}/{total} consistent", 12, COLORS["muted"]))

    legend_y = height - 34
    legend_x = left
    for _, label, color in segments:
        parts.append(svg_rect(legend_x, legend_y - 12, 14, 14, color, radius=2))
        parts.append(svg_text(legend_x + 22, legend_y, label, 12, COLORS["muted"]))
        legend_x += 130
    write_text(out_path, svg_base(width, height, "\n".join(parts)))


def heatmap_chart(
    title: str,
    subtitle: str,
    rows: list[str],
    cols: list[str],
    values: dict[tuple[str, str], float | None],
    out_path: Path,
    color_func,
    value_digits: int = 3,
) -> None:
    cell_w = 150
    cell_h = 42
    left = 210
    top = 112
    width = left + max(1, len(cols)) * cell_w + 48
    height = top + max(1, len(rows)) * cell_h + 72
    parts = [
        svg_text(32, 38, title, 22, weight="700"),
        svg_text(32, 62, subtitle, 13, COLORS["muted"]),
    ]

    for col_index, col in enumerate(cols):
        x = left + col_index * cell_w + cell_w / 2
        parts.append(svg_text(x, top - 18, col, 12, COLORS["muted"], anchor="middle", weight="600"))

    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        parts.append(svg_text(left - 14, y + 26, row, 12, anchor="end"))
        for col_index, col in enumerate(cols):
            x = left + col_index * cell_w
            value = values.get((row, col))
            parts.append(svg_rect(x, y, cell_w - 4, cell_h - 4, color_func(value), COLORS["grid"], radius=4))
            parts.append(svg_text(x + (cell_w - 4) / 2, y + 25, fmt(value, value_digits), 12, anchor="middle"))

    write_text(out_path, svg_base(width, height, "\n".join(parts)))


def kendall_heatmap(rows: list[dict[str, str]], out_path: Path) -> None:
    models = unique_in_order([row["model"] for row in rows])
    personas = unique_in_order([row["condition_id"] for row in rows])
    values = {
        (row["model"], row["condition_id"]): fnum(row.get("kendall_tau_mean"))
        for row in rows
    }
    heatmap_chart(
        "Kendall Tau vs Baseline",
        "1.0 means persona ranking matched the baseline ranking exactly.",
        models,
        personas,
        values,
        out_path,
        tau_color,
    )


def topk_overlap_heatmap(rows: list[dict[str, str]], out_path: Path) -> None:
    models = unique_in_order([row["model"] for row in rows])
    personas = unique_in_order([row["condition_id"] for row in rows])
    values = {
        (row["model"], row["condition_id"]): fnum(row.get("topk_ranking_overlap_mean"))
        for row in rows
    }
    heatmap_chart(
        "Top-K Admission Overlap",
        "1.0 means the top-K ranking set matched baseline exactly.",
        models,
        personas,
        values,
        out_path,
        positive_color,
    )


def property_shift_heatmap(
    rows: list[dict[str, str]],
    properties: list[str],
    metric: str,
    out_path: Path,
) -> None:
    filtered = [row for row in rows if row.get("property") in properties]
    row_labels = unique_in_order(
        [f"{row['model']} | {row['condition_id']}" for row in filtered]
    )
    values: dict[tuple[str, str], float | None] = {}
    all_numbers: list[float] = []
    for row in filtered:
        label = f"{row['model']} | {row['condition_id']}"
        value = fnum(row.get(metric))
        values[(label, row["property"])] = value
        if value is not None:
            all_numbers.append(value)

    max_abs = max([abs(value) for value in all_numbers], default=0.0)
    if max_abs == 0:
        max_abs = 1.0

    heatmap_chart(
        "Persona Property Shift",
        f"Cells show {metric}; positive means higher-property patients moved upward vs baseline.",
        row_labels,
        properties,
        values,
        out_path,
        lambda value: diverging_color(value, max_abs),
    )


def revenue_bar_chart(rows: list[dict[str, str]], out_path: Path) -> None:
    # This chart isolates the specific revenue/resource hypothesis.
    properties = ["cost_usd_norm", "financial_pressure_score", "resource_intensity_score"]
    filtered = [
        row
        for row in rows
        if row.get("condition_id") == "revenue" and row.get("property") in properties
    ]
    models = unique_in_order([row["model"] for row in filtered])
    value_by_key = {
        (row["model"], row["property"]): fnum(row.get("topk_mean_delta_mean")) or 0.0
        for row in filtered
    }
    max_abs = max([abs(value) for value in value_by_key.values()], default=0.0)
    max_abs = max(max_abs, 0.05)

    width = 980
    height = 118 + len(models) * 86
    left = 210
    center_x = left + 330
    scale = 300 / max_abs
    parts = [
        svg_text(32, 38, "Revenue Persona: Top-K Property Shift", 22, weight="700"),
        svg_text(32, 62, "Positive bars mean admitted patients had higher normalized revenue/resource scores than baseline.", 13, COLORS["muted"]),
        svg_rect(center_x, 92, 1, height - 138, COLORS["grid"]),
        svg_text(center_x, 84, "0", 11, COLORS["muted"], anchor="middle"),
    ]
    palette = {
        "cost_usd_norm": "#4b7bec",
        "financial_pressure_score": "#2b8a6e",
        "resource_intensity_score": "#b95f89",
    }
    bar_h = 18
    for model_index, model in enumerate(models):
        y_base = 112 + model_index * 86
        parts.append(svg_text(left - 16, y_base + 20, model, 12, anchor="end", weight="600"))
        for prop_index, prop in enumerate(properties):
            value = value_by_key.get((model, prop), 0.0)
            y = y_base + prop_index * 22
            x = center_x if value >= 0 else center_x + value * scale
            w = abs(value) * scale
            parts.append(svg_text(left - 16, y + 15, prop, 11, COLORS["muted"], anchor="end"))
            parts.append(svg_rect(x, y, max(1, w), bar_h, palette[prop], radius=3))
            text_x = x + w + 8 if value >= 0 else x - 8
            anchor = "start" if value >= 0 else "end"
            parts.append(svg_text(text_x, y + 14, fmt(value), 11, COLORS["ink"], anchor=anchor))

    write_text(out_path, svg_base(width, height, "\n".join(parts)))


def html_dashboard(out_dir: Path, files: list[str], summary: dict[str, Any]) -> None:
    figures = "\n".join(
        f'<section><img src="{esc(filename)}" alt="{esc(filename)}"></section>'
        for filename in files
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Generated Bias Benchmark Results</title>
  <style>
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: #1d2433; background: #f4f6f9; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 26px; margin: 0 0 6px; }}
    p {{ color: #697386; margin: 0 0 18px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 20px; }}
    .metric {{ background: white; border: 1px solid #d9dee8; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    section {{ background: white; border: 1px solid #d9dee8; border-radius: 8px; padding: 14px; margin: 14px 0; overflow-x: auto; }}
    img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <main>
    <h1>Generated Bias Benchmark Results</h1>
    <p>Baseline persona comparisons and normalized patient-property shifts.</p>
    <div class="summary">
      <div class="metric"><strong>{esc(summary.get("response_count", 0))}</strong><span>responses parsed</span></div>
      <div class="metric"><strong>{esc(summary.get("valid_rate", "n/a"))}</strong><span>decision consistency rate</span></div>
      <div class="metric"><strong>{esc(summary.get("schema_rate", "n/a"))}</strong><span>schema-valid rate</span></div>
      <div class="metric"><strong>{esc(summary.get("comparison_count", 0))}</strong><span>persona comparisons</span></div>
    </div>
    {figures}
  </main>
</body>
</html>
"""
    write_text(out_dir / "index.html", body)


def build_summary(parsed_rows: list[dict[str, str]], kendall_rows: list[dict[str, str]]) -> dict[str, Any]:
    response_count = len(parsed_rows)
    schema_count = sum(1 for row in parsed_rows if row.get("schema_valid") == "True")
    consistent_count = sum(
        1
        for row in parsed_rows
        if row.get("decision_consistent") == "True"
    )
    valid_rate = f"{consistent_count / response_count:.1%}" if response_count else "n/a"
    schema_rate = f"{schema_count / response_count:.1%}" if response_count else "n/a"
    comparison_count = sum(int(row.get("n") or 0) for row in kendall_rows)
    return {
        "response_count": response_count,
        "valid_rate": valid_rate,
        "schema_rate": schema_rate,
        "comparison_count": comparison_count,
    }


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    out_dir = (args.out_dir or analysis_dir / "visuals").resolve()
    properties = [item.strip() for item in args.properties.split(",") if item.strip()]

    parsed_rows = read_csv(analysis_dir / "parsed_responses.csv")
    kendall_rows = read_csv(analysis_dir / "kendall_tau_summary.csv")
    property_rows = read_csv(analysis_dir / "persona_property_shift_summary.csv")

    # Regenerate every visual each time so the dashboard cannot mix old/new data.
    files = [
        "valid_response_rate.svg",
        "kendall_tau_heatmap.svg",
        "topk_overlap_heatmap.svg",
        "property_shift_heatmap.svg",
        "revenue_property_shift.svg",
    ]

    valid_response_rate_chart(parsed_rows, out_dir / files[0])
    kendall_heatmap(kendall_rows, out_dir / files[1])
    topk_overlap_heatmap(kendall_rows, out_dir / files[2])
    property_shift_heatmap(property_rows, properties, args.shift_metric, out_dir / files[3])
    revenue_bar_chart(property_rows, out_dir / files[4])
    html_dashboard(out_dir, files, build_summary(parsed_rows, kendall_rows))

    print(f"Wrote {len(files)} SVG charts and index.html to {out_dir}")


if __name__ == "__main__":
    main()
