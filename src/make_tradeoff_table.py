from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_json, percent_change


def parse_args():
    parser = argparse.ArgumentParser(description="Create trade-off table.")
    parser.add_argument("--baseline_eval", type=str, required=True)
    parser.add_argument("--compressed_eval", type=str, required=True)
    parser.add_argument("--baseline_benchmark", type=str, required=True)
    parser.add_argument("--compressed_benchmark", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def pick_bs1_latency(csv_path: str, batch_size: int):
    df = pd.read_csv(csv_path)
    row = df[df["batch_size"] == batch_size].iloc[0]
    return float(row["mean_latency_ms"]), float(row["throughput_img_per_sec"]), float(row["model_size_mb"])


def main():
    args = parse_args()
    base_eval = load_json(args.baseline_eval)
    comp_eval = load_json(args.compressed_eval)

    base_latency, base_thr, base_size = pick_bs1_latency(args.baseline_benchmark, args.batch_size)
    comp_latency, comp_thr, comp_size = pick_bs1_latency(args.compressed_benchmark, args.batch_size)

    rows = [
        {
            "model": "baseline",
            "accuracy": base_eval.get("accuracy"),
            "macro_f1": base_eval.get("macro_f1"),
            "mean_latency_ms": base_latency,
            "throughput_img_per_sec": base_thr,
            "model_size_mb": base_size,
        },
        {
            "model": "compressed",
            "accuracy": comp_eval.get("accuracy"),
            "macro_f1": comp_eval.get("macro_f1"),
            "mean_latency_ms": comp_latency,
            "throughput_img_per_sec": comp_thr,
            "model_size_mb": comp_size,
        },
    ]

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "accuracy_change_percent": percent_change(rows[0]["accuracy"], rows[1]["accuracy"]),
        "macro_f1_change_percent": percent_change(rows[0]["macro_f1"], rows[1]["macro_f1"]),
        "latency_change_percent": percent_change(base_latency, comp_latency),
        "throughput_change_percent": percent_change(base_thr, comp_thr),
        "size_change_percent": percent_change(base_size, comp_size),
    }

    md = df.to_markdown(index=False)
    md += "\n\n## Change summary\n\n"
    for k, v in summary.items():
        md += f"- {k}: {v:.2f}%\n"

    Path(args.output_md).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
