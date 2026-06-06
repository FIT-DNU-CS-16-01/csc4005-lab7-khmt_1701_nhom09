from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.runtime import create_onnx_session, run_onnx
from src.utils import file_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one or more ONNX models.")
    parser.add_argument("--onnx_paths", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def measure(session, batch_np, warmup: int, repeat: int):
    for _ in range(warmup):
        _ = run_onnx(session, batch_np)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = run_onnx(session, batch_np)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times


def main():
    args = parse_args()
    if len(args.onnx_paths) != len(args.names):
        raise ValueError("--onnx_paths and --names must have the same length.")

    rows = []
    for name, onnx_path in zip(args.names, args.onnx_paths):
        session = create_onnx_session(onnx_path)
        size_mb = file_size_mb(onnx_path)

        for bs in args.batch_sizes:
            batch = torch.randn(bs, 3, args.img_size, args.img_size).numpy().astype(np.float32)
            times = measure(session, batch, args.warmup, args.repeat)
            mean_latency = float(np.mean(times))
            rows.append({
                "model": name,
                "onnx_path": onnx_path,
                "batch_size": bs,
                "mean_latency_ms": mean_latency,
                "median_latency_ms": float(np.median(times)),
                "p95_latency_ms": float(np.percentile(times, 95)),
                "throughput_img_per_sec": float(bs / (mean_latency / 1000.0)),
                "model_size_mb": size_mb,
            })

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
