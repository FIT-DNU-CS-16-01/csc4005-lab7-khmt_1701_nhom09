from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run KD grid search with W&B tracking.")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--student_model", type=str, default="mobilenet_v2")

    parser.add_argument("--alpha_list", nargs="+", type=float, required=True)
    parser.add_argument("--temperature_list", nargs="+", type=float, required=True)
    parser.add_argument("--epochs_list", nargs="+", type=int, required=True)
    parser.add_argument("--batch_size_list", nargs="+", type=int, default=[16])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--project", type=str, default="csc4005-lab7-compression")
    parser.add_argument("--wandb_group", type=str, default="kd-grid-search")
    parser.add_argument("--tag", type=str, default="grid")

    parser.add_argument("--output_csv", type=str, default="outputs/kd_grid_search_summary.csv")
    parser.add_argument("--output_json", type=str, default="outputs/kd_grid_search_summary.json")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def format_run_name(alpha: float, temperature: float, epochs: int) -> str:
    # Example: alpha=0.2, temperature=1.0, epochs=50 -> A20_T10_E50
    alpha_pct = int(round(alpha * 100))
    temperature_x10 = int(round(temperature * 10))
    return f"A{alpha_pct}_T{temperature_x10}_E{epochs}"


def run_one_experiment(args, alpha: float, temperature: float, epochs: int, batch_size: int) -> dict:
    run_name = format_run_name(alpha, temperature, epochs)

    cmd = [
        sys.executable,
        "-m",
        "src.kd_train_student",
        "--teacher_checkpoint",
        args.teacher_checkpoint,
        "--data_dir",
        args.data_dir,
        "--classes",
        *args.classes,
        "--student_model",
        args.student_model,
        "--alpha",
        str(alpha),
        "--temperature",
        str(temperature),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--img_size",
        str(args.img_size),
        "--seed",
        str(args.seed),
        "--project",
        args.project,
        "--run_name",
        run_name,
        "--wandb_group",
        args.wandb_group,
        "--wandb_tags",
        args.tag,
        "kd",
        "grid",
        "--use_wandb",
    ]

    completed = subprocess.run(cmd, check=False)
    summary_path = Path("outputs") / run_name / "kd_summary.json"

    result = {
        "run_name": run_name,
        "alpha": alpha,
        "temperature": temperature,
        "epochs": epochs,
        "batch_size": batch_size,
        "return_code": completed.returncode,
        "summary_path": str(summary_path),
    }

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result.update(
            {
                "best_val_macro_f1": summary.get("best_val_macro_f1"),
                "student_checkpoint": summary.get("student_checkpoint"),
            }
        )

    return result


def main() -> None:
    args = parse_args()

    combos = list(
        itertools.product(
            args.alpha_list,
            args.temperature_list,
            args.epochs_list,
            args.batch_size_list,
        )
    )

    print(f"Total experiments: {len(combos)}")
    all_results = []

    for idx, (alpha, temperature, epochs, batch_size) in enumerate(combos, start=1):
        print(
            {
                "progress": f"{idx}/{len(combos)}",
                "alpha": alpha,
                "temperature": temperature,
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )

        result = run_one_experiment(args, alpha, temperature, epochs, batch_size)
        all_results.append(result)

        if result["return_code"] != 0 and not args.continue_on_error:
            print({"stopped_on_error": result})
            break

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    if not df.empty and "best_val_macro_f1" in df.columns:
        df = df.sort_values(by=["best_val_macro_f1"], ascending=False, na_position="last")

    df.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
