from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SmartCampusDataset
from src.metrics import compute_classification_metrics
from src.runtime import create_onnx_session, run_onnx
from src.utils import file_size_mb, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model.")
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SmartCampusDataset(
        data_dir=args.data_dir,
        classes=args.classes,
        img_size=args.img_size,
        augment=False,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    session = create_onnx_session(args.onnx_path)

    y_true, y_pred = [], []
    for images, labels in tqdm(loader, desc="Evaluating ONNX"):
        logits = run_onnx(session, images.numpy().astype(np.float32))
        preds = np.argmax(logits, axis=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics.update({
        "onnx_path": args.onnx_path,
        "model_size_mb": file_size_mb(args.onnx_path),
        "num_samples": len(dataset),
        "classes": dataset.classes,
    })
    save_json(metrics, args.output_json)
    print(metrics)


if __name__ == "__main__":
    main()
