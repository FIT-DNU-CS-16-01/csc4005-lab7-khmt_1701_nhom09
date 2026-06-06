from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SmartCampusDataset
from src.metrics import compute_classification_metrics
from src.models import build_student
from src.utils import file_size_mb, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch student model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--student_model", type=str, default="mobilenet_v2")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SmartCampusDataset(data_dir=args.data_dir, classes=args.classes, img_size=args.img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_student(num_classes=len(args.classes), student_model=args.student_model, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating PyTorch"):
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics.update({
        "checkpoint": args.checkpoint,
        "model_size_mb": file_size_mb(args.checkpoint),
        "num_samples": len(dataset),
        "student_model": args.student_model,
    })
    save_json(metrics, args.output_json)
    print(metrics)


if __name__ == "__main__":
    main()
