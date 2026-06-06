from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import SmartCampusDataset
from src.models import build_student, load_teacher_checkpoint
from src.metrics import compute_classification_metrics
from src.utils import ensure_dir, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation scaffold.")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default="vit_b_16")
    parser.add_argument("--student_model", type=str, default="mobilenet_v2")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--project", type=str, default="csc4005-lab7-compression")
    parser.add_argument("--run_name", type=str, default="kd_student")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def kd_loss(student_logits, teacher_logits, labels, alpha: float, temperature: float):
    ce = F.cross_entropy(student_logits, labels)

    # TODO: Explain this KD term in the report.
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * ce + (1 - alpha) * kd, ce.detach(), kd.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
    return compute_classification_metrics(y_true, y_pred)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(Path("outputs") / args.run_name)

    dataset = SmartCampusDataset(
        data_dir=args.data_dir,
        classes=args.classes,
        img_size=args.img_size,
        augment=True,
    )

    # Scaffold split: students may replace this with stratified split.
    n_total = len(dataset)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    teacher = load_teacher_checkpoint(
        checkpoint_path=args.teacher_checkpoint,
        num_classes=len(args.classes),
        model_name=args.teacher_model,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_student(
        num_classes=len(args.classes),
        student_model=args.student_model,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.project, name=args.run_name, config=vars(args))

    best_val_f1 = -1.0
    best_path = output_dir / "student_best.pt"

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"KD Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss, ce_value, kd_value = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        val_metrics = evaluate(student, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_set),
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        print(row)

        if wandb_run is not None:
            wandb_run.log(row)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "student_model": args.student_model,
                    "classes": args.classes,
                    "args": vars(args),
                },
                best_path,
            )

    summary = {
        "student_checkpoint": str(best_path),
        "best_val_macro_f1": best_val_f1,
        "student_model": args.student_model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "alpha": args.alpha,
        "temperature": args.temperature,
    }
    save_json(summary, output_dir / "kd_summary.json")

    if wandb_run is not None:
        wandb_run.finish()

    print(summary)


if __name__ == "__main__":
    main()
