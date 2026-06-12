from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models import build_student


def parse_args():
    parser = argparse.ArgumentParser(description="Export KD student checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--student_model", type=str, default="mobilenet_v2")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_onnx", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_student(
        num_classes=args.num_classes,
        student_model=args.student_model,
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float32)

    output_path = Path(args.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print({
        "checkpoint": str(ckpt_path),
        "output_onnx": str(output_path),
        "student_model": args.student_model,
        "num_classes": args.num_classes,
        "img_size": args.img_size,
        "opset": args.opset,
    })


if __name__ == "__main__":
    main()
