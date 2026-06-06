from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transform(img_size: int = 224, augment: bool = False):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SmartCampusDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        classes: Iterable[str] | None = None,
        img_size: int = 224,
        augment: bool = False,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.root = Path(data_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root}")

        available = {p.name.lower(): p for p in self.root.iterdir() if p.is_dir()}
        selected = [c.lower() for c in classes] if classes else sorted(available)

        missing = [c for c in selected if c not in available]
        if missing:
            raise ValueError(f"Missing class folders: {missing}. Available: {sorted(available)}")

        self.classes = selected
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        samples = []
        for class_name in self.classes:
            folder = available[class_name]
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((p, self.class_to_idx[class_name]))

        rng = random.Random(seed)
        rng.shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]

        if not samples:
            raise RuntimeError("No images found.")

        self.samples = samples
        self.transform = build_transform(img_size=img_size, augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
