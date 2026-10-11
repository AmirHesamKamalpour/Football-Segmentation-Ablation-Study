from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image as PILImage
    RESAMPLE_BILINEAR = PILImage.Resampling.BILINEAR
    RESAMPLE_NEAREST = PILImage.Resampling.NEAREST
except AttributeError:  # Pillow < 9.1
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


UNIFIED_CLASS_NAMES = [
    "background",
    "player",
    "goalkeeper",
    "referee",
    "ball",
    "goal_bar",
    "advertisement",
    "audience",
    "staff",
]


# Original dataset RGB palettes.
DATASET1_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Advertisement": (237, 34, 236),
    "Field": (201, 158, 74),
    "Football": (96, 32, 192),
    "Goal Bar": (89, 134, 179),
    "Goalkeeper 1": (153, 223, 219),
    "Goalkeeper 2": (255, 106, 77),
    "Referee": (22, 100, 252),
    "Spectators": (143, 182, 45),
    "Team 1": (38, 198, 129),
    "Team 2": (27, 154, 218),
    "Background": (0, 0, 0),
}

DATASET2_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Ground": (137, 126, 126),
    "Team A": (255, 160, 1),
    "Team B": (254, 233, 3),
    "Goalkeeper A": (255, 159, 0),
    "Goalkeeper B": (255, 235, 0),
    "Referee": (238, 171, 171),
    "Ball": (201, 19, 223),
    "Goal Bar": (255, 0, 29),
    "Advertisement": (27, 71, 151),
    "Audience": (111, 48, 253),
    "Coaches & Officials": (0, 0, 0),
}


# Unified label mapping requested in the assignment.
DATASET1_LABEL_TO_ID: Dict[str, int] = {
    "Background": 0,
    "Field": 0,
    "Team 1": 1,
    "Team 2": 1,
    "Goalkeeper 1": 2,
    "Goalkeeper 2": 2,
    "Referee": 3,
    "Football": 4,
    "Goal Bar": 5,
    "Advertisement": 6,
    "Spectators": 7,
}

DATASET2_LABEL_TO_ID: Dict[str, int] = {
    "Ground": 0,
    "Team A": 1,
    "Team B": 1,
    "Goalkeeper A": 2,
    "Goalkeeper B": 2,
    "Referee": 3,
    "Ball": 4,
    "Goal Bar": 5,
    "Advertisement": 6,
    "Audience": 7,
    "Coaches & Officials": 8,
}


# Default visualization palette for unified classes.
UNIFIED_ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (38, 198, 129),
    2: (255, 106, 77),
    3: (22, 100, 252),
    4: (96, 32, 192),
    5: (89, 134, 179),
    6: (237, 34, 236),
    7: (143, 182, 45),
    8: (255, 255, 255),
}


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    dataset_name: str


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _sorted_images(directory: Path, glob_pattern: str = "*") -> List[Path]:
    if not directory.exists():
        return []
    paths = [p for p in directory.rglob(glob_pattern) if p.is_file() and _is_image(p)]
    return sorted(paths, key=lambda p: str(p).lower())


def collect_dataset1_pairs(image_dir: str | Path, mask_dir: str | Path, image_glob: str = "*.png") -> List[SegmentationSample]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    pairs: List[SegmentationSample] = []

    for image_path in _sorted_images(image_dir, image_glob):
        # Dataset 1 uses the same file name for image and mask.
        candidate = mask_dir / image_path.name
        if candidate.exists():
            pairs.append(SegmentationSample(image_path=image_path, mask_path=candidate, dataset_name="dataset1"))

    return pairs


def collect_dataset2_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_glob: str = "*.jpg",
    mask_suffix: str = "___fuse.jpg",
) -> List[SegmentationSample]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    pairs: List[SegmentationSample] = []

    for image_path in _sorted_images(image_dir, image_glob):
        # Fuse masks are stored as "<original_image_name>___fuse.jpg".
        if image_path.name.endswith(mask_suffix):
            continue

        direct_mask = mask_dir / f"{image_path.name}{mask_suffix}"
        if direct_mask.exists():
            pairs.append(SegmentationSample(image_path=image_path, mask_path=direct_mask, dataset_name="dataset2"))
            continue

        # Fallback for unusual nesting: search by exact generated name.
        found = list(mask_dir.rglob(f"{image_path.name}{mask_suffix}")) if mask_dir.exists() else []
        if found:
            pairs.append(SegmentationSample(image_path=image_path, mask_path=found[0], dataset_name="dataset2"))

    return pairs


def collect_all_pairs(cfg: dict) -> List[SegmentationSample]:
    data_cfg = cfg["data"]
    d1 = data_cfg["dataset1"]
    d2 = data_cfg["dataset2"]

    pairs = []
    pairs.extend(collect_dataset1_pairs(d1["image_dir"], d1["mask_dir"], d1.get("image_glob", "*.png")))
    pairs.extend(
        collect_dataset2_pairs(
            d2["image_dir"],
            d2["mask_dir"],
            d2.get("image_glob", "*.jpg"),
            d2.get("mask_suffix", "___fuse.jpg"),
        )
    )

    if not pairs:
        raise FileNotFoundError(
            "No image-mask pairs found. Check config paths under data.dataset1 and data.dataset2."
        )
    return pairs


def balanced_train_val_split(
    samples: Sequence[SegmentationSample],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[SegmentationSample], List[SegmentationSample]]:
    """Split each dataset separately so validation contains both datasets."""
    rng = np.random.default_rng(seed)
    train, val = [], []

    by_dataset: Dict[str, List[SegmentationSample]] = {}
    for sample in samples:
        by_dataset.setdefault(sample.dataset_name, []).append(sample)

    for dataset_name, group in by_dataset.items():
        indices = np.arange(len(group))
        rng.shuffle(indices)

        n_val = int(round(len(group) * val_ratio))
        if len(group) > 1:
            n_val = max(1, n_val)
        else:
            n_val = 0

        val_idx = set(indices[:n_val].tolist())
        for i, sample in enumerate(group):
            if i in val_idx:
                val.append(sample)
            else:
                train.append(sample)

    return train, val


def _palette_for_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_name == "dataset1":
        colors = DATASET1_COLORS
        label_to_id = DATASET1_LABEL_TO_ID
    elif dataset_name == "dataset2":
        colors = DATASET2_COLORS
        label_to_id = DATASET2_LABEL_TO_ID
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    rgb_values = []
    class_ids = []

    for label_name, rgb in colors.items():
        if label_name in label_to_id:
            rgb_values.append(rgb)
            class_ids.append(label_to_id[label_name])

    # IMPORTANT: int32 avoids overflow during squared distance calculation.
    return np.asarray(rgb_values, dtype=np.int32), np.asarray(class_ids, dtype=np.int64)


def rgb_mask_to_label_ids(mask_rgb: np.ndarray, dataset_name: str, ignore_index: int = 255) -> np.ndarray:
    """Convert RGB/RGBA color mask to unified class-ID mask.

    Each pixel is mapped to the nearest known palette color for the dataset.
    int32 is required because squared RGB differences can overflow int16.
    """
    if mask_rgb.ndim != 3:
        raise ValueError(f"Expected HxWxC mask array, got shape={mask_rgb.shape}")

    if mask_rgb.shape[2] == 4:
        mask_rgb = mask_rgb[:, :, :3]
    elif mask_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3 or 4 mask channels, got {mask_rgb.shape[2]}")

    palette_rgb, palette_ids = _palette_for_dataset(dataset_name)

    pixels = mask_rgb.astype(np.int32).reshape(-1, 3)

    distances = ((pixels[:, None, :] - palette_rgb[None, :, :]) ** 2).sum(axis=2)

    nearest = distances.argmin(axis=1)
    labels = palette_ids[nearest].reshape(mask_rgb.shape[:2])

    return labels.astype(np.int64)


class FootballSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = False,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        ignore_index: int = 255,
    ) -> None:
        self.samples = list(samples)
        self.image_size = tuple(image_size)
        self.normalize = normalize
        self.mean = torch.tensor(mean or [0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std or [0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path)

        # Width-height order for PIL.
        size_wh = (self.image_size[1], self.image_size[0])
        image = image.resize(size_wh, RESAMPLE_BILINEAR)
        mask = mask.resize(size_wh, RESAMPLE_NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask)
        label_np = rgb_mask_to_label_ids(mask_np, sample.dataset_name, ignore_index=self.ignore_index)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()
        mask_tensor = torch.from_numpy(label_np).long()

        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "dataset_name": sample.dataset_name,
        }


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, List[SegmentationSample], List[SegmentationSample]]:
    samples = collect_all_pairs(cfg)
    train_samples, val_samples = balanced_train_val_split(
        samples,
        val_ratio=cfg["data"].get("val_ratio", 0.10),
        seed=cfg["project"].get("seed", 42),
    )

    image_size = tuple(cfg["data"].get("image_size", [256, 256]))
    norm_cfg = cfg["data"].get("normalize", {})
    train_ds = FootballSegmentationDataset(
        train_samples,
        image_size=image_size,
        normalize=norm_cfg.get("enabled", False),
        mean=norm_cfg.get("mean"),
        std=norm_cfg.get("std"),
        ignore_index=cfg["data"].get("ignore_index", 255),
    )
    val_ds = FootballSegmentationDataset(
        val_samples,
        image_size=image_size,
        normalize=norm_cfg.get("enabled", False),
        mean=norm_cfg.get("mean"),
        std=norm_cfg.get("std"),
        ignore_index=cfg["data"].get("ignore_index", 255),
    )

    loader_kwargs = dict(
        batch_size=cfg["data"].get("batch_size", 4),
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=cfg["data"].get("pin_memory", True),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_samples, val_samples


@torch.no_grad()
def compute_class_distribution(dataset: FootballSegmentationDataset, num_classes: int = 9) -> dict:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for item in dataset:
        mask = item["mask"]
        valid = (mask >= 0) & (mask < num_classes)
        bincount = torch.bincount(mask[valid].flatten(), minlength=num_classes)
        counts += bincount.cpu()

    total = counts.sum().item()
    percentages = counts.float() / max(total, 1) * 100.0
    return {
        "counts": counts.tolist(),
        "percentages": percentages.tolist(),
        "class_names": UNIFIED_CLASS_NAMES[:num_classes],
    }
