"""Distilled MobileNetV3-Large model loading and multi-label inference."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DistilledMobileNetV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        classifier_dropout: float = 0.2,
        teacher_dim: int = 512,
    ):
        super().__init__()
        backbone = tv_models.mobilenet_v3_large(weights=None)
        backbone.classifier = nn.Identity()
        with torch.no_grad():
            features = backbone(torch.zeros(1, 3, image_size, image_size))
            feature_dim = int(torch.flatten(features, 1).shape[1])

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, teacher_dim),
            nn.LayerNorm(teacher_dim),
            nn.GELU(),
        )
        self.align_head = nn.Linear(teacher_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.classifier(features)


class ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], root: Path | None, transform):
        self.image_paths = list(image_paths)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        relative_path = path.relative_to(self.root).as_posix() if self.root else path.name
        return tensor, str(path), relative_path


def choose_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        return torch.device("cuda:0")
    return torch.device(requested)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model(
    checkpoint_path: Path,
    class_names: Sequence[str],
    device: torch.device,
    image_size: int,
    classifier_dropout: float,
    teacher_dim: int,
) -> nn.Module:
    model = DistilledMobileNetV3(
        num_classes=len(class_names),
        image_size=image_size,
        classifier_dropout=classifier_dropout,
        teacher_dim=teacher_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    model.load_state_dict(normalize_state_dict_keys(checkpoint), strict=True)
    return model.to(device).eval()


def load_thresholds(path: Path, class_count: int, default_threshold: float) -> np.ndarray:
    if not path.is_file():
        return np.full(class_count, default_threshold, dtype=np.float32)
    thresholds = np.load(path).astype(np.float32)
    if thresholds.shape != (class_count,):
        raise ValueError(
            f"Expected {class_count} class thresholds, received {tuple(thresholds.shape)}."
        )
    return thresholds


def collect_images(input_path: Path) -> tuple[List[Path], Path | None]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path], None
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input image or folder not found: {input_path}")
    images = sorted(
        (
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")
    return images, input_path


def format_prediction(
    image_path: str,
    relative_path: str,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
    top_k: int,
    save_probabilities: bool,
) -> Dict[str, object]:
    predicted_indices = np.flatnonzero(probabilities >= thresholds).tolist()
    ranked_indices = np.argsort(-probabilities)[: max(1, min(top_k, len(class_names)))]
    row: Dict[str, object] = {
        "image_path": image_path,
        "relative_path": relative_path,
        "predicted_labels": "; ".join(class_names[index] for index in predicted_indices),
        "num_predicted_labels": len(predicted_indices),
        "top1_label": class_names[int(ranked_indices[0])],
        "top1_probability": float(probabilities[int(ranked_indices[0])]),
        "topk_labels": "; ".join(class_names[int(index)] for index in ranked_indices),
        "topk_probabilities": "; ".join(
            f"{float(probabilities[int(index)]):.8f}" for index in ranked_indices
        ),
    }
    if save_probabilities:
        for class_name, probability in zip(class_names, probabilities):
            row[f"prob_{class_name}"] = float(probability)
    return row


@torch.inference_mode()
def infer(
    model: nn.Module,
    image_paths: Sequence[Path],
    root: Path | None,
    thresholds: np.ndarray,
    class_names: Sequence[str],
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    top_k: int,
    save_probabilities: bool,
) -> List[Dict[str, object]]:
    dataset = ImageDataset(image_paths, root, build_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: List[Dict[str, object]] = []
    for images, paths, relative_paths in loader:
        probabilities = torch.sigmoid(model(images.to(device, non_blocking=True))).cpu().numpy()
        for path, relative_path, probability in zip(paths, relative_paths, probabilities):
            rows.append(
                format_prediction(
                    path,
                    relative_path,
                    probability,
                    thresholds,
                    class_names,
                    top_k,
                    save_probabilities,
                )
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_prediction(
    input_path: Path,
    config_path: Path,
    output_csv: Path | None = None,
    device_override: str | None = None,
    save_probabilities: bool = False,
) -> List[Dict[str, object]]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["mobilenet_v3_distilled"]
    class_names = list(config["class_names"])
    config_dir = config_path.parent

    checkpoint_path = (config_dir / model_config["checkpoint"]).resolve()
    threshold_path = (config_dir / model_config["thresholds"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = choose_device(device_override or model_config["device"])
    image_paths, image_root = collect_images(input_path.resolve())
    thresholds = load_thresholds(
        threshold_path,
        len(class_names),
        float(model_config["default_threshold"]),
    )
    model = load_model(
        checkpoint_path,
        class_names,
        device,
        int(model_config["input_size"]),
        float(model_config["classifier_dropout"]),
        int(model_config["teacher_projection_dimension"]),
    )
    rows = infer(
        model=model,
        image_paths=image_paths,
        root=image_root,
        thresholds=thresholds,
        class_names=class_names,
        device=device,
        image_size=int(model_config["input_size"]),
        batch_size=int(model_config["batch_size"]),
        num_workers=int(model_config["num_workers"]),
        top_k=int(model_config["top_k"]),
        save_probabilities=save_probabilities,
    )
    if output_csv is not None:
        write_csv(output_csv.resolve(), rows)
    return rows


__all__ = ["DistilledMobileNetV3", "run_prediction"]

