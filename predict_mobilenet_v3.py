import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}



DEFAULT_INPUT = r"CCPLD/val"
DEFAULT_OUTPUT = r"predictions.csv"
DEFAULT_JSON_OUTPUT = r""
DEFAULT_DEVICE = "auto"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_TOP_K = 3
DEFAULT_SAVE_PROBABILITIES = False

CLASS_NAMES = [
    "Apple black rot",
    "Apple healthy",
    "Apple scab",
    "Cedar-apple rust",
    "Common corn rust",
    "Corn gray leaf spot",
    "Corn healthy",
    "Grape black measles",
    "Grape black rot",
    "Grape healthy",
    "Grape isariopsis leaf spot",
    "Northern corn leaf blight",
    "Potato early blight",
    "Potato healthy",
    "Potato late blight",
    "Tomato early blight",
    "Tomato healthy",
    "Tomato late blight",
    "Tomato septoria leaf spot",
]


@dataclass
class PredictConfig:
    image_size: int = 224
    classifier_dropout: float = 0.2
    teacher_dim: int = 512


class StudentClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, PredictConfig.teacher_dim),
            nn.LayerNorm(PredictConfig.teacher_dim),
            nn.GELU(),
        )
        self.align_head = nn.Linear(PredictConfig.teacher_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.classifier(features)


class ImagePredictionDataset(Dataset):
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
        if self.root is not None:
            relative_path = path.relative_to(self.root).as_posix()
        else:
            relative_path = path.name
        return tensor, str(path), relative_path


def choose_device(device_text: str) -> torch.device:
    device_text = device_text.strip().lower()
    if device_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_text == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_text)


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


def build_model(num_classes: int, cfg: PredictConfig) -> StudentClassifier:
    mobilenet = tv_models.mobilenet_v3_large(weights=None)
    mobilenet.classifier = nn.Identity()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
        features = mobilenet(dummy)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        feature_dim = int(features.shape[1])
    return StudentClassifier(mobilenet, feature_dim, num_classes, cfg.classifier_dropout)


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model(checkpoint_path: Path, device: torch.device, cfg: PredictConfig) -> nn.Module:
    model = build_model(num_classes=len(CLASS_NAMES), cfg=cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    state = normalize_state_dict_keys(checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_thresholds(threshold_path: Path, default_threshold: float) -> np.ndarray:
    if threshold_path.exists():
        thresholds = np.load(threshold_path).astype(np.float32)
        if thresholds.shape != (len(CLASS_NAMES),):
            raise ValueError(
                f"Threshold shape mismatch: expected {(len(CLASS_NAMES),)}, got {tuple(thresholds.shape)}"
            )
        return thresholds
    return np.full(len(CLASS_NAMES), default_threshold, dtype=np.float32)


def collect_image_paths(input_path: Path) -> tuple[List[Path], Path | None]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path], None
    if input_path.is_dir():
        image_paths = sorted(
            [path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.as_posix().lower(),
        )
        if not image_paths:
            raise RuntimeError(f"No images found under: {input_path}")
        return image_paths, input_path
    raise FileNotFoundError(f"Input path not found: {input_path}")


def topk(probabilities: np.ndarray, top_k: int) -> List[Dict[str, object]]:
    indices = np.argsort(-probabilities)[: max(1, min(top_k, len(CLASS_NAMES)))]
    return [
        {
            "rank": rank + 1,
            "class_index": int(index),
            "class_name": CLASS_NAMES[int(index)],
            "probability": float(probabilities[int(index)]),
        }
        for rank, index in enumerate(indices)
    ]


def format_prediction(
    image_path: str,
    relative_path: str,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    top_k: int,
    save_probabilities: bool,
) -> Dict[str, object]:
    predicted_indices = np.where(probabilities >= thresholds)[0].tolist()
    predicted_labels = [CLASS_NAMES[index] for index in predicted_indices]
    top_predictions = topk(probabilities, top_k)
    row: Dict[str, object] = {
        "image_path": image_path,
        "relative_path": relative_path,
        "predicted_labels": "; ".join(predicted_labels),
        "num_predicted_labels": len(predicted_labels),
        "top1_label": top_predictions[0]["class_name"],
        "top1_probability": top_predictions[0]["probability"],
        "topk_labels": "; ".join(item["class_name"] for item in top_predictions),
        "topk_probabilities": "; ".join(f"{item['probability']:.8f}" for item in top_predictions),
    }
    if save_probabilities:
        for class_name, probability in zip(CLASS_NAMES, probabilities):
            row[f"prob_{class_name}"] = float(probability)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.inference_mode()
def predict(
    model: nn.Module,
    image_paths: Sequence[Path],
    root: Path | None,
    transform,
    thresholds: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    top_k: int,
    save_probabilities: bool,
) -> List[Dict[str, object]]:
    dataset = ImagePredictionDataset(image_paths, root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: List[Dict[str, object]] = []
    for images, paths, relative_paths in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        for path_text, relative_path, probs in zip(paths, relative_paths, probabilities):
            rows.append(
                format_prediction(
                    image_path=path_text,
                    relative_path=relative_path,
                    probabilities=probs,
                    thresholds=thresholds,
                    top_k=top_k,
                    save_probabilities=save_probabilities,
                )
            )
    return rows


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Predict multi-label plant leaf diseases with the distilled MobileNetV3-Large model. "
            "The input can be a single image or a folder of images."
        )
    )
    parser.add_argument("--input", default=None, help="Image path or folder path. If omitted, DEFAULT_INPUT is used.")
    parser.add_argument("--checkpoint", default=str(script_dir / "weight" / "mobilenet_v3.pth"))
    parser.add_argument("--thresholds", default=str(script_dir / "weight" / "best_thresholds.npy"))
    parser.add_argument("--output", default=None, help="CSV path for predictions. If omitted, DEFAULT_OUTPUT is used.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path. If omitted, DEFAULT_JSON_OUTPUT is used.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N. If omitted, DEFAULT_DEVICE is used.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--save-probabilities", action="store_true", help="Save probabilities for all 19 classes.")
    return parser.parse_args()


def resolve_user_path(path_text: str, script_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (script_dir / path).resolve()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_text = args.input if args.input else DEFAULT_INPUT
    output_text = args.output if args.output is not None else DEFAULT_OUTPUT
    json_output_text = args.json_output if args.json_output is not None else DEFAULT_JSON_OUTPUT
    device_text = args.device if args.device is not None else DEFAULT_DEVICE
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    num_workers = args.num_workers if args.num_workers is not None else DEFAULT_NUM_WORKERS
    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    save_probabilities = args.save_probabilities or DEFAULT_SAVE_PROBABILITIES

    if not input_text:
        raise ValueError("No input was provided. Set DEFAULT_INPUT near the top of this script or pass --input.")

    input_path = resolve_user_path(input_text, script_dir)
    checkpoint_path = resolve_user_path(args.checkpoint, script_dir)
    threshold_path = resolve_user_path(args.thresholds, script_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(device_text)
    if device.type == "cpu" and args.cpu_threads is not None and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    cfg = PredictConfig()
    model = load_model(checkpoint_path, device, cfg)
    transform = build_transform(cfg.image_size)
    thresholds = load_thresholds(threshold_path, args.default_threshold)
    image_paths, root = collect_image_paths(input_path)

    rows = predict(
        model=model,
        image_paths=image_paths,
        root=root,
        transform=transform,
        thresholds=thresholds,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        save_probabilities=save_probabilities,
    )

    if input_path.is_file():
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    output_path = resolve_user_path(output_text, script_dir) if output_text else None
    if output_path is not None:
        write_csv(output_path, rows)
        print(f"Saved CSV predictions to: {output_path}")
    if json_output_text:
        json_path = resolve_user_path(json_output_text, script_dir)
        write_json(json_path, rows)
        print(f"Saved JSON predictions to: {json_path}")

    print(f"Images processed: {len(rows)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Thresholds: {threshold_path if threshold_path.exists() else f'default={args.default_threshold}'}")


if __name__ == "__main__":
    main()
