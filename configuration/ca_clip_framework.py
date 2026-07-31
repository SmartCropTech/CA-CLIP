import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import clip
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


EXPERIMENTS = {
    "clip_adapter_vit_b16": {
        "backend": "openai_clip_adapter",
        "model_name": "../weight/ViT-B-16.pt",
        "description": "CLIP-Adapter ViT-B/16 baseline enhanced with CA-gating prompt modulation.",
    },
}


@dataclass
class Config:
    # CLIP-Adapter ViT-B/16 with class-aware prompt gating.
    run_all_experiments: bool = False
    experiments: Tuple[str, ...] = ("clip_adapter_vit_b16",)

    dataset_split_dir: str = "../Dataset_split_7_3"
    prompt_json_path: str = "../prompt_bank/disease_prompt_bank.json"
    prompt_fields: Tuple[str, ...] = ("visible_symptom",)
    device: str = "cuda:0"
    random_seed: int = 10
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    # Windows spawn workers reload torch/clip in each subprocess and can exhaust
    # the page file during final validation/test. Use 0 by default; override on
    # Linux servers with --num-workers when enough RAM is available.
    num_workers: int = 0
    use_amp: bool = True
    base_threshold: float = 0.5
    threshold_search_steps: int = 19
    use_per_class_threshold: bool = True
    patience: int = 20
    grad_clip: float = 1.0
    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    hidden_dim: int = 512
    dropout: float = 0.2
    finetune_mode: str = "visual_last_blocks"  # frozen, visual_last_blocks, all
    unfreeze_visual_blocks: int = 2
    adapter_reduction: int = 4
    adapter_ratio: float = 0.2
    ca_projection_dim: int = 512
    ca_fusion_dropout: float = 0.1
    ca_text_modulation_dropout: float = 0.1
    prompt_attention_temperature: float = 1.0
    gate_sparse_lambda: float = 0.005
    gate_conflict_lambda: float = 0.01
    alignment_loss_lambda: float = 0.1
    pretrained_cache_dir: str = "../pretrained_vlm_weights/hf_cache"
    use_strong_train_augmentation: bool = True
    augmentation_crop_scale_min: float = 0.92
    augmentation_crop_ratio_min: float = 0.95
    augmentation_crop_ratio_max: float = 1.05
    augmentation_horizontal_flip_p: float = 0.5
    augmentation_vertical_flip_p: float = 0.0
    augmentation_affine_p: float = 0.25
    augmentation_rotation_degrees: float = 6.0
    augmentation_translate: float = 0.02
    augmentation_scale_min: float = 0.98
    augmentation_scale_max: float = 1.02
    augmentation_color_jitter_p: float = 0.25
    augmentation_brightness: float = 0.06
    augmentation_contrast: float = 0.06
    augmentation_saturation: float = 0.04
    augmentation_hue: float = 0.005
    augmentation_blur_p: float = 0.0
    output_root: str = "../outputs/ca_clip_p3"


HOST_CLASS_NAMES = {
    "Apple": {"Apple black rot", "Apple healthy", "Apple scab", "Cedar-apple rust"},
    "Corn": {"Common corn rust", "Corn gray leaf spot", "Corn healthy", "Northern corn leaf blight"},
    "Grape": {"Grape black measles", "Grape black rot", "Grape healthy", "Grape isariopsis leaf spot"},
    "Potato": {"Potato early blight", "Potato healthy", "Potato late blight"},
    "Tomato": {"Tomato early blight", "Tomato healthy", "Tomato late blight", "Tomato septoria leaf spot"},
}


class BalancedFocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        return ((1.0 - pt).clamp(min=1e-6).pow(self.gamma) * bce).mean()


def validate_torch_numpy_bridge() -> None:
    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32))
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        raise RuntimeError(
            f"PyTorch cannot use NumPy {np.__version__}. Install a NumPy 1.x-compatible build, e.g. python -m pip install \"numpy<2\"."
        ) from exc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_class_names(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prompt_bank(path: Path, class_names: List[str], prompt_fields: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompt_bank = {}
    for class_name in class_names:
        entry = data.get(class_name)
        if entry is None:
            raise KeyError(f"Prompt bank missing class: {class_name}")
        if isinstance(entry, list):
            prompts = entry
        elif isinstance(entry, dict):
            if prompt_fields is None:
                prompts = entry.get("all") or [p for values in entry.values() if isinstance(values, list) for p in values]
            else:
                prompts = []
                missing_fields = []
                for field in prompt_fields:
                    values = entry.get(field)
                    if not values:
                        missing_fields.append(field)
                        continue
                    if isinstance(values, list):
                        prompts.extend(values)
                    else:
                        prompts.append(values)
                if missing_fields:
                    raise KeyError(f"Prompt fields missing for class {class_name}: {missing_fields}")
        else:
            raise TypeError(f"Unsupported prompt entry for {class_name}: {type(entry)}")
        prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not prompts:
            raise ValueError(f"No valid prompts found for class: {class_name}")
        prompt_bank[class_name] = prompts
    return prompt_bank


def read_onehot_csv(path: Path, split_dir: Path, class_names: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [class_name for class_name in class_names if class_name not in df.columns]
    if missing:
        raise ValueError(f"Missing class columns in {path}: {missing}")
    if "image_path" not in df.columns:
        raise ValueError(f"CSV must contain image_path column: {path}")
    df["label_multi"] = df[class_names].astype(np.float32).values.tolist()
    df["resolved_image_path"] = df["image_path"].apply(lambda x: str((split_dir / str(x)).resolve()))
    missing_files = [p for p in df["resolved_image_path"].head(20) if not Path(p).exists()]
    if missing_files:
        raise FileNotFoundError(f"Example image paths do not exist: {missing_files[:3]}")
    return df


def compute_pos_weight(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    labels = np.asarray(train_df["label_multi"].tolist(), dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / np.clip(positives, 1.0, None)
    return torch.tensor(np.clip(pos_weight, 1.0, 20.0), dtype=torch.float32, device=device)


def compute_multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    thresholds: np.ndarray = None,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    if thresholds is None:
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = (probs >= np.asarray(thresholds)[None, :]).astype(int)
    metrics = {
        "subset_acc": float(np.mean(np.all(y_pred == y_true, axis=1))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["mAP"] = float(average_precision_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["mAP"] = float("nan")
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["macro_auc"] = float("nan")
    return metrics


def search_best_thresholds(
    logits: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 19,
    base_threshold: float = 0.5,
    use_per_class: bool = True,
) -> np.ndarray:
    probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    num_classes = y_true.shape[1]
    if not use_per_class:
        return np.full(num_classes, base_threshold, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, steps)
    thresholds = np.full(num_classes, base_threshold, dtype=np.float32)
    for class_idx in range(num_classes):
        if len(np.unique(y_true[:, class_idx])) < 2:
            continue
        best_f1 = -1.0
        for candidate in grid:
            pred = (probs[:, class_idx] >= candidate).astype(int)
            score = f1_score(y_true[:, class_idx], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                thresholds[class_idx] = candidate
    return thresholds


def build_per_class_report(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    y_pred = (probs >= thresholds[None, :]).astype(int)
    rows = []
    for idx, class_name in enumerate(class_names):
        row = {
            "class_name": class_name,
            "threshold": float(thresholds[idx]),
            "support": int(y_true[:, idx].sum()),
            "precision": float(precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "recall": float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "f1": float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
        }
        try:
            row["average_precision"] = float(average_precision_score(y_true[:, idx], probs[:, idx]))
        except ValueError:
            row["average_precision"] = float("nan")
        try:
            row["auc"] = float(roc_auc_score(y_true[:, idx], probs[:, idx]))
        except ValueError:
            row["auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_host_class_indices(class_names: List[str]) -> Dict[str, List[int]]:
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    mapped_classes = set().union(*HOST_CLASS_NAMES.values())
    missing = sorted(set(class_names) - mapped_classes)
    if missing:
        raise ValueError(f"Host-class mapping does not match labels. missing={missing}")
    return {
        host: [class_to_index[class_name] for class_name in host_classes if class_name in class_to_index]
        for host, host_classes in HOST_CLASS_NAMES.items()
        if any(class_name in class_to_index for class_name in host_classes)
    }


def apply_single_host_constraint(probs: np.ndarray, class_names: List[str]) -> np.ndarray:
    host_indices = build_host_class_indices(class_names)
    host_scores = {host: float(np.max(probs[indices])) for host, indices in host_indices.items()}
    selected_host = max(host_scores, key=host_scores.get)
    constrained = np.zeros_like(probs)
    selected_indices = host_indices[selected_host]
    constrained[selected_indices] = probs[selected_indices]
    return constrained


def apply_single_host_constraint_batch(probs: np.ndarray, class_names: List[str]) -> np.ndarray:
    return np.vstack([apply_single_host_constraint(row, class_names) for row in probs])


def choose_device(device: str) -> torch.device:
    requested = device.lower().strip()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda:0")
    if requested.startswith("cuda:"):
        return torch.device(requested)
    raise ValueError("device must be auto, cpu, cuda, or cuda:N")


class LeafDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_transform, is_train: bool, cfg: Config):
        self.df = dataframe.reset_index(drop=True)
        self.image_transform = image_transform
        self.is_train = is_train
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["resolved_image_path"]) as source_image:
            image = source_image.convert("RGB")
        image = self.image_transform(image, train=self.is_train)
        label = torch.tensor(row["label_multi"], dtype=torch.float32)
        return image, label


class TorchvisionTransform:
    def __init__(self, normalize, cfg: Config):
        from torchvision import transforms

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(cfg.augmentation_crop_scale_min, 1.0),
                ratio=(cfg.augmentation_crop_ratio_min, cfg.augmentation_crop_ratio_max),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=cfg.augmentation_horizontal_flip_p),
            transforms.RandomVerticalFlip(p=cfg.augmentation_vertical_flip_p),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=cfg.augmentation_rotation_degrees,
                        translate=(cfg.augmentation_translate, cfg.augmentation_translate),
                        scale=(cfg.augmentation_scale_min, cfg.augmentation_scale_max),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                ],
                p=cfg.augmentation_affine_p,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=cfg.augmentation_brightness,
                        contrast=cfg.augmentation_contrast,
                        saturation=cfg.augmentation_saturation,
                        hue=cfg.augmentation_hue,
                    )
                ],
                p=cfg.augmentation_color_jitter_p,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=cfg.augmentation_blur_p),
            transforms.ToTensor(),
            normalize,
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image: Image.Image, train: bool):
        return self.train_transform(image) if train else self.eval_transform(image)


class HFImageTransform:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image: Image.Image, train: bool):
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]


def get_clip_normalize(preprocess):
    from torchvision import transforms

    for item in preprocess.transforms[::-1]:
        if isinstance(item, transforms.Normalize):
            return item
    return transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


class TextImageEncoder(nn.Module):
    feature_dim: int

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text_features(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class OpenAIClipEncoder(TextImageEncoder):
    def __init__(self, model, freeze: bool = True, cfg: Optional[Config] = None):
        super().__init__()
        self.model = model
        self.feature_dim = model.visual.output_dim
        self.freeze = freeze
        self.finetune_mode = "frozen"
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        elif cfg is not None:
            self.finetune_mode = cfg.finetune_mode.lower().strip()
            self.configure_finetuning(cfg)

    def configure_finetuning(self, cfg: Config) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        if self.finetune_mode == "frozen":
            self.freeze = True
            self.model.eval()
            return
        if self.finetune_mode == "all":
            for param in self.model.parameters():
                param.requires_grad = True
            return
        if self.finetune_mode != "visual_last_blocks":
            raise ValueError("cfg.finetune_mode must be 'frozen', 'visual_last_blocks', or 'all'.")
        visual = getattr(self.model, "visual", None)
        for name, param in visual.named_parameters():
            if any(key in name for key in ["ln_post", "proj"]):
                param.requires_grad = True
        blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        if blocks is not None:
            n_blocks = len(blocks)
            n_unfreeze = max(0, min(cfg.unfreeze_visual_blocks, n_blocks))
            for block in blocks[n_blocks - n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model.encode_image(images).float()
        return self.model.encode_image(images).float()

    @torch.no_grad()
    def encode_text_features(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        tokens = clip.tokenize(list(prompts), truncate=True).to(device)
        return self.model.encode_text(tokens).float()


class OpenCLIPEncoder(TextImageEncoder):
    def __init__(self, model, tokenizer, freeze: bool = True, cfg: Optional[Config] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.feature_dim = getattr(getattr(model, "visual", None), "output_dim", None) or model.text_projection.shape[1]
        self.freeze = freeze
        self.finetune_mode = "frozen"
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        elif cfg is not None:
            self.finetune_mode = cfg.finetune_mode.lower().strip()
            self.configure_finetuning(cfg)

    def configure_finetuning(self, cfg: Config) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        if self.finetune_mode == "frozen":
            self.freeze = True
            self.model.eval()
            return

        if self.finetune_mode == "all":
            for param in self.model.parameters():
                param.requires_grad = True
            return

        if self.finetune_mode != "visual_last_blocks":
            raise ValueError("cfg.finetune_mode must be 'frozen', 'visual_last_blocks', or 'all'.")

        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise RuntimeError("OpenCLIP model does not expose a visual encoder.")

        for name, param in visual.named_parameters():
            if any(key in name for key in ["ln_post", "proj"]):
                param.requires_grad = True

        blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        if blocks is not None:
            n_blocks = len(blocks)
            n_unfreeze = max(0, min(cfg.unfreeze_visual_blocks, n_blocks))
            for block in blocks[n_blocks - n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            for name, param in visual.named_parameters():
                if any(key in name for key in ["blocks", "trunk", "stages"]):
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model.encode_image(images).float()
        return self.model.encode_image(images).float()

    @torch.no_grad()
    def encode_text_features(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(device)
        return self.model.encode_text(tokens).float()


class HFEncoder(TextImageEncoder):
    def __init__(self, model, tokenizer, freeze: bool = True, cfg: Optional[Config] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.feature_dim = model.config.vision_config.hidden_size
        self.freeze = freeze
        self.finetune_mode = "frozen"
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        elif cfg is not None:
            self.finetune_mode = cfg.finetune_mode.lower().strip()
            self.configure_finetuning(cfg)

    def configure_finetuning(self, cfg: Config) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        if self.finetune_mode == "frozen":
            self.freeze = True
            self.model.eval()
            return
        if self.finetune_mode == "all":
            for param in self.model.parameters():
                param.requires_grad = True
            return
        if self.finetune_mode != "visual_last_blocks":
            raise ValueError("cfg.finetune_mode must be 'frozen', 'visual_last_blocks', or 'all'.")
        vision = getattr(self.model, "vision_model", None)
        if vision is None:
            raise RuntimeError("HF VLM does not expose vision_model for visual_last_blocks fine-tuning.")
        encoder_layers = getattr(getattr(vision, "encoder", None), "layers", None)
        if encoder_layers is None:
            raise RuntimeError("HF VLM vision_model does not expose encoder.layers.")
        n_layers = len(encoder_layers)
        n_unfreeze = max(0, min(cfg.unfreeze_visual_blocks, n_layers))
        for layer in encoder_layers[n_layers - n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        for module_name in ["post_layernorm", "head"]:
            module = getattr(vision, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model.get_image_features(pixel_values=images).float()
        return self.model.get_image_features(pixel_values=images).float()

    @torch.no_grad()
    def encode_text_features(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt").to(device)
        return self.model.get_text_features(**inputs).float()


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, ratio: float = 0.2):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.ratio = ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.ratio * self.net(features)


class PromptClassAwareResidualGate(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, img_proj: torch.Tensor, txt_proto: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_proj.ndim not in (2, 3) or txt_proto.ndim != 2:
            raise ValueError("img_proj must be [B, D] or [B, C, D], and txt_proto must be [C, D].")
        if img_proj.shape[-1] != txt_proto.shape[-1]:
            raise ValueError("Image and text projection dimensions must match.")

        bsz = img_proj.shape[0]
        num_classes = txt_proto.shape[0]
        if img_proj.ndim == 2:
            img_expand = img_proj.unsqueeze(1).expand(bsz, num_classes, -1)
        else:
            if img_proj.shape[1] != num_classes:
                raise ValueError("Class-conditioned img_proj must have shape [B, C, D].")
            img_expand = img_proj

        txt_expand = txt_proto.unsqueeze(0).expand(bsz, num_classes, -1)
        diff = torch.abs(img_expand - txt_expand)
        gate_input = torch.cat([img_expand, txt_expand, diff, img_expand * txt_expand], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = self.out_norm(img_expand + gate * (txt_expand - img_expand))
        return fused, gate


class NativeAlignmentClassifier(nn.Module):
    def __init__(
        self,
        encoder: TextImageEncoder,
        prompt_features: torch.Tensor,
        cfg: Config,
        prompt_mask: Optional[torch.Tensor] = None,
        use_adapter: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("prompt_features", prompt_features.float())
        if prompt_mask is None:
            prompt_mask = torch.ones(prompt_features.shape[:2], dtype=torch.bool) if prompt_features.ndim == 3 else torch.ones(prompt_features.shape[0], dtype=torch.bool)
        self.register_buffer("prompt_mask", prompt_mask.bool())
        feature_dim = prompt_features.shape[-1]
        projection_dim = cfg.ca_projection_dim
        self.prompt_attention_temperature = cfg.prompt_attention_temperature
        self.adapter = ResidualAdapter(feature_dim, cfg.adapter_reduction, cfg.adapter_ratio) if use_adapter else None
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_modulator = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.ca_text_modulation_dropout),
            nn.Linear(projection_dim * 2, projection_dim * 2),
        )
        self.modulation_norm = nn.LayerNorm(projection_dim)
        self.fusion_gate = PromptClassAwareResidualGate(projection_dim, dropout=cfg.ca_fusion_dropout)
        self.class_bias = nn.Parameter(torch.zeros(prompt_features.shape[0]))
        self.ca_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.align_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.last_aux = None

    def get_logit_scale(self, device: torch.device):
        model = getattr(self.encoder, "model", None)
        logit_scale = torch.tensor(1.0, device=device)
        if model is not None and hasattr(model, "logit_scale"):
            raw_scale = model.logit_scale
            logit_scale = raw_scale.exp() if raw_scale.ndim == 0 else raw_scale
        return logit_scale

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder.encode_image_features(images)
        if self.adapter is not None:
            image_features = self.adapter(image_features)
        img_proj = self.image_projection(image_features.float())
        raw_prompt_features = self.prompt_features.to(img_proj.device)
        if raw_prompt_features.ndim == 2:
            raw_prompt_features = raw_prompt_features.unsqueeze(1)
        prompt_mask = self.prompt_mask.to(img_proj.device)
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(1)

        num_classes, num_prompts, _ = raw_prompt_features.shape
        txt_proj = self.text_projection(raw_prompt_features)
        flat_txt = txt_proj.reshape(num_classes * num_prompts, -1)

        gamma_beta = self.text_modulator(txt_proj)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        img_prompt = self.modulation_norm(
            img_proj[:, None, None, :] * gamma.unsqueeze(0) + beta.unsqueeze(0)
        )
        flat_img_prompt = img_prompt.reshape(img_proj.shape[0], num_classes * num_prompts, -1)

        fused, gate = self.fusion_gate(flat_img_prompt, flat_txt)
        fused = fused.reshape(img_proj.shape[0], num_classes, num_prompts, -1)
        gate = gate.reshape(img_proj.shape[0], num_classes, num_prompts, -1)
        fused_norm = F.normalize(fused, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        prompt_logits = self.ca_logit_scale.exp().clamp(max=100.0) * (fused_norm * txt_norm.unsqueeze(0)).sum(dim=-1)
        masked_prompt_logits = prompt_logits.masked_fill(~prompt_mask.unsqueeze(0), -1e4)
        prompt_weights = F.softmax(masked_prompt_logits / max(self.prompt_attention_temperature, 1e-6), dim=-1)
        prompt_weights = prompt_weights * prompt_mask.unsqueeze(0).float()
        prompt_weights = prompt_weights / prompt_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = (prompt_weights * prompt_logits).sum(dim=-1)
        logits = logits + self.class_bias
        align_prompt_logits = self.align_logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bd,ckd->bck",
            F.normalize(img_proj, dim=-1),
            txt_norm,
        )
        align_logits = (prompt_weights * align_prompt_logits).sum(dim=-1)
        self.last_aux = {
            "gate": gate,
            "img_proj": img_proj,
            "txt_proj": txt_proj,
            "prompt_mask": prompt_mask,
            "prompt_weights": prompt_weights,
            "prompt_logits": prompt_logits,
            "align_logits": align_logits,
        }
        return logits


def build_prompt_features(
    encoder: TextImageEncoder,
    prompt_bank: Dict[str, List[str]],
    class_names: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_class_features = []
    max_prompts = max(len(prompt_bank[class_name]) for class_name in class_names)
    for class_name in class_names:
        features = encoder.encode_text_features(prompt_bank[class_name], device)
        features = F.normalize(features, dim=-1)
        per_class_features.append(features.cpu())

    feature_dim = per_class_features[0].shape[-1]
    padded = torch.zeros(len(class_names), max_prompts, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(class_names), max_prompts, dtype=torch.bool)
    for class_idx, features in enumerate(per_class_features):
        num_prompts = features.shape[0]
        padded[class_idx, :num_prompts] = features
        mask[class_idx, :num_prompts] = True
    return padded, mask


def load_experiment(exp_key: str, cfg: Config, base_dir: Path, device: torch.device):
    exp = EXPERIMENTS[exp_key]
    backend = exp["backend"]
    cache_dir = str((base_dir / cfg.pretrained_cache_dir).resolve())
    if backend in {"openai_clip", "openai_clip_adapter"}:
        model_path = (base_dir / exp["model_name"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"OpenAI CLIP local weight is required for offline training: {model_path}. "
                f"Copy the .pt file into CA-CLIP_construct before running {exp_key}."
            )
        model, preprocess = clip.load(str(model_path), device=device)
        if cfg.finetune_mode.lower().strip() != "frozen":
            model.float()
        encoder = OpenAIClipEncoder(model, freeze=False, cfg=cfg)
        transform = TorchvisionTransform(get_clip_normalize(preprocess), cfg)
        return encoder, transform

    if backend == "open_clip":
        model_name = exp["model_name"]
        pretrained = exp.get("pretrained")
        if pretrained is None:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device, cache_dir=cache_dir)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device,
                cache_dir=cache_dir,
            )
        if cfg.finetune_mode.lower().strip() != "frozen":
            model.float()
        tokenizer = open_clip.get_tokenizer(model_name, cache_dir=cache_dir)
        encoder = OpenCLIPEncoder(model, tokenizer, freeze=False, cfg=cfg)
        transform = TorchvisionTransform(get_clip_normalize(preprocess), cfg)
        return encoder, transform

    if backend == "hf_siglip":
        model_name = exp["model_name"]
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        if cfg.finetune_mode.lower().strip() != "frozen":
            model.float()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        encoder = HFEncoder(model, tokenizer, freeze=False, cfg=cfg)
        transform = HFImageTransform(image_processor)
        return encoder, transform

    raise ValueError(f"Unsupported backend: {backend}")


def build_criterion(train_df: pd.DataFrame, class_names: List[str], device: torch.device, cfg: Config):
    pos_weight = compute_pos_weight(train_df, device)
    if cfg.use_focal_loss:
        return BalancedFocalBCEWithLogitsLoss(pos_weight, gamma=cfg.focal_gamma)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_ca_gating_loss(model: nn.Module, logits: torch.Tensor, targets: torch.Tensor, criterion, cfg: Config) -> torch.Tensor:
    cls_loss = criterion(logits, targets)
    aux = getattr(model, "last_aux", None)
    if aux is None:
        return cls_loss
    align_loss = criterion(aux["align_logits"], targets)
    img_proj = F.normalize(aux["img_proj"].float(), dim=-1)
    txt_proj = F.normalize(aux["txt_proj"].float(), dim=-1)
    if txt_proj.ndim == 2:
        disagreement = 1.0 - (img_proj.unsqueeze(1) * txt_proj.unsqueeze(0)).sum(dim=-1)
        valid_prompt = torch.ones_like(disagreement)
    else:
        disagreement = 1.0 - (img_proj[:, None, None, :] * txt_proj.unsqueeze(0)).sum(dim=-1)
        valid_prompt = aux["prompt_mask"].unsqueeze(0).float()
    gate_strength = aux["gate"].float().mean(dim=-1)
    valid_count = valid_prompt.sum().clamp_min(1.0)
    sparse_loss = (gate_strength * valid_prompt).sum() / valid_count
    conflict_loss = (gate_strength * disagreement.detach() * valid_prompt).sum() / valid_count
    reg_loss = cfg.gate_sparse_lambda * sparse_loss + cfg.gate_conflict_lambda * conflict_loss
    return cls_loss + cfg.alignment_loss_lambda * align_loss + reg_loss


def build_optimizer(model: nn.Module, cfg: Config):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("module.encoder."):
            backbone_params.append(param)
            continue
        head_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.backbone_learning_rate})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.learning_rate})
    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def print_trainable_parameters(model: nn.Module) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    encoder_trainable = sum(
        param.numel()
        for name, param in model.named_parameters()
        if (name.startswith("encoder.") or name.startswith("module.encoder.")) and param.requires_grad
    )
    head_trainable = trainable - encoder_trainable
    print(
        f"Trainable parameters: {trainable:,}/{total:,} "
        f"(encoder={encoder_trainable:,}, head={head_trainable:,})"
    )


@torch.no_grad()
def run_inference(model, loader, criterion, device, cfg: Config, desc: str):
    model.eval()
    amp_enabled = cfg.use_amp and device.type == "cuda"
    losses = []
    logits_all = []
    targets_all = []
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = compute_ca_gating_loss(model, logits, targets, criterion, cfg)
        losses.append(loss.item())
        logits_all.append(logits.cpu())
        targets_all.append(targets.cpu())
    return float(np.mean(losses)), torch.cat(logits_all, dim=0), torch.cat(targets_all, dim=0)


class Checkpoint:
    def __init__(self, path: Path, patience: int):
        self.path = path
        self.patience = patience
        self.best_metric = -float("inf")
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, metric: float, loss: float, model: nn.Module):
        improved = metric > self.best_metric or (math.isclose(metric, self.best_metric, rel_tol=1e-6) and loss < self.best_loss)
        if improved:
            self.best_metric = metric
            self.best_loss = loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"[Checkpoint] val macro-F1 improved to {metric:.6f}; saved {self.path}")
        else:
            self.counter += 1
            print(f"[Checkpoint] no improvement. patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one(exp_key: str, cfg: Config, base_dir: Path, device: torch.device):
    seed_everything(cfg.random_seed)
    exp = EXPERIMENTS[exp_key]
    split_dir = (base_dir / cfg.dataset_split_dir).resolve()
    prompt_path = (base_dir / cfg.prompt_json_path).resolve()
    class_names = load_class_names(split_dir / "class_names.txt")
    prompt_bank = load_prompt_bank(prompt_path, class_names, prompt_fields=cfg.prompt_fields)
    train_df = read_onehot_csv(split_dir / "train_labels_onehot.csv", split_dir, class_names)
    val_df = read_onehot_csv(split_dir / "val_labels_onehot.csv", split_dir, class_names)

    out_dir = (base_dir / cfg.output_root / exp_key).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{exp_key}_best.pth"

    print("\n" + "=" * 80)
    print(f"Experiment: {exp_key}")
    print(exp["description"])
    print(f"Output: {out_dir}")
    print("=" * 80)

    encoder, image_transform = load_experiment(exp_key, cfg, base_dir, device)
    prompt_features, prompt_mask = build_prompt_features(encoder, prompt_bank, class_names, device)
    use_adapter = exp["backend"] == "openai_clip_adapter"
    model = NativeAlignmentClassifier(encoder, prompt_features, cfg, prompt_mask=prompt_mask, use_adapter=use_adapter).to(device)
    print_trainable_parameters(model)

    train_dataset = LeafDataset(train_df, image_transform, is_train=True, cfg=cfg)
    val_dataset = LeafDataset(val_df, image_transform, is_train=False, cfg=cfg)
    generator = torch.Generator()
    generator.manual_seed(cfg.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=(cfg.num_workers > 0),
    )
    criterion = build_criterion(train_df, class_names, device, cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    checkpoint = Checkpoint(model_path, cfg.patience)
    history = []

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = asdict(cfg)
        payload["experiment"] = exp_key
        payload["experiment_info"] = exp
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for epoch in range(cfg.num_epochs):
        start = time.time()
        model.train()
        train_losses = []
        train_logits = []
        train_targets = []
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [Train]", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(images)
                loss = compute_ca_gating_loss(model, logits, targets, criterion, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            train_logits.append(logits.detach().cpu())
            train_targets.append(targets.detach().cpu())
        scheduler.step()

        train_logits = torch.cat(train_logits, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_thresholds = search_best_thresholds(
            train_logits,
            train_targets,
            steps=cfg.threshold_search_steps,
            base_threshold=cfg.base_threshold,
            use_per_class=cfg.use_per_class_threshold,
        )
        val_loss, val_logits, val_targets = run_inference(model, val_loader, criterion, device, cfg, f"Epoch {epoch + 1}/{cfg.num_epochs} [Val]")
        val_thresholds = search_best_thresholds(
            val_logits,
            val_targets,
            steps=cfg.threshold_search_steps,
            base_threshold=cfg.base_threshold,
            use_per_class=cfg.use_per_class_threshold,
        )
        train_metrics = compute_multilabel_metrics(train_logits, train_targets, threshold=cfg.base_threshold)
        train_tuned = compute_multilabel_metrics(
            train_logits,
            train_targets,
            threshold=cfg.base_threshold,
            thresholds=train_thresholds,
        )
        val_fixed = compute_multilabel_metrics(val_logits, val_targets, threshold=cfg.base_threshold)
        val_tuned = compute_multilabel_metrics(val_logits, val_targets, threshold=cfg.base_threshold, thresholds=val_thresholds)

        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - start,
            **{f"train_fixed_{key}": value for key, value in train_metrics.items()},
            **{f"train_tuned_{key}": value for key, value in train_tuned.items()},
            **{f"val_fixed_{key}": value for key, value in val_fixed.items()},
            **{f"val_tuned_{key}": value for key, value in val_tuned.items()},
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1:03d} | loss={record['train_loss']:.4f} val_loss={val_loss:.4f} | "
            f"train_fixed_sub_acc={train_metrics['subset_acc']:.4f} "
            f"train_tuned_sub_acc={train_tuned['subset_acc']:.4f} "
            f"train_tuned_macro_f1={train_tuned['macro_f1']:.4f} "
            f"train_mAP={train_metrics['mAP']:.4f} | "
            f"val_fixed_sub_acc={val_fixed['subset_acc']:.4f} val_tuned_sub_acc={val_tuned['subset_acc']:.4f} | "
            f"val_tuned_macro_f1={val_tuned['macro_f1']:.4f} val_mAP={val_fixed['mAP']:.4f}"
        )
        checkpoint.step(val_tuned["macro_f1"], val_loss, model)
        pd.DataFrame(history).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
        if checkpoint.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    val_loss, val_logits, val_targets = run_inference(model, val_loader, criterion, device, cfg, "Final Val")
    thresholds = search_best_thresholds(
        val_logits,
        val_targets,
        steps=cfg.threshold_search_steps,
        base_threshold=cfg.base_threshold,
        use_per_class=cfg.use_per_class_threshold,
    )

    final_val = {
        "val_loss": val_loss,
        "fixed": compute_multilabel_metrics(val_logits, val_targets, threshold=cfg.base_threshold),
        "tuned": compute_multilabel_metrics(val_logits, val_targets, threshold=cfg.base_threshold, thresholds=thresholds),
    }

    val_probs = torch.sigmoid(val_logits.float()).numpy()
    final_val["host_constrained"] = compute_metrics_from_probs(
        apply_single_host_constraint_batch(val_probs, class_names),
        val_targets,
        thresholds,
    )

    np.save(out_dir / "best_thresholds.npy", thresholds)
    with (out_dir / "best_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "thresholds": [float(x) for x in thresholds]}, f, ensure_ascii=False, indent=2)
    with (out_dir / "final_val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_val, f, ensure_ascii=False, indent=2)
    build_per_class_report(val_logits, val_targets, thresholds, class_names).to_csv(out_dir / "val_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": exp_key,
        "val_tuned_subset_acc": final_val["tuned"]["subset_acc"],
        "val_tuned_macro_f1": final_val["tuned"]["macro_f1"],
        "val_mAP": final_val["fixed"]["mAP"],
        "output_dir": str(out_dir),
    }
    print("Final summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def dry_run_one(exp_key: str, cfg: Config, base_dir: Path, device: torch.device):
    seed_everything(cfg.random_seed)
    exp = EXPERIMENTS[exp_key]
    split_dir = (base_dir / cfg.dataset_split_dir).resolve()
    prompt_path = (base_dir / cfg.prompt_json_path).resolve()
    class_names = load_class_names(split_dir / "class_names.txt")
    prompt_bank = load_prompt_bank(prompt_path, class_names, prompt_fields=cfg.prompt_fields)
    train_df = read_onehot_csv(split_dir / "train_labels_onehot.csv", split_dir, class_names)

    print("\n" + "=" * 80)
    print(f"Dry run: {exp_key}")
    print(exp["description"])
    print("=" * 80)

    encoder, image_transform = load_experiment(exp_key, cfg, base_dir, device)
    prompt_features, prompt_mask = build_prompt_features(encoder, prompt_bank, class_names, device)
    use_adapter = exp["backend"] == "openai_clip_adapter"
    model = NativeAlignmentClassifier(encoder, prompt_features, cfg, prompt_mask=prompt_mask, use_adapter=use_adapter).to(device)
    dataset = LeafDataset(train_df, image_transform, is_train=True, cfg=cfg)
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, 2), shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)
    criterion = build_criterion(train_df, class_names, device, cfg)
    model.train()
    with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
        logits = model(images)
        loss = compute_ca_gating_loss(model, logits, targets, criterion, cfg)
    print(f"images={tuple(images.shape)}")
    print(f"prompt_features={tuple(prompt_features.shape)}")
    print(f"prompt_mask={tuple(prompt_mask.shape)}")
    print(f"logits={tuple(logits.shape)}")
    print(f"loss={float(loss):.6f}")
    print(f"finite={bool(torch.isfinite(loss))}")
    return {
        "experiment": exp_key,
        "image_shape": tuple(images.shape),
        "prompt_feature_shape": tuple(prompt_features.shape),
        "prompt_mask_shape": tuple(prompt_mask.shape),
        "logit_shape": tuple(logits.shape),
        "loss": float(loss),
        "finite": bool(torch.isfinite(loss)),
    }


def compute_metrics_from_probs(probs: np.ndarray, targets: torch.Tensor, thresholds: np.ndarray):
    logits = torch.from_numpy(np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1.0)))
    return compute_multilabel_metrics(logits, targets, threshold=0.5, thresholds=thresholds)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP-Adapter ViT-B/16 with CA-gating prompt modulation.")
    parser.add_argument("--dataset-split-dir", type=str, default=None)
    parser.add_argument("--prompt-json-path", type=str, default=None)
    parser.add_argument("--clip-checkpoint", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        choices=[*EXPERIMENTS.keys(), "all"],
        help="Override Config.experiments. Use 'all' for every configured model.",
    )
    parser.add_argument("--run-all", action="store_true", help="Override Config.run_all_experiments=True.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--finetune-mode", choices=["frozen", "visual_last_blocks", "all"], default=None)
    parser.add_argument("--unfreeze-visual-blocks", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--alignment-loss-lambda", type=float, default=None)
    parser.add_argument("--gate-sparse-lambda", type=float, default=None)
    parser.add_argument("--gate-conflict-lambda", type=float, default=None)
    parser.add_argument("--prompt-attention-temperature", type=float, default=None)
    parser.add_argument("--prompt-fields", nargs="+", default=None, help="Prompt bank fields to use. Default is P3: visible_symptom.")
    parser.add_argument("--dry-run", action="store_true", help="Only load each selected model and run one mini-batch.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_torch_numpy_bridge()
    cfg = Config()
    if args.dataset_split_dir is not None:
        cfg.dataset_split_dir = args.dataset_split_dir
    if args.prompt_json_path is not None:
        cfg.prompt_json_path = args.prompt_json_path
    if args.clip_checkpoint is not None:
        EXPERIMENTS["clip_adapter_vit_b16"]["model_name"] = args.clip_checkpoint
    if args.output_root is not None:
        cfg.output_root = args.output_root
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.finetune_mode is not None:
        cfg.finetune_mode = args.finetune_mode
    if args.unfreeze_visual_blocks is not None:
        cfg.unfreeze_visual_blocks = args.unfreeze_visual_blocks
    if args.backbone_lr is not None:
        cfg.backbone_learning_rate = args.backbone_lr
    if args.alignment_loss_lambda is not None:
        cfg.alignment_loss_lambda = args.alignment_loss_lambda
    if args.gate_sparse_lambda is not None:
        cfg.gate_sparse_lambda = args.gate_sparse_lambda
    if args.gate_conflict_lambda is not None:
        cfg.gate_conflict_lambda = args.gate_conflict_lambda
    if args.prompt_attention_temperature is not None:
        cfg.prompt_attention_temperature = args.prompt_attention_temperature
    if args.prompt_fields is not None:
        cfg.prompt_fields = tuple(args.prompt_fields)
    seed_everything(cfg.random_seed)
    base_dir = Path(__file__).resolve().parent
    device = choose_device(cfg.device)
    print(f"Using device: {device}")
    if args.run_all:
        cfg.run_all_experiments = True
    if args.experiments is not None:
        if "all" in args.experiments:
            cfg.run_all_experiments = True
            cfg.experiments = tuple(EXPERIMENTS.keys())
        else:
            cfg.run_all_experiments = False
            cfg.experiments = tuple(args.experiments)

    experiments = list(EXPERIMENTS) if cfg.run_all_experiments else list(cfg.experiments)
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments in Config.experiments: {unknown}. Available: {sorted(EXPERIMENTS)}")
    print(f"Run experiments: {experiments}")
    summaries = []
    for exp_key in experiments:
        if args.dry_run:
            summaries.append(dry_run_one(exp_key, cfg, base_dir, device))
        else:
            summaries.append(train_one(exp_key, cfg, base_dir, device))

    summary_dir = (base_dir / cfg.output_root).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_name = "clip_adapter_vit_b16_ca_gating_p3_dry_run_summary" if args.dry_run else "clip_adapter_vit_b16_ca_gating_p3_summary"
    pd.DataFrame(summaries).to_csv(summary_dir / f"{summary_name}.csv", index=False, encoding="utf-8-sig")
    with (summary_dir / f"{summary_name}.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Standard CLIP-Adapter summary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
