
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np



# DEFAULT_HEALTHY_DIR = r"example_data/apple/Apple healthy"
# DEFAULT_DISEASE_DIRS = {
#     "Apple scab": r"example_data/apple/Apple scab",
#     "Apple black rot": r"example_data/apple/Apple black rot",
#     "Cedar-apple rust": r"example_data/apple/Cedar-apple rust",
# }
# DEFAULT_OUTPUT_DIR = r"synthetic_co_occurring_leaf_images"
# DEFAULT_NUM_IMAGES = 20
# DEFAULT_LESIONS_PER_DISEASE = 3
# DEFAULT_COMBO = ["Apple scab", "Apple black rot"]
# DEFAULT_SEED = 10


# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# LEAF_LABEL = "Complete leaf"


@dataclass
class LesionTemplate:
    disease: str
    image_patch_bgr: np.ndarray
    alpha: np.ndarray
    source_image: str
    source_json: str


@dataclass
class HealthyLeaf:
    image_path: Path
    json_path: Path
    image_bgr: np.ndarray
    annotation: Dict
    leaf_mask: np.ndarray


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_image_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for path in folder.iterdir():
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def shape_to_mask(image_shape: Tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask
    cv2.fillPoly(mask, [pts], 255)
    return mask


def mask_to_polygon_points(mask: np.ndarray, x_offset: int = 0, y_offset: int = 0, max_points: int = 80) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) > max_points:
        step = max(1, len(approx) // max_points)
        approx = approx[::step]
    return [[float(x + x_offset), float(y + y_offset)] for x, y in approx]


def crop_by_mask(image_bgr: np.ndarray, mask: np.ndarray, pad: int = 8) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_bgr.shape[1], x + w + pad)
    y1 = min(image_bgr.shape[0], y + h + pad)
    patch = image_bgr[y0:y1, x0:x1].copy()
    alpha = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
    if patch.size == 0 or alpha.max() <= 0:
        return None
    return patch, alpha


def soft_alpha(alpha: np.ndarray, blur_sigma: float = 1.2) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(np.maximum(alpha * 0.85, blurred), 0.0, 1.0)


def collect_healthy_leaves(healthy_dir: Path) -> List[HealthyLeaf]:
    leaves: List[HealthyLeaf] = []
    for json_path in sorted(healthy_dir.glob("*.json")):
        image_path = find_image_by_stem(healthy_dir, json_path.stem)
        if image_path is None:
            continue
        annotation = load_json(json_path)
        image_bgr = imread_unicode(image_path)
        leaf_shape = next((s for s in annotation.get("shapes", []) if s.get("label") == LEAF_LABEL), None)
        if leaf_shape is None:
            continue
        leaf_mask = shape_to_mask(image_bgr.shape[:2], leaf_shape.get("points", []))
        if leaf_mask.max() == 0:
            continue
        leaves.append(HealthyLeaf(image_path, json_path, image_bgr, annotation, leaf_mask))
    return leaves


def collect_lesion_templates(disease_dirs: Dict[str, Path]) -> Dict[str, List[LesionTemplate]]:
    templates: Dict[str, List[LesionTemplate]] = {disease: [] for disease in disease_dirs}
    for disease, folder in disease_dirs.items():
        for json_path in sorted(folder.glob("*.json")):
            image_path = find_image_by_stem(folder, json_path.stem)
            if image_path is None:
                continue
            annotation = load_json(json_path)
            image_bgr = imread_unicode(image_path)
            h, w = image_bgr.shape[:2]
            for shape in annotation.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                if label == LEAF_LABEL:
                    continue
                if label and label != disease:
                    continue
                mask = shape_to_mask((h, w), shape.get("points", []))
                cropped = crop_by_mask(image_bgr, mask)
                if cropped is None:
                    continue
                patch, alpha = cropped
                if int((alpha > 0.5).sum()) < 20:
                    continue
                templates[disease].append(
                    LesionTemplate(
                        disease=disease,
                        image_patch_bgr=patch,
                        alpha=soft_alpha(alpha),
                        source_image=str(image_path),
                        source_json=str(json_path),
                    )
                )
    return templates


def resize_template(template: LesionTemplate, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    patch = template.image_patch_bgr
    alpha = template.alpha
    new_w = max(3, int(round(patch.shape[1] * scale)))
    new_h = max(3, int(round(patch.shape[0] * scale)))
    resized_patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_patch, np.clip(resized_alpha, 0.0, 1.0)


def sample_position(
    rng: random.Random,
    leaf_mask: np.ndarray,
    occupied_mask: np.ndarray,
    patch_shape: Tuple[int, int],
    max_trials: int = 200,
    min_leaf_coverage: float = 0.85,
    max_overlap: float = 0.12,
) -> Optional[Tuple[int, int]]:
    ph, pw = patch_shape
    h, w = leaf_mask.shape
    if ph >= h or pw >= w:
        return None
    ys, xs = np.where(leaf_mask > 0)
    if len(xs) == 0:
        return None
    for _ in range(max_trials):
        idx = rng.randrange(len(xs))
        cx = int(xs[idx])
        cy = int(ys[idx])
        x0 = int(np.clip(cx - pw // 2, 0, w - pw))
        y0 = int(np.clip(cy - ph // 2, 0, h - ph))
        region_leaf = leaf_mask[y0:y0 + ph, x0:x0 + pw] > 0
        region_occupied = occupied_mask[y0:y0 + ph, x0:x0 + pw] > 0
        leaf_coverage = float(region_leaf.mean())
        overlap = float(region_occupied.mean())
        if leaf_coverage >= min_leaf_coverage and overlap <= max_overlap:
            return x0, y0
    return None


def paste_patch(
    canvas_bgr: np.ndarray,
    occupied_mask: np.ndarray,
    patch_bgr: np.ndarray,
    alpha: np.ndarray,
    x0: int,
    y0: int,
) -> np.ndarray:
    ph, pw = alpha.shape
    roi = canvas_bgr[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    patch = patch_bgr.astype(np.float32)
    a = alpha[..., None].astype(np.float32)
    blended = patch * a + roi * (1.0 - a)
    canvas_bgr[y0:y0 + ph, x0:x0 + pw] = np.clip(blended, 0, 255).astype(np.uint8)
    pasted_mask = (alpha > 0.35).astype(np.uint8) * 255
    occupied_mask[y0:y0 + ph, x0:x0 + pw] = cv2.bitwise_or(
        occupied_mask[y0:y0 + ph, x0:x0 + pw],
        pasted_mask,
    )
    return pasted_mask


def synthesize_one(
    rng: random.Random,
    target: HealthyLeaf,
    templates_by_disease: Dict[str, List[LesionTemplate]],
    combo: Sequence[str],
    lesions_per_disease: int,
    scale_range: Tuple[float, float],
) -> Tuple[np.ndarray, Dict]:
    image = target.image_bgr.copy()
    occupied = np.zeros(target.leaf_mask.shape, dtype=np.uint8)
    output_shapes = [shape for shape in target.annotation.get("shapes", []) if shape.get("label") == LEAF_LABEL]

    for disease in combo:
        disease_templates = templates_by_disease.get(disease, [])
        if not disease_templates:
            raise RuntimeError(f"No lesion templates found for disease: {disease}")
        for _ in range(lesions_per_disease):
            template = rng.choice(disease_templates)
            scale = rng.uniform(*scale_range)
            patch, alpha = resize_template(template, scale)
            position = sample_position(rng, target.leaf_mask, occupied, alpha.shape)
            if position is None:
                continue
            x0, y0 = position
            pasted_mask = paste_patch(image, occupied, patch, alpha, x0, y0)
            points = mask_to_polygon_points(pasted_mask, x_offset=x0, y_offset=y0)
            if len(points) >= 3:
                output_shapes.append(
                    {
                        "label": disease,
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                    }
                )

    output_annotation = {
        "version": target.annotation.get("version", "5.0.1"),
        "flags": {},
        "shapes": output_shapes,
        "imagePath": "",
        "imageData": None,
        "imageHeight": int(image.shape[0]),
        "imageWidth": int(image.shape[1]),
    }
    return image, output_annotation


def parse_disease_dir_args(items: Sequence[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected DISEASE=FOLDER format, got: {item}")
        disease, folder = item.split("=", 1)
        parsed[disease.strip()] = Path(folder.strip()).expanduser().resolve()
    return parsed


def resolve_default_disease_dirs(script_dir: Path) -> Dict[str, Path]:
    return {
        disease: (script_dir / folder).resolve() if not Path(folder).is_absolute() else Path(folder)
        for disease, folder in DEFAULT_DISEASE_DIRS.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize co-occurring disease leaf images from lesion ROIs.")
    parser.add_argument("--healthy-dir", default=None, help="Folder containing healthy images and LabelMe JSON files.")
    parser.add_argument(
        "--disease-dir",
        action="append",
        default=None,
        help='Disease source folder in "Disease name=folder" format. Can be used multiple times.',
    )
    parser.add_argument("--combo", nargs="+", default=None, help="Disease names to synthesize together.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--lesions-per-disease", type=int, default=None)
    parser.add_argument("--scale-min", type=float, default=0.65)
    parser.add_argument("--scale-max", type=float, default=1.20)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    healthy_dir = (
        Path(args.healthy_dir).expanduser().resolve()
        if args.healthy_dir
        else (script_dir / DEFAULT_HEALTHY_DIR).resolve()
    )
    disease_dirs = parse_disease_dir_args(args.disease_dir) if args.disease_dir else resolve_default_disease_dirs(script_dir)
    combo = args.combo if args.combo else DEFAULT_COMBO
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (script_dir / DEFAULT_OUTPUT_DIR).resolve()
    )
    num_images = args.num_images if args.num_images is not None else DEFAULT_NUM_IMAGES
    lesions_per_disease = args.lesions_per_disease if args.lesions_per_disease is not None else DEFAULT_LESIONS_PER_DISEASE
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    rng = random.Random(seed)

    if not healthy_dir.exists():
        raise FileNotFoundError(f"Healthy folder not found: {healthy_dir}")
    for disease in combo:
        if disease not in disease_dirs:
            raise ValueError(f"No source folder configured for disease: {disease}")
        if not disease_dirs[disease].exists():
            raise FileNotFoundError(f"Disease folder not found for {disease}: {disease_dirs[disease]}")

    healthy_leaves = collect_healthy_leaves(healthy_dir)
    if not healthy_leaves:
        raise RuntimeError(f"No valid healthy leaf annotations found in: {healthy_dir}")
    templates_by_disease = collect_lesion_templates({disease: disease_dirs[disease] for disease in combo})

    image_dir = output_dir / "images"
    json_dir = output_dir / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for index in range(num_images):
        target = rng.choice(healthy_leaves)
        synthetic_image, annotation = synthesize_one(
            rng=rng,
            target=target,
            templates_by_disease=templates_by_disease,
            combo=combo,
            lesions_per_disease=lesions_per_disease,
            scale_range=(args.scale_min, args.scale_max),
        )
        stem = f"co_occurring_{'_'.join(d.replace(' ', '_') for d in combo)}_{index + 1:04d}"
        image_path = image_dir / f"{stem}.jpg"
        json_path = json_dir / f"{stem}.json"
        annotation["imagePath"] = image_path.name
        imwrite_unicode(image_path, synthetic_image)
        save_json(json_path, annotation)
        summary.append(
            {
                "image": image_path.as_posix(),
                "json": json_path.as_posix(),
                "target_leaf": target.image_path.as_posix(),
                "diseases": combo,
                "lesions_per_disease": lesions_per_disease,
            }
        )

    save_json(output_dir / "synthesis_summary.json", summary)
    print(f"Synthetic images: {len(summary)}")
    print(f"Output images: {image_dir}")
    print(f"Output annotations: {json_dir}")
    print(f"Summary: {output_dir / 'synthesis_summary.json'}")


if __name__ == "__main__":
    main()
