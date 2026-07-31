# -*- coding: utf-8 -*-
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

"""
Apple co-occurring disease image construction pipeline.

Workflow:
1. Load annotation statistics and sampling matrices for each disease and severity level.
2. Sample lesion counts, positions, and target areas from empirical distributions.
3. Adjust subsequent lesion positions using conservative inter-disease spatial priors.
4. Select real source lesion templates by area, local leaf color, and texture orientation.
5. Preserve the core lesion texture and softly blend only the lesion boundary.

Primary tuning locations:
- Constants near the top control lesion counts, areas, candidate selection,
  color transfer, and boundary blending.
- PRESET_COMBO_TABLE defines commonly used co-occurring disease configurations.
- Inter-disease behavior is controlled by SPATIAL_INTERACTION_TABLE,
  SEVERITY_ROLE_TABLE, COVERAGE_BUDGET_TABLE, and COUNT_SCALE_TABLE.
"""


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
APPLE_LEVEL = ROOT / "Apple_level"
MATRIX_ROOT = APPLE_LEVEL / "_sampling_matrices"
OUTPUT_ROOT = PROJECT_ROOT / "Dataset"
JSON_OUTPUT_ROOT = ROOT / "Json"
TARGET_ROOT = APPLE_LEVEL / "Apple healthy" / "Healthy"

# ---------------------------------------------------------------------------
# Data paths and runtime presets
# ---------------------------------------------------------------------------
IMAGE_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
]

DEFAULT_DEMO_COUNT = 3  # Default number per configuration; batch generation overrides this value.
RESULT_IMAGE_DIR_NAME = "images"  # Output subdirectory for synthesized images.
RESULT_JSON_DIR_NAME = "json"  # Output subdirectory for synthesized annotations.

PRESET_COMBO_TABLE = {
    "C1_blackrot_rust_mild_mild": [
        {"disease": "Apple black rot", "level": "Mild"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C1_blackrot_rust_severe_mild": [
        {"disease": "Apple black rot", "level": "Severe"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C1_blackrot_rust_mild_severe": [
        {"disease": "Apple black rot", "level": "Mild"},
        {"disease": "Cedar-apple rust", "level": "Severe"},
    ],
    "C1_blackrot_rust_moderate_moderate": [
        {"disease": "Apple black rot", "level": "Moderate"},
        {"disease": "Cedar-apple rust", "level": "Moderate"},
    ],
    "C2_scab_rust_mild_mild": [
        {"disease": "Apple scab", "level": "Mild"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C2_scab_rust_moderate_mild": [
        {"disease": "Apple scab", "level": "Moderate"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C2_scab_rust_severe_mild": [
        {"disease": "Apple scab", "level": "Severe"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C2_scab_rust_moderate_moderate": [
        {"disease": "Apple scab", "level": "Moderate"},
        {"disease": "Cedar-apple rust", "level": "Moderate"},
    ],
    "C3_scab_blackrot_mild_mild": [
        {"disease": "Apple scab", "level": "Mild"},
        {"disease": "Apple black rot", "level": "Mild"},
    ],
    "C3_scab_blackrot_moderate_mild": [
        {"disease": "Apple scab", "level": "Moderate"},
        {"disease": "Apple black rot", "level": "Mild"},
    ],
    "C3_scab_blackrot_severe_mild": [
        {"disease": "Apple scab", "level": "Severe"},
        {"disease": "Apple black rot", "level": "Mild"},
    ],
    "C3_scab_blackrot_moderate_moderate": [
        {"disease": "Apple scab", "level": "Moderate"},
        {"disease": "Apple black rot", "level": "Moderate"},
    ],
    "C4_scab_blackrot_rust_moderate_mild_mild": [
        {"disease": "Apple scab", "level": "Moderate"},
        {"disease": "Apple black rot", "level": "Mild"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C4_scab_blackrot_rust_severe_moderate_mild": [
        {"disease": "Apple scab", "level": "Severe"},
        {"disease": "Apple black rot", "level": "Moderate"},
        {"disease": "Cedar-apple rust", "level": "Mild"},
    ],
    "C4_scab_blackrot_rust_moderate_moderate_moderate": [
        {"disease": "Apple scab", "level": "Moderate"},
        {"disease": "Apple black rot", "level": "Moderate"},
        {"disease": "Cedar-apple rust", "level": "Moderate"},
    ],
}

COEXISTENCE_GROUP_TABLE = {
    "blackrot_rust": [
        "C1_blackrot_rust_mild_mild",
        "C1_blackrot_rust_severe_mild",
        "C1_blackrot_rust_mild_severe",
        "C1_blackrot_rust_moderate_moderate",
    ],
    "scab_rust": [
        "C2_scab_rust_mild_mild",
        "C2_scab_rust_moderate_mild",
        "C2_scab_rust_severe_mild",
        "C2_scab_rust_moderate_moderate",
    ],
    "scab_blackrot": [
        "C3_scab_blackrot_mild_mild",
        "C3_scab_blackrot_moderate_mild",
        "C3_scab_blackrot_severe_mild",
        "C3_scab_blackrot_moderate_moderate",
    ],
    "scab_blackrot_rust": [
        "C4_scab_blackrot_rust_moderate_mild_mild",
        "C4_scab_blackrot_rust_severe_moderate_mild",
        "C4_scab_blackrot_rust_moderate_moderate_moderate",
    ],
}

GROUP_FOLDER_NAME_TABLE = {
    "blackrot_rust": "Apple black rot + Cedar-apple rust",
    "scab_rust": "Apple scab + Cedar-apple rust",
    "scab_blackrot": "Apple scab + Apple black rot",
    "scab_blackrot_rust": "Apple scab + Apple black rot + Cedar-apple rust",
}

# ---------------------------------------------------------------------------
# Tuning parameters
#
# Modify this section first when tuning. Functions read these constants to keep
# numerical settings centralized.
# ---------------------------------------------------------------------------

# Candidate locations are determined by lesion occurrence probability. The
# cumulative area-load map is not used during co-occurring disease synthesis.
POSITION_SAMPLING_METRIC = "probability"

# Sample lesion counts from empirical annotations, adjust them mildly by target
# leaf area, and finally apply primary-secondary co-occurrence relationships.
COUNT_AREA_SCALE_MIN = 0.70  # Minimum area-based count scale for small target leaves.
COUNT_AREA_SCALE_MAX = 1.25  # Maximum area-based count scale for large target leaves.
COUNT_TOTAL_CAP_MULTIPLIER = 0.72  # Total multi-disease lesion cap; lower values produce sparser images.
COUNT_TOTAL_CAP_MIN = 3  # Minimum total cap, preventing all lesions from being suppressed.
CODISEASE_PRIMARY_COUNT_SCALE = 0.78  # Base per-disease count reduction to avoid excessive additive density.

# When a severe disease is present, further suppress secondary diseases to
# prevent unrealistically dense coverage across the entire leaf.
SEVERE_CODISEASE_SECONDARY_COUNT_SCALE = 0.5  # Count scale for non-primary diseases.
SEVERE_CODISEASE_SECONDARY_BUDGET_SCALE = 0.5  # Area-budget scale for non-primary diseases.
SEVERE_LEVEL_COUNT_BOOST = 1.00  # A value of 1.0 adds no count boost for a severe disease.

# Derive lesion area from the source lesion-to-leaf ratio and target leaf area,
# then make small adjustments for severity and the remaining coverage budget.
SOURCE_AREA_BLEND_WEIGHT = 0.55  # Weight of the source relative lesion area.
SEVERITY_AREA_BLEND_WEIGHT = 0.45  # Weight of disease-severity area statistics.
BUDGET_RESERVE_Q25_FACTOR = 0.80  # Conservative scale when the remaining budget is below Q1.
BUDGET_MIN_Q25_FACTOR = 0.60  # Minimum per-disease budget relative to Q1 area.
TARGET_AREA_MIN_Q25_FACTOR = 0.45  # Lower lesion-area bound relative to disease Q1.
TARGET_AREA_MAX_Q75_FACTOR = 1.35  # Upper lesion-area bound relative to disease Q3.
TARGET_AREA_MAX_SOURCE_FACTOR = 0.85  # Upper bound relative to the source area mapped to the target leaf.

# Lesion resize limits. A wider range improves budget matching but increases
# the risk of texture distortion.
LESION_RESIZE_MIN = 0.35  # Lower scale limit; very small values lose texture detail.
LESION_RESIZE_MAX = 1.45  # Upper scale limit; very large values stretch lesion texture.

# Source-candidate weights. Area is primary; local color and texture
# orientation reduce visually implausible matches.
CANDIDATE_AREA_WEIGHT = 0.65  # Area-match contribution to candidate ranking.
CANDIDATE_COLOR_WEIGHT = 0.20  # Local leaf-color contribution to candidate ranking.
CANDIDATE_TEXTURE_WEIGHT = 0.15  # Vein and texture-orientation contribution.
CANDIDATE_TOP_K = 10  # Randomly select from the top K candidates to preserve diversity.
TEXTURE_COHERENCE_MIN = 0.08  # Ignore orientation matching below this reliability threshold.
TEXTURE_FALLBACK_SCORE = 0.25  # Default score when texture orientation is unavailable.

# Source lesion extraction parameters.
SOURCE_LOCAL_RING_KERNEL = 31  # Kernel size for estimating healthy color around a lesion.
MIN_LOCAL_HEALTHY_PIXELS = 30  # Minimum local pixels before falling back to whole-leaf statistics.
LESION_DILATION_RATIO = 0.15  # Core-mask dilation ratio used to retain the full lesion boundary.
SUPPORT_EXPAND_RATIO = 0.10  # Expansion ratio for healthy context outside the lesion.

# Source lesion quality filters affect only the synthesis candidate pool and do
# not modify the original annotations.
ENABLE_LESION_QUALITY_FILTER = True  # Enable source-candidate quality filtering.
QUALITY_MIN_AREA_RATIO = 0.00015  # Reject very small, noise-like annotations.
QUALITY_MAX_AREA_RATIO = 0.08000  # Reject abnormally large or contaminated annotations.
QUALITY_MIN_CORE_PIXELS = 12  # Minimum number of valid pixels in a lesion core.
QUALITY_MIN_BBOX_SIDE = 3  # Reject line-like or extremely small bounding boxes.
QUALITY_MIN_SOLIDITY = 0.18  # Reject fragmented or malformed masks.
QUALITY_MIN_MEAN_LUMA = 18.0  # Reject nearly black source regions.
QUALITY_MAX_MEAN_LUMA = 245.0  # Reject nearly white source regions.
QUALITY_MAX_LUMA_GAP_TO_LEAF = 135.0  # Maximum luminance difference from surrounding leaf tissue.
QUALITY_MAX_CHROMA_GAP_TO_LEAF = 95.0  # Maximum chromatic difference from surrounding leaf tissue.
QUALITY_MAX_EDGE_DARK_RATIO = 0.18  # Reject crops with excessive dark boundary pixels.

# Mask thresholds for overlap checks and final lesion instances.
CORE_MASK_THRESHOLD = 0.35  # Pixels above this alpha threshold form the pasted lesion core.
SUPPORT_LEAF_THRESHOLD = 0.08  # Threshold used to ensure the support region remains inside the leaf.

# Color-transfer and boundary-blending parameters. No synthetic halo is
# generated; the lesion core is retained and only its boundary is softened.
COLOR_TRANSFER_L_WEIGHT = 0.9  # Lab luminance adaptation strength.
COLOR_TRANSFER_AB_WEIGHT = 0.9  # Lab chromatic adaptation strength.
LAB_RESIDUAL_CLIP = 34.0  # Limit Lab residuals to prevent excessive color shifts.
EDGE_BAND_RATIO = 0.15  # Boundary-band width relative to the lesion's shorter side.
EDGE_BAND_MIN_PX = 2.0  # Minimum boundary-band width for small lesions.
EDGE_BAND_MAX_PX = 9.0  # Maximum boundary-band width to preserve lesion detail.
EDGE_LOW_FREQ_STRENGTH = 0.72  # Low-frequency color matching at the boundary.
EDGE_GRADIENT_STRENGTH = 0.35  # Local texture-gradient matching at the boundary.
EDGE_LOW_FREQ_SIGMA_RATIO = 0.85  # Low-frequency smoothing relative to boundary width.
EDGE_ALPHA_BLUR_RATIO = 0.22  # Smoothing applied to contextual blend weights.
SOURCE_CONTEXT_EDGE_WEIGHT = 0.70  # Contribution of healthy source context to boundary matching.
PASTE_SOURCE_CONTEXT = True  # Blend healthy source context with the target neighborhood.
SOURCE_CONTEXT_ALPHA_MAX = 0.78  # Maximum source-context weight near the lesion boundary.
SOURCE_CONTEXT_LOW_FREQ_STRENGTH = 0.88  # Low-frequency context adaptation to the target leaf.
SOURCE_CONTEXT_GRADIENT_STRENGTH = 0.45  # High-frequency context adaptation to target texture.

# Minimum lesion interaction radius in normalized unit-circle coordinates.
LESION_RADIUS_NORM_MIN = 0.018  # Prevent very small lesions from having negligible spatial influence.

SEVERITY_AREA_MULTIPLIER = {
    "Mild": 0.95,
    "Moderate": 1.05,
    "Severe": 1.18,
}

TOTAL_BUDGET_MULTIPLIER = {
    "Mild": 1.05,
    "Moderate": 1.18,
    "Severe": 1.35,
}

LEVEL_INTERACTION_SCALE = {
    "Mild": 0.92,
    "Moderate": 1.00,
    "Severe": 1.12,
}

# Conservative inter-disease spatial priors used only as synthesis rules; they
# are not evidence of biological interactions between pathogens.
# Modes:
# - soft_repel: reduce the probability near lesions of previously placed diseases.
# - core_repel_ring_promote: suppress the core and mildly promote the surrounding ring.
SPATIAL_INTERACTION_TABLE = {
    ("Apple black rot", "Cedar-apple rust"): {
        "mode": "core_repel_ring_promote",
        "core_mult": 0.18,
        "core_radius_scale": 2.10,
        "ring_mult": 1.18,
        "ring_radius_scale": 3.20,
        "ring_width_scale": 1.10,
    },
    ("Cedar-apple rust", "Apple black rot"): {
        "mode": "soft_repel",
        "core_mult": 0.55,
        "core_radius_scale": 1.55,
    },
    ("Apple scab", "Cedar-apple rust"): {
        "mode": "core_repel_ring_promote",
        "core_mult": 0.45,
        "core_radius_scale": 1.65,
        "ring_mult": 1.12,
        "ring_radius_scale": 2.70,
        "ring_width_scale": 1.15,
    },
    ("Cedar-apple rust", "Apple scab"): {
        "mode": "soft_repel",
        "core_mult": 0.68,
        "core_radius_scale": 1.25,
    },
    ("Apple scab", "Apple black rot"): {
        "mode": "soft_repel",
        "core_mult": 0.72,
        "core_radius_scale": 1.35,
    },
    ("Apple black rot", "Apple scab"): {
        "mode": "soft_repel",
        "core_mult": 0.72,
        "core_radius_scale": 1.35,
    },
}

# Primary-secondary roles for each disease and severity combination.
# Each role is mapped through COUNT_SCALE_TABLE and COVERAGE_BUDGET_TABLE.
SEVERITY_ROLE_TABLE = {
    ("Apple black rot", "Cedar-apple rust"): {
        ("Mild", "Mild"): "balanced",
        ("Moderate", "Moderate"): "balanced",
        ("Severe", "Mild"): "blackrot_dominant",
        ("Severe", "Moderate"): "blackrot_dominant",
        ("Mild", "Severe"): "rust_dominant",
        ("Moderate", "Severe"): "rust_dominant",
        "default": "balanced",
    },
    ("Apple scab", "Cedar-apple rust"): {
        ("Mild", "Mild"): "balanced",
        ("Moderate", "Mild"): "scab_dominant",
        ("Moderate", "Moderate"): "balanced",
        ("Severe", "Mild"): "scab_strong_dominant",
        ("Severe", "Moderate"): "scab_dominant",
        "default": "balanced",
    },
    ("Apple scab", "Apple black rot"): {
        ("Mild", "Mild"): "balanced",
        ("Moderate", "Mild"): "scab_dominant",
        ("Moderate", "Moderate"): "balanced",
        ("Severe", "Mild"): "scab_strong_dominant",
        ("Severe", "Moderate"): "scab_dominant",
        "default": "balanced",
    },
    ("Apple scab", "Apple black rot", "Cedar-apple rust"): {
        ("Moderate", "Mild", "Mild"): "scab_dominant",
        ("Severe", "Moderate", "Mild"): "scab_strong_dominant",
        ("Moderate", "Moderate", "Moderate"): "balanced_mixed",
        "default": "balanced_mixed",
    },
}

# Per-disease coverage allocation by role. Values are normalized before use and
# therefore represent relative weights.
COVERAGE_BUDGET_TABLE = {
    ("Apple black rot", "Cedar-apple rust"): {
        "balanced": (0.55, 0.45),
        "blackrot_dominant": (0.68, 0.32),
        "rust_dominant": (0.38, 0.62),
        "default": (0.55, 0.45),
    },
    ("Apple scab", "Cedar-apple rust"): {
        "balanced": (0.58, 0.42),
        "scab_dominant": (0.70, 0.30),
        "scab_strong_dominant": (0.80, 0.20),
        "default": (0.60, 0.40),
    },
    ("Apple scab", "Apple black rot"): {
        "balanced": (0.58, 0.42),
        "scab_dominant": (0.70, 0.30),
        "scab_strong_dominant": (0.80, 0.20),
        "default": (0.60, 0.40),
    },
    ("Apple scab", "Apple black rot", "Cedar-apple rust"): {
        "scab_dominant": (0.55, 0.25, 0.20),
        "scab_strong_dominant": (0.60, 0.25, 0.15),
        "balanced_mixed": (0.40, 0.28, 0.32),
        "default": (0.40, 0.28, 0.32),
    },
}

# Scale per-disease counts by role after sampling from empirical count distributions.
COUNT_SCALE_TABLE = {
    ("Apple black rot", "Cedar-apple rust"): {
        "balanced": (1.0, 1.0),
        "blackrot_dominant": (1.0, 0.52),
        "rust_dominant": (0.55, 1.0),
        "default": (1.0, 1.0),
    },
    ("Apple scab", "Cedar-apple rust"): {
        "balanced": (1.0, 1.0),
        "scab_dominant": (1.0, 0.58),
        "scab_strong_dominant": (1.0, 0.42),
        "default": (1.0, 1.0),
    },
    ("Apple scab", "Apple black rot"): {
        "balanced": (1.0, 1.0),
        "scab_dominant": (1.0, 0.55),
        "scab_strong_dominant": (1.0, 0.42),
        "default": (1.0, 1.0),
    },
    ("Apple scab", "Apple black rot", "Cedar-apple rust"): {
        "scab_dominant": (1.0, 0.50, 0.54),
        "scab_strong_dominant": (1.0, 0.42, 0.45),
        "balanced_mixed": (0.82, 0.70, 0.72),
        "default": (1.0, 1.0, 1.0),
    },
}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def path_for_summary(path: Path, base: Path):
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def find_image_by_stem(folder: Path, stem: str):
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_mask(shape_hw, polygon_points):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = np.array(polygon_points, dtype=np.int32)
    if pts.size:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def polygon_centroid(points):
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0, 0.0
    m = cv2.moments(pts.astype(np.float32))
    if m["m00"] != 0:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def polygon_area(points):
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def calc_avg_color(image_bgr, mask):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb[mask > 0]
    if len(pixels) == 0:
        return [0, 0, 0]
    return np.mean(pixels, axis=0).astype(int).tolist()


def ray_boundary_distance(leaf_mask, leaf_centroid, theta_rad):
    h, w = leaf_mask.shape
    direction = np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float64)
    step = 0.0
    max_steps = float(max(h, w) * 2)
    last_inside = 0.0
    while step <= max_steps:
        x = int(round(leaf_centroid[0] + direction[0] * step))
        y = int(round(leaf_centroid[1] + direction[1] * step))
        if not (0 <= x < w and 0 <= y < h) or leaf_mask[y, x] == 0:
            break
        last_inside = step
        step += 0.5
    return last_inside


def get_leaf_info(image, json_data):
    leaf_shape = next((s for s in json_data["shapes"] if s["label"] == "Complete leaf"), None)
    if leaf_shape is None:
        raise ValueError("No Complete leaf shape found")
    leaf_mask = create_mask(image.shape[:2], leaf_shape["points"])
    leaf_centroid = polygon_centroid(leaf_shape["points"])
    leaf_area = int(np.sum(leaf_mask > 0))
    disease_shapes = [s for s in json_data["shapes"] if s["label"] != "Complete leaf"]
    healthy_mask = leaf_mask.copy()
    existing_masks = []
    for disease_shape in disease_shapes:
        disease_mask = create_mask(image.shape[:2], disease_shape["points"])
        if np.sum(disease_mask > 0) == 0:
            continue
        existing_masks.append(disease_mask)
        healthy_mask = cv2.subtract(healthy_mask, disease_mask)
    leaf_color = calc_avg_color(image, healthy_mask if np.sum(healthy_mask > 0) > 0 else leaf_mask)
    return leaf_mask, leaf_centroid, leaf_area, leaf_color, existing_masks


def compute_source_local_leaf_color(image_bgr, leaf_mask, lesion_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SOURCE_LOCAL_RING_KERNEL, SOURCE_LOCAL_RING_KERNEL))
    ring = cv2.dilate(lesion_mask, kernel, iterations=1)
    ring = cv2.subtract(ring, lesion_mask)
    ring = cv2.bitwise_and(ring, leaf_mask)
    if np.count_nonzero(ring) < MIN_LOCAL_HEALTHY_PIXELS:
        ring = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(lesion_mask))
    if np.count_nonzero(ring) < MIN_LOCAL_HEALTHY_PIXELS:
        return calc_avg_color(image_bgr, leaf_mask)
    return calc_avg_color(image_bgr, ring)


def compute_texture_orientation(image_bgr, mask):
    if mask is None or np.count_nonzero(mask) < 30:
        return 0.0, 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    valid = mask > 0
    jxx = float(np.sum((gx[valid] ** 2)))
    jyy = float(np.sum((gy[valid] ** 2)))
    jxy = float(np.sum((gx[valid] * gy[valid])))
    denom = jxx + jyy + 1e-6
    coherence = math.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / denom
    angle = 0.5 * math.atan2(2.0 * jxy, jxx - jyy + 1e-6)
    return float(angle), float(coherence)


def create_dilated_lesion_mask(image_shape, polygon_points):
    """Expand a source lesion mask to retain limited context during cropping."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, np.int32)], 255)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(largest)
    diameter = 2 * radius
    kernel_size = max(1, int(diameter * LESION_DILATION_RATIO))
    kernel_size_odd = kernel_size * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_odd, kernel_size_odd))
    return cv2.dilate(mask, kernel, iterations=1)


def crop_mask_bbox(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, 0, mask.shape[1], mask.shape[0]
    return cv2.boundingRect(coords)


def expand_mask(mask, ratio=0.035):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.copy()
    largest = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(largest)
    diameter = max(2.0, 2.0 * radius)
    kernel_size = max(1, int(diameter * ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def build_lesion_bundle(image_bgr, lesion_mask, lesion_mask_dilated, leaf_mask):
    """Crop a source lesion image and its associated masks.

    ``core`` is the final lesion-instance region. ``outer`` contains real
    healthy leaf context around the source lesion and is used only for
    boundary transitions, not as part of the pasted lesion.
    """
    support_mask = cv2.bitwise_and(expand_mask(lesion_mask_dilated, ratio=SUPPORT_EXPAND_RATIO), leaf_mask)
    ring_mask = cv2.subtract(lesion_mask_dilated, lesion_mask)
    outer_mask = cv2.subtract(support_mask, lesion_mask_dilated)
    x, y, w_rect, h_rect = crop_mask_bbox(support_mask)
    rgb = cv2.cvtColor(image_bgr[y:y + h_rect, x:x + w_rect], cv2.COLOR_BGR2RGB)
    core_crop = (lesion_mask[y:y + h_rect, x:x + w_rect] > 0).astype(np.float32)
    ring_crop = (ring_mask[y:y + h_rect, x:x + w_rect] > 0).astype(np.float32)
    outer_crop = (outer_mask[y:y + h_rect, x:x + w_rect] > 0).astype(np.float32)
    support_crop = (support_mask[y:y + h_rect, x:x + w_rect] > 0).astype(np.float32)
    return {
        "rgb": rgb,
        "core": core_crop,
        "ring": ring_crop,
        "outer": outer_crop,
        "support": support_crop,
    }


def lesion_quality_check(bundle, area_ratio, source_leaf_color):
    """Reject source lesions with implausible color, area, or shape."""
    if not ENABLE_LESION_QUALITY_FILTER:
        return True, "disabled"
    if area_ratio < QUALITY_MIN_AREA_RATIO:
        return False, "area_too_small"
    if area_ratio > QUALITY_MAX_AREA_RATIO:
        return False, "area_too_large"

    core = bundle["core"] > CORE_MASK_THRESHOLD
    core_pixels = int(np.count_nonzero(core))
    if core_pixels < QUALITY_MIN_CORE_PIXELS:
        return False, "core_too_small"

    ys, xs = np.where(core)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    if min(bbox_w, bbox_h) < QUALITY_MIN_BBOX_SIDE:
        return False, "bbox_too_thin"
    solidity = core_pixels / float(max(bbox_w * bbox_h, 1))
    if solidity < QUALITY_MIN_SOLIDITY:
        return False, "shape_too_sparse"

    rgb = bundle["rgb"].astype(np.float32)
    core_rgb = rgb[core]
    if core_rgb.size == 0:
        return False, "empty_core_rgb"
    mean_rgb = core_rgb.mean(axis=0)
    mean_luma = float(np.dot(mean_rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32)))
    if mean_luma < QUALITY_MIN_MEAN_LUMA:
        return False, "too_dark"
    if mean_luma > QUALITY_MAX_MEAN_LUMA:
        return False, "too_bright"

    leaf_rgb = np.array(source_leaf_color, dtype=np.float32)
    leaf_luma = float(np.dot(leaf_rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32)))
    if abs(mean_luma - leaf_luma) > QUALITY_MAX_LUMA_GAP_TO_LEAF:
        return False, "luma_gap_too_large"

    mean_lab = rgb_triplet_to_lab(mean_rgb)
    leaf_lab = rgb_triplet_to_lab(leaf_rgb)
    chroma_gap = float(np.linalg.norm(mean_lab[1:] - leaf_lab[1:]))
    if chroma_gap > QUALITY_MAX_CHROMA_GAP_TO_LEAF:
        return False, "chroma_gap_too_large"

    support = bundle["support"] > 0.02
    edge = support & (~core)
    if np.count_nonzero(edge) > 0:
        edge_rgb = rgb[edge]
        edge_luma = edge_rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        dark_ratio = float(np.mean(edge_luma < 18.0))
        if dark_ratio > QUALITY_MAX_EDGE_DARK_RATIO:
            return False, "dark_edge"

    return True, "ok"


def build_lesion_pool(disease, level):
    folder = APPLE_LEVEL / disease / level
    pool = []
    skipped = {}
    for json_path in sorted(folder.glob("*.json")):
        image_path = find_image_by_stem(folder, json_path.stem)
        if image_path is None:
            continue
        image = imread_unicode(image_path)
        if image is None:
            continue
        data = load_json(json_path)
        leaf_shape = next((s for s in data["shapes"] if s["label"] == "Complete leaf"), None)
        if leaf_shape is None:
            continue
        leaf_mask = create_mask(image.shape[:2], leaf_shape["points"])
        leaf_area = int(np.sum(leaf_mask > 0))
        if leaf_area <= 0:
            continue

        healthy_mask = leaf_mask.copy()
        lesion_shapes = [s for s in data["shapes"] if s["label"] == disease]
        lesion_masks = []
        for lesion_shape in lesion_shapes:
            lesion_mask = create_mask(image.shape[:2], lesion_shape["points"])
            if np.sum(lesion_mask > 0) == 0:
                continue
            lesion_masks.append((lesion_shape, lesion_mask))
            healthy_mask = cv2.subtract(healthy_mask, lesion_mask)
        leaf_color = calc_avg_color(image, healthy_mask if np.sum(healthy_mask > 0) > 0 else leaf_mask)

        for lesion_shape, lesion_mask in lesion_masks:
            lesion_area = int(np.sum(lesion_mask > 0))
            lesion_mask_dilated = create_dilated_lesion_mask(image.shape, lesion_shape["points"])
            source_local_leaf_color = compute_source_local_leaf_color(image, leaf_mask, lesion_mask)
            area_ratio = float(lesion_area / leaf_area)
            lesion_bundle = build_lesion_bundle(image, lesion_mask, lesion_mask_dilated, leaf_mask)
            keep, reason = lesion_quality_check(lesion_bundle, area_ratio, source_local_leaf_color)
            if not keep:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            ring_mask = cv2.bitwise_and(cv2.subtract(lesion_mask_dilated, lesion_mask), leaf_mask)
            texture_angle, texture_coherence = compute_texture_orientation(image, ring_mask if np.count_nonzero(ring_mask) > 20 else leaf_mask)
            pool.append(
                {
                    "type": disease,
                    "level": level,
                    "area_ratio": area_ratio,
                    "leaf_color": source_local_leaf_color,
                    "global_leaf_color": leaf_color,
                    "texture_angle": texture_angle,
                    "texture_coherence": texture_coherence,
                    "lesion_bundle": lesion_bundle,
                }
            )
    if skipped:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(skipped.items()))
        print(f"quality_filter\t{disease}\t{level}\tskipped={sum(skipped.values())}\t{summary}")
    return pool


def pool_stats(pool):
    ratios = sorted([x["area_ratio"] for x in pool if x.get("area_ratio", 0) > 0])
    if not ratios:
        return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def count_stats_for_level(disease, level):
    folder = APPLE_LEVEL / disease / level
    counts = []
    leaf_areas = []
    for json_path in sorted(folder.glob("*.json")):
        image_path = find_image_by_stem(folder, json_path.stem)
        if image_path is None:
            continue
        image = imread_unicode(image_path)
        if image is None:
            continue
        data = load_json(json_path)
        leaf_shape = next((s for s in data["shapes"] if s["label"] == "Complete leaf"), None)
        if leaf_shape is None:
            continue
        leaf_mask = create_mask(image.shape[:2], leaf_shape["points"])
        leaf_area = int(np.sum(leaf_mask > 0))
        count = sum(1 for shape in data["shapes"] if shape["label"] == disease)
        if count > 0 and leaf_area > 0:
            counts.append(count)
            leaf_areas.append(leaf_area)
    if not counts:
        return {"values": [1], "median_leaf_area": 1.0, "median": 1.0, "q75": 1.0}
    arr = np.asarray(sorted(counts), dtype=np.float64)
    leaf_arr = np.asarray(leaf_areas, dtype=np.float64)
    return {
        "values": [int(x) for x in counts],
        "median_leaf_area": float(np.median(leaf_arr)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
    }


def load_matrices(metric, disease, level):
    stem = f"{disease.lower().replace(' ', '_')}_{level.lower()}"
    return np.load(MATRIX_ROOT / metric / f"{stem}.npy")


def load_matrix_metadata():
    with open(MATRIX_ROOT / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def weighted_choice_index(weights, rng):
    flat = weights.reshape(-1).astype(np.float64)
    total = flat.sum()
    if total <= 0:
        return None
    probs = flat / total
    return int(rng.choice(np.arange(flat.size), p=probs))


def sample_norm_point(prob_matrix, xs, ys, rng):
    idx = weighted_choice_index(prob_matrix, rng)
    if idx is None:
        return None
    h, w = prob_matrix.shape
    row = idx // w
    col = idx % w
    return float(xs[col]), float(ys[row]), row, col


def sample_lesion_count(count_stats, target_leaf_area, rng):
    """Sample an empirical lesion count and adjust it mildly by target leaf area."""
    values = count_stats.get("values") or [1]
    raw_count = int(rng.choice(values))
    median_leaf_area = max(count_stats.get("median_leaf_area", 1.0), 1.0)
    area_scale = math.sqrt(max(target_leaf_area, 1.0) / median_leaf_area)
    area_scale = max(COUNT_AREA_SCALE_MIN, min(COUNT_AREA_SCALE_MAX, area_scale))
    sampled = int(round(raw_count * area_scale))
    return max(1, sampled)


def norm_to_pixel(leaf_mask, leaf_centroid, x_norm, y_norm):
    r_norm = math.hypot(x_norm, y_norm)
    theta = math.atan2(y_norm, x_norm)
    boundary_dist = ray_boundary_distance(leaf_mask, leaf_centroid, theta)
    if boundary_dist <= 1e-6:
        return None
    dist = r_norm * boundary_dist
    x = int(round(leaf_centroid[0] + math.cos(theta) * dist))
    y = int(round(leaf_centroid[1] + math.sin(theta) * dist))
    return x, y


def build_xy_grid(xs, ys):
    return np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))


def lesion_radius_norm(area_ratio):
    return max(LESION_RADIUS_NORM_MIN, math.sqrt(max(area_ratio, 1e-8)))


def normalize_sampling_map(arr, global_max):
    """Normalize an exported occurrence-map matrix into sampling weights."""
    return np.clip(arr / max(global_max, 1e-8), 0.0, 1.0)


def apply_probability_complement(base_probability, occupied_probability, disk_mask):
    """Preserve the current disease prior while downweighting occupied regions."""
    remaining_probability = 1.0 - np.clip(occupied_probability, 0.0, 1.0)
    complemented = base_probability * remaining_probability
    complemented *= disk_mask
    return np.clip(complemented, 0.0, None), remaining_probability


def merge_probability_occupancy(occupied_probability, disease_probability, disk_mask):
    """Accumulate previous spatial occupancy using a probabilistic union."""
    merged = 1.0 - (1.0 - np.clip(occupied_probability, 0.0, 1.0)) * (
        1.0 - np.clip(disease_probability, 0.0, 1.0)
    )
    merged *= disk_mask
    return np.clip(merged, 0.0, 1.0)


def apply_interaction_rule(modifier, dist_map, radius_norm, rule, strength_scale):
    mode = rule["mode"]
    core_mult = 1.0 - (1.0 - rule.get("core_mult", 1.0)) * strength_scale
    core_radius = max(0.02, radius_norm * rule.get("core_radius_scale", 1.0))

    if mode in {"soft_repel", "core_repel_ring_promote"}:
        core_sigma = max(core_radius * 0.65, 0.015)
        core_effect = 1.0 - (1.0 - core_mult) * np.exp(-0.5 * (dist_map / core_sigma) ** 2)
        modifier *= core_effect

    if mode == "core_repel_ring_promote":
        ring_mult = 1.0 + (rule.get("ring_mult", 1.0) - 1.0) * strength_scale
        ring_center = max(core_radius, radius_norm * rule.get("ring_radius_scale", 1.0))
        ring_sigma = max(radius_norm * rule.get("ring_width_scale", 1.0), 0.02)
        ring_effect = 1.0 + (ring_mult - 1.0) * np.exp(-0.5 * ((dist_map - ring_center) / ring_sigma) ** 2)
        modifier *= ring_effect

    return modifier


def build_interaction_modifier(current_disease, placed_records, grid_x, grid_y, disk_mask):
    """Adjust sampling weights using lesions already placed for other diseases."""
    modifier = np.ones_like(grid_x, dtype=np.float64)
    for record in placed_records:
        other_disease = record["disease"]
        if other_disease == current_disease:
            continue
        rule = SPATIAL_INTERACTION_TABLE.get((other_disease, current_disease))
        if not rule:
            continue
        dx = grid_x - float(record["x_norm"])
        dy = grid_y - float(record["y_norm"])
        dist_map = np.sqrt(dx * dx + dy * dy)
        radius_norm = lesion_radius_norm(record.get("actual_pasted_area_ratio", record.get("target_area_ratio", 0.0)))
        strength_scale = LEVEL_INTERACTION_SCALE.get(record.get("level", "Moderate"), 1.0)
        modifier = apply_interaction_rule(modifier, dist_map, radius_norm, rule, strength_scale)
    modifier *= disk_mask
    modifier = np.clip(modifier, 0.0, None)
    return modifier


def compute_target_area_ratio(source_ratio, severity_level, remaining_budget_ratio, lesions_left_after_current, stats):
    """Estimate the target relative area for the current lesion.

    ``source_ratio`` is the lesion-to-leaf area ratio in the source image.
    During placement, ``resize_bundle_by_area()`` uses
    ``source_ratio * target_leaf_area`` as the base area and applies the
    severity and remaining-budget adjustment estimated here.
    """
    severity_anchor = {
        "Mild": stats["q25"] if stats["q25"] > 0 else stats["median"],
        "Moderate": stats["median"],
        "Severe": stats["q75"] if stats["q75"] > 0 else stats["median"],
    }[severity_level]
    blended_base = SOURCE_AREA_BLEND_WEIGHT * source_ratio + SEVERITY_AREA_BLEND_WEIGHT * severity_anchor
    severity_factor = SEVERITY_AREA_MULTIPLIER.get(severity_level, 1.0)
    proposal = blended_base * severity_factor

    if lesions_left_after_current > 0:
        reserve_floor = stats["q25"] * BUDGET_RESERVE_Q25_FACTOR * lesions_left_after_current
        usable_budget = max(remaining_budget_ratio - reserve_floor, stats["q25"] * BUDGET_MIN_Q25_FACTOR)
    else:
        usable_budget = remaining_budget_ratio

    budget_limited = min(proposal, usable_budget)
    min_allowed = max(stats["min"], stats["q25"] * TARGET_AREA_MIN_Q25_FACTOR)
    max_allowed = max(stats["q75"] * TARGET_AREA_MAX_Q75_FACTOR, stats["max"] * TARGET_AREA_MAX_SOURCE_FACTOR, min_allowed)
    return min(max(budget_limited, min_allowed), max_allowed)


def choose_lesion_by_target_area(pool, target_area_ratio, target_leaf_color, target_texture_angle, target_texture_coherence, rng):
    """Select primarily by target area, then refine by local color and texture."""
    if not pool:
        return None
    candidates = []
    target_leaf_color = np.array(target_leaf_color, dtype=np.float32)
    for lesion in pool:
        area_diff = abs(lesion.get("area_ratio", 0.0) - target_area_ratio) / max(target_area_ratio, 1e-6)
        color_diff = float(np.linalg.norm(np.array(lesion.get("leaf_color", [0, 0, 0]), dtype=np.float32) - target_leaf_color) / 255.0)
        if target_texture_coherence > TEXTURE_COHERENCE_MIN and lesion.get("texture_coherence", 0.0) > TEXTURE_COHERENCE_MIN:
            angle_diff = abs(target_texture_angle - lesion.get("texture_angle", 0.0))
            angle_diff = min(angle_diff, math.pi - angle_diff if angle_diff > math.pi / 2 else angle_diff)
            angle_score = angle_diff / (math.pi / 2)
        else:
            angle_score = TEXTURE_FALLBACK_SCORE
        score = CANDIDATE_AREA_WEIGHT * area_diff + CANDIDATE_COLOR_WEIGHT * color_diff + CANDIDATE_TEXTURE_WEIGHT * angle_score
        candidates.append((score, lesion))
    candidates.sort(key=lambda x: x[0])
    top_k = [x[1] for x in candidates[: min(CANDIDATE_TOP_K, len(candidates))]]
    return rng.choice(top_k)


def compute_local_target_color(image_bgr, leaf_mask, existing_masks, position, lesion_shape):
    h, w = image_bgr.shape[:2]
    lesion_h, lesion_w = lesion_shape[:2]
    cx, cy = position
    pad_x = max(8, lesion_w // 2)
    pad_y = max(8, lesion_h // 2)
    x0 = max(0, cx - pad_x)
    y0 = max(0, cy - pad_y)
    x1 = min(w, cx + pad_x)
    y1 = min(h, cy + pad_y)
    local_leaf = leaf_mask[y0:y1, x0:x1].copy()
    if local_leaf.size == 0 or np.count_nonzero(local_leaf) == 0:
        return calc_avg_color(image_bgr, leaf_mask)
    occupied = np.zeros_like(local_leaf)
    for mask in existing_masks:
        occupied = cv2.bitwise_or(occupied, mask[y0:y1, x0:x1])
    healthy_local = cv2.bitwise_and(local_leaf, cv2.bitwise_not(occupied))
    if np.count_nonzero(healthy_local) < 20:
        healthy_local = local_leaf
    return calc_avg_color(image_bgr[y0:y1, x0:x1], healthy_local)


def compute_local_target_features(image_bgr, leaf_mask, existing_masks, position, lesion_shape):
    h, w = image_bgr.shape[:2]
    lesion_h, lesion_w = lesion_shape[:2]
    cx, cy = position
    pad_x = max(8, lesion_w // 2)
    pad_y = max(8, lesion_h // 2)
    x0 = max(0, cx - pad_x)
    y0 = max(0, cy - pad_y)
    x1 = min(w, cx + pad_x)
    y1 = min(h, cy + pad_y)
    local_leaf = leaf_mask[y0:y1, x0:x1].copy()
    if local_leaf.size == 0 or np.count_nonzero(local_leaf) == 0:
        return calc_avg_color(image_bgr, leaf_mask), 0.0, 0.0
    occupied = np.zeros_like(local_leaf)
    for mask in existing_masks:
        occupied = cv2.bitwise_or(occupied, mask[y0:y1, x0:x1])
    healthy_local = cv2.bitwise_and(local_leaf, cv2.bitwise_not(occupied))
    if np.count_nonzero(healthy_local) < 20:
        healthy_local = local_leaf
    color = calc_avg_color(image_bgr[y0:y1, x0:x1], healthy_local)
    angle, coherence = compute_texture_orientation(image_bgr[y0:y1, x0:x1], healthy_local)
    return color, angle, coherence


def rgb_triplet_to_lab(color_rgb):
    patch = np.uint8([[np.array(color_rgb, dtype=np.uint8)]])
    return cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


def render_lesion_roi(target_roi_bgr, bundle, source_leaf_color, target_leaf_color):
    """Create a color-adapted source region of interest.

    The result includes the lesion core and healthy source context outside it.
    ``paste_lesion()`` writes only the core into the target image; the outer
    context contributes only to low-frequency and texture transitions.
    """
    core_mask = np.clip(bundle["core"], 0.0, 1.0).astype(np.float32)
    if np.count_nonzero(core_mask > 0.02) == 0:
        return target_roi_bgr.astype(np.float32)

    source_rgb = bundle["rgb"].astype(np.uint8)
    support_mask = np.clip(bundle["support"], 0.0, 1.0).astype(np.float32)
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_leaf_lab = rgb_triplet_to_lab(source_leaf_color).reshape(1, 1, 3)
    target_float = target_roi_bgr.astype(np.float32)
    target_leaf_lab = rgb_triplet_to_lab(target_leaf_color).reshape(1, 1, 3)
    delta_lab = target_leaf_lab - source_leaf_lab
    shift_weight = np.zeros((1, 1, 3), dtype=np.float32)
    shift_weight[:, :, 0] = COLOR_TRANSFER_L_WEIGHT
    shift_weight[:, :, 1] = COLOR_TRANSFER_AB_WEIGHT
    shift_weight[:, :, 2] = COLOR_TRANSFER_AB_WEIGHT
    shifted_source_lab = np.clip(source_lab + delta_lab * shift_weight, 0.0, 255.0)
    shifted_source_bgr = cv2.cvtColor(shifted_source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # Bounding-box corners may contain source background. Replace them with the
    # target ROI so background pixels cannot create dark low-frequency borders.
    support3 = support_mask[:, :, None]
    adapted_source = shifted_source_bgr * support3 + target_float * (1.0 - support3)
    return adapted_source


def resize_bundle_by_area(bundle, source_area_ratio, target_leaf_area, scale_factor=1.0):
    """Resize the lesion image and masks using relative source and target areas.

    Base area equals ``source_area_ratio * target_leaf_area``.
    ``scale_factor`` applies only severity- and budget-dependent adjustment.
    """
    target_lesion_area = source_area_ratio * target_leaf_area * scale_factor
    current_lesion_area = np.sum(bundle["core"] > 0.5)
    if current_lesion_area > 0:
        scale = math.sqrt(target_lesion_area / current_lesion_area)
    else:
        scale = 1.0
    scale = max(LESION_RESIZE_MIN, min(LESION_RESIZE_MAX, scale))
    h, w = bundle["rgb"].shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    rgb_out = cv2.resize(bundle["rgb"], (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    core_out = np.clip(cv2.resize(bundle["core"], (new_w, new_h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    ring_out = np.clip(cv2.resize(bundle["ring"], (new_w, new_h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    outer_out = np.clip(cv2.resize(bundle["outer"], (new_w, new_h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    support_out = np.clip(cv2.resize(bundle["support"], (new_w, new_h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    return {
        "rgb": rgb_out,
        "core": core_out,
        "ring": ring_out,
        "outer": outer_out,
        "support": support_out,
    }, scale


def create_lesion_mask(bundle, position, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    h_lesion, w_lesion = bundle["core"].shape[:2]
    x, y = position
    x_start = max(0, x - w_lesion // 2)
    y_start = max(0, y - h_lesion // 2)
    x_end = min(mask_shape[1], x_start + w_lesion)
    y_end = min(mask_shape[0], y_start + h_lesion)
    lesion_alpha = (bundle["core"] > CORE_MASK_THRESHOLD).astype(np.uint8) * 255
    lesion_y_start = max(0, -(y - h_lesion // 2))
    lesion_y_end = lesion_y_start + (y_end - y_start)
    lesion_x_start = max(0, -(x - w_lesion // 2))
    lesion_x_end = lesion_x_start + (x_end - x_start)
    if lesion_y_start < lesion_y_end and lesion_x_start < lesion_x_end and y_start < y_end and x_start < x_end:
        lesion_roi = lesion_alpha[lesion_y_start:lesion_y_end, lesion_x_start:lesion_x_end]
        mask[y_start:y_end, x_start:x_end] = (lesion_roi > 0) * 255
    return mask


def check_overlap(position, bundle, existing_masks, leaf_mask):
    h_lesion, w_lesion = bundle["support"].shape[:2]
    center_x, center_y = position
    x_start = center_x - w_lesion // 2
    y_start = center_y - h_lesion // 2
    x_end = x_start + w_lesion
    y_end = y_start + h_lesion
    img_height, img_width = leaf_mask.shape
    if x_start < 0 or y_start < 0 or x_end > img_width or y_end > img_height:
        return True

    support_alpha = bundle["support"] > SUPPORT_LEAF_THRESHOLD
    if np.any(support_alpha & (leaf_mask[y_start:y_end, x_start:x_end] == 0)):
        return True

    temp_mask = np.zeros_like(leaf_mask)
    core_alpha = bundle["core"] > CORE_MASK_THRESHOLD
    temp_mask[y_start:y_end, x_start:x_end][core_alpha] = 255
    for existing_mask in existing_masks:
        if np.any((temp_mask > 0) & (existing_mask > 0)):
            return True
    return False


def paste_lesion(target_image, target_mask, position, bundle, source_leaf_color, target_leaf_color):
    """Paste a lesion core and blend healthy source context into the target.

    The pipeline does not generate an artificial pathological ring or halo.
    Boundary width is proportional to lesion size. Core texture is retained,
    while healthy source context is gradient-blended with the healthy target
    neighborhood to form a continuous source-to-target transition.
    """
    x, y = position
    lesion_height, lesion_width = bundle["support"].shape[:2]
    x0 = x - lesion_width // 2
    y0 = y - lesion_height // 2
    x1 = x0 + lesion_width
    y1 = y0 + lesion_height
    if x0 < 0 or y0 < 0 or x1 > target_image.shape[1] or y1 > target_image.shape[0]:
        return target_image

    core_mask = np.clip(bundle["core"], 0.0, 1.0)
    if np.count_nonzero(core_mask > 0.02) == 0:
        return target_image

    target_roi_bgr = target_image[y0:y1, x0:x1].copy()
    lesion_rendered = render_lesion_roi(target_roi_bgr, bundle, source_leaf_color, target_leaf_color)
    min_dim = max(1, min(lesion_height, lesion_width))
    core_binary = (core_mask > 0.12).astype(np.uint8)
    support_alpha = np.clip(bundle["support"], 0.0, 1.0).astype(np.float32)
    source_context = np.clip(support_alpha - core_mask, 0.0, 1.0)

    band_px = int(round(min_dim * EDGE_BAND_RATIO))
    band_px = int(max(EDGE_BAND_MIN_PX, min(EDGE_BAND_MAX_PX, band_px)))
    core_dist = cv2.distanceTransform(core_binary, cv2.DIST_L2, 3).astype(np.float32)
    outside_dist = cv2.distanceTransform((1 - core_binary).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

    inside_edge = np.clip(1.0 - core_dist / max(float(band_px), 1.0), 0.0, 1.0) * core_binary.astype(np.float32)
    outside_edge = np.clip(1.0 - outside_dist / max(float(band_px), 1.0), 0.0, 1.0) * source_context
    inside_edge = cv2.GaussianBlur(inside_edge, (0, 0), sigmaX=max(0.45, band_px * EDGE_ALPHA_BLUR_RATIO))
    outside_edge = cv2.GaussianBlur(outside_edge, (0, 0), sigmaX=max(0.45, band_px * EDGE_ALPHA_BLUR_RATIO))

    # Do not feather core opacity; this preserves the real lesion boundary and texture.
    core_alpha = (core_mask > CORE_MASK_THRESHOLD).astype(np.float32)

    target_float = target_roi_bgr.astype(np.float32)
    blur_sigma = max(1.2, band_px * EDGE_LOW_FREQ_SIGMA_RATIO)
    target_low = cv2.GaussianBlur(target_float, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    lesion_low = cv2.GaussianBlur(lesion_rendered, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    target_high = target_float - target_low
    lesion_high = lesion_rendered - lesion_low

    # Construct the outer boundary band from healthy context around the source lesion.
    context_weight = np.clip(inside_edge + outside_edge * SOURCE_CONTEXT_EDGE_WEIGHT, 0.0, 1.0)
    context_weight = cv2.GaussianBlur(context_weight, (0, 0), sigmaX=max(0.45, band_px * EDGE_ALPHA_BLUR_RATIO))
    edge_zone = np.clip(inside_edge * 0.80 + context_weight * 0.20, 0.0, 1.0)
    lowfreq_matched = np.clip(
        lesion_rendered + (target_low - lesion_low) * edge_zone[:, :, None] * EDGE_LOW_FREQ_STRENGTH,
        0,
        255,
    )
    gradient_matched = np.clip(
        lowfreq_matched + (target_high - lesion_high) * inside_edge[:, :, None] * EDGE_GRADIENT_STRENGTH,
        0,
        255,
    )

    roi_base = target_float.copy()
    if PASTE_SOURCE_CONTEXT:
        context_lowfreq_matched = np.clip(
            lesion_rendered
            + (target_low - lesion_low) * context_weight[:, :, None] * SOURCE_CONTEXT_LOW_FREQ_STRENGTH,
            0,
            255,
        )
        context_gradient_matched = np.clip(
            context_lowfreq_matched
            + (target_high - lesion_high) * outside_edge[:, :, None] * SOURCE_CONTEXT_GRADIENT_STRENGTH,
            0,
            255,
        )
        # Gradient-blend healthy source context into the target neighborhood,
        # favoring source context near the lesion and target tissue farther away.
        context_alpha = np.clip(outside_edge, 0.0, 1.0) * np.clip(source_context, 0.0, 1.0)
        context_alpha = cv2.GaussianBlur(
            context_alpha,
            (0, 0),
            sigmaX=max(0.45, band_px * EDGE_ALPHA_BLUR_RATIO),
        )
        context_alpha = np.clip(context_alpha, 0.0, SOURCE_CONTEXT_ALPHA_MAX)
        context_alpha3 = np.repeat(context_alpha[:, :, None], 3, axis=2)
        roi_base = context_gradient_matched * context_alpha3 + roi_base * (1.0 - context_alpha3)

    core_alpha3 = np.repeat(np.clip(core_alpha, 0.0, 1.0)[:, :, None], 3, axis=2)
    roi = gradient_matched * core_alpha3 + roi_base * (1.0 - core_alpha3)

    result = target_image.copy()
    result[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    outside_leaf = target_mask == 0
    result[outside_leaf] = target_image[outside_leaf]
    return result


def combo_name(combo):
    return "__".join([f"{item['disease'].replace(' ', '_')}__{item['level']}" for item in combo])


def normalize_split(split, combo_len):
    split = tuple(float(x) for x in split)
    if len(split) != combo_len:
        split = tuple([1.0 / combo_len] * combo_len)
    total = sum(split)
    if total <= 0:
        return tuple([1.0 / combo_len] * combo_len)
    return tuple(x / total for x in split)


def get_combo_prior(combo):
    if len(combo) == 0:
        return {
            "role": "none",
            "count_scale": tuple(),
            "coverage_split": tuple(),
        }

    diseases = tuple(item["disease"] for item in combo)
    levels = tuple(item["level"] for item in combo)
    role_cfg = SEVERITY_ROLE_TABLE.get(diseases, {})
    role = role_cfg.get(levels, role_cfg.get("default", "balanced"))

    count_cfg = COUNT_SCALE_TABLE.get(diseases, {})
    count_scale = tuple(count_cfg.get(role, count_cfg.get("default", tuple([1.0] * len(combo)))))
    if len(count_scale) != len(combo):
        count_scale = tuple([1.0] * len(combo))

    coverage_cfg = COVERAGE_BUDGET_TABLE.get(diseases, {})
    coverage_split = normalize_split(coverage_cfg.get(role, coverage_cfg.get("default", tuple([1.0 / len(combo)] * len(combo)))), len(combo))

    return {
        "role": role,
        "count_scale": count_scale,
        "coverage_split": coverage_split,
    }


def apply_severity_competition(combo, counts, budgets):
    """Suppress secondary count and area budgets when a severe disease is present."""
    if not any(item["level"] == "Severe" for item in combo):
        return counts, budgets

    adjusted_counts = []
    adjusted_budgets = []
    severe_indices = [idx for idx, item in enumerate(combo) if item["level"] == "Severe"]
    primary_idx = severe_indices[0] if severe_indices else 0

    for idx, (item, count, budget) in enumerate(zip(combo, counts, budgets)):
        if idx == primary_idx:
            count_scale = SEVERE_LEVEL_COUNT_BOOST * CODISEASE_PRIMARY_COUNT_SCALE
            budget_scale = 1.0
        else:
            count_scale = SEVERE_CODISEASE_SECONDARY_COUNT_SCALE
            budget_scale = SEVERE_CODISEASE_SECONDARY_BUDGET_SCALE
        adjusted_counts.append(max(1, int(round(count * count_scale))))
        adjusted_budgets.append(float(budget * budget_scale))
    return adjusted_counts, adjusted_budgets


def cap_total_counts(counts, count_stats_list):
    """Cap total lesions to prevent excessive density from additive counts."""
    if not counts:
        return counts
    raw_total = int(sum(counts))
    reference_total = sum(max(stats.get("median", 1.0), 1.0) for stats in count_stats_list)
    total_cap = max(COUNT_TOTAL_CAP_MIN, int(round(reference_total * COUNT_TOTAL_CAP_MULTIPLIER)))
    if raw_total <= total_cap:
        return counts

    scale = total_cap / float(max(raw_total, 1))
    capped = [max(1, int(round(x * scale))) for x in counts]
    while sum(capped) > total_cap:
        idx = int(np.argmax(capped))
        if capped[idx] <= 1:
            break
        capped[idx] -= 1
    return capped


def disease_level_root(disease, level):
    return APPLE_LEVEL / disease / level


def safe_name(text):
    return text.lower().replace("apple ", "").replace("cedar-apple ", "cedar_apple_").replace(" ", "_").replace("-", "_")


def generate_demo(
    combo,
    demo_count=DEFAULT_DEMO_COUNT,
    seed=42,
    out_name=None,
    output_dir=None,
    json_output_dir=None,
    file_prefix="demo",
    start_index=1,
    summary_name=None,
    target_root=TARGET_ROOT,
    base_mode="healthy",
    base_disease=None,
    base_level=None,
    skip_diseases=None,
):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    skip_diseases = set(skip_diseases or [])
    active_combo = [item for item in combo if item["disease"] not in skip_diseases]

    metadata = load_matrix_metadata()
    xs = np.load(MATRIX_ROOT / "xs.npy")
    ys = np.load(MATRIX_ROOT / "ys.npy")
    disk_mask = np.load(MATRIX_ROOT / "disk_mask.npy").astype(np.float64)
    grid_x, grid_y = build_xy_grid(xs, ys)
    position_global = metadata["metrics"][POSITION_SAMPLING_METRIC]["global_max"]

    pools = {}
    matrices = {}
    stats_by_key = {}
    count_stats_by_key = {}
    for item in active_combo:
        key = (item["disease"], item["level"])
        pools[key] = build_lesion_pool(item["disease"], item["level"])
        stats_by_key[key] = pool_stats(pools[key])
        count_stats_by_key[key] = count_stats_for_level(item["disease"], item["level"])
        matrices[key] = {
            "position": load_matrices(POSITION_SAMPLING_METRIC, item["disease"], item["level"]),
        }
        stats = stats_by_key[key]
        count_stats = count_stats_by_key[key]
        print(
            f"pool\t{item['disease']}\t{item['level']}\tlesions={len(pools[key])}"
            f"\tmedian_area={stats['median']:.5f}\tq75_area={stats['q75']:.5f}"
            f"\tmedian_count={count_stats['median']:.2f}\tq75_count={count_stats['q75']:.2f}"
        )

    target_root = Path(target_root)
    target_jsons = sorted(target_root.glob("*.json"))
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / (out_name or combo_name(combo))
    if json_output_dir is None:
        image_out_dir = out_dir / RESULT_IMAGE_DIR_NAME
        json_out_dir = out_dir / RESULT_JSON_DIR_NAME
    else:
        image_out_dir = out_dir
        json_out_dir = Path(json_output_dir)
    image_out_dir.mkdir(parents=True, exist_ok=True)
    json_out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    if len(target_jsons) == 0:
        raise RuntimeError(f"No target json files found in: {target_root}")
    replace_targets = demo_count > len(target_jsons)
    chosen_targets = rng.choice(np.arange(len(target_jsons)), size=demo_count, replace=replace_targets)
    for i, target_idx in enumerate(chosen_targets, start=start_index):
        json_path = target_jsons[int(target_idx)]
        image_path = find_image_by_stem(target_root, json_path.stem)
        if image_path is None:
            continue
        image = imread_unicode(image_path)
        if image is None:
            continue
        data = load_json(json_path)
        leaf_mask, leaf_centroid, leaf_area, _, existing_masks = get_leaf_info(image, data)
        result = image.copy()
        placement_records = []
        planned_combo = []
        sampled_count_by_key = {}
        remaining_budget_ratio = {}

        combo_prior = get_combo_prior(active_combo)
        sampled_counts_raw = []
        budget_bases = []
        keys_in_order = []

        for item in active_combo:
            key = (item["disease"], item["level"])
            keys_in_order.append(key)
            raw_count = sample_lesion_count(count_stats_by_key[key], leaf_area, rng)
            sampled_counts_raw.append(raw_count)
            stats = stats_by_key[key]
            budget_bases.append(max(stats["median"], stats["q25"]) * raw_count * TOTAL_BUDGET_MULTIPLIER.get(item["level"], 1.0))

        total_budget = float(sum(budget_bases))
        count_scale = combo_prior["count_scale"]
        coverage_split = combo_prior["coverage_split"]

        preliminary_counts = []
        preliminary_budgets = []
        for idx, item in enumerate(active_combo):
            key = keys_in_order[idx]
            sampled_count = max(1, int(round(sampled_counts_raw[idx] * count_scale[idx])))
            preliminary_counts.append(sampled_count)
            preliminary_budgets.append(total_budget * coverage_split[idx])

        preliminary_counts, preliminary_budgets = apply_severity_competition(active_combo, preliminary_counts, preliminary_budgets)
        preliminary_counts = cap_total_counts(preliminary_counts, [count_stats_by_key[key] for key in keys_in_order])

        for idx, item in enumerate(active_combo):
            key = keys_in_order[idx]
            sampled_count = preliminary_counts[idx]
            sampled_count_by_key[key] = sampled_count
            remaining_budget_ratio[key] = preliminary_budgets[idx]
            planned_combo.append(
                {
                    "disease": item["disease"],
                    "level": item["level"],
                    "count": sampled_count,
                    "budget_ratio": remaining_budget_ratio[key],
                    "role": combo_prior["role"],
                }
            )

        occupied_probability = np.zeros_like(disk_mask, dtype=np.float64)
        for item in active_combo:
            key = (item["disease"], item["level"])
            position_map = matrices[key]["position"]
            base_position = normalize_sampling_map(position_map, position_global)
            complemented_position, probability_complement = apply_probability_complement(
                base_position, occupied_probability, disk_mask
            )
            if complemented_position.sum() <= 0:
                complemented_position = base_position
            pool = pools[key]
            stats = stats_by_key[key]
            target_count = sampled_count_by_key[key]
            placed = 0
            attempts = 0

            while placed < target_count and attempts < max(target_count * 120, 50):
                attempts += 1
                interaction_modifier = build_interaction_modifier(item["disease"], placement_records, grid_x, grid_y, disk_mask)
                effective_prob = np.clip(complemented_position * interaction_modifier, 0.0, None)
                if effective_prob.sum() <= 0:
                    effective_prob = complemented_position
                sampled = sample_norm_point(effective_prob, xs, ys, rng)
                if sampled is None:
                    break
                x_norm, y_norm, row, col = sampled
                pos = norm_to_pixel(leaf_mask, leaf_centroid, x_norm, y_norm)
                if pos is None:
                    continue

                provisional_source = py_rng.choice(pool) if pool else None
                if provisional_source is None:
                    break
                lesions_left_after = target_count - placed - 1
                target_ratio = compute_target_area_ratio(
                    source_ratio=provisional_source["area_ratio"],
                    severity_level=item["level"],
                    remaining_budget_ratio=remaining_budget_ratio[key],
                    lesions_left_after_current=lesions_left_after,
                    stats=stats,
                )
                provisional_shape = provisional_source["lesion_bundle"]["rgb"].shape
                local_target_color, target_texture_angle, target_texture_coherence = compute_local_target_features(
                    result, leaf_mask, existing_masks, pos, provisional_shape
                )
                lesion = choose_lesion_by_target_area(
                    pool,
                    target_ratio,
                    local_target_color,
                    target_texture_angle,
                    target_texture_coherence,
                    py_rng,
                )
                if lesion is None:
                    break

                adjusted, resize_scale = resize_bundle_by_area(
                    lesion["lesion_bundle"],
                    lesion["area_ratio"],
                    leaf_area,
                    scale_factor=max(0.72, min(1.22, target_ratio / max(lesion["area_ratio"], 1e-8))),
                )
                local_target_color = compute_local_target_color(result, leaf_mask, existing_masks, pos, adjusted["rgb"].shape)
                if check_overlap(pos, adjusted, existing_masks, leaf_mask):
                    continue

                result = paste_lesion(result, leaf_mask, pos, adjusted, lesion["leaf_color"], local_target_color)
                new_mask = create_lesion_mask(adjusted, pos, leaf_mask.shape)
                existing_masks.append(new_mask)

                actual_ratio = float(np.sum(adjusted["core"] > 0.35) / max(leaf_area, 1))
                remaining_budget_ratio[key] = max(0.0, remaining_budget_ratio[key] - actual_ratio)
                placement_records.append(
                    {
                        "disease": item["disease"],
                        "level": item["level"],
                        "position": [int(pos[0]), int(pos[1])],
                        "x_norm": x_norm,
                        "y_norm": y_norm,
                        "interaction_modifier_value": float(interaction_modifier[row, col]),
                        "position_sampling_value": float(base_position[row, col]),
                        "probability_complement_value": float(probability_complement[row, col]),
                        "effective_position_sampling_value": float(complemented_position[row, col]),
                        "target_area_ratio": target_ratio,
                        "chosen_source_area_ratio": lesion["area_ratio"],
                        "resize_scale": resize_scale,
                        "actual_pasted_area_ratio": actual_ratio,
                        "remaining_budget_ratio": remaining_budget_ratio[key],
                        "local_target_color": local_target_color,
                    }
                )
                placed += 1
            occupied_probability = merge_probability_occupancy(
                occupied_probability, base_position, disk_mask
            )

        out_image = image_out_dir / f"{file_prefix}_{i:03d}.jpg"
        out_json = json_out_dir / f"{file_prefix}_{i:03d}.json"
        imwrite_unicode(out_image, result)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_mode": base_mode,
                    "base_disease": base_disease,
                    "base_level": base_level,
                    "target_image": image_path.name,
                    "skipped_existing_diseases": sorted(skip_diseases),
                    "requested_combo": combo,
                    "added_combo": planned_combo,
                    "combo": planned_combo,
                    "placements": placement_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        summary.append(
            {
                "image": path_for_summary(out_image, PROJECT_ROOT),
                "json": path_for_summary(out_json, PROJECT_ROOT),
                "base_mode": base_mode,
                "base_disease": base_disease,
                "base_level": base_level,
                "placements": len(placement_records),
                "requested_combo": combo,
                "added_combo": planned_combo,
                "combo": planned_combo,
            }
        )
        print(f"generated\t{out_image}")

    summary_file = summary_name or f"{file_prefix}_summary.json"
    with open(json_out_dir / summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved demos to: {out_dir}")
    return summary


def strip_preset_code(preset_name):
    return preset_name.split("_", 1)[1] if "_" in preset_name else preset_name


def split_total_count(total_count, part_count):
    if part_count <= 0:
        return []
    base = total_count // part_count
    remainder = total_count % part_count
    return [base + (1 if idx < remainder else 0) for idx in range(part_count)]


def generate_coexistence_groups(total_count_per_group=200, healthy_base_count=100, diseased_base_count=100, base_seed=900):
    """Generate synthesized images by co-occurring disease group.

    By default, each group contains 200 images: 100 with healthy-leaf
    backgrounds and 100 with single-disease backgrounds. Single-disease
    backgrounds are sampled evenly across diseases in the group and matched
    to the current severity configuration.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    seed_offset = 0

    for group_name, preset_names in COEXISTENCE_GROUP_TABLE.items():
        folder_name = GROUP_FOLDER_NAME_TABLE.get(group_name, group_name)
        group_dir = OUTPUT_ROOT / folder_name
        if group_dir.exists():
            shutil.rmtree(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)

        healthy_counts = split_total_count(healthy_base_count, len(preset_names))
        diseased_counts = split_total_count(diseased_base_count, len(preset_names))
        group_summary = []
        print(
            f"Generating co-occurrence group: {folder_name}"
            f"\ttotal={total_count_per_group}"
            f"\thealthy backgrounds={healthy_base_count}"
            f"\tsingle-disease backgrounds={diseased_base_count}"
        )

        for preset_idx, preset_name in enumerate(preset_names, start=1):
            combo = PRESET_COMBO_TABLE[preset_name]
            file_prefix = strip_preset_code(preset_name)
            healthy_count = healthy_counts[preset_idx - 1]
            diseased_count = diseased_counts[preset_idx - 1]
            print(
                f"  Severity configuration: {file_prefix}"
                f"\thealthy backgrounds={healthy_count}"
                f"\tsingle-disease backgrounds={diseased_count}"
            )

            if healthy_count > 0:
                summary = generate_demo(
                    combo=combo,
                    demo_count=healthy_count,
                    seed=base_seed + seed_offset + preset_idx,
                    output_dir=group_dir,
                    json_output_dir=JSON_OUTPUT_ROOT,
                    file_prefix=f"{file_prefix}__healthy",
                    start_index=1,
                    summary_name=f"{group_name}__{file_prefix}__healthy_summary.json",
                    target_root=TARGET_ROOT,
                    base_mode="healthy",
                )
                group_summary.extend(summary)

            base_counts = split_total_count(diseased_count, len(combo))
            for base_idx, (base_item, base_count) in enumerate(zip(combo, base_counts), start=1):
                if base_count <= 0:
                    continue
                source_root = disease_level_root(base_item["disease"], base_item["level"])
                base_tag = safe_name(base_item["disease"])
                print(
                    f"    Single-disease background: "
                    f"{base_item['disease']} {base_item['level']}"
                    f"\tcount={base_count}"
                )
                summary = generate_demo(
                    combo=combo,
                    demo_count=base_count,
                    seed=base_seed + seed_offset + preset_idx * 10 + base_idx,
                    output_dir=group_dir,
                    json_output_dir=JSON_OUTPUT_ROOT,
                    file_prefix=f"{file_prefix}__base_{base_tag}",
                    start_index=1,
                    summary_name=f"{group_name}__{file_prefix}__base_{base_tag}_summary.json",
                    target_root=source_root,
                    base_mode="diseased",
                    base_disease=base_item["disease"],
                    base_level=base_item["level"],
                    skip_diseases={base_item["disease"]},
                )
                group_summary.extend(summary)

        with open(JSON_OUTPUT_ROOT / f"{group_name}__summary_all.json", "w", encoding="utf-8") as f:
            json.dump(group_summary, f, ensure_ascii=False, indent=2)
        all_summaries[folder_name] = group_summary
        seed_offset += 100

    with open(JSON_OUTPUT_ROOT / "apple_coexistence_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_coexistence_groups(total_count_per_group=200, healthy_base_count=100, diseased_base_count=100, base_seed=900)
