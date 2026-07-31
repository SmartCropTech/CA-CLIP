"""Predict one image or every image under a folder.

Set INPUT_PATH below, then run this file directly from an IDE.
"""

import argparse
import json
from pathlib import Path

from configuration.mobilenet_v3_distilled import run_prediction


# A single image and a folder are both supported.
INPUT_PATH = r"CCPLD/val"
OUTPUT_CSV = r"predictions.csv"
DEVICE = "auto"
SAVE_CLASS_PROBABILITIES = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distilled MobileNetV3 multi-label prediction.")
    parser.add_argument("--input", default=None, help="Optional image or folder path override.")
    parser.add_argument("--output", default=None, help="Optional CSV output path override.")
    parser.add_argument("--device", default=None, help="Optional auto, cpu, cuda, or cuda:N override.")
    parser.add_argument("--save-class-probabilities", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parent
    input_path = resolve_path(args.input or INPUT_PATH, repository_root)
    output_csv = resolve_path(args.output or OUTPUT_CSV, repository_root)
    config_path = repository_root / "configuration" / "config.json"

    rows = run_prediction(
        input_path=input_path,
        config_path=config_path,
        output_csv=output_csv,
        device_override=args.device or DEVICE,
        save_probabilities=args.save_class_probabilities or SAVE_CLASS_PROBABILITIES,
    )

    if input_path.is_file():
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    print(f"Images processed: {len(rows)}")
    print(f"Predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()

