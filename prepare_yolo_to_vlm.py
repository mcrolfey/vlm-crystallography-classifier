#!/usr/bin/env python3
"""
prepare_yolo_to_vlm.py
----------------------
Converts a YOLO bounding-box dataset into a multi-label classification JSONL
dataset suitable for Qwen2.5-VL fine-tuning.

Strategy:
  For each image, extract the UNIQUE set of YOLO class IDs present across all
  bounding boxes in the corresponding label file.  This reduces the spatial
  detection task to an image-level "which phases are present?" question that
  a VLM can answer via free-text generation.

Output (JSONL):
  Each line is a JSON object:
    {
      "image_path": "C:/abs/path/to/image.jpg",
      "label_ids":   [3],
      "label_names": ["Phase_3"],
      "prompt":      "<system + user prompt string>",
      "response":    "Phase_3"
    }

Usage:
  python prepare_yolo_to_vlm.py
  python prepare_yolo_to_vlm.py --images path/to/images --labels path/to/labels
  python prepare_yolo_to_vlm.py --val_split 0.15 --seed 123
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SYSTEM_PROMPT = (
    "You are an expert geologist specialising in polarised light microscopy "
    "and scanning electron microscopy (SEM) for crystallographic phase classification."
)

USER_PROMPT_TEMPLATE = """\
Examine this microscopy image carefully and identify every crystallographic \
phase that is visible.

Available classes:
{class_list}

Reply with ONLY the names of the classes present in the image, separated by \
commas. Do not explain. Do not add extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes(classes_file: Path) -> dict[int, str]:
    """Load class names from a one-per-line text file."""
    with open(classes_file, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        sys.exit(f"[ERROR] classes.txt is empty: {classes_file}")
    return {i: name for i, name in enumerate(lines)}


def parse_yolo_label(label_path: Path) -> list[int]:
    """Return sorted list of unique integer class IDs from a YOLO .txt file."""
    found: set[int] = set()
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                found.add(int(parts[0]))
            except ValueError:
                continue
    return sorted(found)


def find_image(image_dir: Path, stem: str) -> Path | None:
    """Locate the image file whose stem matches the label stem."""
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_user_prompt(class_names: dict[int, str]) -> str:
    class_list = "\n".join(f"  - {name}" for name in class_names.values())
    return USER_PROMPT_TEMPLATE.format(class_list=class_list)


def build_response(class_ids: list[int], class_names: dict[int, str]) -> str:
    return ", ".join(class_names.get(cid, f"class_{cid}") for cid in class_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to VLM multi-label classification JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        default=r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\images",
        help="Root directory containing image files (.jpg / .png).",
    )
    parser.add_argument(
        "--labels",
        default=r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\labels",
        help="Root directory containing YOLO label .txt files.",
    )
    parser.add_argument(
        "--classes",
        default="data/classes.txt",
        help="Path to classes.txt (one class name per line).",
    )
    parser.add_argument(
        "--output_train",
        default="data/train.jsonl",
        help="Output path for training JSONL.",
    )
    parser.add_argument(
        "--output_val",
        default="data/val.jsonl",
        help="Output path for validation JSONL.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.10,
        help="Fraction of data to hold out for validation (0.0 to 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap the total number of samples processed (0 = no cap). Useful for quick tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    classes_file = Path(args.classes)

    # --- Validate paths ---
    for p, name in [(image_dir, "--images"), (label_dir, "--labels"), (classes_file, "--classes")]:
        if not p.exists():
            sys.exit(f"[ERROR] {name} path does not exist: {p}")

    class_names = load_classes(classes_file)
    print(f"[INFO] Loaded {len(class_names)} classes: {class_names}")

    user_prompt = build_user_prompt(class_names)

    # --- Discover label files (single-threaded, Windows-safe) ---
    label_files = sorted(label_dir.glob("*.txt"))
    print(f"[INFO] Found {len(label_files)} label files in {label_dir}")

    if args.max_samples > 0:
        label_files = label_files[: args.max_samples]
        print(f"[INFO] Capped to {args.max_samples} samples (--max_samples).")

    # --- Build records ---
    records: list[dict] = []
    skipped_no_image = 0
    skipped_empty_label = 0
    class_distribution: dict[int, int] = {cid: 0 for cid in class_names}

    for idx, label_file in enumerate(label_files):
        if idx % 1000 == 0:
            print(f"  Processing {idx}/{len(label_files)} ...", end="\r")

        stem = label_file.stem
        image_path = find_image(image_dir, stem)

        if image_path is None:
            skipped_no_image += 1
            continue

        class_ids = parse_yolo_label(label_file)
        if not class_ids:
            skipped_empty_label += 1
            continue

        for cid in class_ids:
            class_distribution[cid] = class_distribution.get(cid, 0) + 1

        records.append(
            {
                "image_path": str(image_path.resolve()),
                "label_ids": class_ids,
                "label_names": [class_names.get(cid, f"class_{cid}") for cid in class_ids],
                "prompt": user_prompt,
                "response": build_response(class_ids, class_names),
            }
        )

    print(f"\n[INFO] Valid records : {len(records)}")
    print(f"[INFO] Skipped (no matching image)  : {skipped_no_image}")
    print(f"[INFO] Skipped (empty label file)   : {skipped_empty_label}")
    print("[INFO] Class distribution (# images where class appears):")
    for cid, count in sorted(class_distribution.items()):
        print(f"       {cid}: {class_names.get(cid, '?'):15s} -> {count}")

    if not records:
        sys.exit("[ERROR] No valid records produced. Check --images and --labels paths.")

    # --- Train / Val split ---
    random.seed(args.seed)
    random.shuffle(records)

    val_size = max(1, int(len(records) * args.val_split))
    val_records = records[:val_size]
    train_records = records[val_size:]
    print(f"[INFO] Split -> train: {len(train_records)}, val: {len(val_records)}")

    # --- Write output ---
    for out_path_str, data in [
        (args.output_train, train_records),
        (args.output_val, val_records),
    ]:
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Wrote {len(data):>6,} records -> {out_path}")

    print("[DONE] Dataset preparation complete.")


if __name__ == "__main__":
    main()
