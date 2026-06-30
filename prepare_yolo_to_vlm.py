#!/usr/bin/env python3
"""
prepare_yolo_to_vlm.py
----------------------
Converts a YOLO bounding-box dataset into a per-box VLM grounding dataset.

For every bounding box annotation:
  1. Extracts a padded crop of that region from the full image
  2. Saves the crop to data/crops/
  3. Writes a JSONL record with crop_path, image_path, class info, bbox, response

Training prompt  (two images):
  [CROP IMAGE]  [FULL IMAGE]
  "The first image is a zoomed crop from the microscopy image.
   What phase is in the crop and where is it in the full image?"

Response format (one line, easy to parse at inference):
  A-CF at [0.21, 0.39, 0.29, 0.49]

At inference the model receives only the full image and outputs all detections
in the same single-line format, which test_model.py parses into SVG overlays.

Usage:
    python prepare_yolo_to_vlm.py
    python prepare_yolo_to_vlm.py --images path/to/images --labels path/to/labels
    python prepare_yolo_to_vlm.py --crop_padding 0.15 --min_crop_px 160
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TRAIN_PROMPT = (
    "The first image is a zoomed crop extracted from the second microscopy image. "
    "Identify the crystallographic phase visible in the crop and state where it is "
    "located in the full image.\n\n"
    "Reply in this exact format (one line only):\n"
    "CLASS at [x1, y1, x2, y2]\n\n"
    "Where x1 y1 x2 y2 are the bounding box corners normalised to 0–1 "
    "(0,0 = top-left, 1,1 = bottom-right)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        sys.exit(f"[ERROR] classes.txt is empty: {path}")
    return {i: name for i, name in enumerate(lines)}


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def extract_crop(
    image: Image.Image,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    padding: float,
    min_px: int,
) -> tuple[Image.Image, list[float]]:
    """
    Crop the bounding-box region with padding from the full image.

    Returns:
        crop        – PIL image of the padded region
        bbox_norm   – [x1, y1, x2, y2] of the ORIGINAL (unpadded) box, normalised 0-1
    """
    W, H = image.size

    # Original normalised box
    ox1 = max(0.0, x_center - width  / 2)
    oy1 = max(0.0, y_center - height / 2)
    ox2 = min(1.0, x_center + width  / 2)
    oy2 = min(1.0, y_center + height / 2)

    # Padded crop region
    pad_x = width  * padding
    pad_y = height * padding
    cx1 = max(0.0, ox1 - pad_x)
    cy1 = max(0.0, oy1 - pad_y)
    cx2 = min(1.0, ox2 + pad_x)
    cy2 = min(1.0, oy2 + pad_y)

    # Convert to pixels
    px1, py1 = int(cx1 * W), int(cy1 * H)
    px2, py2 = int(cx2 * W), int(cy2 * H)
    px2, py2 = max(px1 + 1, px2), max(py1 + 1, py2)

    crop = image.crop((px1, py1, px2, py2))

    # Enforce minimum crop size
    cw, ch = crop.size
    if cw < min_px or ch < min_px:
        scale = max(min_px / cw, min_px / ch)
        crop = crop.resize(
            (max(min_px, int(cw * scale)), max(min_px, int(ch * scale))),
            Image.LANCZOS,
        )

    bbox_norm = [round(ox1, 4), round(oy1, 4), round(ox2, 4), round(oy2, 4)]
    return crop, bbox_norm


def parse_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return list of (class_id, x_c, y_c, w, h) from a YOLO label file."""
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                boxes.append((cid, xc, yc, bw, bh))
            except ValueError:
                continue
    return boxes


def oversample_records(
    records: list[dict],
    max_ratio: int,
) -> list[dict]:
    """Repeat minority-class records so all classes appear proportionally."""
    from collections import Counter
    counts: Counter = Counter(r["class_id"] for r in records)
    if not counts:
        return records
    sorted_counts = sorted(counts.values())
    target = sorted_counts[len(sorted_counts) // 2]  # median

    result = []
    for rec in records:
        cnt = counts[rec["class_id"]]
        repeat = min(max(1, math.ceil(target / cnt)), max_ratio)
        result.extend([rec] * repeat)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert YOLO dataset to per-box VLM grounding JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images",  default=r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\images")
    p.add_argument("--labels",  default=r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\labels")
    p.add_argument("--classes", default="data/classes.txt")
    p.add_argument("--crops_dir",     default="data/crops")
    p.add_argument("--output_train",  default="data/train.jsonl")
    p.add_argument("--output_val",    default="data/val.jsonl")
    p.add_argument("--val_split",     type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_samples",   type=int,   default=0,
                   help="Cap total boxes processed (0 = all). For quick tests.")
    p.add_argument("--crop_padding",  type=float, default=0.10,
                   help="Fractional padding added around each bounding box crop.")
    p.add_argument("--min_crop_px",   type=int,   default=112,
                   help="Minimum crop side length in pixels.")
    p.add_argument("--max_oversample_ratio", type=int, default=15,
                   help="Max times a minority-class box may be repeated. 1 = off.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    image_dir   = Path(args.images)
    label_dir   = Path(args.labels)
    classes_file = Path(args.classes)
    crops_dir   = Path(args.crops_dir)

    for p, name in [(image_dir, "--images"), (label_dir, "--labels"),
                    (classes_file, "--classes")]:
        if not p.exists():
            sys.exit(f"[ERROR] {name} path not found: {p}")

    class_names = load_classes(classes_file)
    print(f"[INFO] Classes: {class_names}")

    crops_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.txt"))
    print(f"[INFO] Found {len(label_files)} label files")

    if args.max_samples > 0:
        label_files = label_files[: args.max_samples]

    records: list[dict] = []
    skipped_no_image = 0
    skipped_small    = 0
    class_dist: dict[int, int] = {i: 0 for i in class_names}

    for file_idx, label_file in enumerate(label_files):
        if file_idx % 500 == 0:
            print(f"  {file_idx}/{len(label_files)} ...", end="\r")

        stem = label_file.stem
        image_path = find_image(image_dir, stem)
        if image_path is None:
            skipped_no_image += 1
            continue

        boxes = parse_yolo_boxes(label_file)
        if not boxes:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"\n[WARN] Cannot open {image_path}: {e}")
            continue

        for box_idx, (cid, xc, yc, bw, bh) in enumerate(boxes):
            # Skip degenerate boxes (< 0.5% of image dimension)
            if bw < 0.005 or bh < 0.005:
                skipped_small += 1
                continue

            crop, bbox = extract_crop(
                image, xc, yc, bw, bh,
                padding=args.crop_padding,
                min_px=args.min_crop_px,
            )

            crop_filename = f"{stem}_{box_idx:04d}.jpg"
            crop_path = crops_dir / crop_filename
            crop.save(str(crop_path), "JPEG", quality=92)

            class_name = class_names.get(cid, f"class_{cid}")
            response   = f"{class_name} at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            class_dist[cid] = class_dist.get(cid, 0) + 1

            records.append({
                "crop_path":  str(crop_path.resolve()),
                "image_path": str(image_path.resolve()),
                "class_id":   cid,
                "class_name": class_name,
                "bbox":       bbox,
                "prompt":     TRAIN_PROMPT,
                "response":   response,
            })

    print(f"\n[INFO] Total box records   : {len(records)}")
    print(f"[INFO] Skipped (no image)  : {skipped_no_image}")
    print(f"[INFO] Skipped (tiny box)  : {skipped_small}")
    print("[INFO] Class distribution (# boxes):")
    for cid, cnt in sorted(class_dist.items()):
        print(f"       {cid}: {class_names.get(cid,'?'):10s} -> {cnt}")

    # Shuffle before split so val images are representative
    random.seed(args.seed)
    random.shuffle(records)

    # Oversample training set only (applied after split)
    val_size   = max(1, int(len(records) * args.val_split))
    val_raw    = records[:val_size]
    train_raw  = records[val_size:]

    if args.max_oversample_ratio > 1:
        train_records = oversample_records(train_raw, args.max_oversample_ratio)
        print(f"[INFO] Training: {len(train_raw)} -> {len(train_records)} after oversampling")
    else:
        train_records = train_raw

    val_records = val_raw
    print(f"[INFO] Val: {len(val_records)} boxes (no oversampling)")

    for out_path, data in [
        (args.output_train, train_records),
        (args.output_val,   val_records),
    ]:
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Wrote {len(data):>7,} records -> {out_path}")

    print("[DONE] Dataset preparation complete.")


if __name__ == "__main__":
    main()
