#!/usr/bin/env python3
"""
test_model.py
-------------
Run the fine-tuned crystallography grounding VLM on one or more images.

For each image the model outputs all detected phases plus their bounding box
coordinates (normalised 0-1).  An SVG file is then generated with coloured
boxes and labels overlaid on the original image.

Usage:
    # Single image -> creates image.svg alongside image.jpg
    python test_model.py path/to/image.jpg

    # Directory -> processes every image in the folder
    python test_model.py path/to/folder/

    # Custom adapter
    python test_model.py image.jpg --adapter outputs/qwen_crystallography/lora_adapter

    # More tokens if model is verbose
    python test_model.py image.jpg --max_new_tokens 256

    # Skip SVG and just print detections
    python test_model.py image.jpg --no_svg
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# One distinctive colour per class (RGB hex).  Up to 16 classes; extras cycle.
_PALETTE = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E67E22",  # dark orange
    "#34495E",  # dark grey
    "#E91E63",  # pink
    "#00BCD4",  # cyan
    "#8BC34A",  # light green
    "#FF5722",  # deep orange
    "#607D8B",  # blue grey
    "#CDDC39",  # lime
    "#795548",  # brown
    "#9E9E9E",  # grey
]

# Detect line like: "A-CF at [0.21, 0.39, 0.29, 0.49]"
# Also tolerates minor format variations: parens, no "at", spaces, etc.
_DETECTION_RE = re.compile(
    r"([A-Za-z0-9_\-]+)"            # class name
    r"\s*(?:at|:)?\s*"              # optional "at" or ":"
    r"[\[\(]"                        # opening bracket
    r"\s*([\d.]+)\s*,\s*"           # x1
    r"([\d.]+)\s*,\s*"              # y1
    r"([\d.]+)\s*,\s*"              # x2
    r"([\d.]+)\s*"                  # y2
    r"[\]\)]"                        # closing bracket
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect crystallographic phases and generate SVG overlays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="*", help="Image file(s) or director(ies) to process.")
    p.add_argument("--image_list", default=None,
                   help="Path to a text file with one image path per line "
                        "(produced by prepare_yolo_to_vlm.py as data/test_images.txt).")
    p.add_argument("--adapter", default="outputs/qwen_crystallography/lora_adapter")
    p.add_argument("--classes", default="data/classes.txt")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens the model may generate (set higher for dense images).")
    p.add_argument("--no_svg", action="store_true", help="Skip SVG generation.")
    p.add_argument("--verbose", action="store_true", help="Print raw model output.")
    p.add_argument("--max_side", type=int, default=1120,
                   help="Resize images whose longest side exceeds this before inference.")
    p.add_argument("--results_json", default=None,
                   help="Path to save per-image detailed results as JSON.")
    p.add_argument("--test_log", default="outputs/test_log.json",
                   help="Cumulative test-run log (new runs appended below previous ones).")
    p.add_argument("--no_test_log", action="store_true",
                   help="Skip updating the cumulative test log.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return {i: name for i, name in enumerate(lines)}


def class_color(class_name: str, class_names: dict[int, str]) -> str:
    """Return a consistent hex colour for a class name."""
    names = list(class_names.values()) if class_names else []
    try:
        idx = names.index(class_name)
    except ValueError:
        idx = abs(hash(class_name)) % len(_PALETTE)
    return _PALETTE[idx % len(_PALETTE)]


def collect_images(p: Path) -> list[Path]:
    if p.is_dir():
        found = sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        if not found:
            print(f"[SKIP] No images found in: {p}")
        return found
    if p.is_file():
        return [p]
    print(f"[SKIP] Path not found: {p}")
    return []


def parse_detections(text: str) -> list[dict]:
    """
    Parse model output into a list of detection dicts.
    Each dict: {class_name, x1, y1, x2, y2}  (coords normalised 0-1)
    """
    detections = []
    for m in _DETECTION_RE.finditer(text):
        cls  = m.group(1)
        x1, y1, x2, y2 = float(m.group(2)), float(m.group(3)), \
                          float(m.group(4)), float(m.group(5))
        # Sanity clamp to [0, 1]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 > x1 and y2 > y1:
            detections.append({"class_name": cls, "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2})
    return detections


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_svg(
    image_path: Path,
    detections: list[dict],
    class_names: dict[int, str],
    max_side: int = 1120,
) -> str:
    """
    Build an SVG string with the original image embedded and coloured
    bounding boxes + labels drawn on top.

    The SVG uses an <image> element referencing the original file by a
    relative path, so open it from the same directory as the image.
    """
    img = Image.open(image_path)
    W, H = img.size

    # Scale display size so it fits in a browser without scrolling
    if max(W, H) > max_side:
        scale = max_side / max(W, H)
        dW, dH = int(W * scale), int(H * scale)
    else:
        dW, dH = W, H

    label_h  = 22   # height of label bar in px (SVG units)
    font_sz  = 13
    pad      = 4    # text padding inside label bar
    stroke_w = 2

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{dW}" height="{dH}" viewBox="0 0 {dW} {dH}">',
        f'  <title>Detections: {_escape(image_path.name)}</title>',
        f'  <image href="{_escape(image_path.name)}" '
        f'x="0" y="0" width="{dW}" height="{dH}" '
        f'preserveAspectRatio="none"/>',
    ]

    for det in detections:
        color = class_color(det["class_name"], class_names)
        bx1 = int(det["x1"] * dW)
        by1 = int(det["y1"] * dH)
        bx2 = int(det["x2"] * dW)
        by2 = int(det["y2"] * dH)
        bw  = max(1, bx2 - bx1)
        bh  = max(1, by2 - by1)

        # Box outline
        lines.append(
            f'  <rect x="{bx1}" y="{by1}" width="{bw}" height="{bh}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_w}" '
            f'stroke-opacity="0.95"/>'
        )

        # Label background — place above box, or inside if box is at top
        lby = by1 - label_h if by1 >= label_h else by1
        label_text = _escape(det["class_name"])
        label_w = len(det["class_name"]) * (font_sz * 0.65) + pad * 2
        lines.append(
            f'  <rect x="{bx1}" y="{lby}" width="{label_w:.0f}" height="{label_h}" '
            f'fill="{color}" opacity="0.92"/>'
        )
        lines.append(
            f'  <text x="{bx1 + pad}" y="{lby + label_h - pad}" '
            f'font-family="Arial,sans-serif" font-size="{font_sz}" '
            f'fill="white" font-weight="bold">{label_text}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

def load_model(adapter_path: Path):
    if not adapter_path.exists():
        sys.exit(
            f"[ERROR] Adapter not found: {adapter_path}\n"
            "        Make sure training completed and the path is correct."
        )
    print(f"[INFO] Loading adapter from: {adapter_path}")
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        str(adapter_path),
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastVisionModel.for_inference(model)
    if hasattr(processor, "image_processor"):
        try:
            processor.image_processor.min_pixels = 64 * 28 * 28
            processor.image_processor.max_pixels = 200704
        except AttributeError:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                str(adapter_path),
                min_pixels=64 * 28 * 28,
                max_pixels=200704,
            )
    print("[INFO] Model ready.\n")
    return model, processor


def build_detection_prompt(class_names: dict[int, str]) -> str:
    if class_names:
        class_list = ", ".join(class_names.values())
    else:
        class_list = "unknown"

    return (
        "You are an expert geologist specialising in polarised light microscopy.\n\n"
        f"Available crystallographic phases: {class_list}\n\n"
        "Identify and locate EVERY instance of each phase visible in this microscopy image.\n\n"
        "For each detection output exactly one line in this format:\n"
        "CLASS at [x1, y1, x2, y2]\n\n"
        "Where x1, y1, x2, y2 are the bounding box corners normalised to 0-1 "
        "(0,0 = top-left corner, 1,1 = bottom-right corner).\n\n"
        "Output only detection lines — no explanation, no preamble. "
        "If nothing is detected write: none"
    )


def run_detection(
    image_path: Path,
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
    max_side: int,
) -> str:
    import torch

    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    except Exception as e:
        return f"[ERROR opening image: {e}]"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", truncation=False
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=1.0,
            do_sample=False,
        )

    generated = processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    class_names = load_classes(Path(args.classes))
    if class_names:
        print(f"[INFO] Classes: {list(class_names.values())}")
    else:
        print("[WARN] classes.txt not found — class list will not be included in prompt.")

    prompt = build_detection_prompt(class_names)
    model, processor = load_model(Path(args.adapter))

    # Expand paths — from positional args and/or --image_list file
    all_images: list[Path] = []
    for img_arg in args.images:
        all_images.extend(collect_images(Path(img_arg)))

    if args.image_list:
        list_path = Path(args.image_list)
        if not list_path.exists():
            sys.exit(f"[ERROR] --image_list file not found: {list_path}")
        with open(list_path, encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                if p.is_file():
                    all_images.append(p)
                elif line.strip():
                    print(f"[SKIP] Not found: {p}")

    if not all_images:
        sys.exit("[ERROR] No valid image files found.")

    print(f"[INFO] Processing {len(all_images)} image(s)...\n")
    summary: list[dict] = []

    for image_path in all_images:
        print(f"Image : {image_path.name}")

        raw_output = run_detection(
            image_path, model, processor, prompt,
            args.max_new_tokens, args.max_side,
        )

        if args.verbose:
            print(f"Raw   : {raw_output}")

        detections = parse_detections(raw_output)

        if not detections:
            print("Result: (no detections parsed)")
        else:
            for det in detections:
                print(f"  {det['class_name']:10s}  [{det['x1']:.3f}, {det['y1']:.3f}, "
                      f"{det['x2']:.3f}, {det['y2']:.3f}]")

        svg_path = None
        if not args.no_svg:
            svg_str = generate_svg(
                image_path, detections, class_names, max_side=args.max_side
            )
            svg_path = image_path.with_suffix(".svg")
            svg_path.write_text(svg_str, encoding="utf-8")
            print(f"SVG   : {svg_path}")

        summary.append({
            "image":      str(image_path),
            "raw_output": raw_output,
            "detections": detections,
            "svg":        str(svg_path) if svg_path else None,
        })
        print()

    # Summary table
    if len(summary) > 1:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        for r in summary:
            n = len(r["detections"])
            name = Path(r["image"]).name
            classes_found = list({d["class_name"] for d in r["detections"]})
            print(f"  {name:35s}  {n:2d} detection(s)  {classes_found}")

    # -----------------------------------------------------------------------
    # Compute aggregate stats for this run
    # -----------------------------------------------------------------------
    total_detections = sum(len(r["detections"]) for r in summary)
    images_with_det  = sum(1 for r in summary if r["detections"])
    per_class: dict[str, int] = defaultdict(int)
    for r in summary:
        for det in r["detections"]:
            per_class[det["class_name"]] += 1

    print(f"\n[RESULTS] {total_detections} detection(s) across "
          f"{len(summary)} image(s)  "
          f"({images_with_det} with detections, "
          f"{len(summary) - images_with_det} empty)")

    # -----------------------------------------------------------------------
    # Save per-image detailed results JSON  (--results_json)
    # -----------------------------------------------------------------------
    if args.results_json:
        results_path = Path(args.results_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[RESULTS] Detailed results -> {results_path}")

    # -----------------------------------------------------------------------
    # Update cumulative test log  (--test_log)
    # -----------------------------------------------------------------------
    if not args.no_test_log:
        log_path = Path(args.test_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing entries (if any)
        existing: list[dict] = []
        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = []

        run_entry = {
            "run_date":              datetime.now().isoformat(timespec="seconds"),
            "adapter":               str(Path(args.adapter).resolve()),
            "total_images":          len(summary),
            "images_with_detections": images_with_det,
            "images_no_detection":   len(summary) - images_with_det,
            "total_detections":      total_detections,
            "per_class":             dict(sorted(per_class.items())),
            "results_file":          str(Path(args.results_json).resolve())
                                     if args.results_json else None,
        }

        existing.append(run_entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[RESULTS] Test log updated ({len(existing)} run(s) total) -> {log_path}")


if __name__ == "__main__":
    main()
