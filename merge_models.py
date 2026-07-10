#!/usr/bin/env python3
"""
merge_models.py
---------------
Merge two or more fine-tuned Qwen2.5-VL LoRA adapters using MergeKit.

After the self-improvement loop produces multiple cycle checkpoints, this
script picks the best-performing ones, converts each LoRA to a full 16-bit
model, then uses MergeKit to blend them.  The merged model often outperforms
any single cycle because it combines complementary strengths.

Install MergeKit first:
    pip install mergekit

Usage:
    # Merge two cycle adapters (SLERP, 50/50 blend)
    python merge_models.py \\
        outputs/self_improve/cycle_01/lora_adapter \\
        outputs/self_improve/cycle_02/lora_adapter

    # Merge three adapters with TIES (better for 3+ models)
    python merge_models.py \\
        outputs/self_improve/cycle_01/lora_adapter \\
        outputs/self_improve/cycle_02/lora_adapter \\
        outputs/self_improve/cycle_03/lora_adapter \\
        --method ties

    # Custom weights (model A weighted 0.7, model B weighted 0.3)
    python merge_models.py model_a/ model_b/ --weights 0.7 0.3

    # Custom output location
    python merge_models.py model_a/ model_b/ --output outputs/merged/best_blend

Merge methods:
    slerp  Spherical linear interpolation — best for two models, smooth blend.
    ties   Task vector merging — handles 2+ models, prunes conflicting weights.
    dare   DARE + TIES — best for models with different fine-tuning domains.
    linear Simple weighted average — fast but less principled than slerp/ties.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge fine-tuned Qwen2.5-VL adapters with MergeKit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("models", nargs="+",
                   help="Paths to LoRA adapter directories or merged 16-bit model directories.")
    p.add_argument("--method", default="slerp",
                   choices=["slerp", "ties", "dare", "linear"],
                   help="MergeKit merge method.")
    p.add_argument("--weights", type=float, nargs="*", default=None,
                   help="Per-model blend weights (must match number of models). "
                        "Defaults to equal weights.")
    p.add_argument("--slerp_t", type=float, default=0.5,
                   help="SLERP interpolation parameter (0=model_1, 1=model_2). "
                        "Only used with --method slerp.")
    p.add_argument("--density", type=float, default=0.5,
                   help="TIES/DARE density: fraction of task-vector weights to keep.")
    p.add_argument("--output", default="outputs/merged",
                   help="Output directory for the merged model.")
    p.add_argument("--base_model_id", default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
                   help="HuggingFace model ID of the base model (needed to merge LoRA).")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--skip_lora_merge", action="store_true",
                   help="Skip LoRA-to-16bit conversion; treat input paths as full models.")
    p.add_argument("--keep_temp", action="store_true",
                   help="Keep temporary merged-16bit directories after merging.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LoRA -> full model conversion
# ---------------------------------------------------------------------------

def is_lora_adapter(path: Path) -> bool:
    """True if the directory looks like a LoRA adapter (has adapter_config.json)."""
    return (path / "adapter_config.json").exists()


def merge_lora_to_full(
    adapter_path: Path,
    base_model_id: str,
    output_path: Path,
) -> None:
    """
    Load a 4-bit LoRA adapter with Unsloth, merge the LoRA weights into the
    base model, and save as a standard 16-bit HuggingFace model.
    """
    print(f"  [LoRA->16bit] {adapter_path.name} ...")

    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        str(adapter_path),
        load_in_4bit=True,
        device_map={"": 0},
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(output_path),
        processor,
        save_method="merged_16bit",
    )
    processor.save_pretrained(str(output_path))
    print(f"  [LoRA->16bit] Saved -> {output_path}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MergeKit config generation
# ---------------------------------------------------------------------------

def build_slerp_config(
    model_paths: list[str],
    t: float,
    dtype: str,
) -> dict:
    if len(model_paths) != 2:
        sys.exit("[ERROR] SLERP requires exactly 2 models.")
    return {
        "merge_method": "slerp",
        "base_model": model_paths[0],
        "models": [{"model": p} for p in model_paths],
        "parameters": {"t": t},
        "dtype": dtype,
    }


def build_ties_config(
    model_paths: list[str],
    weights: list[float],
    density: float,
    dtype: str,
) -> dict:
    return {
        "merge_method": "ties",
        "base_model": model_paths[0],
        "models": [
            {
                "model": p,
                "parameters": {"density": density, "weight": w},
            }
            for p, w in zip(model_paths, weights)
        ],
        "parameters": {"normalize": True},
        "dtype": dtype,
    }


def build_dare_config(
    model_paths: list[str],
    weights: list[float],
    density: float,
    dtype: str,
) -> dict:
    return {
        "merge_method": "dare_ties",
        "base_model": model_paths[0],
        "models": [
            {
                "model": p,
                "parameters": {"density": density, "weight": w},
            }
            for p, w in zip(model_paths, weights)
        ],
        "parameters": {"normalize": True},
        "dtype": dtype,
    }


def build_linear_config(
    model_paths: list[str],
    weights: list[float],
    dtype: str,
) -> dict:
    return {
        "merge_method": "linear",
        "models": [
            {"model": p, "parameters": {"weight": w}}
            for p, w in zip(model_paths, weights)
        ],
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate inputs
    model_dirs = [Path(m) for m in args.models]
    for d in model_dirs:
        if not d.exists():
            sys.exit(f"[ERROR] Model path not found: {d}")

    n = len(model_dirs)
    weights = args.weights or [1.0 / n] * n
    if len(weights) != n:
        sys.exit(f"[ERROR] --weights has {len(weights)} values but {n} models were given.")

    # Normalise weights to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights]

    print(f"\n{'='*60}")
    print(f"  MergeKit Model Merger")
    print(f"  Method : {args.method}")
    print(f"  Models : {n}")
    for d, w in zip(model_dirs, weights):
        print(f"    {d.name}  (weight={w:.3f})")
    print(f"  Output : {args.output}")
    print(f"{'='*60}\n")

    # Check MergeKit is installed
    try:
        import mergekit  # noqa: F401
    except ImportError:
        sys.exit(
            "[ERROR] mergekit is not installed.\n"
            "        Run: pip install mergekit\n"
            "        Then re-run this script."
        )

    # -----------------------------------------------------------------------
    # Step 1: Convert any LoRA adapters to full 16-bit models
    # -----------------------------------------------------------------------
    temp_dirs: list[Path] = []
    full_model_paths: list[str] = []

    if args.skip_lora_merge:
        full_model_paths = [str(d) for d in model_dirs]
    else:
        temp_root = Path(args.output) / "_temp_merged"
        for i, adapter_dir in enumerate(model_dirs):
            if is_lora_adapter(adapter_dir):
                temp_out = temp_root / f"model_{i:02d}"
                merge_lora_to_full(adapter_dir, args.base_model_id, temp_out)
                full_model_paths.append(str(temp_out))
                temp_dirs.append(temp_out)
            else:
                print(f"  [INFO] {adapter_dir.name} — already a full model, using directly.")
                full_model_paths.append(str(adapter_dir))

    # -----------------------------------------------------------------------
    # Step 2: Build MergeKit YAML config
    # -----------------------------------------------------------------------
    method = args.method
    if method == "slerp":
        config = build_slerp_config(full_model_paths, args.slerp_t, args.dtype)
    elif method == "ties":
        config = build_ties_config(full_model_paths, weights, args.density, args.dtype)
    elif method == "dare":
        config = build_dare_config(full_model_paths, weights, args.density, args.dtype)
    else:  # linear
        config = build_linear_config(full_model_paths, weights, args.dtype)

    # Write config to a temp YAML file
    try:
        import yaml
        yaml_available = True
    except ImportError:
        yaml_available = False

    config_path = Path(args.output) / "mergekit_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml_available:
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: write JSON (MergeKit also accepts JSON)
        config_path = config_path.with_suffix(".json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    print(f"[INFO] MergeKit config written -> {config_path}")

    # -----------------------------------------------------------------------
    # Step 3: Run MergeKit
    # -----------------------------------------------------------------------
    output_dir = Path(args.output) / "merged_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Running MergeKit merge -> {output_dir}")
    cmd = [
        sys.executable, "-m", "mergekit.scripts.run_yaml",
        str(config_path),
        str(output_dir),
        "--copy-tokenizer",
        "--allow-crimes",          # needed for VLM / non-standard architectures
        "--out-shard-size", "5B",  # keep shards manageable
        "--lazy-unpickle",
    ]

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print("\n[ERROR] MergeKit failed.")
        print("        Check the config above and ensure all model paths exist.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 4: Clean up temp directories
    # -----------------------------------------------------------------------
    if temp_dirs and not args.keep_temp:
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        print("[INFO] Temporary 16-bit models removed.")

    print(f"\n{'='*60}")
    print(f"  MERGE COMPLETE")
    print(f"  Merged model -> {output_dir}")
    print(f"{'='*60}")
    print(f"\nTo test the merged model:")
    print(f"  python test_model.py your_image.jpg --adapter {output_dir}")


if __name__ == "__main__":
    main()
