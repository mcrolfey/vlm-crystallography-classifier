#!/usr/bin/env python3
"""
self_improve.py
---------------
Recursive self-improvement loop for Qwen2.5-VL crystallography fine-tuning.

After each training cycle the local LLM (served by LM Studio on localhost:1234)
reviews the metrics and suggests hyperparameter / prompt improvements. Those
suggestions are applied to the next training run automatically.

Usage:
    python self_improve.py
    python self_improve.py --cycles 5 --epochs_per_cycle 2
    python self_improve.py --lm_studio_model "your-model-name" --dry_run

Requirements:
    pip install openai          # for LM Studio API calls
    LM Studio running locally with a model loaded and server started on port 1234.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-improving VLM fine-tuning loop via LM Studio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Loop control
    p.add_argument("--cycles", type=int, default=3,
                   help="Number of self-improvement cycles to run.")
    p.add_argument("--epochs_per_cycle", type=int, default=2,
                   help="Training epochs per cycle.")
    p.add_argument("--steps_per_cycle", type=int, default=-1,
                   help="If > 0, use max_steps instead of full epochs per cycle. "
                        "E.g. 100 steps @ ~9s/it = ~15 min per cycle.")
    # Initial training config (mirrors train_qwen_classifier.py defaults)
    p.add_argument("--model_id", default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    p.add_argument("--train_jsonl", default="data/train.jsonl")
    p.add_argument("--val_jsonl",   default="data/val.jsonl")
    p.add_argument("--output_dir",  default="outputs/self_improve")
    p.add_argument("--learning_rate",  type=float, default=2e-4)
    p.add_argument("--lora_r",         type=int,   default=16)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--grad_accum",     type=int,   default=8)
    p.add_argument("--max_seq_length", type=int,   default=768,
                   help="Token budget; 768 handles crop+full image + coordinate response.")
    p.add_argument("--max_pixels",     type=int,   default=200704,
                   help="Max image pixels. Keep <= 200704 on 8 GB with two images per sample.")
    p.add_argument("--warmup_ratio",   type=float, default=0.03)
    # LM Studio
    p.add_argument("--lm_studio_url",   default="http://localhost:1234/v1",
                   help="LM Studio OpenAI-compatible API base URL.")
    p.add_argument("--lm_studio_model", default="local-model",
                   help="Model name as shown in LM Studio (or 'local-model').")
    p.add_argument("--llm_temperature", type=float, default=0.3)
    p.add_argument("--llm_max_tokens",  type=int,   default=2048)
    # Misc
    p.add_argument("--dry_run", action="store_true",
                   help="Skip actual training; only test the LM Studio suggestion loop.")
    p.add_argument("--log_dir", default="outputs/self_improve/loop_logs",
                   help="Directory for per-cycle JSON logs.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use for subprocess training calls.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

def run_training_cycle(
    cycle: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    metrics_path: Path,
) -> dict[str, Any] | None:
    """
    Launch train_qwen_classifier.py as a subprocess with the given config.
    Returns the metrics dict written to metrics_path, or None on failure.
    """
    cycle_output = Path(args.output_dir) / f"cycle_{cycle:02d}"
    cycle_output.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parent / "train_qwen_classifier.py"
    cmd = [
        args.python, str(script),
        "--model_id",       config["model_id"],
        "--train_jsonl",    args.train_jsonl,
        "--val_jsonl",      args.val_jsonl,
        "--output_dir",     str(cycle_output),
        "--epochs",         str(args.epochs_per_cycle),
        "--max_steps",      str(args.steps_per_cycle),
        "--learning_rate",  str(config["learning_rate"]),
        "--lora_r",         str(config["lora_r"]),
        "--lora_alpha",     str(config["lora_alpha"]),
        "--batch_size",     str(config["batch_size"]),
        "--grad_accum",     str(config["grad_accum"]),
        "--max_seq_length", str(config["max_seq_length"]),
        "--max_pixels",     str(config["max_pixels"]),
        "--warmup_ratio",   str(config["warmup_ratio"]),
        "--metrics_output", str(metrics_path),
    ]
    if config.get("system_message"):
        cmd += ["--system_message", config["system_message"]]

    print(f"\n{'='*60}")
    print(f"  CYCLE {cycle}: launching training subprocess")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd[:6])} ...")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Training subprocess exited with code {result.returncode}")
        return None

    if not metrics_path.exists():
        print(f"[ERROR] Metrics file not written: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LM Studio suggestion engine
# ---------------------------------------------------------------------------

SUGGESTION_SCHEMA = """\
{
  "reasoning": "<1-3 sentences explaining the diagnosis and rationale>",
  "suggested_config": {
    "learning_rate":  <float>,
    "lora_r":         <int>,
    "lora_alpha":     <int>,
    "batch_size":     <int>,
    "grad_accum":     <int>,
    "max_seq_length": <int>,
    "warmup_ratio":   <float>,
    "system_message": "<string>"
  },
  "data_suggestions": ["<optional list of data/prompt improvement ideas>"]
}"""

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert ML engineer specialising in fine-tuning vision-language models
    on limited VRAM (8 GB RTX 3070 Laptop). You review training metrics and suggest
    concrete hyperparameter or prompt changes to improve the next training cycle.

    Task context: the model is trained as a crystallographic phase detector. Each
    training sample contains TWO images (a bounding-box crop and the full microscopy
    image) and the model must output: CLASS at [x1, y1, x2, y2]. This is a grounding
    task, not simple classification — the model must learn precise coordinate output.

    Rules:
    - Keep batch_size=1 (hard GPU constraint).
    - Keep max_pixels <= 200704 (8 GB constraint with two images per sample).
    - Prefer small, targeted changes over large sweeps.
    - Return ONLY valid JSON matching the schema — no markdown fences, no explanation outside the JSON.
""")


def build_user_prompt(cycle: int, metrics: dict[str, Any], current_config: dict[str, Any]) -> str:
    train_curve = metrics.get("train_loss_curve", [])
    eval_curve  = metrics.get("eval_loss_curve", [])
    trend = ""
    if len(train_curve) >= 2:
        delta = train_curve[-1] - train_curve[0]
        trend = f"Train loss moved {delta:+.4f} over the cycle (start={train_curve[0]:.4f}, end={train_curve[-1]:.4f})."
    if len(eval_curve) >= 2:
        delta = eval_curve[-1] - eval_curve[0]
        trend += f" Eval loss moved {delta:+.4f} (start={eval_curve[0]:.4f}, end={eval_curve[-1]:.4f})."

    return textwrap.dedent(f"""\
        ## Training cycle {cycle} results

        Final train loss : {metrics.get('final_train_loss')}
        Final eval loss  : {metrics.get('final_eval_loss')}
        Epochs trained   : {metrics.get('epochs_trained')}
        {trend}

        ## Current config
        {json.dumps(current_config, indent=2)}

        ## Task
        Suggest improvements for cycle {cycle + 1}. Return JSON matching this schema exactly:
        {SUGGESTION_SCHEMA}
    """)


def ask_lm_studio(
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """
    Call LM Studio's OpenAI-compatible API and parse the JSON response.
    Returns the parsed suggestion dict or None on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 'openai' package not installed. Run: pip install openai")
        return None

    client = OpenAI(base_url=args.lm_studio_url, api_key="lm-studio")

    print(f"\n[LLM] Asking LM Studio ({args.lm_studio_model}) for suggestions ...")
    try:
        response = client.chat.completions.create(
            model=args.lm_studio_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
    except Exception as exc:
        print(f"[ERROR] LM Studio API call failed: {exc}")
        print("        Is LM Studio running with a model loaded and the local server started?")
        return None

    raw = response.choices[0].message.content.strip()
    print(f"[LLM] Raw response:\n{raw}\n")

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Response was truncated mid-JSON — try trimming to the last complete object
        last_brace = raw.rfind("}")
        if last_brace != -1:
            trimmed = raw[:last_brace + 1]
            # Walk back to find the matching opening brace for the root object
            depth = 0
            for i in range(len(trimmed) - 1, -1, -1):
                if trimmed[i] == "}":
                    depth += 1
                elif trimmed[i] == "{":
                    depth -= 1
                if depth == 0:
                    candidate = trimmed[i:]
                    try:
                        parsed = json.loads(candidate)
                        print("[WARN] LLM response was truncated — recovered partial JSON.")
                        return parsed
                    except json.JSONDecodeError:
                        break
        print(f"[WARN] Could not parse LLM response as JSON: {exc}")
        return None


# ---------------------------------------------------------------------------
# Apply suggestions safely
# ---------------------------------------------------------------------------

ALLOWED_CONFIG_KEYS = {
    "learning_rate", "lora_r", "lora_alpha", "batch_size",
    "grad_accum", "max_seq_length", "warmup_ratio", "system_message",
}

VRAM_GUARDS = {
    "batch_size":     (1, 1),        # must stay 1 on 8 GB
    "max_seq_length": (256, 2048),
    "max_pixels":     (65536, 200704),   # 8 GB ceiling with two images per sample
    "lora_r":         (4, 64),
    "lora_alpha":     (4, 128),
    "grad_accum":     (1, 32),
    "learning_rate":  (1e-6, 1e-2),
    "warmup_ratio":   (0.0, 0.2),
}


def apply_suggestions(
    current_config: dict[str, Any],
    suggestion: dict[str, Any],
) -> dict[str, Any]:
    """Merge LLM suggestions into config with safety clamping."""
    new_config = deepcopy(current_config)
    suggested = suggestion.get("suggested_config", {})

    for key, value in suggested.items():
        if key not in ALLOWED_CONFIG_KEYS:
            print(f"[WARN] Ignoring unknown suggestion key: {key}")
            continue
        if key in VRAM_GUARDS:
            lo, hi = VRAM_GUARDS[key]
            clamped = max(lo, min(hi, value))
            if clamped != value:
                print(f"[GUARD] {key}: {value} -> clamped to {clamped}")
            value = clamped
        old = current_config.get(key)
        if old != value:
            print(f"  {key}: {old} -> {value}")
        new_config[key] = value

    return new_config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_cycle_log(
    log_dir: Path,
    cycle: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
    suggestion: dict[str, Any] | None,
    next_config: dict[str, Any],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "cycle":       cycle,
        "timestamp":   datetime.now().isoformat(),
        "config":      config,
        "metrics":     metrics,
        "suggestion":  suggestion,
        "next_config": next_config,
    }
    log_path = log_dir / f"cycle_{cycle:02d}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    print(f"[LOG] Cycle log -> {log_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    log_dir     = Path(args.log_dir)
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Starting config
    config: dict[str, Any] = {
        "model_id":       args.model_id,
        "learning_rate":  args.learning_rate,
        "lora_r":         args.lora_r,
        "lora_alpha":     args.lora_alpha,
        "batch_size":     args.batch_size,
        "grad_accum":     args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "max_pixels":     args.max_pixels,
        "warmup_ratio":   args.warmup_ratio,
        "system_message": None,
    }

    print("\n" + "="*60)
    print("  SELF-IMPROVEMENT LOOP")
    print(f"  Cycles: {args.cycles}  |  Epochs/cycle: {args.epochs_per_cycle}")
    print(f"  LM Studio: {args.lm_studio_url}  model={args.lm_studio_model}")
    print("="*60)

    best_eval_loss = float("inf")
    best_config:    dict[str, Any] = deepcopy(config)
    best_cycle:     int = 0

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'#'*60}")
        print(f"  CYCLE {cycle} / {args.cycles}")
        print(f"{'#'*60}")
        print(f"  Config: lr={config['learning_rate']}  lora_r={config['lora_r']}  "
              f"grad_accum={config['grad_accum']}")

        metrics_path = metrics_dir / f"cycle_{cycle:02d}_metrics.json"

        # --- Training ---
        if args.dry_run:
            print("[DRY RUN] Skipping training. Using fake metrics.")
            metrics = {
                "epochs_trained":    args.epochs_per_cycle,
                "final_train_loss":  round(1.5 - cycle * 0.2, 4),
                "final_eval_loss":   round(1.6 - cycle * 0.15, 4),
                "train_loss_curve":  [round(1.5 - cycle * 0.2 + i * 0.01, 4) for i in range(5)],
                "eval_loss_curve":   [round(1.6 - cycle * 0.15 + i * 0.01, 4) for i in range(3)],
                "config":            config,
            }
        else:
            metrics = run_training_cycle(cycle, config, args, metrics_path)

        if metrics is None:
            print(f"[ERROR] Cycle {cycle} failed — stopping loop.")
            break

        eval_loss = metrics.get("final_eval_loss") or float("inf")
        print(f"\n[METRICS] train_loss={metrics.get('final_train_loss')}  "
              f"eval_loss={eval_loss}")

        # Track the best config seen so far
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_config    = deepcopy(config)
            best_cycle     = cycle
            print(f"[BEST]  New best eval_loss={best_eval_loss:.4f} at cycle {best_cycle}")

        if cycle == args.cycles:
            save_cycle_log(log_dir, cycle, config, metrics, None, config)
            print("\n[DONE] Final cycle complete. No further suggestions needed.")
            break

        # --- Ask LM Studio for suggestions ---
        user_prompt = build_user_prompt(cycle, metrics, config)
        suggestion  = ask_lm_studio(user_prompt, args)

        if suggestion is None:
            print("[WARN] No valid suggestion received — keeping current config for next cycle.")
            next_config = deepcopy(config)
        else:
            reasoning = suggestion.get("reasoning", "")
            if reasoning:
                print(f"\n[LLM REASONING] {reasoning}")
            data_tips = suggestion.get("data_suggestions", [])
            if data_tips:
                print("[LLM DATA TIPS]")
                for tip in data_tips:
                    print(f"  - {tip}")
            print("\n[CONFIG CHANGES]")
            next_config = apply_suggestions(config, suggestion)

        save_cycle_log(log_dir, cycle, config, metrics, suggestion, next_config)
        config = next_config

    # -----------------------------------------------------------------------
    # Save the best config so train_qwen_classifier.py picks it up next run
    # -----------------------------------------------------------------------
    best_config_path = Path(args.output_dir) / "best_config.json"
    best_config_out  = {k: v for k, v in best_config.items() if k != "system_message" or v}
    best_config_out["_best_cycle"]     = best_cycle
    best_config_out["_best_eval_loss"] = best_eval_loss
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config_out, f, indent=2)

    print(f"\n[BEST CONFIG] Cycle {best_cycle} had lowest eval_loss={best_eval_loss:.4f}")
    print(f"[BEST CONFIG] Saved -> {best_config_path}")
    print("[BEST CONFIG] train_qwen_classifier.py will use these as defaults next run.\n")
    for k, v in best_config_out.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("  SELF-IMPROVEMENT LOOP COMPLETE")
    print(f"  Logs -> {log_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
