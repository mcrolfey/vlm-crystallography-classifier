#!/usr/bin/env python3
"""
j_self_improve.py
-----------------
Self-improvement loop with Jacobian-space (j-space) class-salience analysis.

Performs all operations of self_improve.py (training cycles, LM Studio
hyperparameter suggestions, full cycle history) and adds a Jacobian-Lens
interpretability step after each cycle.  The j-space analysis measures
how salient each class name is in the model's intermediate layer
representations — deeper diagnostic signal than loss alone.

After each training cycle:
  1.  Train for N steps  (same subprocess as self_improve.py)
  2.  Load the cycle checkpoint and run j-space analysis:
        - PRIMARY: Jacobian Lens (anthropics/jacobian-lens) via jlens.fit() +
          jlens.apply(), transporting residual-stream vectors into final-layer
          basis before decoding
        - FALLBACK: Logit Lens — projects each layer's hidden state through the
          final RMS norm + unembedding matrix; always works with any VLM
  3.  For each class name, report mean rank across layers and which layers
      first place the class in the top-20 / top-50 vocabulary predictions
  4.  Save a per-cycle salience JSON to outputs/j_self_improve/jspace/
  5.  Include salience table in the LM Studio prompt alongside loss history

Install jacobian-lens for the primary method:
    pip install git+https://github.com/anthropics/jacobian-lens.git

Usage:
    python j_self_improve.py --cycles 4 --steps_per_cycle 100 \\
        --lm_studio_model "google/gemma-4-26b-a4b"

    # Skip j-space (identical behaviour to self_improve.py):
    python j_self_improve.py --skip_jspace ...

    # Dry run — no training, test LM Studio + j-space pipeline:
    python j_self_improve.py --dry_run --lm_studio_model "your-model"
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import textwrap
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-improving VLM loop with Jacobian-space class-salience",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Loop control
    p.add_argument("--cycles",           type=int,   default=3)
    p.add_argument("--epochs_per_cycle", type=int,   default=2)
    p.add_argument("--steps_per_cycle",  type=int,   default=-1,
                   help="If > 0, use max_steps instead of full epochs per cycle.")
    # Initial training config (mirrors train_qwen_classifier.py defaults)
    p.add_argument("--model_id",        default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    p.add_argument("--train_jsonl",     default="data/train.jsonl")
    p.add_argument("--val_jsonl",       default="data/val.jsonl")
    p.add_argument("--classes",         default="data/classes.txt",
                   help="Path to classes.txt — one class name per line.")
    p.add_argument("--output_dir",      default="outputs/j_self_improve")
    p.add_argument("--learning_rate",   type=float, default=2e-4)
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=16)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument("--grad_accum",      type=int,   default=8)
    p.add_argument("--max_seq_length",  type=int,   default=768)
    p.add_argument("--max_pixels",      type=int,   default=200704)
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    # LM Studio
    p.add_argument("--lm_studio_url",   default="http://localhost:1234/v1")
    p.add_argument("--lm_studio_model", default="local-model")
    p.add_argument("--llm_temperature", type=float, default=0.3)
    p.add_argument("--llm_max_tokens",  type=int,   default=2048)
    # J-space
    p.add_argument("--skip_jspace",    action="store_true",
                   help="Skip j-space analysis (same behaviour as self_improve.py).")
    p.add_argument("--jspace_fit",     type=int, default=80,
                   help="Prompts used to fit the Jacobian Lens.")
    p.add_argument("--jspace_apply",   type=int, default=30,
                   help="Prompts to apply the lens to for salience scoring.")
    p.add_argument("--jspace_max_len", type=int, default=128,
                   help="Max tokens per j-space text prompt.")
    p.add_argument("--jspace_seed",    type=int, default=42)
    # Misc
    p.add_argument("--dry_run",    action="store_true",
                   help="Skip training; test LM Studio and j-space pipeline only.")
    p.add_argument("--log_dir",    default="outputs/j_self_improve/loop_logs")
    p.add_argument("--python",     default=sys.executable)
    p.add_argument("--best_config_path", default="outputs/self_improve/best_config.json",
                   help="Shared best-config path read by train_qwen_classifier.py.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training subprocess  (identical to self_improve.py)
# ---------------------------------------------------------------------------

def run_training_cycle(
    cycle: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    metrics_path: Path,
) -> dict[str, Any] | None:
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
# J-space: class helpers
# ---------------------------------------------------------------------------

def load_classes(path: Path) -> list[str]:
    if not path.exists():
        print(f"[WARN] classes.txt not found: {path} — j-space will use generic labels.")
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_val_records(val_jsonl: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    with open(val_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rng.shuffle(records)
    return records[:n]


def get_class_token_ids(class_names: list[str], tokenizer) -> dict[str, int]:
    """Map each class name to the token ID of its first subword."""
    result = {}
    for name in class_names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if ids:
            result[name] = ids[0]
    return result


# ---------------------------------------------------------------------------
# J-space: logit lens  (always works, used as fallback)
# ---------------------------------------------------------------------------

def run_logit_lens(
    model,
    tokenizer,
    records: list[dict],
    class_token_ids: dict[str, int],
    max_len: int,
) -> dict[str, Any]:
    """
    Standard logit lens: project each layer's last-position hidden state
    through the final RMS norm + unembedding matrix, then rank class tokens.

    Uses text-only prompts (the 'response' field from val records) so this
    works with any VLM without requiring image inputs.

    Returns: class_name -> {mean_rank, per_layer, layers_top20, layers_top50}
    """
    import torch

    # layer_idx -> class_name -> [ranks across prompts]
    layer_ranks: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    # Try to locate the final norm and lm_head
    final_norm = None
    lm_head    = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head

    if lm_head is None:
        print("[JSPACE] Could not locate lm_head — logit lens skipped.")
        return {}

    model.eval()
    n_layers_seen = 0

    with torch.no_grad():
        for rec in records:
            # Use the response text as the prompt so the model is in its
            # "normal" output distribution (class name + coordinates).
            text = rec.get("response", "")
            if not text:
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                ).to("cuda")

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except Exception as e:
                print(f"[JSPACE] Forward pass failed: {e}")
                continue

            hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
            if not hidden_states:
                continue

            n_layers_seen = max(n_layers_seen, len(hidden_states) - 1)

            for layer_idx, h in enumerate(hidden_states[1:]):  # skip embedding
                last_h = h[0, -1, :]  # (hidden_dim,)

                # Apply final norm before projecting (standard logit-lens recipe)
                if final_norm is not None:
                    last_h = final_norm(last_h.unsqueeze(0).unsqueeze(0)).squeeze()

                logits = lm_head(last_h.unsqueeze(0)).squeeze()  # (vocab,)
                sorted_ids = logits.argsort(descending=True).cpu().tolist()
                rank_map   = {tid: rank for rank, tid in enumerate(sorted_ids)}

                for name, tid in class_token_ids.items():
                    layer_ranks[layer_idx][name].append(
                        rank_map.get(tid, len(sorted_ids))
                    )

    if not layer_ranks:
        return {}

    result: dict[str, Any] = {}
    for name in class_token_ids:
        per_layer: list[float] = []
        for layer_idx in range(n_layers_seen):
            ranks = layer_ranks[layer_idx].get(name, [])
            per_layer.append(sum(ranks) / len(ranks) if ranks else float("inf"))

        if not per_layer:
            continue
        finite = [r for r in per_layer if r != float("inf")]
        overall_mean = sum(finite) / len(finite) if finite else float("inf")

        result[name] = {
            "mean_rank":   round(overall_mean, 1) if overall_mean != float("inf") else None,
            "per_layer":   [round(r, 1) if r != float("inf") else None for r in per_layer],
            "layers_top20": [i for i, r in enumerate(per_layer) if r < 20],
            "layers_top50": [i for i, r in enumerate(per_layer) if r < 50],
        }

    return result


# ---------------------------------------------------------------------------
# J-space: Jacobian Lens  (primary method)
# ---------------------------------------------------------------------------

def run_jacobian_lens(
    model,
    tokenizer,
    fit_records:   list[dict],
    apply_records: list[dict],
    class_token_ids: dict[str, int],
    ckpt_path: Path,
) -> dict[str, Any] | None:
    """
    Fit a Jacobian Lens on fit_records, apply it on apply_records.

    Uses jlens.from_hf() to wrap the model — works when the backbone
    follows the standard Qwen2 / transformer layout (model.model.layers,
    model.lm_head).  Returns None if the library is unavailable or the
    model layout is incompatible.

    Returns: class_name -> {mean_rank, per_layer, layers_top20, layers_top50}
    """
    try:
        import jlens
    except ImportError:
        print("[JSPACE] jacobian-lens not installed — falling back to logit lens.")
        print("         Install: pip install git+https://github.com/anthropics/jacobian-lens.git")
        return None

    import torch

    fit_texts   = [r.get("response", "") for r in fit_records   if r.get("response")]
    apply_texts = [r.get("response", "") for r in apply_records if r.get("response")]

    if not fit_texts:
        print("[JSPACE] No fit prompts — skipping Jacobian Lens.")
        return None

    try:
        print(f"[JSPACE] Wrapping model for jlens ...")
        jlens_model = jlens.from_hf(model, tokenizer)

        lens_ckpt = str(ckpt_path / "jlens_ckpt.pt")
        print(f"[JSPACE] Fitting Jacobian Lens on {len(fit_texts)} prompts ...")
        lens = jlens.fit(jlens_model, prompts=fit_texts, checkpoint_path=lens_ckpt)

        print(f"[JSPACE] Applying lens on {len(apply_texts)} prompts ...")
    except Exception as e:
        print(f"[JSPACE] Jacobian Lens setup failed ({e}) — falling back to logit lens.")
        return None

    # layer -> class -> [ranks]
    layer_ranks: dict[Any, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    vocab_size = tokenizer.vocab_size or 150000

    for text in apply_texts:
        if not text:
            continue
        try:
            lens_logits, _, _ = lens.apply(jlens_model, text, positions=[-1])
        except Exception as e:
            print(f"[JSPACE] lens.apply failed: {e}")
            continue

        for layer_key, logits in lens_logits.items():
            # logits may be (1, vocab) or (vocab,)
            logits_1d  = logits.view(-1)
            sorted_ids = logits_1d.argsort(descending=True).cpu().tolist()
            rank_map   = {tid: rank for rank, tid in enumerate(sorted_ids)}

            for name, tid in class_token_ids.items():
                layer_ranks[layer_key][name].append(rank_map.get(tid, vocab_size))

    if not layer_ranks:
        return None

    sorted_layers = sorted(layer_ranks.keys())
    result: dict[str, Any] = {}

    for name in class_token_ids:
        per_layer: list[float] = []
        for lk in sorted_layers:
            ranks = layer_ranks[lk].get(name, [])
            per_layer.append(sum(ranks) / len(ranks) if ranks else float("inf"))

        finite = [r for r in per_layer if r != float("inf")]
        overall_mean = sum(finite) / len(finite) if finite else float("inf")

        result[name] = {
            "mean_rank":    round(overall_mean, 1) if finite else None,
            "per_layer":    [round(r, 1) if r != float("inf") else None for r in per_layer],
            "layers_top20": [i for i, r in enumerate(per_layer) if r < 20],
            "layers_top50": [i for i, r in enumerate(per_layer) if r < 50],
        }

    return result


# ---------------------------------------------------------------------------
# J-space: orchestrator
# ---------------------------------------------------------------------------

def run_jspace_analysis(
    checkpoint_path: Path,
    args: argparse.Namespace,
    class_names: list[str],
    cycle: int,
    jspace_dir: Path,
) -> dict[str, Any] | None:
    """
    Load the cycle checkpoint, run Jacobian Lens (or logit lens fallback),
    save the salience report, and return it.
    """
    import torch

    adapter_path = checkpoint_path / "lora_adapter"
    if not adapter_path.exists():
        print(f"[JSPACE] Adapter not found at {adapter_path} — skipping j-space.")
        return None

    print(f"\n{'='*60}")
    print(f"  J-SPACE ANALYSIS — cycle {cycle}")
    print(f"{'='*60}")

    # Load checkpoint — training subprocess has already exited so VRAM is free
    try:
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            str(adapter_path),
            load_in_4bit=True,
            device_map={"": 0},
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        print(f"[JSPACE] Checkpoint loaded: {adapter_path.name}")
    except Exception as e:
        print(f"[JSPACE] Could not load checkpoint: {e}")
        return None

    class_token_ids = get_class_token_ids(class_names, tokenizer)
    if not class_token_ids:
        print("[JSPACE] No class token IDs resolved — check classes.txt and tokenizer.")
        del model
        torch.cuda.empty_cache()
        return None

    total_needed = args.jspace_fit + args.jspace_apply
    all_records  = load_val_records(args.val_jsonl, total_needed, args.jspace_seed)
    fit_records   = all_records[:args.jspace_fit]
    apply_records = all_records[args.jspace_fit:]

    # Try Jacobian Lens first, fall back to logit lens
    ckpt_jspace = jspace_dir / f"cycle_{cycle:02d}"
    ckpt_jspace.mkdir(parents=True, exist_ok=True)

    salience = run_jacobian_lens(
        model, tokenizer,
        fit_records, apply_records,
        class_token_ids,
        ckpt_path=ckpt_jspace,
    )
    method = "jacobian"

    if salience is None:
        print("[JSPACE] Running logit lens ...")
        salience = run_logit_lens(
            model, tokenizer,
            fit_records + apply_records,
            class_token_ids,
            max_len=args.jspace_max_len,
        )
        method = "logit"

    del model
    torch.cuda.empty_cache()

    if not salience:
        print("[JSPACE] No salience data computed.")
        return None

    report = {
        "cycle":           cycle,
        "method":          method,
        "timestamp":       datetime.now().isoformat(),
        "class_salience":  salience,
    }

    report_path = ckpt_jspace / "salience.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[JSPACE] Salience report -> {report_path}")

    # Print summary table
    print(f"\n[JSPACE] Class salience (method={method}):")
    print(f"  {'Class':<14} {'Mean rank':>10}  Top-20 layers")
    print("  " + "-" * 50)
    for cls, stats in sorted(salience.items(), key=lambda x: x[1].get("mean_rank") or 9999):
        mr  = stats.get("mean_rank")
        t20 = stats.get("layers_top20", [])
        print(f"  {cls:<14} {str(mr):>10}  {t20 if t20 else '(none)'}")

    return report


def format_salience_block(report: dict[str, Any]) -> str:
    """Format the salience report as a section for the LM Studio prompt."""
    if not report:
        return ""
    method = report.get("method", "?")
    cycle  = report.get("cycle", "?")
    data   = report.get("class_salience", {})
    if not data:
        return ""

    lines = [
        f"\n## J-space class salience (cycle {cycle}, method={method})\n",
        "Mean rank of each class-name token in intermediate model layers.",
        "Lower rank = class is more salient (model internalised it at that layer).",
        "Top-20 layers = layers where the class falls in the top-20 predicted tokens.",
        "(null = not enough data for that layer)\n",
        f"{'Class':<14} {'Mean rank':>10}  Top-20 layers",
        "-" * 55,
    ]
    for cls, stats in sorted(data.items(), key=lambda x: x[1].get("mean_rank") or 9999):
        mr  = stats.get("mean_rank")
        t20 = stats.get("layers_top20", [])
        lines.append(f"{cls:<14} {str(mr):>10}  {t20 if t20 else '(none)'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LM Studio suggestion engine  (extended to include j-space data)
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
    on limited VRAM (8 GB RTX 3070 Laptop). You review training metrics and
    Jacobian-space class-salience data, then suggest concrete hyperparameter or
    prompt changes to improve the next training cycle.

    Task context: the model is trained as a crystallographic phase detector. Each
    training sample contains TWO images (a bounding-box crop and the full microscopy
    image) and the model must output: CLASS at [x1, y1, x2, y2]. This is a grounding
    task — the model must learn precise coordinate output AND class names.

    J-space interpretation: a class with mean_rank < 50 in early layers is well
    internalised. A class absent from top-50 in all layers is not learned yet —
    suggest targeted changes (more LoRA capacity, lower LR, longer warmup).

    Rules:
    - Keep batch_size=1 (hard GPU constraint).
    - Keep max_pixels <= 200704 (8 GB constraint with two images per sample).
    - Prefer small, targeted changes over large sweeps.
    - Return ONLY valid JSON matching the schema — no markdown, no extra text.
""")


def build_user_prompt(
    cycle: int,
    metrics: dict[str, Any],
    current_config: dict[str, Any],
    history: list[dict[str, Any]],
    salience_report: dict[str, Any] | None,
) -> str:
    train_curve = metrics.get("train_loss_curve", [])
    eval_curve  = metrics.get("eval_loss_curve",  [])
    trend = ""
    if len(train_curve) >= 2:
        delta  = train_curve[-1] - train_curve[0]
        trend  = (f"Train loss moved {delta:+.4f} "
                  f"(start={train_curve[0]:.4f}, end={train_curve[-1]:.4f}).")
    if len(eval_curve) >= 2:
        delta  = eval_curve[-1] - eval_curve[0]
        trend += (f" Eval loss moved {delta:+.4f} "
                  f"(start={eval_curve[0]:.4f}, end={eval_curve[-1]:.4f}).")

    history_block = ""
    if history:
        lines = ["## History of completed cycles (oldest first)\n"]
        for h in history:
            cfg = h["config"]
            m   = h["metrics"] or {}
            lines.append(
                f"Cycle {h['cycle']}: "
                f"lr={cfg.get('learning_rate')}, lora_r={cfg.get('lora_r')}, "
                f"lora_alpha={cfg.get('lora_alpha')}, grad_accum={cfg.get('grad_accum')}, "
                f"warmup={cfg.get('warmup_ratio')} "
                f"-> train_loss={m.get('final_train_loss')}, "
                f"eval_loss={m.get('final_eval_loss')}"
            )
            if h.get("reasoning"):
                lines.append(f"  Your reasoning then: {h['reasoning']}")
            if h.get("jspace_summary"):
                lines.append(f"  J-space then: {h['jspace_summary']}")
        history_block = "\n".join(lines) + "\n\n"

    salience_block = format_salience_block(salience_report) if salience_report else ""

    return textwrap.dedent(f"""\
        {history_block}## Training cycle {cycle} results

        Final train loss : {metrics.get('final_train_loss')}
        Final eval loss  : {metrics.get('final_eval_loss')}
        Epochs trained   : {metrics.get('epochs_trained')}
        {trend}
        {salience_block}

        ## Current config
        {json.dumps(current_config, indent=2)}

        ## Task
        Suggest improvements for cycle {cycle + 1}. Return JSON matching this schema exactly:
        {SUGGESTION_SCHEMA}
    """)


def ask_lm_studio(prompt: str, args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 'openai' not installed. Run: pip install openai")
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

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        last_brace = raw.rfind("}")
        if last_brace != -1:
            trimmed = raw[:last_brace + 1]
            depth = 0
            for i in range(len(trimmed) - 1, -1, -1):
                if trimmed[i] == "}":
                    depth += 1
                elif trimmed[i] == "{":
                    depth -= 1
                if depth == 0:
                    candidate = trimmed[i:]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        print(f"[WARN] Could not parse LLM response as JSON: {exc}")
        return None


# ---------------------------------------------------------------------------
# Config application  (identical to self_improve.py)
# ---------------------------------------------------------------------------

ALLOWED_CONFIG_KEYS = {
    "learning_rate", "lora_r", "lora_alpha", "batch_size",
    "grad_accum", "max_seq_length", "warmup_ratio", "system_message",
}

VRAM_GUARDS = {
    "batch_size":     (1, 1),
    "max_seq_length": (256, 2048),
    "max_pixels":     (65536, 200704),
    "lora_r":         (4, 64),
    "lora_alpha":     (4, 128),
    "grad_accum":     (1, 32),
    "learning_rate":  (1e-6, 1e-2),
    "warmup_ratio":   (0.0, 0.2),
}


def apply_suggestions(current_config: dict, suggestion: dict) -> dict:
    new_config = deepcopy(current_config)
    for key, value in suggestion.get("suggested_config", {}).items():
        if key not in ALLOWED_CONFIG_KEYS:
            print(f"[GUARD] Ignoring unknown key: {key}")
            continue
        if key in VRAM_GUARDS:
            lo, hi = VRAM_GUARDS[key]
            clamped = max(lo, min(hi, value))
            if clamped != value:
                print(f"[GUARD] {key}: {value} -> clamped to {clamped}")
            value = clamped
        if current_config.get(key) != value:
            print(f"  {key}: {current_config.get(key)} -> {value}")
        new_config[key] = value
    return new_config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_cycle_log(
    log_dir: Path,
    cycle: int,
    config: dict,
    metrics: dict | None,
    suggestion: dict | None,
    next_config: dict,
    salience_report: dict | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "cycle":           cycle,
        "timestamp":       datetime.now().isoformat(),
        "config":          config,
        "metrics":         metrics,
        "suggestion":      suggestion,
        "next_config":     next_config,
        "jspace_salience": salience_report,
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
    jspace_dir  = Path(args.output_dir) / "jspace"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(Path(args.classes))

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
    print("  J-SELF-IMPROVEMENT LOOP")
    print(f"  Cycles: {args.cycles}  |  Epochs/cycle: {args.epochs_per_cycle}")
    print(f"  LM Studio: {args.lm_studio_url}  model={args.lm_studio_model}")
    print(f"  J-space: {'DISABLED' if args.skip_jspace else 'ENABLED'}"
          f"  (fit={args.jspace_fit}, apply={args.jspace_apply})")
    if class_names:
        print(f"  Classes: {len(class_names)} loaded from {args.classes}")
    print("="*60)

    best_eval_loss = float("inf")
    best_config:  dict[str, Any] = deepcopy(config)
    best_cycle:   int = 0
    history:      list[dict[str, Any]] = []

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'#'*60}")
        print(f"  CYCLE {cycle} / {args.cycles}")
        print(f"{'#'*60}")
        print(f"  Config: lr={config['learning_rate']}  lora_r={config['lora_r']}  "
              f"grad_accum={config['grad_accum']}")

        metrics_path = metrics_dir / f"cycle_{cycle:02d}_metrics.json"

        # --- Training ---
        if args.dry_run:
            print("[DRY RUN] Using synthetic metrics.")
            metrics = {
                "epochs_trained":   args.epochs_per_cycle,
                "final_train_loss": round(1.5 - cycle * 0.2, 4),
                "final_eval_loss":  round(1.6 - cycle * 0.15, 4),
                "train_loss_curve": [round(1.5 - cycle * 0.2 + i * 0.01, 4) for i in range(5)],
                "eval_loss_curve":  [round(1.6 - cycle * 0.15 + i * 0.01, 4) for i in range(3)],
                "config":           config,
            }
        else:
            metrics = run_training_cycle(cycle, config, args, metrics_path)

        if metrics is None:
            print(f"[ERROR] Cycle {cycle} failed — stopping loop.")
            break

        eval_loss = metrics.get("final_eval_loss") or float("inf")
        print(f"\n[METRICS] train_loss={metrics.get('final_train_loss')}  "
              f"eval_loss={eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_config    = deepcopy(config)
            best_cycle     = cycle
            print(f"[BEST] New best eval_loss={best_eval_loss:.4f} at cycle {best_cycle}")

        # --- J-space analysis ---
        salience_report = None
        if not args.skip_jspace and class_names:
            cycle_output = Path(args.output_dir) / f"cycle_{cycle:02d}"
            salience_report = run_jspace_analysis(
                checkpoint_path=cycle_output,
                args=args,
                class_names=class_names,
                cycle=cycle,
                jspace_dir=jspace_dir,
            )

        if cycle == args.cycles:
            save_cycle_log(log_dir, cycle, config, metrics, None, config, salience_report)
            print("\n[DONE] Final cycle complete. No further suggestions needed.")
            break

        # --- LM Studio suggestions (with history + j-space) ---
        user_prompt = build_user_prompt(cycle, metrics, config, history, salience_report)
        suggestion  = ask_lm_studio(user_prompt, args)

        reasoning = ""
        if suggestion is None:
            print("[WARN] No valid suggestion — keeping current config for next cycle.")
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

        # Compact j-space summary for future history entries
        jspace_summary = None
        if salience_report:
            top = sorted(
                salience_report["class_salience"].items(),
                key=lambda x: x[1].get("mean_rank") or 9999,
            )[:3]
            jspace_summary = ", ".join(
                f"{cls}=rank{s.get('mean_rank')}" for cls, s in top
            )

        history.append({
            "cycle":         cycle,
            "config":        deepcopy(config),
            "metrics":       metrics,
            "reasoning":     reasoning,
            "jspace_summary": jspace_summary,
        })

        save_cycle_log(log_dir, cycle, config, metrics, suggestion, next_config, salience_report)
        config = next_config

    # -----------------------------------------------------------------------
    # Save best config to the shared location read by train_qwen_classifier.py
    # -----------------------------------------------------------------------
    best_config_path = Path(args.best_config_path)
    best_config_out  = {k: v for k, v in best_config.items() if k != "system_message" or v}
    best_config_out["_best_cycle"]     = best_cycle
    best_config_out["_best_eval_loss"] = best_eval_loss
    best_config_out["_source"]         = "j_self_improve"
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
    print("  J-SELF-IMPROVEMENT LOOP COMPLETE")
    print(f"  Cycle logs    -> {log_dir}")
    print(f"  J-space data  -> {jspace_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
