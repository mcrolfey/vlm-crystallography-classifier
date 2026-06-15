#!/usr/bin/env python3
"""
train_qwen_classifier.py
------------------------
Fine-tunes Qwen2.5-VL-7B-Instruct (4-bit, via Unsloth) as a multi-label
crystallography phase classifier using LoRA + SFTTrainer.

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

Usage:
    python train_qwen_classifier.py
    python train_qwen_classifier.py --train_jsonl data/train.jsonl --val_jsonl data/val.jsonl
    python train_qwen_classifier.py --epochs 3 --batch_size 1 --grad_accum 8

Tested on: RTX 3060 12 GB, RTX 4090 24 GB, A100 40 GB.
Minimum VRAM: ~8 GB with batch_size=1, grad_accum=8, max_seq_length=1024.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Windows-specific: prevent multiprocessing spawn issues in DataLoader
# ---------------------------------------------------------------------------
import torch.multiprocessing as _mp
_mp.set_start_method("spawn", force=True)  # noqa: E402

import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Argument parsing (before heavy imports so --help is fast)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL-7B as a crystallography classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
    p.add_argument("--train_jsonl", default="data/train.jsonl")
    p.add_argument("--val_jsonl", default="data/val.jsonl")
    p.add_argument("--output_dir", default="outputs/qwen_crystallography")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size.")
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=1536)
    p.add_argument("--max_pixels", type=int, default=1003520,
                   help="Max total image pixels sent to the vision encoder (controls VRAM).")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--download_retries", type=int, default=5,
                   help="Number of retry attempts for model download.")
    p.add_argument("--no_finetune_vision", action="store_true",
                   help="Freeze vision encoder, only train the LLM layers.")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a checkpoint directory to resume training from.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fault-tolerant model loader
# ---------------------------------------------------------------------------

def load_model_with_retry(
    model_id: str,
    max_seq_length: int,
    max_pixels: int,
    retries: int = 5,
):
    """
    Load Qwen2.5-VL via Unsloth with exponential-backoff retry.

    Returns (model, processor).
    Raises RuntimeError if all attempts fail.
    """
    from unsloth import FastVisionModel

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Loading model (attempt {attempt}/{retries}): {model_id}")
            model, processor = FastVisionModel.from_pretrained(
                model_id,
                load_in_4bit=True,
                use_gradient_checkpointing="unsloth",
                max_seq_length=max_seq_length,
                # Qwen2.5-VL vision kwargs forwarded through **kwargs
                min_pixels=256 * 28 * 28,
                max_pixels=max_pixels,
            )
            print("[INFO] Model loaded successfully.")
            return model, processor
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
            print(f"[WARN] Load attempt {attempt} failed: {exc}")
            if attempt < retries:
                print(f"       Retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to load {model_id} after {retries} attempts. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert mineralogist specialising in polarised light microscopy "
    "and scanning electron microscopy (SEM) for asbestos fibre identification "
    "and crystallographic phase classification."
)


class CrystallographyDataset(Dataset):
    """
    Reads a JSONL file produced by prepare_yolo_to_vlm.py and returns
    Qwen2.5-VL conversation dicts with PIL images embedded.

    Each sample format expected by UnslothVisionDataCollator:
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": <PIL.Image>},
                        {"type": "text",  "text": "<prompt>"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "<response>",
                },
            ]
        }
    """

    def __init__(self, jsonl_path: str, max_samples: int = 0):
        self.records: list[dict] = []
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            sys.exit(f"[ERROR] Dataset not found: {jsonl_path}\n"
                     "        Run prepare_yolo_to_vlm.py first.")

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if max_samples > 0:
            self.records = self.records[:max_samples]

        print(f"[INFO] Dataset '{jsonl_path.name}' loaded: {len(self.records)} samples.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]

        # Load image — skip broken files gracefully
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Cannot open image {rec['image_path']}: {exc}. Using blank.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": rec["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": rec["response"],
                },
            ]
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Load model + processor
    # -----------------------------------------------------------------------
    model, processor = load_model_with_retry(
        model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        max_pixels=args.max_pixels,
        retries=args.download_retries,
    )

    # -----------------------------------------------------------------------
    # 2. Apply LoRA
    # -----------------------------------------------------------------------
    from unsloth import FastVisionModel

    finetune_vision = not args.no_finetune_vision
    print(f"[INFO] LoRA rank={args.lora_r}, alpha={args.lora_alpha}, "
          f"finetune_vision_layers={finetune_vision}")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # -----------------------------------------------------------------------
    # 3. Datasets
    # -----------------------------------------------------------------------
    train_dataset = CrystallographyDataset(args.train_jsonl)
    val_dataset = CrystallographyDataset(args.val_jsonl)

    # -----------------------------------------------------------------------
    # 4. Collator
    # -----------------------------------------------------------------------
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(model, processor)

    # -----------------------------------------------------------------------
    # 5. Trainer configuration
    # -----------------------------------------------------------------------
    from unsloth import is_bf16_supported
    from trl import SFTConfig, SFTTrainer

    use_bf16 = is_bf16_supported()
    print(f"[INFO] Mixed precision: {'bf16' if use_bf16 else 'fp16'}")

    # Effective batch size = batch_size * grad_accum * num_gpus
    effective_batch = args.batch_size * args.grad_accum
    print(f"[INFO] Effective batch size: {effective_batch}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        # Optimizer — Unsloth ships a paged 8-bit Adam for low VRAM
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Precision
        fp16=not use_bf16,
        bf16=use_bf16,
        # Windows / single-GPU safe
        dataloader_num_workers=0,      # CRITICAL: 0 prevents Windows spawn freeze
        dataloader_pin_memory=False,   # avoids OOM on constrained systems
        # Required for VLM datasets
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Sequence length
        max_seq_length=args.max_seq_length,
        # Misc
        seed=3407,
        report_to="none",              # set to "wandb" to enable W&B logging
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    # -----------------------------------------------------------------------
    # 6. Train
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Starting fine-tuning …")
    print("=" * 60 + "\n")

    resume = args.resume_from_checkpoint
    if resume and not Path(resume).exists():
        print(f"[WARN] --resume_from_checkpoint path not found: {resume}. Starting fresh.")
        resume = None

    try:
        trainer.train(resume_from_checkpoint=resume)
    except torch.cuda.OutOfMemoryError:
        print(
            "\n[FATAL] CUDA Out of Memory.\n"
            "  Try: --grad_accum 16, --max_seq_length 1024, --max_pixels 501760, "
            "or --no_finetune_vision"
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 7. Save
    # -----------------------------------------------------------------------
    out_path = Path(args.output_dir)
    model.save_pretrained(str(out_path / "lora_adapter"))
    processor.save_pretrained(str(out_path / "lora_adapter"))
    print(f"\n[INFO] LoRA adapter saved → {out_path / 'lora_adapter'}")

    # Optional: merge LoRA into model weights (requires more RAM/VRAM)
    try:
        print("[INFO] Merging LoRA weights into base model (16-bit) ...")
        model.save_pretrained_merged(
            str(out_path / "merged_16bit"),
            processor,
            save_method="merged_16bit",
        )
        print(f"[INFO] Merged model saved → {out_path / 'merged_16bit'}")
    except Exception as exc:
        print(f"[WARN] Merge failed (likely OOM): {exc}")
        print("       The LoRA adapter at 'lora_adapter/' is still usable for inference.")

    print("\n[DONE] Training complete.")


# ---------------------------------------------------------------------------
# Inference helper (run separately after training)
# ---------------------------------------------------------------------------

def run_inference(
    adapter_path: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Load the fine-tuned LoRA adapter and classify a single image.

    Example:
        from train_qwen_classifier import run_inference
        result = run_inference(
            adapter_path="outputs/qwen_crystallography/lora_adapter",
            image_path="sample.jpg",
            prompt="<your prompt from classes.txt>",
        )
        print(result)
    """
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(adapter_path, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

    generated = processor.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return generated.strip()


if __name__ == "__main__":
    main()
