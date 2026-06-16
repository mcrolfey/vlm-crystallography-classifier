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
        description="Fine-tune Qwen2.5-VL as a crystallography classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Model ---
    # Default is 3B: fits comfortably on 8 GB VRAM with frozen vision encoder.
    # Switch to 7B only if you have ≥ 16 GB VRAM.
    p.add_argument("--model_id", default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
                   help="HuggingFace model ID. 3B works on 8 GB; 7B needs ≥ 16 GB.")
    p.add_argument("--train_jsonl", default="data/train.jsonl")
    p.add_argument("--val_jsonl", default="data/val.jsonl")
    p.add_argument("--output_dir", default="outputs/qwen_crystallography")
    # --- Training ---
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size.")
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="Max token sequence length. Reduce to save VRAM.")
    p.add_argument("--max_pixels", type=int, default=602112,
                   help="Max image pixels fed to the vision encoder "
                        "(602112 ≈ 784×768, safe for 8 GB). Reduce further if OOM.")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    # --- Download ---
    p.add_argument("--download_retries", type=int, default=20,
                   help="Retry attempts for model download (resumable on dropout).")
    p.add_argument("--download_timeout", type=int, default=300,
                   help="Seconds before a single HTTP request times out.")
    # --- VRAM knobs ---
    # Frozen vision encoder is the default for 8 GB GPUs: the 3B LLM layers
    # already hold the class-label knowledge; the vision encoder only needs
    # to extract patch features, which it does well out-of-the-box.
    p.add_argument("--finetune_vision", action="store_true",
                   help="Also train vision encoder LoRA layers (needs ≥ 16 GB VRAM).")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a checkpoint directory to resume training from.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fault-tolerant model downloader + loader (two separate phases)
# ---------------------------------------------------------------------------

# Network exception types we want to retry on.
# Populated lazily in _network_exceptions() so imports stay optional.
def _network_exceptions() -> tuple[type[Exception], ...]:
    """Return a tuple of exception types that indicate a transient network error."""
    import socket
    exc_types: list[type[Exception]] = [
        ConnectionError,       # built-in: covers ConnectionResetError, BrokenPipeError, etc.
        TimeoutError,          # built-in
        OSError,               # covers ECONNRESET, EHOSTUNREACH on Windows
        socket.timeout,
    ]
    # requests / urllib3 (HF Hub uses these internally)
    try:
        import requests.exceptions as req_exc
        exc_types += [
            req_exc.ConnectionError,
            req_exc.Timeout,
            req_exc.ChunkedEncodingError,   # connection drops mid-stream
            req_exc.ReadTimeout,
            req_exc.ConnectTimeout,
        ]
    except ImportError:
        pass
    try:
        import urllib3.exceptions as u3_exc
        exc_types += [
            u3_exc.ProtocolError,
            u3_exc.ReadTimeoutError,
            u3_exc.ConnectionError,
        ]
    except ImportError:
        pass
    return tuple(set(exc_types))


def download_model_with_retry(
    model_id: str,
    timeout: int = 300,
    retries: int = 20,
) -> str:
    """
    Download every model shard to the local HuggingFace cache using
    snapshot_download, with per-file retry on connection drops.

    Key properties:
      - Files already in the HF cache are skipped entirely — a dropout
        only costs the file currently in-flight, not the whole download.
      - Each HTTP request has a hard `timeout`-second deadline.
      - Backs off exponentially up to 60 s between attempts.
      - Skips TF/Flax weights (saves ~7 GB for this model).

    Returns the local cache directory path (pass to from_pretrained).
    """
    import socket
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        EntryNotFoundError,
    )

    # Tell the HF Hub HTTP client how long to wait for a single response.
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)

    network_exc = _network_exceptions() + (HfHubHTTPError,)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Downloading model [{attempt}/{retries}]: {model_id}")
            print(f"       Timeout per request: {timeout}s  |  Already-cached files are skipped.")
            local_dir = snapshot_download(
                repo_id=model_id,
                repo_type="model",
                # Skip TF/Flax variants — saves ~7 GB
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"[INFO] Download complete. Cache dir: {local_dir}")
            return local_dir

        except (RepositoryNotFoundError, EntryNotFoundError) as exc:
            # Fatal — wrong model ID or file deleted. Don't retry.
            raise RuntimeError(
                f"Model '{model_id}' not found on HuggingFace Hub.\n"
                f"Check the model ID and your internet connection. Detail: {exc}"
            ) from exc

        except network_exc as exc:
            last_exc = exc
            wait = min(2 ** attempt, 60)   # cap at 60 s
            print(
                f"\n[WARN] Network error on attempt {attempt}/{retries}: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < retries:
                print(f"       Waiting {wait}s before retry ... (Ctrl+C to abort)")
                time.sleep(wait)

        except Exception as exc:
            # Unknown error — still retry, but warn loudly.
            last_exc = exc
            wait = min(2 ** attempt, 60)
            print(f"\n[WARN] Unexpected error on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                print(f"       Waiting {wait}s before retry ...")
                time.sleep(wait)

    raise RuntimeError(
        f"Download of '{model_id}' failed after {retries} attempts.\n"
        f"Last error: {last_exc}\n"
        f"Tip: increase --download_retries or --download_timeout, "
        f"or manually download via: huggingface-cli download {model_id}"
    )


def load_model_from_cache(
    local_dir: str,
    max_seq_length: int,
    max_pixels: int,
):
    """
    Load Qwen2.5-VL from the local HF cache directory.
    No network access required after download_model_with_retry() succeeds.

    Returns (model, processor).
    """
    from unsloth import FastVisionModel

    # Free any scratch memory left over from the download phase.
    torch.cuda.empty_cache()

    free_gb = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) / 1e9
    print(f"[INFO] Free VRAM before load: {free_gb:.1f} GB")
    print(f"[INFO] Loading model from local cache: {local_dir}")

    # device_map={"": 0} forces all layers onto GPU 0 without running
    # the FP16-based size estimation that incorrectly concludes the 4-bit
    # model (~4.5 GB) won't fit in 8 GB and tries to offload to CPU.
    model, processor = FastVisionModel.from_pretrained(
        local_dir,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        device_map={"": 0},
    )
    # min_pixels / max_pixels are image-processor settings, not model args.
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = max_pixels
    print("[INFO] Model loaded successfully.")
    return model, processor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert geologist specialising in polarised light microscopy "
    "and scanning electron microscopy (SEM) for crystallographic phase classification."
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
    # 1. Download model files (resumable, dropout-tolerant), then load
    # -----------------------------------------------------------------------
    local_dir = download_model_with_retry(
        model_id=args.model_id,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )
    model, processor = load_model_from_cache(
        local_dir=local_dir,
        max_seq_length=args.max_seq_length,
        max_pixels=args.max_pixels,
    )

    # -----------------------------------------------------------------------
    # 2. Apply LoRA
    # -----------------------------------------------------------------------
    from unsloth import FastVisionModel

    finetune_vision = args.finetune_vision
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
            "or remove --finetune_vision"
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
