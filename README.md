# VLM Crystallography Classifier

Fine-tune **Qwen2.5-VL** as a crystallography phase **detector** — the model
finds, classifies, and localises phases in microscopy images in a single
forward pass, with no secondary object detector needed at inference time.

---

## Overview

This repo turns a YOLO-labelled microscopy dataset into a vision-language
model that can look at a new image and output every detected phase with its
bounding box coordinates:

```
Phase_3 at [0.21, 0.39, 0.29, 0.49]
Phase_1 at [0.67, 0.12, 0.71, 0.19]
```

`test_model.py` parses that output and generates an SVG overlay — coloured
boxes and labels drawn directly over the original image.

### Pipeline at a glance

```
1. prepare_yolo_to_vlm.py   Extract one crop per bounding box + stratified split
           |
           v
2. self_improve.py          Short training cycles + LLM hyperparameter search
           |                Saves best_config.json when done
           v
3. train_qwen_classifier.py Full training run (auto-loads best_config.json)
           |
           v
4. test_model.py            Run on held-out test images -> .svg overlays
           |
           v (optional)
   merge_models.py          Blend multiple cycle checkpoints with MergeKit
```

### Key features

| Feature | Detail |
|---------|--------|
| **Single-model detection** | VLM detects and classifies in one pass — no YOLO at inference |
| **Grounding training** | Each sample: crop + full image → `CLASS at [x1,y1,x2,y2]` |
| **Stratified split** | Train / val / test split preserves per-class ratios |
| **Class balancing** | Minority classes oversampled up to 15× in training |
| **Self-improvement** | LLM reviews loss curves and tunes hyperparameters between cycles |
| **Auto best config** | Best hyperparameters saved and loaded automatically next run |
| **Model merging** | MergeKit blends multiple trained checkpoints (SLERP / TIES / DARE) |
| **SVG output** | Coloured bounding boxes + labels overlaid on original image |
| **8 GB VRAM** | 3B model trains comfortably on a laptop GPU |
| **Windows safe** | Single-threaded data loading, no spawn issues |

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.11 |
| PyTorch | >= 2.7.0 (see install note below) |
| CUDA driver | 12.6+ |
| GPU VRAM | 8 GB minimum |
| LM Studio | Only needed for self-improvement loop |

### VRAM guide

| GPU VRAM | Recommended model | Vision encoder |
|----------|------------------|----------------|
| 8 GB | `Qwen2.5-VL-3B` (default) | Frozen (default) |
| 12 GB | `Qwen2.5-VL-3B` | Can unfreeze (`--finetune_vision`) |
| 16 GB+ | `Qwen2.5-VL-7B` | Can unfreeze |

> The 7B model (~4.5 GB in 4-bit) plus training activations exhausts 8 GB
> before the loss function runs.  The 3B model (~2 GB in 4-bit) trains
> reliably on 8 GB and matches 7B accuracy for this task.

---

## Environment Setup

### 1 — Create a conda environment

```bash
conda create -n vlm-crystallography python=3.11 -y
conda activate vlm-crystallography
```

### 2 — Install PyTorch >= 2.7.0

Check your CUDA version:
```bash
nvidia-smi
```

Then install the matching wheel:
```bash
# CUDA 12.6
pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.8
pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu128
```

> PyTorch 2.7.0 is the minimum — earlier versions lack
> `torch.utils._pytree.register_constant` which is required by
> `torchao >= 0.16.0` (a hard dependency of recent `peft`).

### 3 — Install Unsloth + dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install Pillow openai
```

> **Windows / triton:** if the install fails on `triton`, run
> `pip install triton-windows` first.

---

## Class Map

Classes are defined in `data/classes.txt` — one name per line, where the
line index matches the YOLO class ID.  Edit this file before running
`prepare_yolo_to_vlm.py`.

Example `classes.txt` for a 7-class dataset:
```
Phase_0
Phase_1
Phase_2
Phase_3
Phase_4
Phase_5
Phase_6
```

Replace each `Phase_N` with your own class names.

---

## How It Works

```
Training
  YOLO labels  ->  extract one crop per bounding box  ->  save to data/crops/
  [CROP IMAGE] + [FULL IMAGE]  ->  VLM learns  ->  "Phase_3 at [x1,y1,x2,y2]"

Inference
  [FULL IMAGE]  ->  VLM outputs all detections  ->  parse coordinates
             ->  SVG file with coloured boxes + labels
```

The crop teaches the model what each class looks like up close.  The full
image gives it spatial context to report accurate coordinates.  At inference,
only the full image is needed — the model finds and localises everything itself.

---

## Repository Layout

```
.
├── data/
│   ├── classes.txt           <- class names (one per line, index = YOLO class ID)
│   ├── crops/                <- bounding-box crops (generated, gitignored)
│   ├── train.jsonl           <- generated by prepare_yolo_to_vlm.py (gitignored)
│   ├── val.jsonl             <- generated (gitignored)
│   ├── test.jsonl            <- generated (gitignored)
│   └── test_images.txt       <- list of held-out image paths (gitignored)
├── prepare_yolo_to_vlm.py    <- Step 1: crop extraction + stratified JSONL split
├── self_improve.py           <- Step 2: hyperparameter search loop via LM Studio
├── train_qwen_classifier.py  <- Step 3: full fine-tuning run
├── test_model.py             <- Step 4: inference + SVG overlay generation
├── merge_models.py           <- Optional: MergeKit blend of cycle checkpoints
└── outputs/
    ├── qwen_crystallography/
    │   ├── lora_adapter/     <- LoRA weights — always saved, use for inference
    │   ├── merged_16bit/     <- Full bf16 model (saved if VRAM allows)
    │   └── checkpoint-*/     <- Rolling checkpoints (3 kept)
    ├── self_improve/
    │   ├── best_config.json  <- Best hyperparameters (auto-loaded next training run)
    │   ├── cycle_01/         <- Per-cycle outputs
    │   ├── metrics/          <- Loss metrics per cycle
    │   └── loop_logs/        <- Full JSON logs (config + metrics + LLM suggestion)
    └── merged/
        ├── mergekit_config.yaml
        └── merged_model/     <- Blended model from merge_models.py
```

---

## Step 1 — Prepare the Dataset

Reads YOLO label files, extracts one padded crop per bounding box, and writes
three stratified JSONL files.  Single-threaded — safe on Windows.

```bash
python prepare_yolo_to_vlm.py \
    --images "C:\path\to\images" \
    --labels "C:\path\to\labels" \
    --classes data/classes.txt \
    --crops_dir    data/crops \
    --output_train data/train.jsonl \
    --output_val   data/val.jsonl \
    --output_test  data/test.jsonl \
    --val_split  0.10 \
    --test_split 0.10
```

Quick test with 100 label files:
```bash
python prepare_yolo_to_vlm.py --max_samples 100
```

**Stratified split:** the dataset is split at the image level (never box
level — that causes data leakage) and every class keeps the same proportion
across all three splits.

| Split | Default | Used for |
|-------|---------|----------|
| train | 80% | Training + class-balanced oversampling |
| val   | 10% | Eval loss during training |
| test  | 10% | Held-out — run `test_model.py` after training |

Also writes `data/test_images.txt` — a plain text list of test image paths
for use with `test_model.py --image_list`.

---

## Step 2 — Find the Best Hyperparameters (Self-Improvement Loop)

Run this before the full training run.  The loop trains for short cycles, has
a local LLM analyse the loss curves, adjusts the hyperparameters, and repeats.
At the end it saves the best config to `outputs/self_improve/best_config.json`,
which Step 3 loads automatically.

### Set up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai)
2. Load any chat model (Gemma, Llama, Mistral, etc.)
3. Go to **Local Server** tab → click **Start Server** (port 1234)
4. Confirm the model name:

```bash
curl http://localhost:1234/v1/models
```

### Dry run first (confirm connection)

```bash
python self_improve.py --dry_run --lm_studio_model "YOUR_MODEL_NAME"
```

If `[LLM REASONING]` appears in the output, you are connected and ready.

### Run the loop

```bash
python self_improve.py --cycles 3 --steps_per_cycle 100 --lm_studio_model "YOUR_MODEL_NAME"
```

Each cycle is ~17 min on an RTX 3070 at 100 steps.

| `--steps_per_cycle` | Time/cycle (RTX 3070) | Signal quality |
|---|---|---|
| 50  | ~8 min  | Minimal — early loss only |
| 100 | ~17 min | Good — enough to show trend |
| 200 | ~34 min | Strong |
| 500 | ~85 min | Full early-epoch coverage |

### What the LLM can adjust

| Parameter | Range |
|---|---|
| `learning_rate` | 1e-6 to 1e-2 |
| `lora_r` | 4 to 64 |
| `lora_alpha` | 4 to 128 |
| `grad_accum` | 1 to 32 |
| `max_seq_length` | 256 to 2048 |
| `warmup_ratio` | 0.0 to 0.2 |
| `batch_size` | locked at 1 (8 GB constraint) |
| `max_pixels` | locked at ≤ 200704 (8 GB with two images) |
| `system_message` | free text — prompt engineering |

---

## Step 3 — Full Training Run

Run after Step 2.  The best hyperparameters from `best_config.json` are loaded
automatically — no manual copying needed.

```bash
python train_qwen_classifier.py
```

At startup you will see:
```
[INFO] Loaded best config from self-improvement loop (cycle 2, eval_loss=0.1843).
```

If you skip Step 2, built-in defaults are used — the script works either way.

### Common overrides

| Scenario | Command |
|----------|---------|
| 8 GB default | `python train_qwen_classifier.py` |
| 8 GB, if OOM | `python train_qwen_classifier.py --max_pixels 150000 --max_seq_length 512` |
| Resume after crash | `python train_qwen_classifier.py --resume_from_checkpoint outputs/qwen_crystallography/checkpoint-400` |
| Run only N steps | `python train_qwen_classifier.py --max_steps 200` |
| 12 GB, unfreeze vision | `python train_qwen_classifier.py --finetune_vision` |
| 16 GB+, 7B model | `python train_qwen_classifier.py --model_id unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit --finetune_vision` |
| More epochs | `python train_qwen_classifier.py --epochs 5` |
| Bigger LoRA | `python train_qwen_classifier.py --lora_r 32 --lora_alpha 32` |

> CLI flags override `best_config.json` for that run only — the file is not modified.

### If you get CUDA Out of Memory

```bash
rmdir /s /q unsloth_compiled_cache
python train_qwen_classifier.py --max_pixels 150000 --max_seq_length 512
```

---

## Step 4 — Run Inference and Generate SVG Overlays

```bash
# Single image — creates image.svg in the same directory
python test_model.py path\to\image.jpg

# Run on the full held-out test split
python test_model.py --image_list data/test_images.txt

# Whole folder
python test_model.py path\to\folder\

# See raw model output (useful for debugging)
python test_model.py image.jpg --verbose

# Skip SVG and just print detections
python test_model.py image.jpg --no_svg --verbose
```

Open the `.svg` file in any browser to see coloured bounding boxes and class
labels drawn over the original image.  Each class gets a consistent colour
across all images.

---

## Optional — Merge Multiple Cycle Checkpoints

After the self-improvement loop, you can blend two or more cycle checkpoints
with MergeKit.  The merged model often outperforms any single cycle because it
combines complementary strengths.

```bash
pip install mergekit
```

Merge two cycles (SLERP — smooth 50/50 blend, best for two models):
```bash
python merge_models.py \
    outputs/self_improve/cycle_01/lora_adapter \
    outputs/self_improve/cycle_02/lora_adapter
```

Merge three cycles (TIES — better for three or more):
```bash
python merge_models.py \
    outputs/self_improve/cycle_01/lora_adapter \
    outputs/self_improve/cycle_02/lora_adapter \
    outputs/self_improve/cycle_03/lora_adapter \
    --method ties
```

Test the merged model:
```bash
python test_model.py your_image.jpg --adapter outputs/merged/merged_model
```

| Method | Best for | Notes |
|--------|----------|-------|
| `slerp` | 2 models | Smooth interpolation between weights |
| `ties`  | 2–4 models | Prunes conflicting weight updates before blending |
| `dare`  | Models from different training stages | DARE pruning + TIES |
| `linear`| Quick test | Simple weighted average |

> Merging runs on CPU (~6 GB RAM per 3B model).  No VRAM needed.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Training very slow (>10 s/it) | Add `--max_pixels 200704 --max_seq_length 512` |
| `No or negligible GPU memory` | Delete `unsloth_compiled_cache\`, then reduce `--max_pixels 150000 --max_seq_length 512` |
| OOM with two images per sample | Each sample contains crop + full image. Try `--max_pixels 150000 --max_seq_length 512` |
| `Some modules dispatched on CPU` | Fixed via `device_map={"": 0}`. Delete compiled cache and rerun |
| `register_constant` / torchao error | PyTorch too old. Upgrade: `pip install "torch>=2.7.0" torchvision --index-url https://download.pytorch.org/whl/cu126` |
| SVG shows no boxes | Model not trained yet, or output format unexpected. Run `--verbose` to see raw output |
| Download hangs / drops | Script retries up to 20× with exponential back-off. Try `--download_timeout 600` |
| `triton` import error on Windows | `pip install triton-windows` |
| LM Studio connection refused | Start the Local Server in LM Studio on port 1234. Check: `curl http://localhost:1234/v1/models` |
| LLM returns non-JSON | Try a larger instruction-tuned model in LM Studio |
| `CUDA Out of Memory` during model merge | Normal on <16 GB VRAM — `lora_adapter/` is still saved and usable |
| Image has spaces in filename | Wrap the path in quotes: `python test_model.py "image (1).jpg"` |

---

## Customising for Your Dataset

1. Edit `data/classes.txt` — one class name per line, line index = YOLO class ID
2. Re-run `prepare_yolo_to_vlm.py` to rebuild crops and JSONL files
3. Re-run the self-improvement loop and then the full training run

---

## Citation

- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- Unsloth: https://github.com/unslothai/unsloth
- LM Studio: https://lmstudio.ai
- MergeKit: https://github.com/arcee-ai/mergekit
