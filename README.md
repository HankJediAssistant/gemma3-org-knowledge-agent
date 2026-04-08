# 🧠 Gemma 3 4B Org Knowledge Agent

Fine-tune Google's **Gemma 3 4B** model on organizational Q&A data using **QLoRA + Unsloth**, then run inference locally — no cloud, no API keys, no cost per query.

> Built and trained on a single **NVIDIA RTX 4070 (12GB VRAM)**.

---

## What It Does

Turns a general-purpose LLM into an **organizational knowledge expert** — a model that knows your team structure, tooling, workflows, pain points, and processes. Ask it anything your new hires would struggle to find in a wiki.

### Historical Training Results (Gemma 3 4B)

> The results below were from the initial proof-of-concept using Gemma 3 4B.
> These results are from the proof-of-concept using Gemma 3 4B.

```
Step 10:  loss=1.37
Step 50:  loss=0.27
Step 100: loss=0.13
Step 200: loss=0.08
Final:    loss=0.075 ✅
```

---

## The 8 Org Knowledge Questions

The dataset is built around core questions employees answer about their team:

1. What does your team/organization do?
2. What tools and software do you use daily?
3. What are the biggest roadblocks or pain points?
4. What operating procedures does your team follow?
5. Who do you collaborate with most, and for what?
6. What knowledge is hardest to find or most often asked about?
7. What would a new employee most need to know?
8. What processes have changed recently that people might not know about?

---

## Quick Start

### 1. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA (adjust cu124/cu121 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies (unsloth last to avoid torchao conflicts)
pip install "unsloth[colab-new]" transformers datasets peft trl accelerate bitsandbytes huggingface_hub pyyaml tqdm numpy
pip uninstall torchao -y   # prevents torch version conflicts
```

> ⚠️ **Important:** After install, run `pip uninstall torchao -y` — it conflicts with torch 2.5/2.6.

### 2. Set HuggingFace token

Gemma 3 is a gated model. Accept the license at [huggingface.co/unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it), then:

```bash
export HF_TOKEN="hf_your_token_here"
```

### 3. Download the base model

```bash
python scripts/download_model.py
```

Downloads `unsloth/gemma-3-4b-it` to `models/`.

### 4. Generate synthetic training data (POC)

```bash
python scripts/generate_synthetic_data.py
python scripts/prepare_dataset.py
```

Creates 640 synthetic Q&A examples across 10 fictional semiconductor/tech orgs → `data/synthetic/`.

### 5. Fine-tune

```bash
python scripts/train.py
```

LoRA adapter saved to `output/final/`.

### 6. Run inference

```bash
# Single question
python scripts/inference.py --question "What does the SDE team do?"

# Interactive mode
python scripts/inference.py

# Custom adapter
python scripts/inference.py --adapter ./output/final
```

### 7. Evaluate (optional)

```bash
python scripts/evaluate.py
```

Compares base model vs fine-tuned on test questions.

---

## Hardware Requirements

| Component | Minimum | Used Here |
|-----------|---------|-----------|
| GPU VRAM  | 8 GB    | RTX 4070 (12GB) |
| RAM       | 16 GB   | — |
| Disk      | 15 GB   | ~13 GB total |
| CUDA      | 11.8+   | 12.4 |

---

## Project Structure

```
gemma3-org-agent/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_model.py          # Download Gemma 3 4B from HuggingFace
│   ├── generate_synthetic_data.py # Generate synthetic Q&A dataset
│   ├── prepare_dataset.py         # Format into instruction JSONL
│   ├── train.py                   # QLoRA fine-tuning with Unsloth
│   ├── inference.py               # Load model + adapter, run queries
│   └── evaluate.py                # Compare base vs fine-tuned
├── data/
│   ├── synthetic/                 # Generated POC dataset
│   └── real/                      # Drop real employee data here
├── models/                        # Base model weights (gitignored)
├── output/                        # Fine-tuned adapters (gitignored)
└── configs/
    └── training_config.yaml       # All hyperparameters
```

---

## Using Real Data

When real employee responses are ready:

1. Place them in `data/real/` as JSONL matching the schema in `data/synthetic/raw_responses.jsonl`
2. Run `python scripts/prepare_dataset.py --input data/real/your_data.jsonl`
3. Re-run `python scripts/train.py`

---

## Troubleshooting

**`torch has no attribute 'int1'`**
```bash
pip uninstall torchao -y
```

**`ValueError: requires torch>=2.6` during inference**
```bash
pip install "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124
pip uninstall torchao -y
```

**`TorchDynamo InternalTorchDynamoError`**
Add to the top of `train.py` after imports:
```python
torch._dynamo.config.disable = True
```

**OOM during training**
Reduce `per_device_train_batch_size` to 1 in `configs/training_config.yaml`.

---

## Tech Stack

- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster QLoRA fine-tuning
- [Google Gemma 3 4B-IT](https://huggingface.co/unsloth/gemma-3-4b-it) — base model
- [HuggingFace TRL](https://github.com/huggingface/trl) — SFTTrainer
- [PEFT](https://github.com/huggingface/peft) — LoRA adapters
- PyTorch 2.6 + CUDA 12.4

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
