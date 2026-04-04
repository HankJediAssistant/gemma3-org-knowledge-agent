#!/usr/bin/env python3
"""
QLoRA fine-tuning of Gemma3 4B using Unsloth.

Loads training config from configs/training_config.yaml, then fine-tunes
the model on the prepared instruction dataset using 4-bit quantization.

Optimized for NVIDIA RTX 4070 (12GB VRAM).
"""

import argparse
import json
import os

import unsloth  # must be first
import torch
# Disable torch.compile to avoid TorchDynamo bugs with Gemma3/Unsloth
torch._dynamo.config.disable = True
import yaml
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return config


def format_chat_template(examples, tokenizer):
    """Apply the chat template to a batch of examples."""
    texts = []
    for messages in examples["messages"]:
        # Parse messages if they're stored as strings
        if isinstance(messages, str):
            messages = json.loads(messages)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma3 4B with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to training config YAML (default: ./configs/training_config.yaml)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["dataset"]

    # -------------------------------------------------------------------------
    # 1. Load model with Unsloth (handles 4-bit quantization automatically)
    # -------------------------------------------------------------------------
    print(f"\nLoading model: {model_cfg['name']}")
    print(f"  Max sequence length: {model_cfg['max_seq_length']}")
    print(f"  4-bit quantization: {model_cfg['load_in_4bit']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        # Unsloth handles dtype selection automatically
    )

    # -------------------------------------------------------------------------
    # 2. Apply LoRA adapters
    # -------------------------------------------------------------------------
    print(f"\nApplying LoRA adapters:")
    print(f"  Rank: {lora_cfg['r']}, Alpha: {lora_cfg['alpha']}")
    print(f"  Target modules: {lora_cfg['target_modules']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=train_cfg["seed"],
    )

    # -------------------------------------------------------------------------
    # 3. Load and format dataset
    # -------------------------------------------------------------------------
    print(f"\nLoading dataset from: {data_cfg['train_file']}")

    dataset = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    print(f"  Training examples: {len(dataset)}")

    # Apply chat template formatting
    dataset = dataset.map(
        lambda examples: format_chat_template(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # -------------------------------------------------------------------------
    # 4. Set up trainer
    # -------------------------------------------------------------------------
    print("\nConfiguring trainer...")

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to="none",  # Disable wandb/tensorboard for POC
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        packing=False,  # Disable packing for cleaner training on short examples
    )

    # -------------------------------------------------------------------------
    # 5. Train!
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    gpu_stats_before = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_stats_before = torch.cuda.get_device_properties(0)
            print(f"  GPU: {gpu_stats_before.name}")
            print(f"  VRAM: {gpu_stats_before.total_mem / 1024**3:.1f} GB")
    except Exception:
        pass

    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # -------------------------------------------------------------------------
    # 6. Save the fine-tuned LoRA adapter
    # -------------------------------------------------------------------------
    output_path = os.path.join(train_cfg["output_dir"], "final")
    print(f"\nSaving LoRA adapter to: {output_path}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\nTraining complete!")
    print(f"  Adapter saved to: {output_path}")
    print(f"  To run inference: python scripts/inference.py --adapter {output_path}")


if __name__ == "__main__":
    main()
