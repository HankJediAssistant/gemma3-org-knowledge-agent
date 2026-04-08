#!/usr/bin/env python3
"""
Download the Gemma 3 4B instruction-tuned model from HuggingFace.

Gemma 3 is a gated model — you need to:
1. Accept the license at https://huggingface.co/unsloth/gemma-3-4b-it
2. Set your HuggingFace token: export HF_TOKEN=hf_xxxxx

The model will be saved to ./models/ for offline use.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download
from tqdm import tqdm


def check_hf_token():
    """Verify HuggingFace token is available."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("=" * 60)
        print("ERROR: HF_TOKEN environment variable not set!")
        print()
        print("Gemma 3 is a gated model. You need to:")
        print("  1. Create a HuggingFace account at https://huggingface.co")
        print("  2. Accept the Gemma 3 license at:")
        print("     https://huggingface.co/unsloth/gemma-3-4b-it")
        print("  3. Create an access token at:")
        print("     https://huggingface.co/settings/tokens")
        print("  4. Set the token:")
        print('     export HF_TOKEN="hf_your_token_here"')
        print("=" * 60)
        sys.exit(1)
    return token


def download_model(model_name: str, output_dir: str, token: str):
    """Download model snapshot from HuggingFace Hub."""
    print(f"Downloading {model_name} to {output_dir}...")
    print("This may take a while.")
    print()

    local_dir = os.path.join(output_dir, model_name.replace("/", "--"))

    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        token=token,
        # Skip large safetensors shards we don't need for QLoRA
        # (Unsloth handles quantized loading directly from the Hub)
        ignore_patterns=["*.gguf", "*.bin"],
    )

    print()
    print(f"Model downloaded to: {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 4B model from HuggingFace"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-4b-it",
        help="Model name on HuggingFace Hub (default: unsloth/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save the model (default: ./models)",
    )
    args = parser.parse_args()

    token = check_hf_token()
    os.makedirs(args.output_dir, exist_ok=True)
    download_model(args.model, args.output_dir, token)
    print("Done! You can now run training with: python scripts/train.py")


if __name__ == "__main__":
    main()
