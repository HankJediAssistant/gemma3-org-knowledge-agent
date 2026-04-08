#!/usr/bin/env python3
"""Export fine-tuned LoRA adapter to Ollama-compatible GGUF format."""

import os
from pathlib import Path


def main():
    adapter_path = "output/final"
    merged_path = "output/merged"
    gguf_path = "output/gguf"
    modelfile_path = "output/Modelfile"

    if not Path(adapter_path).exists():
        print(f"Error: Adapter not found at {adapter_path}/")
        print("Run training first: python scripts/train.py")
        return

    print("Loading fine-tuned model...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Save merged model in 16-bit
    print(f"Merging LoRA weights and saving to {merged_path}/...")
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print("Merged model saved.")

    # Export to GGUF using Unsloth built-in
    print(f"Exporting to GGUF (Q4_K_M) at {gguf_path}/...")
    os.makedirs(gguf_path, exist_ok=True)
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    print("GGUF export complete.")

    # Create Modelfile
    modelfile_content = """\
FROM ./output/gguf/model-Q4_K_M.gguf
SYSTEM "You are an organizational knowledge expert. Answer questions about team structure, tools, workflows, and processes concisely and helpfully."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    print(f"Modelfile written to {modelfile_path}")

    print()
    print("=" * 60)
    print("Export complete! To load into Ollama, run:")
    print()
    print("  ollama create org-agent -f output/Modelfile && ollama run org-agent")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
