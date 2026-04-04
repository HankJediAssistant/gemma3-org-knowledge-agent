#!/usr/bin/env python3
"""
Format raw synthetic responses into instruction→response JSONL for training.

Takes the raw_responses.jsonl from generate_synthetic_data.py and converts it
into the chat format that Gemma3 expects:

  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Also does a train/eval split (90/10 by default).
"""

import argparse
import json
import os
import random


# System prompt that gives the model its persona
SYSTEM_PROMPT = (
    "You are an internal organizational knowledge assistant for a semiconductor "
    "technology company. You answer questions about teams, tools, processes, and "
    "organizational knowledge based on what employees have shared. Be helpful, "
    "specific, and conversational. If you reference a specific team or person, "
    "provide context about their role."
)


def format_instruction(record: dict) -> dict:
    """Convert a raw response record into a chat-formatted training example."""
    # Build a contextual user question
    user_content = record["question"]

    # Build the assistant response with attribution context
    assistant_content = record["response"]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def prepare_dataset(
    input_path: str,
    output_dir: str,
    eval_fraction: float = 0.1,
    seed: int = 42,
):
    """Load raw responses, format them, and write train/eval splits."""
    random.seed(seed)

    # Load raw data
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} raw records from {input_path}")

    # Format into instruction tuning examples
    formatted = [format_instruction(r) for r in records]

    # Shuffle and split
    random.shuffle(formatted)
    split_idx = int(len(formatted) * (1 - eval_fraction))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    # Write output files
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    print(f"Train set: {len(train_data)} examples → {train_path}")
    print(f"Eval set:  {len(eval_data)} examples → {eval_path}")

    # Show a sample
    print()
    print("=" * 60)
    print("Sample training example:")
    print("=" * 60)
    sample = train_data[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        print(f"\n[{role}]")
        print(msg["content"])

    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser(
        description="Format raw responses into instruction-tuning JSONL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/synthetic/raw_responses.jsonl",
        help="Path to raw responses JSONL (default: ./data/synthetic/raw_responses.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/synthetic",
        help="Directory for train/eval JSONL files (default: ./data/synthetic)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for eval (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    prepare_dataset(args.input, args.output_dir, args.eval_fraction, args.seed)


if __name__ == "__main__":
    main()
