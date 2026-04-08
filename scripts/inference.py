#!/usr/bin/env python3
"""
Run inference with the fine-tuned Gemma 3 4B model.

Loads the base model + LoRA adapter and generates responses to questions.
Works as both a CLI tool and an importable module.

Usage:
  # Single question via CLI argument
  python scripts/inference.py --question "What does the SDE team do?"

  # Interactive mode (default)
  python scripts/inference.py

  # Custom adapter path
  python scripts/inference.py --adapter ./output/final

  # As a module
  from scripts.inference import OrgAgent
  agent = OrgAgent("./output/final")
  answer = agent.ask("What tools does DevOps use?")
"""

import argparse
import sys

from unsloth import FastLanguageModel


# Default system prompt (should match what was used in training)
SYSTEM_PROMPT = (
    "You are an internal organizational knowledge assistant for a semiconductor "
    "technology company. You answer questions about teams, tools, processes, and "
    "organizational knowledge based on what employees have shared. Be helpful, "
    "specific, and conversational. If you reference a specific team or person, "
    "provide context about their role."
)


class OrgAgent:
    """Org knowledge Q&A agent backed by a fine-tuned Gemma 3 4B model."""

    def __init__(
        self,
        adapter_path: str = "./output/final",
        base_model: str = "unsloth/gemma-3-4b-it",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """Load the base model and LoRA adapter."""
        print(f"Loading model: {base_model}")
        print(f"Loading adapter: {adapter_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        # Put model in inference mode (2x faster with Unsloth)
        FastLanguageModel.for_inference(self.model)

        self.system_prompt = system_prompt
        self.max_seq_length = max_seq_length
        print("Model loaded and ready.")

    def ask(
        self,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Ask the model a question and get a response."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode only the generated part (skip the input tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


def interactive_mode(agent: OrgAgent):
    """Run an interactive Q&A session."""
    print()
    print("=" * 60)
    print("Org Knowledge Agent — Interactive Mode")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = agent.ask(question)
        print(f"\nAgent: {response}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Gemma 3 4B org knowledge agent"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./output/final",
        help="Path to LoRA adapter directory (default: ./output/final)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-3-4b-it",
        help="Base model name (default: unsloth/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question (omit for interactive mode)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    args = parser.parse_args()

    agent = OrgAgent(
        adapter_path=args.adapter,
        base_model=args.base_model,
    )

    if args.question:
        # Single question mode
        response = agent.ask(
            args.question,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)
    else:
        # Interactive mode
        interactive_mode(agent)


if __name__ == "__main__":
    main()
