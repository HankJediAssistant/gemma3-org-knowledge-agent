#!/usr/bin/env python3
"""
Basic evaluation of the fine-tuned model.

Runs a set of test questions through both the base model and the fine-tuned
model, then prints side-by-side comparisons so you can eyeball the difference.

Also computes simple metrics:
  - Response length (are answers more detailed after fine-tuning?)
  - Keyword hit rate (does the model mention org-specific terms?)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from unsloth import FastLanguageModel


# System prompt (same as training)
SYSTEM_PROMPT = (
    "You are an internal organizational knowledge assistant for a semiconductor "
    "technology company. You answer questions about teams, tools, processes, and "
    "organizational knowledge based on what employees have shared. Be helpful, "
    "specific, and conversational. If you reference a specific team or person, "
    "provide context about their role."
)

# Test questions — a mix of the training questions and novel variations
TEST_QUESTIONS = [
    # Direct training questions
    "What does the Silicon Design Engineering team do?",
    "What tools does the DevOps team use?",
    "What are the biggest pain points for Fabrication Operations?",
    # Novel questions (should still benefit from fine-tuning)
    "Who should I talk to about chip testing issues?",
    "What's the onboarding process like for new engineers?",
    "How do teams coordinate during a tape-out?",
    "What changed recently in the quality assurance process?",
    "What's the hardest knowledge to find in this company?",
]

# Keywords the fine-tuned model should be more likely to mention
ORG_KEYWORDS = [
    "SDE", "FABOPS", "TE", "PM", "DEVOPS", "QA", "CE", "HR", "ESW", "SCP",
    "tape-out", "wafer", "ATE", "EDA", "LoRA", "yield", "cleanroom",
    "Cadence", "Synopsys", "JIRA", "Confluence", "Splunk", "SAP",
    "8D", "JEDEC", "Zephyr", "firmware", "SDK",
]


def load_model(model_path: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    """Load a model (base or fine-tuned)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """Generate a response to a question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


def count_keyword_hits(text: str) -> int:
    """Count how many org-specific keywords appear in the text."""
    text_lower = text.lower()
    return sum(1 for kw in ORG_KEYWORDS if kw.lower() in text_lower)


def run_evaluation(
    base_model_name: str,
    adapter_path: str,
    eval_file: str = None,
    output_path: str = None,
):
    """Run the full evaluation comparing base vs fine-tuned model."""

    # Use eval file questions if provided, otherwise use TEST_QUESTIONS
    questions = list(TEST_QUESTIONS)
    if eval_file and os.path.exists(eval_file):
        print(f"Loading eval questions from {eval_file}")
        with open(eval_file) as f:
            for line in f:
                data = json.loads(line.strip())
                # Extract the user question from the messages
                for msg in data.get("messages", []):
                    if msg["role"] == "user":
                        questions.append(msg["content"])
                        break
        # Take a sample if we have too many
        if len(questions) > 20:
            import random
            random.seed(42)
            questions = random.sample(questions, 20)

    print("=" * 70)
    print("EVALUATION: Base Model vs Fine-Tuned Model")
    print("=" * 70)

    # Load base model
    print(f"\nLoading base model: {base_model_name}")
    base_model, base_tokenizer = load_model(base_model_name)

    # Load fine-tuned model
    print(f"Loading fine-tuned model: {adapter_path}")
    ft_model, ft_tokenizer = load_model(adapter_path)

    results = []
    base_keyword_total = 0
    ft_keyword_total = 0
    base_length_total = 0
    ft_length_total = 0

    for i, question in enumerate(questions):
        print(f"\n{'─' * 70}")
        print(f"Question {i+1}/{len(questions)}: {question}")
        print(f"{'─' * 70}")

        # Generate from both models
        base_response = generate_response(base_model, base_tokenizer, question)
        ft_response = generate_response(ft_model, ft_tokenizer, question)

        # Metrics
        base_kw = count_keyword_hits(base_response)
        ft_kw = count_keyword_hits(ft_response)
        base_keyword_total += base_kw
        ft_keyword_total += ft_kw
        base_length_total += len(base_response.split())
        ft_length_total += len(ft_response.split())

        print(f"\n  [BASE MODEL] ({len(base_response.split())} words, {base_kw} keywords)")
        print(f"  {base_response[:300]}{'...' if len(base_response) > 300 else ''}")
        print(f"\n  [FINE-TUNED] ({len(ft_response.split())} words, {ft_kw} keywords)")
        print(f"  {ft_response[:300]}{'...' if len(ft_response) > 300 else ''}")

        results.append({
            "question": question,
            "base_response": base_response,
            "finetuned_response": ft_response,
            "base_keywords": base_kw,
            "finetuned_keywords": ft_kw,
        })

    # Summary
    n = len(questions)
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Questions evaluated: {n}")
    print(f"Avg response length — Base: {base_length_total/n:.0f} words, Fine-tuned: {ft_length_total/n:.0f} words")
    print(f"Total keyword hits  — Base: {base_keyword_total}, Fine-tuned: {ft_keyword_total}")
    print(f"Avg keywords/answer — Base: {base_keyword_total/n:.1f}, Fine-tuned: {ft_keyword_total/n:.1f}")

    if ft_keyword_total > base_keyword_total:
        improvement = ((ft_keyword_total - base_keyword_total) / max(base_keyword_total, 1)) * 100
        print(f"\nKeyword improvement: +{improvement:.0f}% more org-specific terms in fine-tuned responses")
    else:
        print("\nNote: Fine-tuned model didn't show keyword improvement — may need more training data or epochs.")

    # Save results if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model vs base model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-3-4b-it",
        help="Base model name (default: unsloth/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./output/final",
        help="Path to fine-tuned LoRA adapter (default: ./output/final)",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="./data/synthetic/eval.jsonl",
        help="Path to eval JSONL for additional questions (default: ./data/synthetic/eval.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save detailed results to this JSON file",
    )
    args = parser.parse_args()

    run_evaluation(args.base_model, args.adapter, args.eval_file, args.output)


if __name__ == "__main__":
    main()
