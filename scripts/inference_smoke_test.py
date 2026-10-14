"""Inference smoke test for DistilGPT-2.

This script is intentionally simple. It shows the basic HuggingFace workflow:

1. Load a tokenizer.
2. Load a pretrained causal language model.
3. Convert a text prompt into token IDs.
4. Generate new text.
5. Decode token IDs back into text.
"""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small GPT-2 inference test.")
    parser.add_argument(
        "--model-name",
        default="distilgpt2",
        help="HuggingFace model name. Default: distilgpt2",
    )
    parser.add_argument(
        "--prompt",
        default="Machine learning is",
        help="Text prompt for the model to continue.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower is more conservative; higher is more random.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model from the local HuggingFace cache without contacting the Hub.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt")

    sampling_options = {} if args.greedy else {"temperature": args.temperature, "top_p": args.top_p}
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        **sampling_options,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nPrompt:")
    print(args.prompt)
    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()
