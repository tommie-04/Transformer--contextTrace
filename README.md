# ContextTrace

ContextTrace studies how a causal Transformer retrieves one relevant fact from a prompt containing multiple facts.

## Research question

Given a prompt such as:

```text
Alice's code is blue.
Bob's code is green.
Charlie's code is red.

Bob's code is
```

where does the answer `green` become decodable inside the model, and which internal activations causally influence that prediction?

ContextTrace compares two measurements over the same **layer × token-position** grid:

- **Decodability:** whether a linear probe can recover the correct answer from a hidden representation.
- **Causal influence:** whether replacing an activation with one from a matched counterfactual prompt changes the model's preference for the correct answer.

## Method

The project uses controlled synthetic prompts to reduce confounding from memorized world knowledge. The analysis pipeline:

1. builds matched multi-fact prompts with distractors, queries, and counterfactual answers;
2. records residual-stream activations from a GPT-2-family causal language model;
3. measures answer decodability at each layer and token position;
4. measures causal effects with residual-stream activation patching;
5. compares the decodability and causal-effect maps.

Probe accuracy and intervention effects are reported as distinct forms of evidence: decodable information is not necessarily information the model uses for its prediction.

## Setup

Python 3.11 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference check

The repository includes a small command-line check for model loading, tokenization, and generation:

```bash
python scripts/inference_smoke_test.py \
  --model-name distilgpt2 \
  --prompt "Bob's code is" \
  --greedy \
  --max-new-tokens 10
```

Model weights are downloaded from Hugging Face on first use. Pass `--local-files-only` to load only from the local cache.

## Configuration

The inference command accepts:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-name` | `distilgpt2` | Hugging Face causal language model |
| `--prompt` | `Machine learning is` | Input text to continue |
| `--max-new-tokens` | `40` | Maximum generated token count |
| `--seed` | `42` | Random seed |
| `--greedy` | disabled | Use deterministic greedy decoding |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-p` | `0.95` | Nucleus-sampling threshold |
| `--local-files-only` | disabled | Avoid Hugging Face network access |

Experiment configurations added to this repository record the model, dataset construction, seed, prediction metric, probe settings, and intervention settings required to reproduce each result.

## Scope

ContextTrace focuses on controlled contextual retrieval, residual-stream representations, linear probes, and residual-stream activation patching in a small GPT-2-family model. Head-level circuit discovery, sparse autoencoders, model editing, and broad cross-model benchmarking are outside the core analysis.
