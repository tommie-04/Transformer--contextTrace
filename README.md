# Mini Assistant Axis

A lightweight interpretability pipeline for analyzing persona-related directions in the hidden states of GPT-2-style language models.

## Overview

Mini Assistant Axis extracts hidden-state representations from a GPT-2-style language model and constructs a contrastive activation direction — the *assistant axis* — that separates standard assistant-style responses from roleplay-style (persona-shifted) responses. Any new model response is pooled into a single vector and projected onto this axis, producing a scalar score that measures how closely the response sits near the model default assistant-like style. The pipeline also includes PCA visualization, projection-score analysis, threshold-based label prediction, and a logistic-regression probe that compares a learned classifier direction with the mean-difference axis.

The project implements, on a small laptop-runnable model, the central idea of the Assistant Axis work from Anthropic (Lu et al., 2026): a dominant direction in hidden-state space tracks how far a model has drifted from its default Assistant persona. The original study maps a persona space over 275 role archetypes on 27B–70B models; this repository demonstrates the same core mechanism with a two-group contrast (assistant vs roleplay) on `distilgpt2`.

## Method

1. **Data** — two labeled prompt groups, `assistant` and `roleplay`, plus an unlabeled evaluation set.
2. **Response generation** — `distilgpt2` via HuggingFace Transformers, with configurable `temperature`, `top_p`, `max_new_tokens`, and random seed.
3. **Hidden-state extraction** — hidden states read from a selected transformer layer; token-level states are pooled (mean / last-token / last-k) into one vector per response.
4. **Axis construction** — `assistant_axis = normalize(mean(assistant_vectors) - mean(roleplay_vectors))`.
5. **Scoring** — the projection of a response vector onto the normalized axis yields the assistant-style score.
6. **Analysis** — PCA scatter of response vectors, projection-score histograms, threshold-based label prediction, and a logistic-regression probe.

## Repository structure

```text
mini_assistant_axis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── seed_prompts.jsonl        # 10 labeled prompts: 5 assistant, 5 roleplay
│   └── eval_prompts.jsonl        # 5 unlabeled evaluation prompts
└── scripts/
    └── inference_smoke_test.py   # end-to-end inference smoke test for distilgpt2
```

Further pipeline modules (response generation, hidden-state extraction, axis construction, visualization, CLI demo) are added incrementally; see the roadmap below.

## Environment setup

Python 3.11+ with the packages listed in `requirements.txt` (`torch`, `transformers`, `accelerate`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `jupyter`). No GPU is required: `distilgpt2` (~82M parameters) runs comfortably on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model weights are downloaded from the HuggingFace Hub on first use. To run from a local cache without network access, pass `--local-files-only`.

## Quick start

```bash
python scripts/inference_smoke_test.py \
    --prompt "A transformer model is" \
    --max-new-tokens 30 \
    --temperature 0.8
```

This loads the `distilgpt2` tokenizer and model, tokenizes the prompt, generates a continuation, and decodes it back to text. Run with `--help` to see all options (`--model-name`, `--top-p`, `--local-files-only`, and others).

## Data format

Each line of `data/*.jsonl` is one JSON object:

```json
{"id": "a001", "label": "assistant", "prompt": "You are a helpful AI assistant. Explain what overfitting means in machine learning in simple terms."}
```

- `seed_prompts.jsonl` — 10 labeled prompts (5 `assistant`, 5 `roleplay`) used to construct the axis.
- `eval_prompts.jsonl` — 5 unlabeled prompts reserved for evaluation.

## Roadmap

- [ ] Response-generation pipeline (JSONL I/O, generation, command-line interface).
- [ ] Hidden-state extraction and pooling.
- [ ] Assistant-axis construction and projection scoring.
- [ ] Visualization and analysis (PCA, score histograms).
- [ ] CLI demo, logistic-regression probe, and project report.

## References

- Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey. *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*. arXiv:2601.10387. https://arxiv.org/abs/2601.10387
- Anthropic Research — The assistant axis. https://www.anthropic.com/research/assistant-axis