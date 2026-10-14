# ContextTrace

**Tracing how a causal Transformer retrieves the relevant fact from context.**

ContextTrace is a lightweight mechanistic-interpretability project for studying **contextual fact retrieval** in GPT-2-style language models.

## Research question

Given several facts in the prompt and a query that refers to one of them:

```text
Alice's code is blue.
Bob's code is green.
Charlie's code is red.

Bob's code is
```

how does the model select the relevant fact and use it to predict `green`?

ContextTrace compares two kinds of evidence across **Transformer layer × token position**:

1. **Decodability** — can the correct answer be recovered from the hidden representation with a simple probe?
2. **Causal influence** — does intervening on that activation change the model's preference for the correct answer?

The central question is:

> **Does where information is decodable match where it is causally important for prediction?**

## Method

ContextTrace uses controlled synthetic and counterfactual prompts so that the relevant fact is supplied by the context rather than assumed to come from memorized world knowledge.

The main analysis pipeline is:

1. Construct matched multi-fact retrieval prompts with distractors and a target query.
2. Run a GPT-2-family causal language model and collect hidden activations.
3. Measure answer **decodability** across layers and token positions with a simple probe.
4. Measure **causal influence** with matched activation interventions / patching.
5. Compare the resulting decodability and causal-effect maps under controlled conditions.

Attention weights, PCA, probes, and patching are treated as different forms of evidence rather than interchangeable explanations of model behavior.

## Repository structure

```text
contexttrace/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/             # experiment and model settings
├── data/                # tracked prompt templates / small reproducible datasets
├── src/
│   ├── data/            # task construction and dataset utilities
│   ├── model/           # model loading, inference, activation collection
│   ├── analysis/        # probing, patching, metrics, comparison
│   └── utils/
├── scripts/             # reproducible experiment entry points
├── tests/               # focused tests for core utilities
└── results/             # selected figures / summaries used in the project
```

Only files needed to run, reproduce, or understand the project belong in the public repository.

## Setup

Python 3.11+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Experiment settings should be kept in version-controlled configuration files under `configs/` rather than hard-coded across scripts. A typical configuration may specify:

```yaml
model_name: distilgpt2
seed: 42

dataset:
  template: multi_fact_codes
  num_distractors: 2

analysis:
  metric: logit_difference
  probe: linear
  intervention: residual_stream_patch
```

Exact fields may evolve with the implementation, but each reported result should record the model, dataset/task settings, random seed, analysis metric, and intervention configuration required to reproduce it.

## Scope

The core project focuses on:

- one small GPT-2-family causal language model at a time;
- controlled contextual fact retrieval;
- layer × token-position representations;
- one primary decodability method;
- one primary causal intervention method;
- quantitative comparison between the two.

Head-level circuit discovery, sparse autoencoders, model editing, and cross-model generalization are outside the initial core scope.

## References

Methodological references and external tools are documented when they are used in the implementation or experiments.
