<div align="center">
  <img src="Q_Logo.svg" width="72" alt="Q Project logo">
  <h1>QBench</h1>
  <h3>A small, model-agnostic benchmark for small (&lt;1B parameter) language models</h3>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" alt="License"></a>
    <a href="https://huggingface.co/datasets/q-project/qbench-results"><img src="https://img.shields.io/badge/Leaderboard-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface" alt="Leaderboard"></a>
  </p>
</div>

## Why this exists

We built [Q-U-164M](https://github.com/Q-Project-LM/Q-164M) and wanted to compare it fairly against other
small models. Our first attempt copied phrasing straight from Q-Agent's own training-data generator into
the eval questions — an in-distribution advantage that made Q-U-164M look far better than it actually is.
A user who didn't believe the result caught it. QBench is the fixed version: every question is phrased from
scratch and checked against this project's own training corpora for overlap, so no model in the Q family
gets an unfair head start from wording alone. It benchmarks any Hugging Face model, not just ours.

## What it measures

- **Agentic / tool-use** — single-turn tasks (arithmetic, percentages, primality, sorting) plus an 8-turn
  conversation using natural conversational connectives ("By the way, ...", "Following up, ..."), scored on
  whether the model's final answer is actually correct. Also flags repetition-collapse (a model spiraling
  into repeating itself instead of answering).
- **General knowledge / reasoning** — [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) and
  [TruthfulQA MC1](https://huggingface.co/datasets/truthfulqa/truthful_qa), scored via loglikelihood over
  each answer choice (the standard technique — no generation/parsing ambiguity).

**No model gets forced-correct tool execution, including ours.** Some of our own models (Q-Agent family)
normally run with a deterministic executor that corrects arithmetic mistakes after the model decides to
call a tool — QBench turns that off for everyone, so what's measured is what the model itself actually
gets right, comparably across architectures.

## Usage

```bash
pip install -r requirements.txt
python cli.py run Qwen/Qwen2.5-0.5B-Instruct
```

For a custom architecture (like Q-U-164M):

```bash
python cli.py run q-project/Q-U-164M --trust-remote-code
```

Submit your result to the public leaderboard (needs `hf auth login` with a token that has write access to
`q-project/qbench-results`, or fork the dataset and point `RESULTS_REPO` in `cli.py` at your own copy):

```bash
python cli.py run <model-id> --submit --submitted-by "your name"
```

Each run also computes a short **fingerprint** — a hash of the model's config plus its behavior on a few
fixed canonical prompts, included with every submitted result. It's not airtight proof, but a mismatched
fingerprint on rerunning the same `model_id` is an easy, checkable red flag for a leaderboard where scores
are computed locally and only the result is uploaded.

## Leaderboard

The shared leaderboard lives at
[huggingface.co/datasets/q-project/qbench-results](https://huggingface.co/datasets/q-project/qbench-results)
and renders at [huggingface.co/spaces/q-project/QBench](https://huggingface.co/spaces/q-project/QBench) (static HTML, reads the dataset directly — no
server, no inference backend).

Current results (see the dataset for the live version):

| model | params | single-turn | multi-turn | ARC-Easy | TruthfulQA MC1 |
|---|---|---|---|---|---|
| GPT-2 | 124M | 0% | 0% | 42% | 26% |
| DistilGPT-2 | 82M | 0% | 6% | 36% | 25% |
| SmolLM2-135M-Instruct | 135M | 5% | 6% | 53% | 23% |
| SmolLM2-360M-Instruct | 360M | 50% | 50% | 50% | 18% |
| Qwen2.5-0.5B-Instruct | 500M | 45% | 50% | 62% | 24% |
| Q-U-164M (`q-project/Q-U-164M`) | 164M | 35% | 25% | 43% | 26% |

## Repository contents

```
qbench/
  loading.py           generic HF model wrapper: chat-template prompting, loglikelihood scoring, fingerprint
  modules/
    agentic.py          tool-use tasks (single-turn + 8-turn multi-turn), scoring
    knowledge.py         ARC-Easy + TruthfulQA MC1
  report.py              renders a results dict into a short Markdown scorecard
cli.py                    entry point: `python cli.py run <model-id> [--submit]`
```

## License

Apache-2.0 (see `LICENSE`).
