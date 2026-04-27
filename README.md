# cost-benchmark

A tiny benchmark for measuring **what an LLM call actually costs** — not what
its price page says it costs.

Built to support the argument in *The Token Is a Lie*: that `$/token` has
become a fiction because token volume, reasoning surcharges, and per-task
correctness all decouple from the number on the invoice. The scripts here run
the same prompt against many models via [OpenRouter](https://openrouter.ai),
record per-trial input/output/reasoning tokens, real billed cost, latency, and
correctness, and produce publication-ready charts.

## Quick start

You'll need [`uv`](https://docs.astral.sh/uv/) (handles deps inline via PEP 723
— no `pip install`, no venv) and an OpenRouter API key.

```bash
export OPENROUTER_API_KEY=sk-or-...

# Default: send "What is 4 + 8?" to a built-in model lineup
uv run token_bench.py

# Run the 3-prompt suite (apples / CRT / sibling-trap) with reasoning forced on
uv run token_bench.py --suite --reasoning-effort high --csv suite_run.csv

# Same suite, reasoning disabled
uv run token_bench.py --suite --reasoning-effort none --csv suite_run_nothink.csv
```

Both CSVs feed the comparison chart:

```bash
uv run compare_chart.py suite_run.csv suite_run_nothink.csv suite_compare.png
```

For a per-token list-price chart in the same model order:

```bash
uv run price_chart.py suite_run.csv price_chart.png
```

## What's in here

| File | What it does |
|---|---|
| `token_bench.py` | The runner. Sends one query (or a suite) to N models, records tokens/cost/latency/correctness, writes CSV. Supports `--reasoning-effort {none,low,medium,high}` mapped through OpenRouter's unified reasoning parameter. |
| `make_chart.py` | Single-prompt chart: tokens used vs. cost spent, per model. |
| `compare_chart.py` | 2×3 grid: reasoning ON vs. OFF across three prompts. The headline figure. |
| `price_chart.py` | List-price chart pulled live from OpenRouter's `/api/v1/models`, ordered to match `compare_chart`. |
| `suite_compare.png` | Pre-generated example output. |
| `price_chart.png` | Pre-generated example output. |
| `token_bench_chart.png` | Pre-generated example output for a single prompt. |

## Design notes

**Why `uv run`.** Each script declares its dependencies inline (PEP 723) so
the whole thing is shareable as standalone files — no `requirements.txt`,
no virtualenv, no global `pip install`. First run resolves and caches; later
runs are instant.

**Why OpenRouter.** Switching providers is a slug change, and OpenRouter
returns a consistent `usage` block including a `cost` field when you pass
`usage: {include: true}`. That's what makes "what did this call really cost"
a one-line measurement instead of a per-provider parsing exercise.

**Why `reasoning_effort: high` by default.** OpenRouter's unified reasoning
parameter maps to each provider's native control:

- Anthropic → `thinking: {type: enabled, budget_tokens: ...}`
- OpenAI → `reasoning_effort`
- Google → `thinking_config`
- DeepSeek → reasoning mode
- Qwen → `enable_thinking`

Non-reasoning models silently ignore it. Set `--reasoning-effort none` to
skip the parameter entirely for a clean comparison.

**Why the correctness check looks for the *last* number in the answer.** Many
models (especially DeepSeek's, which emit visible chain-of-thought) restate
prompt numbers mid-response. Naive substring matching gives false positives.
The script's `correct()` function compares the final numeric token in the
output against the expected value.

## Adding your own prompt suite

Pass a JSON file to `--suite`:

```json
[
  {"label": "math", "query": "What is 17 × 23?", "expect": "391"},
  {"label": "code", "query": "Write a one-line Python lambda for factorial.", "expect": "lambda"}
]
```

```bash
uv run token_bench.py --suite my_suite.json --csv my_run.csv
```

## License

MIT.
