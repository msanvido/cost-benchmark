#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
# ]
# ///
"""
token_bench.py — Run one query across many LLMs via OpenRouter and compare
                 tokens, cost, latency, and answer.

Setup
-----
    # Install uv (one-time): https://docs.astral.sh/uv/getting-started/installation/
    # macOS / Linux:
    #     curl -LsSf https://astral.sh/uv/install.sh | sh

    export OPENROUTER_API_KEY=sk-or-...

Quick start (with uv — no manual venv, no pip install)
------------------------------------------------------
    # Default query: "What is the result of the sum of 4 and 8?"
    uv run token_bench.py

    # Custom query
    uv run token_bench.py --query "What is 17 * 23?"

    # Read query from file
    uv run token_bench.py --query-file prompt.txt

    # Different model set
    uv run token_bench.py --models anthropic/claude-opus-4 openai/gpt-4o deepseek/deepseek-chat

    # Check correctness against an expected substring, save CSV, run 3 trials
    uv run token_bench.py --expect "12" --trials 3 --csv results.csv

    # Or, since the shebang is `uv run --script`, you can also do:
    chmod +x token_bench.py
    ./token_bench.py --expect "12"

Notes
-----
- Model IDs use OpenRouter's `vendor/model` format. See https://openrouter.ai/models
- Cost is read from OpenRouter's response (`usage.cost`) when available.
- `reasoning_tokens` is reported when the provider exposes it
  (OpenAI o-series, Anthropic extended thinking, etc.).
- Dependencies are declared inline (PEP 723); uv resolves them on first run
  and caches the env so subsequent runs are instant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_QUERY = "What is the result of the sum of 4 and 8?"

# Built-in 3-prompt suite spanning easy multi-step → hard math → logic trap.
# Reach for this with --suite (no value) to run all three.
BUILTIN_SUITE = [
    {
        "label": "apples",
        "query": (
            "I have 4 apples and buy 8 more. I give half of my apples to a friend, "
            "then I triple what I have left. How many apples do I have? "
            "Just give the number."
        ),
        "expect": "18",
    },
    {
        "label": "crt",
        "query": (
            "Find the smallest positive integer N greater than 100 such that N leaves "
            "a remainder of 1 when divided by 3, a remainder of 2 when divided by 5, "
            "and a remainder of 3 when divided by 7. Just give the number."
        ),
        "expect": "157",
    },
    {
        "label": "siblings",
        "query": (
            "Alice has 3 brothers and 2 sisters. How many sisters does each of "
            "Alice's brothers have? Answer with only the number, nothing else."
        ),
        "expect": "3",
    },
]

# Default lineup: current generation + 2 previous generations for each family.
# Slugs follow OpenRouter format (vendor/model). Verify on
# https://openrouter.ai/models — vendors deprecate and rename frequently.
# Override the whole list (or any subset) with --models.
DEFAULT_MODELS = [
    # Anthropic — Claude Opus
    "anthropic/claude-opus-4.7",     # current
    "anthropic/claude-opus-4.6",     # -1
    "anthropic/claude-opus-4.5",     # -2

    # Google — Gemini Pro
    "google/gemini-3.1-pro",         # current
    "google/gemini-3-pro",           # -1
    "google/gemini-2.5-pro",         # -2

    # OpenAI — GPT
    "openai/gpt-5.5",                # current
    "openai/gpt-5.4",                # -1
    "openai/gpt-5.2",                # -2

    # DeepSeek
    "deepseek/deepseek-v4-pro",      # current
    "deepseek/deepseek-v3.2",        # -1
    "deepseek/deepseek-chat",        # -2  (V3)

    # Qwen
    "qwen/qwen3.6-max",              # current
    "qwen/qwen3.5-max",              # -1
    "qwen/qwen3-max",                # -2
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_s: float = 0.0
    answer: str = ""
    error: Optional[str] = None


@dataclass
class Result:
    model: str
    query_label: str = "default"
    trials: list[Trial] = field(default_factory=list)

    @property
    def ok_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.error is None]

    def avg(self, attr: str) -> Optional[float]:
        vals = [getattr(t, attr) for t in self.ok_trials if getattr(t, attr) is not None]
        return statistics.mean(vals) if vals else None

    @property
    def first_answer(self) -> str:
        for t in self.trials:
            if t.error is None:
                return t.answer
        return ""

    @property
    def first_error(self) -> Optional[str]:
        for t in self.trials:
            if t.error:
                return t.error
        return None


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def run_one(model: str, query: str, api_key: str, timeout: int = 60,
            reasoning_effort: str = "high") -> Trial:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/token-bench",
        "X-Title": "token-bench",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        # Ask OpenRouter to embed cost + detailed usage in the response.
        "usage": {"include": True},
    }
    # OpenRouter's unified reasoning parameter — providers map this to:
    #   Anthropic  → thinking: {type: enabled, budget_tokens: ...}
    #   OpenAI     → reasoning_effort
    #   Google     → thinking_config
    #   DeepSeek   → reasoning mode
    #   Qwen       → enable_thinking
    # Non-reasoning models silently ignore it.
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning"] = {"effort": reasoning_effort}
    t0 = time.perf_counter()
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            return Trial(
                latency_s=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        answer = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        input_t = usage.get("prompt_tokens", 0)
        output_t = usage.get("completion_tokens", 0)
        total_t = usage.get("total_tokens", input_t + output_t)
        details = usage.get("completion_tokens_details") or {}
        reasoning_t = details.get("reasoning_tokens")
        cost = usage.get("cost")
        return Trial(
            input_tokens=input_t,
            output_tokens=output_t,
            reasoning_tokens=reasoning_t,
            total_tokens=total_t,
            cost_usd=cost,
            latency_s=latency,
            answer=answer.strip(),
        )
    except Exception as e:
        return Trial(latency_s=time.perf_counter() - t0, error=str(e))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def correct(answer: str, expect: Optional[str]) -> Optional[bool]:
    """Compare model answer to expected value.

    For numeric expects, finds the last number in the answer and compares —
    robust to chain-of-thought traces that mention earlier numbers from the prompt.
    For non-numeric expects, falls back to substring match.
    """
    if expect is None:
        return None
    expect = expect.strip()
    if _NUM_RE.fullmatch(expect):
        nums = _NUM_RE.findall(answer)
        if nums:
            return nums[-1] == expect
        return False
    return expect.lower() in answer.lower()


def fmt(v, width=8, decimals=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _trial_correctness_rate(r: Result, expect: Optional[str]) -> Optional[float]:
    if expect is None or not r.ok_trials:
        return None
    hits = sum(1 for t in r.ok_trials if correct(t.answer, expect))
    return hits / len(r.ok_trials)


def print_table(results: list[Result], expect: Optional[str], trials: int,
                label: str = "default"):
    print(f"\n--- {label} ---")
    headers = ["model", "✓", "answer", "in", "out", "reason", "cost $", "lat s"]
    rows = []
    for r in results:
        err = r.first_error
        if err and not r.ok_trials:
            rows.append([r.model, "✗", f"ERROR: {err[:50]}", "-", "-", "-", "-", "-"])
            continue
        ans = r.first_answer.replace("\n", " ")
        ans_short = ans[:60] + ("…" if len(ans) > 60 else "")
        rate = _trial_correctness_rate(r, expect)
        if rate is None:
            ok_mark = "·"
        elif rate >= 1.0:
            ok_mark = "✓"
        elif rate > 0:
            ok_mark = f"{int(rate*len(r.ok_trials))}/{len(r.ok_trials)}"
        else:
            ok_mark = "✗"
        rows.append([
            r.model,
            ok_mark,
            ans_short,
            fmt(r.avg("input_tokens"), decimals=0),
            fmt(r.avg("output_tokens"), decimals=0),
            fmt(r.avg("reasoning_tokens"), decimals=0),
            fmt(r.avg("cost_usd"), decimals=6),
            fmt(r.avg("latency_s"), decimals=2),
        ])
    widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    def line(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(line(headers))
    print(sep)
    for row in rows:
        print(line(row))
    if trials > 1:
        print(f"(metrics averaged over {trials} trials per model)")


def write_csv(path: str, all_results: list[tuple[Result, Optional[str]]]):
    """Write per-trial CSV. all_results is a list of (Result, expect) tuples."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "query_label", "model", "trial", "correct", "answer",
            "input_tokens", "output_tokens", "reasoning_tokens",
            "total_tokens", "cost_usd", "latency_s", "error",
        ])
        for r, expect in all_results:
            for i, t in enumerate(r.trials, 1):
                w.writerow([
                    r.query_label, r.model, i,
                    correct(t.answer, expect) if t.error is None else "",
                    t.answer,
                    t.input_tokens, t.output_tokens, t.reasoning_tokens,
                    t.total_tokens,
                    f"{t.cost_usd:.8f}" if t.cost_usd is not None else "",
                    f"{t.latency_s:.4f}",
                    t.error or "",
                ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_suite(arg: Optional[str]) -> list[dict]:
    """Resolve --suite arg: 'builtin' (or None when --suite passed bare) → built-in,
    else treat as a JSON file path."""
    if arg in (None, "", "builtin"):
        return BUILTIN_SUITE
    with open(arg) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark LLMs on one query (or a suite) via OpenRouter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query", default=DEFAULT_QUERY,
                   help=f"Prompt to send. Default: {DEFAULT_QUERY!r}")
    p.add_argument("--query-file",
                   help="Read prompt from file (overrides --query).")
    p.add_argument("--suite", nargs="?", const="builtin", default=None,
                   help="Run a suite of prompts. Pass with no value for the built-in "
                        "3-prompt suite (apples / crt / siblings), or pass a path to "
                        "a JSON file with [{label, query, expect}, ...].")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="OpenRouter model slugs, e.g. anthropic/claude-opus-4.7")
    p.add_argument("--expect",
                   help="Expected answer for correctness (single-query mode only).")
    p.add_argument("--reasoning-effort", default="high",
                   choices=["none", "low", "medium", "high"],
                   help="OpenRouter unified reasoning effort. Default: high. "
                        "'none' disables it (legacy behavior).")
    p.add_argument("--trials", type=int, default=1,
                   help="Run each (model, query) combo N times and average (default 1).")
    p.add_argument("--csv", help="Write per-trial results to this CSV file.")
    p.add_argument("--timeout", type=int, default=120, help="Per-request timeout (s).")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    # Resolve query plan: either suite (multiple) or single query.
    if args.suite is not None:
        suite = _load_suite(args.suite)
    else:
        if args.query_file:
            with open(args.query_file) as f:
                query = f.read()
        else:
            query = args.query
        suite = [{"label": "default", "query": query, "expect": args.expect}]

    print(f"Models  : {len(args.models)}")
    print(f"Prompts : {len(suite)}  ({', '.join(s['label'] for s in suite)})")
    print(f"Trials  : {args.trials} per (model, prompt)")
    print(f"Reason  : effort={args.reasoning_effort}")
    print(f"Total   : {len(args.models) * len(suite) * args.trials} API calls")
    print()

    all_results: list[tuple[Result, Optional[str]]] = []
    for prompt in suite:
        label = prompt["label"]
        query = prompt["query"]
        expect = prompt.get("expect")
        print(f"=== {label}  (expect: {expect}) ===")
        results: list[Result] = []
        for model in args.models:
            r = Result(model=model, query_label=label)
            for i in range(args.trials):
                tag = f" t{i+1}" if args.trials > 1 else ""
                print(f"  → {model}{tag}", end="", flush=True)
                t = run_one(model, query, api_key, timeout=args.timeout,
                            reasoning_effort=args.reasoning_effort)
                r.trials.append(t)
                if t.error:
                    print(f"  ✗ {t.error[:60]}")
                else:
                    cost_str = f", ${t.cost_usd:.6f}" if t.cost_usd is not None else ""
                    rsn = f", {t.reasoning_tokens}r" if t.reasoning_tokens else ""
                    print(f"  ✓ {t.latency_s:.2f}s, {t.output_tokens} out tok{rsn}{cost_str}")
            results.append(r)
            all_results.append((r, expect))
        print_table(results, expect, args.trials, label=label)
        # Write CSV incrementally after each prompt — survives mid-run kills.
        if args.csv:
            write_csv(args.csv, all_results)
        print()

    if args.csv:
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
