#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///
"""
make_chart.py — turn token_bench CSV into a publication-ready figure.

    uv run make_chart.py apples_run.csv chart.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


# Family palette — colorblind-friendly, distinct
FAMILY_COLOR = {
    "anthropic": "#cc785c",   # Anthropic orange
    "google":    "#4285f4",   # Google blue
    "openai":    "#10a37f",   # OpenAI green
    "deepseek":  "#7b3ff2",   # DeepSeek purple
    "qwen":      "#e8b339",   # Qwen amber
}


def family_of(model: str) -> str:
    return model.split("/")[0]


def short_label(model: str) -> str:
    name = model.split("/")[1]
    # Trim '-preview' suffix and similar noise to keep labels tight
    for suffix in ("-preview", "-001", "-customtools"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def aggregate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["error"].isna() | (df["error"] == "")]
    grp = df.groupby("model").agg(
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        reasoning_tokens=("reasoning_tokens", "mean"),
        cost_usd=("cost_usd", "mean"),
        latency_s=("latency_s", "mean"),
        correct=("correct", lambda s: (s.astype(str).str.lower() == "true").mean()),
    ).reset_index()
    grp["family"] = grp["model"].apply(family_of)
    grp["short"] = grp["model"].apply(short_label)
    return grp


def make_figure(df: pd.DataFrame, out_path: Path, query: str | None = None):
    df = df.sort_values("cost_usd", ascending=True).reset_index(drop=True)
    n = len(df)
    colors = [FAMILY_COLOR.get(f, "#888") for f in df["family"]]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, max(5, 0.42 * n + 1.6)),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45},
    )

    y = range(n)

    # --- Left: output tokens (with reasoning portion stacked darker)
    out_total = df["output_tokens"].fillna(0).values
    reasoning = df["reasoning_tokens"].fillna(0).values
    visible = out_total - reasoning

    ax1.barh(y, reasoning, color=colors, alpha=0.55, label="reasoning (hidden)")
    ax1.barh(y, visible, left=reasoning, color=colors, alpha=1.0, label="visible output")
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(df["short"], fontsize=10)
    ax1.set_xlabel("Output tokens (avg per query)", fontsize=10)
    ax1.set_title("Tokens used", fontsize=12, weight="bold", loc="left", pad=10)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.25, linestyle="--")
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # Legend explaining the two-tone bars
    from matplotlib.patches import Patch
    ax1.legend(
        handles=[
            Patch(facecolor="#888", alpha=1.0, label="visible output tokens"),
            Patch(facecolor="#888", alpha=0.55, label="reasoning tokens (hidden)"),
        ],
        loc="lower right", frameon=False, fontsize=8,
    )

    # Annotate token totals at end of bars
    for i, v in enumerate(out_total):
        ax1.text(v + max(out_total) * 0.015, i, f"{int(v)}",
                 va="center", fontsize=9, color="#333")

    # --- Right: cost
    cost = df["cost_usd"].fillna(0).values * 1000  # show as $/1K queries for readability
    ax2.barh(y, cost, color=colors)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Cost per 1,000 queries (USD)", fontsize=10)
    ax2.set_title("Dollars spent", fontsize=12, weight="bold", loc="left", pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.25, linestyle="--")
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))

    # Annotate cost at end of bars
    for i, c in enumerate(cost):
        ax2.text(c + max(cost) * 0.015, i, f"${c:.2f}",
                 va="center", fontsize=9, color="#333")

    # Mark wrong answers with red label suffix (avoid overlapping y-tick text)
    wrong = df["correct"] < 1.0
    new_labels = []
    for short, w in zip(df["short"], wrong):
        if w:
            new_labels.append(f"{short}  (wrong ✗)")
        else:
            new_labels.append(short)
    ax1.set_yticklabels(new_labels, fontsize=10)
    # Color the wrong-answer tick red
    for tick, w in zip(ax1.get_yticklabels(), wrong):
        if w:
            tick.set_color("#d62728")
            tick.set_weight("bold")

    # --- Title block
    title = "Same question. Same answer. Wildly different bills."
    subtitle = (
        "14 frontier LLMs solving “4 apples + 8 more, halve, triple” → 18.\n"
        "Output tokens vary 186×; cost varies 86×. The cheapest model got it wrong."
    )
    fig.suptitle(title, fontsize=15, weight="bold", x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.93, subtitle, fontsize=10, color="#555", ha="left", va="top")

    # Family color legend
    legend_handles = [Patch(color=c, label=f.capitalize())
                      for f, c in FAMILY_COLOR.items() if f in df["family"].values]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fig.text(
        0.06, -0.03,
        f"Source: token_bench.py via OpenRouter, 3 trials per model, April 2026."
        + (f"  Prompt: “{query}”" if query else ""),
        fontsize=8, color="#888", ha="left",
    )

    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.18, right=0.95)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) < 3:
        print("usage: make_chart.py <input.csv> <output.png> [query-text]", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    query = sys.argv[3] if len(sys.argv) > 3 else None
    df = aggregate(csv_path)
    make_figure(df, out_path, query=query)


if __name__ == "__main__":
    main()
