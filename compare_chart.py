#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///
"""
compare_chart.py — 2-row × 3-column comparison chart for the suite.

    uv run compare_chart.py suite_run.csv suite_run_nothink.csv suite_compare.png

Top row: with reasoning (effort=high)
Bottom row: without reasoning (effort=none)
Each column is one prompt from the suite.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


FAMILY_COLOR = {
    "anthropic": "#cc785c",
    "google":    "#4285f4",
    "openai":    "#10a37f",
    "deepseek":  "#7b3ff2",
    "qwen":      "#e8b339",
}


def family_of(model: str) -> str:
    return model.split("/")[0]


def short_label(model: str) -> str:
    name = model.split("/")[1]
    for suffix in ("-preview", "-001", "-customtools"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_and_aggregate(path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mode"] = mode
    df["had_error"] = df["error"].fillna("").astype(str).str.len() > 0
    df["correct_b"] = df["correct"].astype(str).str.lower() == "true"

    # Aggregate per (mode, query_label, model)
    grp = df.groupby(["mode", "query_label", "model"]).agg(
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        reasoning_tokens=("reasoning_tokens", "mean"),
        cost_usd=("cost_usd", "mean"),
        latency_s=("latency_s", "mean"),
        n_total=("trial", "count"),
        n_ok=("had_error", lambda s: (~s).sum()),
        n_correct=("correct_b", "sum"),
    ).reset_index()
    grp["family"] = grp["model"].apply(family_of)
    grp["short"] = grp["model"].apply(short_label)
    grp["any_error"] = grp["n_ok"] < grp["n_total"]
    grp["any_wrong"] = (grp["n_correct"] < grp["n_ok"])
    return grp


def panel(ax, df_panel, all_models_in_order, max_cost, *, show_yticks=False,
          col_title=None, row_title=None):
    df_panel = df_panel.set_index("model").reindex(all_models_in_order).reset_index()

    n = len(df_panel)
    y = list(range(n))

    colors = [FAMILY_COLOR.get(f, "#888") for f in df_panel["family"]]
    cost = df_panel["cost_usd"].fillna(0).values * 1000  # $/1K queries

    bars = ax.barh(y, cost, color=colors)

    # Annotate cost at end of bars
    for i, c in enumerate(cost):
        if c > 0:
            ax.text(c + max_cost * 0.02, i, f"${c:.2f}",
                    va="center", fontsize=8, color="#333")

    # Mark errored / wrong rows
    for i, row in df_panel.iterrows():
        if pd.isna(row["model"]):
            continue
        if row.get("any_error", False) and row["n_ok"] == 0:
            ax.text(max_cost * 0.5, i, "errored",
                    va="center", ha="center", fontsize=8,
                    color="#d62728", weight="bold",
                    bbox=dict(facecolor="#fee", edgecolor="#d62728", boxstyle="round,pad=0.2"))
        elif row.get("any_wrong", False):
            ax.text(max_cost * 0.5, i, "wrong",
                    va="center", ha="center", fontsize=8,
                    color="#d62728", weight="bold",
                    bbox=dict(facecolor="#fee", edgecolor="#d62728", boxstyle="round,pad=0.2"))

    ax.set_xlim(0, max_cost * 1.2)
    ax.set_yticks(y)
    if show_yticks:
        ax.set_yticklabels([short_label(m) for m in all_models_in_order], fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.1f}"))
    ax.tick_params(axis="x", labelsize=8)

    if col_title:
        ax.set_title(col_title, fontsize=12, weight="bold", loc="left", pad=8)
    if row_title:
        ax.text(-0.42, 0.5, row_title, transform=ax.transAxes,
                fontsize=12, weight="bold", color="#444",
                ha="center", va="center", rotation=90)


def make_figure(thinking_csv: Path, nothink_csv: Path, out_path: Path):
    df_t = load_and_aggregate(thinking_csv, "thinking")
    df_n = load_and_aggregate(nothink_csv, "nothink")
    df = pd.concat([df_t, df_n], ignore_index=True)

    prompts = ["apples", "crt", "siblings"]
    prompt_titles = {
        "apples":   'Easy: "4 apples + 8, halve, triple" → 18',
        "crt":      'Hard math (CRT) → 157',
        "siblings": 'Logic trap (Alice\'s sisters) → 3',
    }

    # Stable model order: family, then by aggregate cost ascending across all panels
    all_models = sorted(df["model"].unique(),
                        key=lambda m: (family_of(m),
                                       df.loc[df["model"] == m, "cost_usd"].mean()))

    # Per-prompt max cost across both modes for shared x-axis within a column
    col_max = {p: df.loc[df["query_label"] == p, "cost_usd"].max() * 1000 * 1.0
               for p in prompts}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col, p in enumerate(prompts):
        for row, mode in enumerate(["thinking", "nothink"]):
            ax = axes[row, col]
            sub = df[(df["mode"] == mode) & (df["query_label"] == p)]
            panel(
                ax, sub, all_models, col_max[p],
                show_yticks=(col == 0),
                col_title=prompt_titles[p] if row == 0 else None,
                row_title=("reasoning = high" if mode == "thinking" else "reasoning = off")
                          if col == 0 else None,
            )

    # Shared X label on bottom row
    for ax in axes[1, :]:
        ax.set_xlabel("Cost per 1,000 queries (USD)", fontsize=9)

    # Family color legend
    families_present = [f for f in FAMILY_COLOR if f in df["family"].values]
    legend_handles = [Patch(color=FAMILY_COLOR[f], label=f.capitalize())
                      for f in families_present]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Reasoning ON vs. OFF — same prompts, same models",
                 fontsize=15, weight="bold", x=0.06, ha="left", y=0.99)
    fig.text(0.06, 0.955,
             "Top row: OpenRouter reasoning_effort=high.   Bottom row: reasoning_effort=none.   "
             "Each bar is the average cost per 1,000 calls.",
             fontsize=9.5, color="#555", ha="left")

    fig.text(0.06, -0.025,
             "Source: token_bench.py via OpenRouter, April 2026. 15 models × 3 prompts × 1 trial. "
             "Wrong/errored runs flagged in red.",
             fontsize=8, color="#888", ha="left")

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.20, right=0.98,
                        hspace=0.30, wspace=0.20)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) != 4:
        print("usage: compare_chart.py <thinking.csv> <nothink.csv> <output.png>",
              file=sys.stderr)
        sys.exit(1)
    make_figure(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
