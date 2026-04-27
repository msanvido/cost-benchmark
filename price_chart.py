#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "requests",
# ]
# ///
"""
price_chart.py — per-token list prices from OpenRouter, plotted in the
                 same model order as suite_compare.png.

    export OPENROUTER_API_KEY=sk-or-...
    uv run price_chart.py suite_run.csv price_chart.png
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


FAMILY_COLOR = {
    "anthropic": "#cc785c",
    "google":    "#4285f4",
    "openai":    "#10a37f",
    "deepseek":  "#7b3ff2",
    "qwen":      "#e8b339",
}


def family_of(m): return m.split("/")[0]


def short_label(m):
    name = m.split("/")[1]
    for s in ("-preview", "-001", "-customtools"):
        if name.endswith(s):
            name = name[: -len(s)]
    return name


def fetch_prices(models: list[str]) -> dict[str, dict]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get("https://openrouter.ai/api/v1/models",
                     headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    wanted = set(models)
    out = {}
    for m in data:
        if m["id"] in wanted:
            p = m.get("pricing", {}) or {}
            # Convert OpenRouter's per-token strings to $/M tokens
            try:
                inp = float(p.get("prompt") or 0) * 1_000_000
                outp = float(p.get("completion") or 0) * 1_000_000
            except (TypeError, ValueError):
                inp, outp = 0.0, 0.0
            out[m["id"]] = {"input_per_m": inp, "output_per_m": outp}
    return out


def model_order(suite_csv: Path) -> list[str]:
    """Replicate compare_chart.py's ordering: family, then avg cost within family."""
    df = pd.read_csv(suite_csv)
    df["family"] = df["model"].apply(family_of)
    return sorted(df["model"].unique(),
                  key=lambda m: (family_of(m),
                                 df.loc[df["model"] == m, "cost_usd"].mean()))


def make_figure(suite_csv: Path, out_path: Path):
    models = model_order(suite_csv)
    prices = fetch_prices(models)

    rows = []
    for m in models:
        p = prices.get(m, {"input_per_m": 0, "output_per_m": 0})
        rows.append({
            "model": m,
            "family": family_of(m),
            "short": short_label(m),
            "input": p["input_per_m"],
            "output": p["output_per_m"],
        })
    df = pd.DataFrame(rows)
    colors = [FAMILY_COLOR.get(f, "#888") for f in df["family"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8),
                                   gridspec_kw={"wspace": 0.30})

    y = list(range(len(df)))

    # --- Input prices
    in_max = df["input"].max() * 1.25
    ax1.barh(y, df["input"], color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["short"], fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Input price ($ per 1M tokens)", fontsize=10)
    ax1.set_title("Input tokens", fontsize=12, weight="bold", loc="left", pad=8)
    ax1.set_xlim(0, in_max)
    ax1.grid(axis="x", alpha=0.25, linestyle="--")
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
    for i, v in enumerate(df["input"]):
        ax1.text(v + in_max * 0.015, i, f"${v:.2f}",
                 va="center", fontsize=9, color="#333")

    # --- Output prices
    out_max = df["output"].max() * 1.25
    ax2.barh(y, df["output"], color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Output price ($ per 1M tokens)", fontsize=10)
    ax2.set_title("Output tokens", fontsize=12, weight="bold", loc="left", pad=8)
    ax2.set_xlim(0, out_max)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
    for i, v in enumerate(df["output"]):
        ax2.text(v + out_max * 0.015, i, f"${v:.2f}",
                 va="center", fontsize=9, color="#333")

    # Family legend
    families_present = [f for f in FAMILY_COLOR if f in df["family"].values]
    legend_handles = [Patch(color=FAMILY_COLOR[f], label=f.capitalize())
                      for f in families_present]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Per-token list prices — same models, same order",
                 fontsize=15, weight="bold", x=0.06, ha="left", y=0.97)
    fig.text(0.06, 0.93,
             "OpenRouter list price per million tokens, April 2026. "
             "Same y-axis order as the reasoning ON/OFF chart.",
             fontsize=10, color="#555", ha="left")
    fig.text(0.06, -0.01,
             "Source: OpenRouter /api/v1/models, pulled live.",
             fontsize=8, color="#888", ha="left")

    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.18, right=0.97)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) != 3:
        print("usage: price_chart.py <suite_run.csv> <output.png>", file=sys.stderr)
        sys.exit(1)
    make_figure(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
