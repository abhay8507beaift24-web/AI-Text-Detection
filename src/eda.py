"""
eda.py - Exploratory Data Analysis for AI Text Detection dataset
"""

import re
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from pathlib import Path


OUTPUT_DIR = Path("eda_plots")


def run_eda(df: pd.DataFrame, save_plots: bool = True) -> dict:
    """
    Run full EDA on the dataset and return a summary dict.
    Optionally saves plots to disk.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = {}

    # --- Basic stats ---
    total = len(df)
    human_count = (df["label"] == 0).sum()
    ai_count = (df["label"] == 1).sum()
    summary["total"] = total
    summary["human"] = int(human_count)
    summary["ai"] = int(ai_count)
    summary["balance_ratio"] = round(human_count / ai_count, 3)

    print(f"\n[EDA] Total samples   : {total}")
    print(f"[EDA] Human (0)       : {human_count} ({100*human_count/total:.1f}%)")
    print(f"[EDA] AI-generated (1): {ai_count} ({100*ai_count/total:.1f}%)")

    # --- Text length stats ---
    df = df.copy()
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))

    for label, name in [(0, "Human"), (1, "AI")]:
        lens = df[df["label"] == label]["text_len"]
        print(f"\n[EDA] {name} - word count: mean={lens.mean():.1f}, "
              f"median={lens.median():.1f}, std={lens.std():.1f}")
        summary[f"{name.lower()}_mean_len"] = round(float(lens.mean()), 2)

    if save_plots:
        _plot_class_distribution(df, human_count, ai_count)
        _plot_text_length_distribution(df)
        _plot_top_words(df)
        print(f"\n[EDA] Plots saved to '{OUTPUT_DIR}/' directory.")

    return summary


def _plot_class_distribution(df, human_count, ai_count):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Human", "AI-Generated"], [human_count, ai_count],
                  color=["#4CAF50", "#F44336"], edgecolor="black", width=0.5)
    for bar, count in zip(bars, [human_count, ai_count]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(human_count, ai_count) * 1.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()


def _plot_text_length_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colors = {0: "#4CAF50", 1: "#F44336"}
    labels = {0: "Human", 1: "AI-Generated"}
    for label, ax in zip([0, 1], axes):
        data = df[df["label"] == label]["text_len"]
        ax.hist(data, bins=30, color=colors[label], edgecolor="black", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", label=f"Mean: {data.mean():.1f}")
        ax.set_title(f"{labels[label]} Text Length (words)", fontweight="bold")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "text_length_distribution.png", dpi=150)
    plt.close()


def _plot_top_words(df, top_n: int = 15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stopwords = {"the", "a", "an", "is", "in", "it", "of", "to", "and", "that",
                 "this", "for", "with", "i", "be", "was", "are", "on", "at", "by"}
    for label, ax, color in zip([0, 1], axes, ["#4CAF50", "#F44336"]):
        texts = " ".join(df[df["label"] == label]["text"].tolist()).lower()
        words = [w.strip(string.punctuation) for w in texts.split()
                 if w.strip(string.punctuation) not in stopwords and len(w) > 2]
        counter = Counter(words).most_common(top_n)
        words_sorted, counts = zip(*counter)
        ax.barh(list(reversed(words_sorted)), list(reversed(counts)), color=color, edgecolor="black")
        ax.set_title(f"Top Words: {'Human' if label == 0 else 'AI-Generated'}", fontweight="bold")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_words.png", dpi=150)
    plt.close()