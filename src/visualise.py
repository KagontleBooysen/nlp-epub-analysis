"""
src/visualise.py
All chart generation for the NLP pipeline.

Charts produced
---------------
  1.  sentiment_arc      – Chapter sentiment trajectory (line plot)
  2.  sentiment_bars     – Mean VADER score per text (horizontal bar chart)
  3.  ner_bars           – Entity category distribution per text
  4.  ner_heatmap        – Entity %-heatmap across corpus
  5.  tfidf_wordcloud    – Word cloud from TF-IDF scores
  6.  tfidf_bars         – Top-10 keywords per text (horizontal bars)
  7.  lda_coherence      – K vs coherence curve
  8.  lda_topic_heatmap  – Doc x topic probability heatmap
  9.  lexical_diversity  – MTLD/HD-D grouped bars
  10. sentence_hist      – Sentence length bar chart
"""
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = "Blues_d"
FIG_DPI   = 150
TITLE_PAD = 12
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _short(title: str, n: int = 35) -> str:
    """Truncate long book titles for axis labels."""
    return title[:n] + "…" if len(title) > n else title


# ── 1. Sentiment arc ──────────────────────────────────────────────────────────

def plot_sentiment_arc(chapter_scores: list,
                       title: str,
                       output_path: str):
    """Line plot of VADER compound score per chapter."""
    fig, ax = plt.subplots(figsize=(10, 4))
    chapters = range(1, len(chapter_scores) + 1)
    ax.plot(chapters, chapter_scores, marker="o", markersize=4,
            linewidth=1.5, color="#2E75B6")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.fill_between(chapters, chapter_scores, 0,
                    where=[s >= 0 for s in chapter_scores],
                    alpha=0.15, color="#2E75B6", label="Positive")
    ax.fill_between(chapters, chapter_scores, 0,
                    where=[s < 0 for s in chapter_scores],
                    alpha=0.15, color="#C0392B", label="Negative")
    ax.set_title(f"Sentiment Arc – {title[:60]}", pad=TITLE_PAD,
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Chapter", fontsize=11)
    ax.set_ylabel("VADER Compound Score", fontsize=11)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    _save(fig, output_path)


# ── 2. Corpus sentiment bar chart ─────────────────────────────────────────────

def plot_sentiment_bars(stats_df: pd.DataFrame, output_path: str):
    """Horizontal bar chart of mean VADER score per text — fixed for 99 texts."""
    df = stats_df.copy()
    df["short_title"] = df["title"].apply(lambda t: _short(t, 40))
    df = df.sort_values("mean_vader_compound")

    colors = ["#C0392B" if v < 0 else "#2E75B6" for v in df["mean_vader_compound"]]

    # Tall enough to show all 99 titles without overlap
    fig, ax = plt.subplots(figsize=(12, max(14, len(df) * 0.28)))
    bars = ax.barh(df["short_title"], df["mean_vader_compound"],
                   color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Mean VADER Compound Score by Text",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean VADER Compound Score  (−1 = most negative, +1 = most positive)",
                  fontsize=10)
    ax.set_ylabel("Text Title", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)

    for bar, val in zip(bars, df["mean_vader_compound"]):
        ax.text(val + (0.003 if val >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=6)
    _save(fig, output_path)


# ── 3. NER category stacked bar ───────────────────────────────────────────────

def plot_ner_bars(ner_df: pd.DataFrame, output_path: str):
    """
    Horizontal stacked bar chart of NER macro-category percentages.
    Fixed for 99 texts — uses horizontal bars so titles are readable.
    """
    cats     = ["PERSON", "GPE/LOC", "ORG", "DATE/TIME", "OTHER"]
    pct_cols = ["pct_PERSON", "pct_GPE_LOC", "pct_ORG", "pct_DATE_TIME", "pct_OTHER"]
    use_pct  = all(c in ner_df.columns for c in pct_cols)
    plot_cols = pct_cols if use_pct else cats

    df = ner_df.copy()
    df["short_title"] = df["title"].apply(lambda t: _short(t, 40))
    df = df.set_index("short_title")[plot_cols]
    df.columns = cats

    colors = ["#1F4E79", "#2E75B6", "#5BA3D0", "#A8D1E7", "#D6EAF8"]

    # Tall figure — one row per text
    fig, ax = plt.subplots(figsize=(13, max(16, len(df) * 0.28)))

    left = np.zeros(len(df))
    for col, color in zip(cats, colors):
        vals = df[col].values
        ax.barh(df.index, vals, left=left, color=color,
                label=col, edgecolor="white", height=0.7)
        left += vals

    ax.set_title("NER Entity Distribution by Text",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    ax.set_xlabel("% of Total Entities" if use_pct else "Entity Count", fontsize=10)
    ax.set_ylabel("Text Title", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(loc="lower right", fontsize=9, title="Entity Type",
              title_fontsize=9)
    ax.set_xlim(0, 101 if use_pct else None)
    _save(fig, output_path)


# ── 4. NER heatmap ────────────────────────────────────────────────────────────

def plot_ner_heatmap(ner_df: pd.DataFrame, output_path: str):
    """Heatmap of NER counts — fixed label sizes for 99 texts."""
    cats = ["PERSON", "GPE/LOC", "ORG", "DATE/TIME", "OTHER"]
    df = ner_df.copy()
    df["short_title"] = df["title"].apply(lambda t: _short(t, 40))
    matrix = df.set_index("short_title")[cats]

    fig, ax = plt.subplots(figsize=(10, max(18, len(df) * 0.28)))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.4, ax=ax,
                annot_kws={"size": 6})
    ax.set_title("Named Entity Counts — Corpus Heatmap",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    ax.set_xlabel("Entity Category", fontsize=10)
    ax.set_ylabel("Text Title", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)
    _save(fig, output_path)


# ── 5. TF-IDF word cloud ──────────────────────────────────────────────────────

def plot_tfidf_wordcloud(kw_dict: dict, title: str, output_path: str):
    """Word cloud for one text using TF-IDF scores as weights."""
    if title not in kw_dict:
        return
    freq = {kw: score for kw, score in kw_dict[title] if score > 0}
    if not freq:
        return

    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="Blues", max_words=60, prefer_horizontal=0.9)
    wc.generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"TF-IDF Word Cloud – {title[:60]}",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    _save(fig, output_path)


# ── 6. TF-IDF keyword bars ────────────────────────────────────────────────────

def plot_tfidf_bars(kw_df: pd.DataFrame, title: str, output_path: str, n: int = 10):
    """Top-N TF-IDF keywords for one text."""
    df = kw_df[kw_df["title"] == title].head(n).sort_values("tfidf_score")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["keyword"], df["tfidf_score"],
            color="#2E75B6", edgecolor="white")
    ax.set_title(f"Top {n} Keywords (TF-IDF) – {title[:60]}",
                 pad=TITLE_PAD, fontsize=12, fontweight="bold")
    ax.set_xlabel("TF-IDF Score", fontsize=10)
    ax.set_ylabel("Keyword", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    _save(fig, output_path)


# ── 7. LDA coherence curve ────────────────────────────────────────────────────

def plot_coherence_curve(coherence_scores: dict, output_path: str):
    """K vs C_v coherence line plot."""
    if not coherence_scores:
        return
    ks   = sorted(coherence_scores.keys())
    cvs  = [coherence_scores[k] for k in ks]
    best = max(coherence_scores, key=coherence_scores.get)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, cvs, "o-", color="#2E75B6", linewidth=2, markersize=7)
    ax.axvline(best, color="#C0392B", linestyle="--", linewidth=1.5,
               label=f"Best K={best}  (C_v={coherence_scores[best]:.3f})")
    ax.set_title("LDA Coherence Score vs Number of Topics (K)",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    ax.set_xlabel("K (Number of Topics)", fontsize=11)
    ax.set_ylabel("C_v Coherence Score", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=10)
    _save(fig, output_path)


# ── 8. LDA topic heatmap ──────────────────────────────────────────────────────

def plot_topic_heatmap(doc_topic_df: pd.DataFrame, output_path: str):
    """Document × topic probability heatmap — fixed for 99 texts."""
    topic_cols = [c for c in doc_topic_df.columns if c.startswith("topic_")]
    df = doc_topic_df.copy()
    df["short_title"] = df["title"].apply(lambda t: _short(t, 35))
    matrix = df.set_index("short_title")[topic_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(topic_cols) * 0.8),
                                    max(18, len(df) * 0.28)))
    sns.heatmap(matrix, annot=False, cmap="Blues",
                linewidths=0.3, ax=ax, vmin=0, vmax=0.5)
    ax.set_title("Document–Topic Probability Matrix (LDA)",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    ax.set_xlabel("Topic", fontsize=10)
    ax.set_ylabel("Text Title", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)
    _save(fig, output_path)


# ── 9. Lexical diversity bars ─────────────────────────────────────────────────

def plot_lexical_diversity(stats_df: pd.DataFrame, output_path: str):
    """
    Two horizontal bar charts side by side — MTLD and HD-D.
    Fixed for 99 texts: tall figure, readable y-axis labels.
    """
    df = stats_df[["title", "mtld", "hdd"]].dropna().copy()
    df["short_title"] = df["title"].apply(lambda t: _short(t, 38))
    df = df.sort_values("mtld")

    # Tall enough for all 99 titles
    fig, axes = plt.subplots(1, 2, figsize=(18, max(16, len(df) * 0.28)))

    for ax, col, label, xlabel in zip(
        axes,
        ["mtld", "hdd"],
        ["MTLD Score", "HD-D Score"],
        ["MTLD Score (higher = more lexically diverse)",
         "HD-D Score (0–1, higher = more diverse)"]
    ):
        vals = df.set_index("short_title")[col]
        ax.barh(vals.index, vals.values,
                color="#2E75B6", edgecolor="white", height=0.7)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Text Title", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Lexical Diversity Metrics — All 99 Texts",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, output_path)


# ── 10. Sentence length bar chart ─────────────────────────────────────────────

def plot_sentence_histogram(stats_df: pd.DataFrame, output_path: str):
    """
    Horizontal bar chart of mean sentence length per text.
    Fixed for 99 texts: tall figure, readable labels, proper axis labels.
    """
    df = stats_df[["title", "mean_sent_len", "stdev_sent_len"]].dropna().copy()
    df["short_title"] = df["title"].apply(lambda t: _short(t, 38))
    df = df.sort_values("mean_sent_len")

    fig, ax = plt.subplots(figsize=(12, max(16, len(df) * 0.28)))
    ax.barh(df["short_title"], df["mean_sent_len"],
            xerr=df["stdev_sent_len"],
            color="#2E75B6", edgecolor="white",
            capsize=3, error_kw={"elinewidth": 0.8})
    ax.set_title("Mean Sentence Length per Text (tokens ± 1 SD)",
                 pad=TITLE_PAD, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Tokens per Sentence", fontsize=10)
    ax.set_ylabel("Text Title", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)
    _save(fig, output_path)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores = [0.1, -0.2, 0.3, 0.4, -0.1, 0.05, -0.3, 0.2]
    plot_sentiment_arc(scores, "Demo Text", "outputs/plots/demo_arc.png")
    print("Demo chart saved.")