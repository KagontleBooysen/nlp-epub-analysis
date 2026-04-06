"""
src/descriptive.py
Corpus-level and document-level descriptive statistics.

Metrics
-------
  - Token count, type count, TTR
  - MTLD  (Measure of Textual Lexical Diversity)
  - HD-D  (Hypergeometric Distribution D)
  - Mean / median / std sentence length
  - Top-N word frequencies
  - Zipf fit coefficient
"""
import math
import statistics
from collections import Counter

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from lexicalrichness import LexicalRichness


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_mtld(lex: LexicalRichness, threshold: float = 0.72) -> float:
    try:
        return round(lex.mtld(threshold=threshold), 2)
    except Exception:
        return float("nan")


def _safe_hdd(lex: LexicalRichness, draws: int = 42) -> float:
    try:
        return round(lex.hdd(draws=draws), 4)
    except Exception:
        return float("nan")


# ── Core stats ────────────────────────────────────────────────────────────────

def token_stats(full_text: str) -> dict:
    """
    Compute token/type/TTR + lexical diversity metrics.

    Parameters
    ----------
    full_text : the complete text of one document

    Returns
    -------
    dict with keys: num_tokens, num_types, ttr, mtld, hdd
    """
    lex = LexicalRichness(full_text)
    tokens = word_tokenize(full_text.lower())
    freq   = Counter(t for t in tokens if t.isalpha())

    return {
        "num_tokens": lex.words,
        "num_types" : lex.terms,
        "ttr"       : round(lex.ttr, 4),
        "mtld"      : _safe_mtld(lex),
        "hdd"       : _safe_hdd(lex),
        "top_20"    : freq.most_common(20),
    }


def sentence_length_stats(full_text: str) -> dict:
    """
    Sentence length statistics (in tokens).
    """
    sentences = sent_tokenize(full_text)
    lengths   = [len(word_tokenize(s)) for s in sentences]

    if not lengths:
        return {}

    return {
        "num_sentences"       : len(lengths),
        "mean_sent_len"       : round(statistics.mean(lengths), 2),
        "median_sent_len"     : round(statistics.median(lengths), 2),
        "stdev_sent_len"      : round(statistics.stdev(lengths) if len(lengths) > 1 else 0, 2),
        "min_sent_len"        : min(lengths),
        "max_sent_len"        : max(lengths),
        "sentence_lengths_raw": lengths,  # for histogram
    }


def zipf_fit(full_text: str, top_n: int = 200) -> dict:
    """
    Fit Zipf's Law: f(r) ≈ C / r^α
    Returns {'alpha': float, 'r_squared': float}
    """
    tokens = word_tokenize(full_text.lower())
    freq   = Counter(t for t in tokens if t.isalpha())
    counts = [c for _, c in freq.most_common(top_n)]

    if len(counts) < 10:
        return {"alpha": float("nan"), "r_squared": float("nan")}

    ranks  = np.arange(1, len(counts) + 1, dtype=float)
    log_r  = np.log(ranks)
    log_f  = np.log(np.array(counts, dtype=float))

    # Linear regression in log-log space
    coef   = np.polyfit(log_r, log_f, 1)
    y_pred = np.polyval(coef, log_r)
    ss_res = np.sum((log_f - y_pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha"    : round(-coef[0], 4),  # Zipf exponent (expect ~1.0)
        "r_squared": round(r2, 4),
    }


# ── Corpus-level summary ──────────────────────────────────────────────────────

def corpus_stats(documents: list[dict]) -> pd.DataFrame:
    """
    Compute stats for every document in a corpus list.

    Parameters
    ----------
    documents : list of dicts with at least {"title", "author", "full_text"}

    Returns
    -------
    pd.DataFrame with one row per document
    """
    rows = []
    for doc in documents:
        print(f"  Stats for: {doc['title']}")
        ts   = token_stats(doc["full_text"])
        sls  = sentence_length_stats(doc["full_text"])
        zf   = zipf_fit(doc["full_text"])

        row  = {
            "title"          : doc["title"],
            "author"         : doc["author"],
            "num_tokens"     : ts["num_tokens"],
            "num_types"      : ts["num_types"],
            "ttr"            : ts["ttr"],
            "mtld"           : ts["mtld"],
            "hdd"            : ts["hdd"],
            "num_sentences"  : sls.get("num_sentences"),
            "mean_sent_len"  : sls.get("mean_sent_len"),
            "median_sent_len": sls.get("median_sent_len"),
            "stdev_sent_len" : sls.get("stdev_sent_len"),
            "zipf_alpha"     : zf["alpha"],
            "zipf_r2"        : zf["r_squared"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = " ".join(["the quick brown fox jumps over the lazy dog"] * 50)
    print(token_stats(sample))
    print(sentence_length_stats(sample))
    print(zipf_fit(sample))
