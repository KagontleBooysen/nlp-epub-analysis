"""
src/sentiment.py
Sentence-level sentiment analysis using:
  1. VADER  – rule-based, fast, no GPU required
  2. DistilBERT – fine-tuned transformer (SST-2)

Both can be run independently or together.
"""
import warnings
warnings.filterwarnings("ignore")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── VADER ─────────────────────────────────────────────────────────────────────

_vader = None

def _get_vader():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def vader_sentence(sentence: str) -> dict:
    """Return VADER scores for a single sentence."""
    scores = _get_vader().polarity_scores(sentence)
    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral" : scores["neu"],
        "label"   : "positive" if scores["compound"] >= 0.05
                    else "negative" if scores["compound"] <= -0.05
                    else "neutral",
    }


def vader_document(text: str) -> pd.DataFrame:
    """
    Run VADER on every sentence in a text.
    Returns a DataFrame with columns: sentence, compound, positive, negative, neutral, label.
    """
    sentences = sent_tokenize(text)
    rows = [{"sentence": s, **vader_sentence(s)} for s in sentences]
    return pd.DataFrame(rows)


def vader_chapters(chapters: list[str]) -> list[float]:
    """Return mean VADER compound score per chapter."""
    chapter_scores = []
    for ch in chapters:
        sents = sent_tokenize(ch)
        if not sents:
            chapter_scores.append(0.0)
            continue
        scores = [_get_vader().polarity_scores(s)["compound"] for s in sents]
        chapter_scores.append(round(float(np.mean(scores)), 4))
    return chapter_scores


# ── DistilBERT ────────────────────────────────────────────────────────────────

_bert_pipeline = None

def _get_bert():
    global _bert_pipeline
    if _bert_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"  Loading DistilBERT on {'GPU' if device == 0 else 'CPU'} …")
        _bert_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            truncation=True,
            max_length=512,
        )
    return _bert_pipeline


def bert_sentence(sentence: str) -> dict:
    """Return BERT label + score for a single sentence (≤512 tokens)."""
    result = _get_bert()(sentence[:1024])[0]
    return {
        "bert_label": result["label"].lower(),
        "bert_score": round(result["score"], 4),
    }


def bert_document(text: str, batch_size: int = 32) -> pd.DataFrame:
    """
    Run DistilBERT on every sentence in a text.
    Batched for speed. Returns DataFrame with: sentence, bert_label, bert_score.
    """
    bert = _get_bert()
    sentences = sent_tokenize(text)
    # Truncate each to 512 chars to stay within token budget
    truncated = [s[:512] for s in sentences]

    labels, scores = [], []
    for i in tqdm(range(0, len(truncated), batch_size), desc="BERT"):
        batch   = truncated[i: i + batch_size]
        results = bert(batch)
        for r in results:
            labels.append(r["label"].lower())
            scores.append(round(r["score"], 4))

    return pd.DataFrame({
        "sentence"  : sentences,
        "bert_label": labels,
        "bert_score": scores,
    })


# ── Combined ──────────────────────────────────────────────────────────────────

def full_sentiment(text: str, use_bert: bool = True) -> pd.DataFrame:
    """
    Run VADER + (optionally) BERT on all sentences.
    Returns a merged DataFrame.
    """
    vader_df = vader_document(text)

    if use_bert:
        bert_df  = bert_document(text)
        df = vader_df.merge(bert_df[["sentence", "bert_label", "bert_score"]],
                            on="sentence", how="left")
    else:
        df = vader_df

    return df


def sentiment_summary(df: pd.DataFrame) -> dict:
    """
    Summarise a sentiment DataFrame from full_sentiment().
    Returns aggregate stats.
    """
    summary = {
        "mean_vader_compound" : round(df["compound"].mean(), 4),
        "pct_positive_vader"  : round((df["label"] == "positive").mean() * 100, 1),
        "pct_negative_vader"  : round((df["label"] == "negative").mean() * 100, 1),
        "pct_neutral_vader"   : round((df["label"] == "neutral").mean() * 100, 1),
    }
    if "bert_label" in df.columns:
        summary["pct_positive_bert"] = round(
            (df["bert_label"] == "positive").mean() * 100, 1)
    return summary


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    text = (
        "It was a wonderful, glorious morning! "
        "The dark clouds gathered ominously overhead. "
        "She felt nothing in particular about the grey sky."
    )
    df = full_sentiment(text, use_bert=False)
    print(df[["sentence", "compound", "label"]])
    print(sentiment_summary(df))
