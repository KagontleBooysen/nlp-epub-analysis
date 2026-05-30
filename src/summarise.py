"""
src/summarise.py
Text summarisation:
  1. Extractive TextRank (sumy) – fast, no GPU
  2. Abstractive BART (HuggingFace) – higher quality, GPU recommended

Both functions take plain text strings and return summary strings.
"""
import warnings
warnings.filterwarnings("ignore")

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import pandas as pd
from nltk.tokenize import sent_tokenize

LANGUAGE = "english"

# ── Extractive (TextRank) ─────────────────────────────────────────────────────

def textrank_summary(text: str, n_sentences: int = 10) -> str:
    """
    Extractive summary using TextRank algorithm.

    Parameters
    ----------
    text        : full document text
    n_sentences : number of sentences to extract

    Returns
    -------
    str – extracted summary
    """
    if not text.strip():
        return ""

    stemmer    = Stemmer(LANGUAGE)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    parser  = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summary = summarizer(parser.document, n_sentences)

    return " ".join(str(sentence) for sentence in summary)


def lsa_summary(text: str, n_sentences: int = 10) -> str:
    """
    Extractive summary using Latent Semantic Analysis.
    Alternative to TextRank.
    """
    if not text.strip():
        return ""

    stemmer    = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    parser  = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summary = summarizer(parser.document, n_sentences)

    return " ".join(str(sentence) for sentence in summary)


# ── Abstractive (BART) ────────────────────────────────────────────────────────

_bart_pipeline = None

def _get_bart():
    global _bart_pipeline
    if _bart_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"  Loading BART-large-CNN on {'GPU' if device == 0 else 'CPU'} …")
        _bart_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
        )
    return _bart_pipeline


def bart_summary(text: str,
                 max_input_tokens: int = 1024,
                 max_length: int = 150,
                 min_length: int = 40) -> str:
    """
    Abstractive summary using BART-large-CNN.
    Input is truncated to max_input_tokens characters × ~5 chars/token heuristic.
    """
    # Rough character budget: 1024 tokens × ~5 chars/token
    truncated = text[: max_input_tokens * 5]
    result    = _get_bart()(
        truncated,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return result[0]["summary_text"]


def bart_chapter_summaries(chapters: list[str],
                           max_input_tokens: int = 900) -> list[str]:
    """
    Summarise each chapter individually (short chapters are kept as-is).
    """
    summaries = []
    for i, ch in enumerate(chapters):
        tokens_approx = len(ch.split())
        if tokens_approx < 50:
            summaries.append(ch)
        elif tokens_approx > max_input_tokens:
            summaries.append(bart_summary(ch, max_input_tokens=max_input_tokens))
        else:
            summaries.append(bart_summary(ch))
        print(f"  Chapter {i+1}/{len(chapters)} summarised")
    return summaries


# ── ROUGE evaluation ──────────────────────────────────────────────────────────

def rouge_score(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Parameters
    ----------
    hypothesis : generated summary
    reference  : reference (human-written) summary

    Returns
    -------
    dict: {rouge1, rouge2, rougeL}  (F1 scores)
    """
    try:
        from rouge_score import rouge_scorer
        scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores  = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        return {"rouge1": None, "rouge2": None, "rougeL": None}


# ── Corpus-level summarisation ────────────────────────────────────────────────

def summarise_corpus(documents: list[dict],
                     method: str = "textrank",
                     n_sentences: int = 10) -> pd.DataFrame:
    """
    Summarise all documents in a corpus.

    Parameters
    ----------
    documents  : list of dicts with "title", "author", "full_text"
    method     : "textrank" | "lsa" | "bart"
    n_sentences: sentences to extract (TextRank / LSA only)

    Returns
    -------
    pd.DataFrame with columns: title, author, method, summary
    """
    rows = []
    for doc in documents:
        print(f"  Summarising ({method}): {doc['title']}")
        text = doc["full_text"]

        if method == "bart":
            summary = bart_summary(text)
        elif method == "lsa":
            summary = lsa_summary(text, n_sentences=n_sentences)
        else:
            summary = textrank_summary(text, n_sentences=n_sentences)

        rows.append({
            "title" : doc["title"],
            "author": doc["author"],
            "method": method,
            "summary": summary,
        })

    return pd.DataFrame(rows)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    text = (
        "It was a bright cold day in April, and the clocks were striking thirteen. "
        "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, "
        "slipped quickly through the glass doors of Victory Mansions, though not quickly enough "
        "to prevent a swirl of gritty dust from entering along with him. "
        "The hallway smelt of boiled cabbage and old rag mats. "
        "At one end of it a coloured poster, too large for the wall, had been tacked to the plaster, "
        "its edges curled in the damp. "
        "It depicted simply an enormous face, more than a metre wide: the face of a man of about "
        "forty-five, with a heavy black moustache and ruggedly handsome features."
    )
    print("TextRank:", textrank_summary(text, n_sentences=2))
    print("LSA     :", lsa_summary(text, n_sentences=2))
