"""
src/preprocess.py
Task-aware text pre-processing using spaCy.

Tasks
-----
  tfidf   : lowercase lemmas, no stops, no punct, alpha only
  lda     : same as tfidf (bag-of-words)
  ner     : returns spaCy Doc (no string transformation)
  general : whitespace-tokenised lowercase tokens
  raw     : sentence-split only, text preserved
"""
import re
import spacy
from nltk.tokenize import sent_tokenize

# Load the lightweight model by default.
# Swap to en_core_web_trf for higher NER accuracy.
_nlp = None

def _get_nlp(model: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None or _nlp.meta["name"] != model.replace("en_core_web_", ""):
        _nlp = spacy.load(model)
    return _nlp


# Maximum characters spaCy processes in one pass
_SPACY_MAX = 999_999


def preprocess(text: str, task: str = "tfidf", model: str = "en_core_web_sm"):
    """
    Pre-process text for a given downstream task.

    Returns
    -------
    list[str]   for tfidf / lda / general
    spacy.Doc   for ner
    list[str]   of sentences for raw
    """
    nlp = _get_nlp(model)
    text = text[:_SPACY_MAX]

    if task == "raw":
        return sent_tokenize(text)

    doc = nlp(text)

    if task == "ner":
        return doc

    if task in ("tfidf", "lda"):
        return [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop
            and not tok.is_punct
            and tok.is_alpha
            and len(tok.text) > 1
        ]

    # general / fallback
    return [tok.text.lower() for tok in doc if tok.is_alpha]


def split_into_windows(text: str, window_size: int = 100) -> list[list[str]]:
    """
    Split a text into overlapping sentence windows for sliding-window sentiment.
    Returns list of sentence lists.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= window_size:
        return [sentences]
    step = window_size // 2
    windows = []
    for start in range(0, len(sentences) - window_size + 1, step):
        windows.append(sentences[start: start + window_size])
    return windows


def chapter_sentences(chapters: list[str]) -> list[list[str]]:
    """Sentence-tokenise each chapter, return list of sentence lists."""
    return [sent_tokenize(ch) for ch in chapters]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "It was the best of times, it was the worst of times. "
        "Mr. Darcy walked into the room. Elizabeth Bennet watched him carefully."
    )
    print("TF-IDF tokens:", preprocess(sample, "tfidf"))
    print("Sentences    :", preprocess(sample, "raw"))
    doc = preprocess(sample, "ner")
    print("Entities     :", [(e.text, e.label_) for e in doc.ents])
