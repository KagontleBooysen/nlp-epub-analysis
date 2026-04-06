"""
src/tfidf_keywords.py
TF-IDF keyword extraction using Scikit-learn.

Supports:
  - Unigrams + bigrams
  - Top-N keywords per document
  - Corpus-level IDF for cross-document comparison
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def build_tfidf_matrix(documents: list[dict],
                       max_features: int = 15_000,
                       ngram_range: tuple = (1, 2),
                       min_df: int = 2,
                       max_df: float = 0.9):
    """
    Build a TF-IDF matrix over all documents in the corpus.

    Parameters
    ----------
    documents   : list of dicts with "title" and "full_text"
    max_features: vocabulary size cap
    ngram_range : (min_n, max_n) for n-gram extraction
    min_df      : ignore terms appearing in fewer than min_df docs
    max_df      : ignore terms appearing in more than max_df fraction of docs

    Returns
    -------
    vectorizer  : fitted TfidfVectorizer
    X           : sparse matrix (n_docs × n_features)
    titles      : list of document titles in row order
    """
    print("  Tokenising corpus for TF-IDF …")
    corpus = []
    titles = []
    for doc in documents:
        tokens = preprocess(doc["full_text"], task="tfidf")
        corpus.append(_join_tokens(tokens))
        titles.append(doc["title"])

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,   # apply log(1 + tf) instead of raw tf
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X, titles


def top_keywords(vectorizer: TfidfVectorizer,
                 X,
                 titles: list[str],
                 n: int = 20) -> pd.DataFrame:
    """
    Extract top-N keywords per document.

    Returns
    -------
    pd.DataFrame with columns: title, rank, keyword, tfidf_score
    """
    terms = vectorizer.get_feature_names_out()
    rows  = []

    for idx, title in enumerate(titles):
        row_arr = np.asarray(X[idx].todense()).ravel()
        top_idx = row_arr.argsort()[::-1][:n]
        for rank, i in enumerate(top_idx, start=1):
            rows.append({
                "title"      : title,
                "rank"       : rank,
                "keyword"    : terms[i],
                "tfidf_score": round(float(row_arr[i]), 6),
            })

    return pd.DataFrame(rows)


def keywords_dict(vectorizer, X, titles, n=20) -> dict:
    """
    Return {title: [(keyword, score), ...]} for quick access.
    """
    df = top_keywords(vectorizer, X, titles, n)
    result = {}
    for title, group in df.groupby("title"):
        result[title] = list(zip(group["keyword"], group["tfidf_score"]))
    return result


def tfidf_pipeline(documents: list[dict], n_keywords: int = 20) -> tuple:
    """
    Convenience wrapper: runs full TF-IDF pipeline and returns
    (keywords_df, keywords_dict, vectorizer, matrix).
    """
    vec, X, titles = build_tfidf_matrix(documents)
    kw_df   = top_keywords(vec, X, titles, n=n_keywords)
    kw_dict = keywords_dict(vec, X, titles, n=n_keywords)
    return kw_df, kw_dict, vec, X


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [
        {"title": "Doc A", "full_text": "The whale swam in the dark deep ocean near the ship"},
        {"title": "Doc B", "full_text": "Elizabeth danced with Mr Darcy at the ball in Netherfield"},
        {"title": "Doc C", "full_text": "The monster fled into the arctic ice away from civilisation"},
    ]
    kw_df, kw_dict, _, _ = tfidf_pipeline(docs, n_keywords=5)
    print(kw_df.to_string(index=False))
