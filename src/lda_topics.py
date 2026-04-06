"""
src/lda_topics.py
Latent Dirichlet Allocation topic modelling using Gensim.

Features
--------
  - Auto-selects K via C_v coherence score
  - Saves pyLDAvis interactive visualisation
  - Returns per-document topic distribution
"""
import os
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm

from src.preprocess import preprocess


# ── Dictionary + Corpus ───────────────────────────────────────────────────────

def build_gensim_corpus(documents: list[dict]) -> tuple:
    """
    Tokenise documents, build Gensim dictionary and BoW corpus.

    Returns
    -------
    tokenised_docs  : list[list[str]]
    dictionary      : gensim.corpora.Dictionary
    bow_corpus      : list of BoW vectors
    """
    print("  Tokenising for LDA …")
    tokenised_docs = []
    for doc in documents:
        tokens = preprocess(doc["full_text"], task="lda")
        tokenised_docs.append(tokens)

    dictionary = corpora.Dictionary(tokenised_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # prune rare/common terms

    bow_corpus = [dictionary.doc2bow(t) for t in tokenised_docs]

    print(f"  Vocabulary size after filtering: {len(dictionary):,} terms")
    return tokenised_docs, dictionary, bow_corpus


# ── Coherence sweep ───────────────────────────────────────────────────────────

def find_best_k(tokenised_docs, dictionary, bow_corpus,
                k_range=None, passes: int = 10, random_state: int = 42) -> tuple:
    """
    Fit LDA for each K in k_range and return (best_k, coherence_scores_dict).
    """
    if k_range is None:
        k_range = [5, 10, 12, 15, 20, 25]

    coherence_scores = {}
    best_k   = k_range[0]
    best_cv  = -1.0

    for k in tqdm(k_range, desc="  Coherence sweep"):
        lda = models.LdaModel(
            bow_corpus,
            num_topics=k,
            id2word=dictionary,
            passes=passes,
            random_state=random_state,
            alpha="auto",
            eta="auto",
        )
        cm = CoherenceModel(
            model=lda,
            texts=tokenised_docs,
            dictionary=dictionary,
            coherence="c_v",
        )
        cv = cm.get_coherence()
        coherence_scores[k] = round(cv, 4)
        print(f"    K={k:>3}  C_v={cv:.4f}")

        if cv > best_cv:
            best_cv = cv
            best_k  = k

    print(f"  ✓ Best K = {best_k}  (C_v = {best_cv:.4f})")
    return best_k, coherence_scores


# ── Fit final model ───────────────────────────────────────────────────────────

def fit_lda(tokenised_docs, dictionary, bow_corpus,
            num_topics: int = 12,
            passes: int = 15,
            random_state: int = 42) -> models.LdaModel:
    """
    Fit final LDA model with the chosen number of topics.
    """
    print(f"  Fitting LDA with K={num_topics}, passes={passes} …")
    lda = models.LdaModel(
        bow_corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=random_state,
        alpha="auto",
        eta="auto",
    )
    return lda


# ── Topic descriptions ────────────────────────────────────────────────────────

def topic_words(lda: models.LdaModel, n_words: int = 10) -> list[dict]:
    """
    Return top words for each topic.
    Returns list of dicts: {topic_id, words: [(word, prob), …]}
    """
    results = []
    for i in range(lda.num_topics):
        words = lda.show_topic(i, topn=n_words)
        results.append({"topic_id": i, "words": words})
    return results


def print_topics(lda: models.LdaModel, n_words: int = 10):
    for t in topic_words(lda, n_words):
        words_str = ", ".join(w for w, _ in t["words"])
        print(f"  Topic {t['topic_id']:>2}: {words_str}")


# ── Document-topic distributions ─────────────────────────────────────────────

def doc_topic_matrix(lda: models.LdaModel,
                     bow_corpus: list,
                     titles: list[str]) -> pd.DataFrame:
    """
    Compute topic probability distribution for each document.
    Returns a DataFrame (rows = docs, columns = topic_0 … topic_K).
    """
    rows = []
    for bow in bow_corpus:
        dist = dict(lda.get_document_topics(bow, minimum_probability=0))
        row  = {f"topic_{i}": dist.get(i, 0.0) for i in range(lda.num_topics)}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "title", titles)
    return df


# ── pyLDAvis export ───────────────────────────────────────────────────────────

def save_pyldavis(lda, bow_corpus, dictionary, output_path: str):
    """Save an interactive pyLDAvis HTML visualisation."""
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
        vis = gensimvis.prepare(lda, bow_corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, output_path)
        print(f"  pyLDAvis saved → {output_path}")
    except ImportError:
        print("  pyLDAvis not installed; skipping visualisation.")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def lda_pipeline(documents: list[dict],
                 auto_k: bool = True,
                 k_range=None,
                 num_topics: int = 12,
                 output_dir: str = "outputs") -> dict:
    """
    End-to-end LDA pipeline.

    Returns
    -------
    {
        "lda"              : LdaModel,
        "dictionary"       : Dictionary,
        "bow_corpus"       : list,
        "tokenised_docs"   : list,
        "topic_words"      : list[dict],
        "doc_topic_matrix" : pd.DataFrame,
        "coherence_scores" : dict   (if auto_k=True)
    }
    """
    tokenised, dictionary, bow = build_gensim_corpus(documents)
    titles = [d["title"] for d in documents]

    if auto_k:
        best_k, coh_scores = find_best_k(tokenised, dictionary, bow, k_range=k_range)
        num_topics = best_k
    else:
        coh_scores = {}

    lda = fit_lda(tokenised, dictionary, bow, num_topics=num_topics)

    # Save visualisation
    os.makedirs(output_dir, exist_ok=True)
    save_pyldavis(lda, bow, dictionary, os.path.join(output_dir, "lda_vis.html"))

    return {
        "lda"             : lda,
        "dictionary"      : dictionary,
        "bow_corpus"      : bow,
        "tokenised_docs"  : tokenised,
        "topic_words"     : topic_words(lda),
        "doc_topic_matrix": doc_topic_matrix(lda, bow, titles),
        "coherence_scores": coh_scores,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [
        {"title": "A", "full_text": " ".join(["whale ocean ship sea dark"] * 60)},
        {"title": "B", "full_text": " ".join(["love dance ball lady gentleman"] * 60)},
        {"title": "C", "full_text": " ".join(["monster creature ice arctic flee"] * 60)},
    ]
    result = lda_pipeline(docs, auto_k=False, num_topics=3, output_dir="outputs")
    print_topics(result["lda"])
