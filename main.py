"""
main.py
Entry point for the NLP Text Analysis Pipeline.

Usage
-----
  python main.py --epub_dir data/ --output_dir outputs/

Flags
-----
  --epub_dir    str   Directory containing .epub files          [default: data/]
  --output_dir  str   Where to save results and plots           [default: outputs/]
  --no_bert          Skip DistilBERT (faster, CPU-friendly)
  --no_lda_sweep     Use fixed K=12 instead of coherence sweep
  --no_bart          Skip abstractive summarisation
  --spacy_model str  spaCy model to use                         [default: en_core_web_sm]
"""
import argparse
import os
import sys
import json
import time

# Add project root to path so src.* imports resolve
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.epub_extract   import load_corpus
from src.descriptive    import corpus_stats
from src.sentiment      import vader_chapters, full_sentiment, sentiment_summary
from src.ner            import ner_corpus, top_entities, extract_entities
from src.tfidf_keywords import tfidf_pipeline
from src.lda_topics     import lda_pipeline
from src.summarise      import summarise_corpus, rouge_score
from src.visualise      import (
    plot_sentiment_arc, plot_sentiment_bars,
    plot_ner_bars, plot_ner_heatmap,
    plot_tfidf_wordcloud, plot_tfidf_bars,
    plot_coherence_curve, plot_topic_heatmap,
    plot_lexical_diversity, plot_sentence_histogram,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NLP Text Analysis Pipeline")
    p.add_argument("--epub_dir",     default="data/",    help="Directory of .epub files")
    p.add_argument("--output_dir",   default="outputs/", help="Output directory")
    p.add_argument("--no_bert",      action="store_true", help="Skip BERT sentiment")
    p.add_argument("--no_lda_sweep", action="store_true", help="Use K=12 without sweep")
    p.add_argument("--no_bart",      action="store_true", help="Skip BART summarisation")
    p.add_argument("--spacy_model",  default="en_core_web_sm", help="spaCy model name")
    return p.parse_args()


# ── Utility ───────────────────────────────────────────────────────────────────

def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def makedirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def safe_filename(title: str, max_len: int = 30) -> str:
    """
    Convert a book title into a safe Windows filename by:
    - Truncating to max_len characters
    - Removing all characters Windows forbids in filenames
    - Replacing spaces with underscores
    """
    name = title[:max_len]
    for ch in ['/', '\\', ':', '?', '*', '"', '<', '>', '|', '\n', '\r', '\t']:
        name = name.replace(ch, '')
    name = name.replace(' ', '_')
    return name.strip('_')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    OUT   = args.output_dir
    PLOTS = os.path.join(OUT, "plots")
    SUMS  = os.path.join(OUT, "summaries")

    makedirs(OUT, PLOTS, SUMS)
    t0 = time.time()

    # ── Step 1: Load corpus ───────────────────────────────────────────────────
    banner("STEP 1 / 7  –  Loading EPUB corpus")
    documents = load_corpus(args.epub_dir)

    if not documents:
        print(f"\n⚠  No .epub files found in '{args.epub_dir}'")
        print("   Add .epub files to that directory and re-run.")
        sys.exit(0)

    titles = [d["title"] for d in documents]
    print(f"\n  Loaded {len(documents)} document(s):")
    for d in documents:
        print(f"    • {d['title']} by {d['author']}  ({len(d['full_text']):,} chars)")

    # ── Step 2: Descriptive statistics ───────────────────────────────────────
    banner("STEP 2 / 7  –  Descriptive statistics")
    stats_df = corpus_stats(documents)
    stats_df.to_csv(os.path.join(OUT, "corpus_stats.csv"), index=False)
    print(stats_df[["title", "num_tokens", "ttr", "mtld", "hdd",
                     "mean_sent_len"]].to_string(index=False))

    plot_lexical_diversity(stats_df, os.path.join(PLOTS, "lexical_diversity.png"))
    plot_sentence_histogram(stats_df, os.path.join(PLOTS, "sentence_length.png"))

    # ── Step 3: Sentiment analysis ────────────────────────────────────────────
    banner("STEP 3 / 7  –  Sentiment analysis")
    sent_summaries = []

    for doc in documents:
        print(f"\n  [{doc['title']}]")
        use_bert = not args.no_bert

        # Chapter-level VADER arc
        arc = vader_chapters(doc["chapters"])
        plot_sentiment_arc(
            arc, doc["title"],
            os.path.join(PLOTS, f"sentiment_arc_{safe_filename(doc['title'])}.png")
        )

        # Sentence-level full analysis (sample first 50k chars for speed)
        sent_df = full_sentiment(doc["full_text"][:50_000], use_bert=use_bert)
        sent_df["title"] = doc["title"]
        sent_df.to_csv(
            os.path.join(OUT, f"sentiment_{safe_filename(doc['title'])}.csv"),
            index=False
        )

        summary = sentiment_summary(sent_df)
        summary["title"] = doc["title"]
        sent_summaries.append(summary)
        print(f"    Mean VADER: {summary['mean_vader_compound']:+.4f}  "
              f"Positive: {summary['pct_positive_vader']}%  "
              f"Negative: {summary['pct_negative_vader']}%")

    sent_summary_df = pd.DataFrame(sent_summaries)
    sent_summary_df.to_csv(os.path.join(OUT, "sentiment_summary.csv"), index=False)
    plot_sentiment_bars(sent_summary_df, os.path.join(PLOTS, "sentiment_bars.png"))

    # ── Step 4: Named Entity Recognition ─────────────────────────────────────
    banner("STEP 4 / 7  –  Named Entity Recognition")
    ner_df = ner_corpus(documents, model=args.spacy_model)
    ner_df.to_csv(os.path.join(OUT, "ner_results.csv"), index=False)

    # Top characters per document
    top_chars = {}
    for doc in documents:
        ents = extract_entities(doc["full_text"], model=args.spacy_model)
        top  = top_entities(ents, label="PERSON", n=10)
        top_chars[doc["title"]] = top
        print(f"  {doc['title'][:30]:<30}  Top characters: {[n for n,_ in top[:5]]}")

    with open(os.path.join(OUT, "top_characters.json"), "w") as f:
        json.dump(top_chars, f, indent=2)

    plot_ner_bars(ner_df,    os.path.join(PLOTS, "ner_bars.png"))
    plot_ner_heatmap(ner_df, os.path.join(PLOTS, "ner_heatmap.png"))

    # ── Step 5: TF-IDF keyword extraction ────────────────────────────────────
    banner("STEP 5 / 7  –  TF-IDF keyword extraction")
    kw_df, kw_dict, _, _ = tfidf_pipeline(documents, n_keywords=20)
    kw_df.to_csv(os.path.join(OUT, "tfidf_keywords.csv"), index=False)

    for title in titles:
        top5 = [kw for kw, _ in kw_dict.get(title, [])[:5]]
        print(f"  {title[:30]:<30}  {top5}")
        plot_tfidf_wordcloud(
            kw_dict, title,
            os.path.join(PLOTS, f"wordcloud_{safe_filename(title)}.png")
        )
        plot_tfidf_bars(
            kw_df, title,
            os.path.join(PLOTS, f"tfidf_bars_{safe_filename(title)}.png")
        )

    # ── Step 6: LDA topic modelling ───────────────────────────────────────────
    banner("STEP 6 / 7  –  LDA topic modelling")
    lda_result = lda_pipeline(
        documents,
        auto_k=not args.no_lda_sweep,
        output_dir=OUT,
    )
    lda_result["doc_topic_matrix"].to_csv(
        os.path.join(OUT, "doc_topic_matrix.csv"), index=False)

    # Save topic descriptions
    with open(os.path.join(OUT, "lda_topics.txt"), "w") as f:
        for t in lda_result["topic_words"]:
            words = ", ".join(w for w, _ in t["words"])
            f.write(f"Topic {t['topic_id']:>2}: {words}\n")
            print(f"  Topic {t['topic_id']:>2}: {words}")

    if lda_result["coherence_scores"]:
        plot_coherence_curve(
            lda_result["coherence_scores"],
            os.path.join(PLOTS, "lda_coherence.png")
        )
    plot_topic_heatmap(
        lda_result["doc_topic_matrix"],
        os.path.join(PLOTS, "topic_heatmap.png")
    )

    # ── Step 7: Summarisation ─────────────────────────────────────────────────
    banner("STEP 7 / 7  –  Summarisation")
    sum_df = summarise_corpus(documents, method="textrank", n_sentences=10)
    sum_df.to_csv(os.path.join(OUT, "summaries.csv"), index=False)

    for _, row in sum_df.iterrows():
        out_file = os.path.join(SUMS, f"{safe_filename(row['title'])}_summary.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"TITLE : {row['title']}\n")
            f.write(f"AUTHOR: {row['author']}\n")
            f.write(f"METHOD: {row['method']}\n\n")
            f.write(row["summary"])

    if not args.no_bart:
        print("\n  Running BART abstractive summaries (first chapter of each text) …")
        from src.summarise import bart_summary
        for doc in documents:
            first_chapter = doc["chapters"][0] if doc["chapters"] else doc["full_text"][:2000]
            abstract = bart_summary(first_chapter)
            fname = os.path.join(SUMS, f"{safe_filename(doc['title'])}_bart.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(abstract)
            print(f"  {doc['title'][:40]:<40}  {abstract[:80]}…")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"\n  All outputs saved to: {os.path.abspath(OUT)}/")
    print(f"""
  Key files:
    {OUT}corpus_stats.csv
    {OUT}sentiment_summary.csv
    {OUT}ner_results.csv
    {OUT}tfidf_keywords.csv
    {OUT}doc_topic_matrix.csv
    {OUT}lda_topics.txt
    {OUT}lda_vis.html        <- interactive topic visualisation
    {OUT}plots/              <- {len(os.listdir(PLOTS))} charts
    {OUT}summaries/          <- per-text summaries
""")


if __name__ == "__main__":
    main()