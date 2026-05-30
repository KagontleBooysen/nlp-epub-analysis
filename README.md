# Text Analysis & NLP Project
### EPUB Corpus Analysis Pipeline

A complete, modular Python pipeline for literary text analysis using NLP.

---

## Quick Start

```bash
# 1. Clone / unzip this project folder
cd nlp_project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required models
python setup_models.py

# 5. Place your EPUB files in  data/
#    (or run the demo with built-in sample text)

# 6. Run the full pipeline
python main.py --epub_dir data/ --output_dir outputs/

# 7. (Optional) Open the interactive notebook
jupyter notebook notebooks/analysis_walkthrough.ipynb
```

---

## Project Structure

```
nlp_project/
├── main.py                  # Entry point – runs the full pipeline
├── setup_models.py          # Downloads NLTK data + spaCy models
├── requirements.txt
│
├── src/
│   ├── epub_extract.py      # EPUB → plain text extraction
│   ├── preprocess.py        # Tokenisation, lemmatisation, cleaning
│   ├── descriptive.py       # TTR, MTLD, HD-D, frequency stats
│   ├── sentiment.py         # VADER + DistilBERT sentence sentiment
│   ├── ner.py               # Named Entity Recognition (spaCy)
│   ├── tfidf_keywords.py    # TF-IDF keyword extraction
│   ├── lda_topics.py        # LDA topic modelling (Gensim)
│   ├── summarise.py         # TextRank + BART summarisation
│   └── visualise.py         # All charts & word clouds
│
├── data/                    # Place your .epub files here
├── outputs/                 # Generated reports, plots, CSVs
└── notebooks/
    └── analysis_walkthrough.ipynb
```

---

## Outputs

After running `main.py` you will find in `outputs/`:

| File | Description |
|------|-------------|
| `corpus_stats.csv` | Descriptive statistics per text |
| `sentiment_results.csv` | Sentence-level VADER + BERT scores |
| `ner_results.csv` | Entity counts per text |
| `tfidf_keywords.csv` | Top-20 keywords per text |
| `lda_topics.txt` | Topic–word distributions |
| `plots/` | All visualisations (PNG) |
| `summaries/` | Per-text extractive summaries |
| `report.html` | Auto-generated HTML report |

---

## Module Overview

### epub_extract.py
Reads EPUB spine items, strips HTML tags, normalises encoding.

### preprocess.py
Configurable pre-processing: lowercasing, lemmatisation, stopword removal.
Task-aware: retains punctuation for sentiment, strips for TF-IDF/LDA.

### descriptive.py
Computes token/type counts, TTR, MTLD (threshold=0.72), HD-D (draws=42),
sentence length distribution, and Zipf fit.

### sentiment.py
- **VADER**: compound score per sentence, chapter means.
- **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`):
  binary positive/negative with confidence score.
- GPU auto-detected; falls back to CPU.

### ner.py
spaCy `en_core_web_sm` (fast) or `en_core_web_trf` (accurate, requires GPU).
Collapses 18 OntoNotes labels into 5 macro-categories.

### tfidf_keywords.py
Scikit-learn `TfidfVectorizer` with unigrams + bigrams, min_df=2, max_df=0.9.

### lda_topics.py
Gensim LDA with Gibbs sampling. Auto-selects K via C_v coherence score
over K ∈ {5,10,15,20,25}. Saves pyLDAvis HTML.

### summarise.py
TextRank (sumy) for extractive summaries. Optional BART for chapters ≤1024 tokens.

### visualise.py
Generates: sentiment arc line plots, NER bar charts, TF-IDF word clouds,
LDA coherence curve, topic heatmap.
