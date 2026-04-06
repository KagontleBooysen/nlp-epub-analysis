"""
src/ner.py
Named Entity Recognition using spaCy.

Collapses 18 OntoNotes categories into 5 macro-categories:
  PERSON    – people, fictional characters
  GPE/LOC   – countries, cities, geographical features
  ORG       – companies, institutions, parties
  DATE/TIME – dates, times, durations
  OTHER     – everything else (MONEY, WORK_OF_ART, etc.)
"""
import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm

MACRO_MAP = {
    "PERSON"  : "PERSON",
    "ORG"     : "ORG",
    "GPE"     : "GPE/LOC",
    "LOC"     : "GPE/LOC",
    "FACILITY": "GPE/LOC",
    "DATE"    : "DATE/TIME",
    "TIME"    : "DATE/TIME",
}

_nlp = None

def _get_nlp(model: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_entities(text: str,
                     model: str = "en_core_web_sm",
                     chunk_size: int = 500_000) -> list[dict]:
    """
    Extract all named entities from text.
    Processes in chunks to handle long texts.

    Returns
    -------
    list of dicts: {text, label, macro_label, start_char, end_char}
    """
    nlp      = _get_nlp(model)
    entities = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start: start + chunk_size]
        doc   = nlp(chunk)
        for ent in doc.ents:
            macro = MACRO_MAP.get(ent.label_, "OTHER")
            entities.append({
                "text"       : ent.text,
                "label"      : ent.label_,
                "macro_label": macro,
                "start_char" : start + ent.start_char,
                "end_char"   : start + ent.end_char,
            })

    return entities


def entity_counts(entities: list[dict]) -> dict:
    """
    Count entities by macro_label.
    Returns dict: {macro_label: count}
    """
    return dict(Counter(e["macro_label"] for e in entities))


def top_entities(entities: list[dict],
                 label: str = "PERSON",
                 n: int = 20) -> list[tuple]:
    """
    Most frequent entities for a given macro_label.
    Returns list of (entity_text, count).
    """
    names = [e["text"].strip() for e in entities if e["macro_label"] == label]
    return Counter(names).most_common(n)


# ── Corpus-level ──────────────────────────────────────────────────────────────

def ner_corpus(documents: list[dict],
               model: str = "en_core_web_sm") -> pd.DataFrame:
    """
    Run NER on a corpus and return a summary DataFrame.

    Columns: title, author, PERSON, GPE/LOC, ORG, DATE/TIME, OTHER, total
    """
    rows = []
    for doc in documents:
        print(f"  NER: {doc['title']}")
        ents   = extract_entities(doc["full_text"], model=model)
        counts = entity_counts(ents)

        total  = sum(counts.values())
        rows.append({
            "title"    : doc["title"],
            "author"   : doc["author"],
            "PERSON"   : counts.get("PERSON",   0),
            "GPE/LOC"  : counts.get("GPE/LOC",  0),
            "ORG"      : counts.get("ORG",       0),
            "DATE/TIME": counts.get("DATE/TIME", 0),
            "OTHER"    : counts.get("OTHER",     0),
            "total"    : total,
            # percentage of total for each macro category
            "pct_PERSON"   : round(counts.get("PERSON",   0) / max(total,1) * 100, 1),
            "pct_GPE_LOC"  : round(counts.get("GPE/LOC",  0) / max(total,1) * 100, 1),
            "pct_ORG"      : round(counts.get("ORG",       0) / max(total,1) * 100, 1),
            "pct_DATE_TIME": round(counts.get("DATE/TIME", 0) / max(total,1) * 100, 1),
            "pct_OTHER"    : round(counts.get("OTHER",     0) / max(total,1) * 100, 1),
        })

    return pd.DataFrame(rows)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Elizabeth Bennet met Mr. Darcy at Netherfield Park in Hertfordshire. "
        "Lady Catherine de Bourgh arrived from Rosings on Tuesday."
    )
    ents = extract_entities(sample)
    for e in ents:
        print(f"  {e['text']:<30} {e['label']:<12} → {e['macro_label']}")

    print("\nCounts:", entity_counts(ents))
    print("Top PERSON:", top_entities(ents, "PERSON"))
