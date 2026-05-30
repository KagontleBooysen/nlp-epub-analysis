"""
src/epub_extract.py
Extracts clean plain text from EPUB files.
Returns a dict with metadata + list of chapter strings.
"""
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


# Characters that should be normalised
_CURLY_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "--",
    "\u00a0": " ",
})


def _clean(text: str) -> str:
    """Normalise whitespace, curly quotes, and non-ASCII dashes."""
    text = text.translate(_CURLY_MAP)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_epub(epub_path: str, min_chapter_len: int = 150) -> dict:
    """
    Parse an EPUB file and return structured text data.

    Parameters
    ----------
    epub_path      : path to the .epub file
    min_chapter_len: minimum character count to keep a spine item
                     (filters out cover, TOC, copyright pages)

    Returns
    -------
    {
        "title"    : str,
        "author"   : str,
        "chapters" : list[str],   # one entry per spine item
        "full_text": str,         # all chapters joined
    }
    """
    book = epub.read_epub(epub_path)

    title  = book.get_metadata("DC", "title")
    author = book.get_metadata("DC", "creator")
    title  = title[0][0]  if title  else "Unknown"
    author = author[0][0] if author else "Unknown"

    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content()
        soup = BeautifulSoup(html, "lxml")

        # Remove script / style noise
        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = _clean(text)

        if len(text) >= min_chapter_len:
            chapters.append(text)

    full_text = " ".join(chapters)

    return {
        "title"    : title,
        "author"   : author,
        "chapters" : chapters,
        "full_text": full_text,
    }


def load_corpus(epub_dir: str) -> list[dict]:
    """
    Load all .epub files in a directory.
    Returns a list of extract_epub() dicts, sorted by title.
    """
    import os
    results = []
    for fname in sorted(os.listdir(epub_dir)):
        if fname.lower().endswith(".epub"):
            path = os.path.join(epub_dir, fname)
            print(f"  Loading: {fname}")
            try:
                data = extract_epub(path)
                data["filename"] = fname
                results.append(data)
            except Exception as exc:
                print(f"  ⚠ Failed to load {fname}: {exc}")
    return results


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.epub"
    result = extract_epub(path)
    print(f"Title  : {result['title']}")
    print(f"Author : {result['author']}")
    print(f"Chapters: {len(result['chapters'])}")
    print(f"Total chars: {len(result['full_text']):,}")
    print("\nFirst 300 chars:\n", result["full_text"][:300])
