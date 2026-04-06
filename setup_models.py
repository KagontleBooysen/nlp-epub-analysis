"""
setup_models.py
Run once after pip install to download all required models and data.
"""
import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("=" * 60)
print(" NLP Project – Model Setup")
print("=" * 60)

# ── NLTK resources ────────────────────────────────────────────
import nltk
resources = [
    "punkt", "punkt_tab", "stopwords",
    "vader_lexicon", "averaged_perceptron_tagger", "wordnet"
]
for r in resources:
    print(f"  Downloading NLTK: {r}")
    nltk.download(r, quiet=True)

# ── spaCy models ──────────────────────────────────────────────
run(f"{sys.executable} -m spacy download en_core_web_sm")

print("""
Optional (accurate NER, requires more RAM):
  python -m spacy download en_core_web_trf

Optional (GPU sentiment, requires CUDA):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
""")

print("\n✓ Setup complete. Run:  python main.py --epub_dir data/ --output_dir outputs/")
