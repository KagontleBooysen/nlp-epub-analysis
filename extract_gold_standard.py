"""
extract_gold_standard.py
Extracts 1,000 sentences for gold standard annotation.
Option A: 10 genres × 5 texts × 20 sentences = 1,000 sentences

Run from your nlp_project/nlp_project/ folder:
    python extract_gold_standard.py
"""

import pandas as pd
import numpy as np
import os
import random

random.seed(42)
np.random.seed(42)

# ── Genre assignments — 5 texts per genre ─────────────────────────────────────
GENRES = {
    "Gothic/Horror": [
        "sentiment_Frankenstein;_or,_the_modern_p.csv",
        "sentiment_Dracula.csv",
        "sentiment_Carmilla.csv",
        "sentiment_The_Picture_of_Dorian_Gray.csv",
        "sentiment_The_Yellow_Wallpaper.csv",
    ],
    "Adventure": [
        "sentiment_The_call_of_the_wild.csv",
        "sentiment_White_Fang.csv",
        "sentiment_Tarzan_of_the_Apes.csv",
        "sentiment_The_Lost_World.csv",
        "sentiment_Kim.csv",
    ],
    "Romance": [
        "sentiment_Anne_of_Green_Gables.csv",
        "sentiment_The_Age_of_Innocence.csv",
        "sentiment_The_House_of_Mirth.csv",
        "sentiment_Howards_End.csv",
        "sentiment_My_Ántonia.csv",
    ],
    "Science Fiction": [
        "sentiment_The_Time_Machine.csv",
        "sentiment_The_war_of_the_worlds.csv",
        "sentiment_The_First_Men_in_the_Moon.csv",
        "sentiment_Herland.csv",
        "sentiment_The_Invisible_Man_A_Grotesque.csv",
    ],
    "Children's Fiction": [
        "sentiment_Peter_Pan.csv",
        "sentiment_The_Secret_Garden.csv",
        "sentiment_Pollyanna.csv",
        "sentiment_Daddy-Long-Legs.csv",
        "sentiment_The_Wind_in_the_Willows.csv",
    ],
    "Mystery/Crime": [
        "sentiment_The_Adventures_of_Sherlock_Hol.csv",
        "sentiment_The_Moonstone.csv",
        "sentiment_The_Hound_of_the_Baskervilles.csv",
        "sentiment_The_Red_House_Mystery.csv",
        "sentiment_The_Man_Who_Was_Thursday_A_Ni.csv",
    ],
    "Philosophy/Essays": [
        "sentiment_Pragmatism_A_New_Name_for_Som.csv",
        "sentiment_Beyond_Good_and_Evil.csv",
        "sentiment_Thus_Spake_Zarathustra_A_Book.csv",
        "sentiment_The_Will_to_Believe,_and_Other.csv",
        "sentiment_Siddhartha.csv",
    ],
    "Historical/Political": [
        "sentiment_The_Jungle.csv",
        "sentiment_Up_from_Slavery_An_Autobiogra.csv",
        "sentiment_The_Ragged_Trousered_Philanthr.csv",
        "sentiment_The_Souls_of_Black_Folk.csv",
        "sentiment_The_Eighteenth_Brumaire_of_Lou.csv",
    ],
    "Poetry/Drama": [
        "sentiment_Dubliners.csv",
        "sentiment_The_Waste_Land.csv",
        "sentiment_Pygmalion.csv",
        "sentiment_Mrs._Warren's_Profession.csv",
        "sentiment_A_Shropshire_Lad.csv",
    ],
    "Science/Non-fiction": [
        "sentiment_On_the_Origin_of_Species_By_Me.csv",
        "sentiment_The_Descent_of_Man,_and_Select.csv",
        "sentiment_The_Natural_History_of_Selborn.csv",
        "sentiment_Flatland_A_Romance_of_Many_Di.csv",
        "sentiment_Insectivorous_Plants.csv",
    ],
}

OUTPUTS_DIR = "outputs"
SENTENCES_PER_TEXT = 20
MIN_WORDS = 8    # minimum words per sentence
MAX_WORDS = 60   # maximum words per sentence

# ── Extract sentences ──────────────────────────────────────────────────────────
all_rows = []
sentence_id = 1
errors = []

for genre, files in GENRES.items():
    print(f"\n{'='*50}")
    print(f"Genre: {genre}")
    print(f"{'='*50}")

    for fname in files:
        fpath = os.path.join(OUTPUTS_DIR, fname)

        if not os.path.exists(fpath):
            print(f"  ⚠ File not found: {fname}")
            errors.append(f"Missing: {fname}")
            continue

        try:
            df = pd.read_csv(fpath, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"  ⚠ Could not read {fname}: {e}")
            errors.append(f"Read error: {fname}")
            continue

        # Find sentence column
        sent_col = None
        for col in ['sentence', 'text', 'Sentence', 'Text', 'sent']:
            if col in df.columns:
                sent_col = col
                break

        if sent_col is None:
            print(f"  ⚠ No sentence column found in {fname}")
            print(f"    Columns: {df.columns.tolist()}")
            errors.append(f"No sentence col: {fname}")
            continue

        # Find VADER column
        vader_col = None
        for col in ['vader_compound', 'compound', 'vader', 'score']:
            if col in df.columns:
                vader_col = col
                break

        # Clean sentences
        df = df.dropna(subset=[sent_col])
        df['_text'] = df[sent_col].astype(str).str.strip()
        df['_words'] = df['_text'].str.split().str.len()
        df = df[(df['_words'] >= MIN_WORDS) & (df['_words'] <= MAX_WORDS)]
        df = df[df['_text'].str.len() > 20]

        # Remove Gutenberg boilerplate
        boilerplate = ['Project Gutenberg', 'gutenberg', 'www.', 'http', 
                      'chapter', 'Chapter', 'CHAPTER', '***']
        for bp in boilerplate:
            df = df[~df['_text'].str.contains(bp, na=False, regex=False)]

        if len(df) < SENTENCES_PER_TEXT:
            print(f"  ⚠ Only {len(df)} valid sentences in {fname} (need {SENTENCES_PER_TEXT})")
            sample = df
        else:
            # Stratified sample — spread across the text
            df = df.reset_index(drop=True)
            indices = np.linspace(0, len(df)-1, SENTENCES_PER_TEXT, dtype=int)
            sample = df.iloc[indices]

        # Get title from filename
        title = fname.replace('sentiment_', '').replace('.csv', '').replace('_', ' ').strip()

        vader_score = None
        for _, row in sample.iterrows():
            if vader_col:
                try:
                    vader_score = round(float(row[vader_col]), 4)
                except:
                    vader_score = None

            all_rows.append({
                'Sentence_ID': sentence_id,
                'Genre': genre,
                'Title': title,
                'Sentence': row['_text'],
                'VADER_Compound': vader_score,
                'Rater_1': '',
                'Rater_2': '',
                'Rater_3': '',
            })
            sentence_id += 1

        print(f"  ✓ {title[:45]:<45} {len(sample)} sentences extracted")

# ── Save to CSV ────────────────────────────────────────────────────────────────
result_df = pd.DataFrame(all_rows)
result_df.to_csv('gold_standard_sentences.csv', index=False, encoding='utf-8')
print(f"\n{'='*50}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*50}")
print(f"Total sentences extracted: {len(result_df)}")
print(f"Target: 1,000")
print(f"Breakdown by genre:")
print(result_df.groupby('Genre').size().to_string())
if errors:
    print(f"\nErrors encountered:")
    for e in errors:
        print(f"  - {e}")
print(f"\nSaved to: gold_standard_sentences.csv")
print(f"Next step: open gold_standard_rating_sheet.py to build the Excel rating sheet")