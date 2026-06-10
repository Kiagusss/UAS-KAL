import argparse
import html
import os
import re
from functools import lru_cache

import pandas as pd

DEFAULT_POSITIVE_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
DEFAULT_NEGATIVE_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_lexicon_term(term):
    term = str(term).strip().lower()
    term = re.sub(r"\s+", " ", term)
    return term


def _parse_weight(value):
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def load_inset_lexicon(path_or_url):
    df = pd.read_csv(path_or_url, sep="\t", engine="python")

    if len(df.columns) < 2:
        raise ValueError(f"Lexicon InSet tidak valid: {path_or_url}")

    word_col = df.columns[0]
    weight_col = df.columns[1]

    lexicon = {}
    for _, row in df.iterrows():
        term = _clean_lexicon_term(row[word_col])
        if not term:
            continue
        lexicon[term] = lexicon.get(term, 0.0) + _parse_weight(row[weight_col])

    return lexicon


@lru_cache(maxsize=1)
def load_default_lexicons(positive_source, negative_source):
    positive = load_inset_lexicon(positive_source)
    negative = load_inset_lexicon(negative_source)
    return positive, negative


def score_text(text, positive_lexicon, negative_lexicon, max_ngram=4):
    tokens = normalize_text(text).split()
    if not tokens:
        return 0.0, 0, 0

    score = 0.0
    matched_positive = 0
    matched_negative = 0
    used_spans = set()

    for n in range(max_ngram, 0, -1):
        for start in range(0, len(tokens) - n + 1):
            span = tuple(range(start, start + n))
            if any(index in used_spans for index in span):
                continue

            term = " ".join(tokens[start:start + n])
            if term in positive_lexicon:
                weight = positive_lexicon[term]
                score += weight
                matched_positive += 1
                used_spans.update(span)
            elif term in negative_lexicon:
                weight = negative_lexicon[term]
                score += weight
                matched_negative += 1
                used_spans.update(span)

    return score, matched_positive, matched_negative


def assign_label(score):
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def detect_text_column(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred

    for candidate in ("cleaned_indobert", "cleaned_lstm", "cleaned_svm", "cleaned_text", "text"):
        if candidate in df.columns:
            return candidate

    raise ValueError("Kolom teks tidak ditemukan. Pastikan ada salah satu kolom teks pada CSV.")


def main():
    parser = argparse.ArgumentParser(description="Label otomatis komentar menggunakan InSet lexicon")
    parser.add_argument("--input", default="comments_cleaned.csv", help="CSV input berisi kolom teks")
    parser.add_argument("--output", default="comments_labeled_inset.csv", help="CSV output dengan kolom label")
    parser.add_argument("--text_column", default=None, help="Kolom teks yang akan diberi label")
    parser.add_argument("--positive_lexicon", default=DEFAULT_POSITIVE_URL, help="Path atau URL lexicon positif InSet")
    parser.add_argument("--negative_lexicon", default=DEFAULT_NEGATIVE_URL, help="Path atau URL lexicon negatif InSet")
    parser.add_argument("--max_ngram", type=int, default=4, help="Panjang frasa maksimum untuk pencocokan")
    parser.add_argument("--neutral_threshold", type=float, default=0.0, help="Ambang skor untuk label netral")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    df = pd.read_csv(args.input)
    text_column = detect_text_column(df, preferred=args.text_column)

    print(f"[*] Memuat lexicon InSet dari:\n    + {args.positive_lexicon}\n    + {args.negative_lexicon}")
    positive_lexicon, negative_lexicon = load_default_lexicons(args.positive_lexicon, args.negative_lexicon)
    print(f"[*] Lexicon positif: {len(positive_lexicon)} entri")
    print(f"[*] Lexicon negatif : {len(negative_lexicon)} entri")
    print(f"[*] Menggunakan kolom teks: {text_column}")

    scores = []
    labels = []
    positive_hits = []
    negative_hits = []

    for text in df[text_column].astype(str):
        score, pos_hits, neg_hits = score_text(
            text,
            positive_lexicon=positive_lexicon,
            negative_lexicon=negative_lexicon,
            max_ngram=args.max_ngram,
        )
        scores.append(score)
        positive_hits.append(pos_hits)
        negative_hits.append(neg_hits)

        if score > args.neutral_threshold:
            labels.append("positive")
        elif score < -args.neutral_threshold:
            labels.append("negative")
        else:
            labels.append("neutral")

    result = df.copy()
    result["inset_score"] = scores
    result["inset_positive_hits"] = positive_hits
    result["inset_negative_hits"] = negative_hits
    result["label"] = labels

    result.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[+] Labeling selesai. File disimpan ke: {os.path.abspath(args.output)}")
    print("[*] Distribusi label:")
    print(result["label"].value_counts().to_string())


if __name__ == "__main__":
    main()