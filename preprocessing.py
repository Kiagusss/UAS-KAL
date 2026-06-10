import argparse
import html
import os
import re
import emoji
import pandas as pd
from tqdm import tqdm

# Kamus slang / singkatan default untuk bahasa Indonesia.
DEFAULT_SLANG_DICT = {
    "yg": "yang",
    "dgn": "dengan",
    "bgt": "banget",
    "gk": "tidak",
    "gajelas": "tidak jelas",
    "ga": "tidak",
    "gak": "tidak",
    "kalo": "kalau",
    "kl": "kalau",
    "utk": "untuk",
    "tp": "tapi",
    "dpt": "dapat",
    "sy": "saya",
    "gw": "saya",
    "gua": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "elo": "kamu",
    "aja": "saja",
    "aj": "saja",
    "bs": "bisa",
    "sm": "sama",
    "krn": "karena",
    "udh": "sudah",
    "udah": "sudah",
    "jd": "jadi",
    "dr": "dari",
    "kmrn": "kemarin",
    "pd": "pada",
    "tgl": "tanggal",
    "tpi": "tapi",
    "skrg": "sekarang",
    "skg": "sekarang",
    "blm": "belum",
    "bisaa": "bisa",
    "ny": "nya",
    "lg": "lagi",
    "sdh": "sudah",
    "msh": "masih",
    "bnyak": "banyak",
    "bnyk": "banyak",
    "karna": "karena",
    "trs": "terus",
    "makasih": "terima kasih",
    "thx": "terima kasih",
    "tq": "terima kasih",
    "makasi": "terima kasih",
    "kpd": "kepada",
    "sbg": "sebagai",
    "dlm": "dalam",
    "bener": "benar",
    "bnr": "benar",
}

PROFILE_DEFAULTS = {
    "svm": {"lowercase": True, "remove_punctuation": True},
    "lstm": {"lowercase": True, "remove_punctuation": True},
    "indobert": {"lowercase": False, "remove_punctuation": False},
}


def load_slang_dict(filepath):
    """Memuat kamus slang tambahan dari CSV dengan minimal dua kolom."""
    if not filepath or not os.path.exists(filepath):
        print(f"[*] Menggunakan kamus slang bawaan ({len(DEFAULT_SLANG_DICT)} kata).")
        return DEFAULT_SLANG_DICT

    try:
        df = pd.read_csv(filepath)
        if len(df.columns) < 2:
            print("[!] CSV slang harus memiliki minimal 2 kolom (slang, formal). Menggunakan default.")
            return DEFAULT_SLANG_DICT

        custom_dict = dict(zip(df.iloc[:, 0].astype(str).str.lower(), df.iloc[:, 1].astype(str)))
        merged_dict = {**DEFAULT_SLANG_DICT, **custom_dict}
        print(f"[+] Berhasil memuat {len(custom_dict)} kata slang dari {filepath}.")
        return merged_dict
    except Exception as exc:
        print(f"[!] Gagal membaca file slang: {exc}. Menggunakan default.")
        return DEFAULT_SLANG_DICT


def _normalize_slang_words(text, slang_dict):
    words = text.split()
    normalized_words = []
    for word in words:
        clean_word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
        clean_word_lower = clean_word.lower()

        if clean_word_lower in slang_dict and clean_word:
            formal_word = slang_dict[clean_word_lower]
            prefix = word[: word.find(clean_word)] if clean_word in word else ""
            suffix = word[word.find(clean_word) + len(clean_word):] if clean_word in word else ""
            normalized_words.append(prefix + formal_word + suffix)
        else:
            normalized_words.append(word)

    return " ".join(normalized_words)


def clean_text(text, slang_dict, profile="indobert", lowercase=None, remove_punctuation=None):
    """Membersihkan teks untuk profil SVM, LSTM, atau IndoBERT."""
    if not isinstance(text, str):
        return ""

    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Profil tidak dikenal: {profile}")

    defaults = PROFILE_DEFAULTS[profile]
    if lowercase is None:
        lowercase = defaults["lowercase"]
    if remove_punctuation is None:
        remove_punctuation = defaults["remove_punctuation"]

    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    text = _normalize_slang_words(text, slang_dict)

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    else:
        text = re.sub(r"[^\w\s.,!?\'\"-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_bert(text, slang_dict, lowercase=False, remove_punctuation=False):
    """Alias kompatibel untuk kode lama."""
    return clean_text(
        text,
        slang_dict=slang_dict,
        profile="indobert",
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
    )


def preprocess_dataframe(df, slang_dict, profile="indobert", lowercase=None, remove_punctuation=None):
    """Mengembalikan DataFrame baru dengan kolom teks bersih."""
    if "text" not in df.columns:
        raise ValueError("Kolom 'text' tidak ditemukan dalam file CSV.")

    result = df.copy()
    cleaned_texts = []
    for text in tqdm(result["text"], desc=f"Pembersihan Teks ({profile})"):
        cleaned_texts.append(
            clean_text(
                text,
                slang_dict=slang_dict,
                profile=profile,
                lowercase=lowercase,
                remove_punctuation=remove_punctuation,
            )
        )

    if profile == "all":
        result["cleaned_svm"] = [clean_text(text, slang_dict, profile="svm") for text in result["text"]]
        result["cleaned_lstm"] = [clean_text(text, slang_dict, profile="lstm") for text in result["text"]]
        result["cleaned_indobert"] = [clean_text(text, slang_dict, profile="indobert") for text in result["text"]]
        result["cleaned_text"] = result["cleaned_indobert"]
    else:
        result["cleaned_text"] = cleaned_texts

    return result


def test_indobert_tokenization(sample_original, sample_cleaned):
    """Menampilkan perbandingan token sebelum dan sesudah preprocessing."""
    try:
        from transformers import AutoTokenizer

        print("\n" + "=" * 50)
        print("[*] MEMULAI UJI COBA TOKENIZER INDOBERT")
        print("=" * 50)
        print("Sedang memuat tokenizer 'indobenchmark/indobert-base-p1'...")

        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

        print("\n--- Sampel 1: Teks Asli (Sebelum Preprocessing) ---")
        print(f"Teks: {sample_original}")
        tokens_orig = tokenizer.tokenize(sample_original)
        ids_orig = tokenizer.encode(sample_original)
        print(f"Tokens: {tokens_orig}")
        print(f"Token IDs: {ids_orig}")

        print("\n--- Sampel 2: Teks Bersih (Setelah Preprocessing) ---")
        print(f"Teks: {sample_cleaned}")
        tokens_clean = tokenizer.tokenize(sample_cleaned)
        ids_clean = tokenizer.encode(sample_cleaned)
        print(f"Tokens: {tokens_clean}")
        print(f"Token IDs: {ids_clean}")

        unk_orig = tokens_orig.count("[UNK]")
        unk_clean = tokens_clean.count("[UNK]")
        print("\n[Analisis Tokenizer]:")
        print(f"- Jumlah token [UNK] pada teks asli: {unk_orig}")
        print(f"- Jumlah token [UNK] pada teks bersih: {unk_clean}")
        print("- Catatan: normalisasi slang dan emoji membantu IndoBERT menangkap konteks lebih baik.")
        print("=" * 50 + "\n")
    except ImportError:
        print("\n[!] Library 'transformers' atau 'torch' tidak terdeteksi.")
        print("[!] Untuk menguji tokenisasi IndoBERT secara langsung, jalankan: pip install transformers torch")
    except Exception as exc:
        print(f"\n[!] Gagal menguji tokenization: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Preprocessing komentar YouTube untuk SVM, LSTM, dan IndoBERT")
    parser.add_argument("--input", default="comments_raw.csv", help="File CSV input hasil scraping")
    parser.add_argument("--output", default="comments_cleaned.csv", help="File CSV output hasil pembersihan")
    parser.add_argument("--slang_file", default=None, help="File CSV kamus slang tambahan (opsional)")
    parser.add_argument(
        "--profile",
        choices=["all", "svm", "lstm", "indobert"],
        default="all",
        help="Profil preprocessing yang akan dibuat",
    )
    parser.add_argument("--lowercase", action="store_true", help="Paksa semua teks menjadi lowercase")
    parser.add_argument("--remove_punct", action="store_true", help="Paksa hapus semua tanda baca")
    parser.add_argument("--test_tokenizer", action="store_true", help="Jalankan simulasi tokenisasi IndoBERT")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] ERROR: File input '{args.input}' tidak ditemukan.")
        print("[!] Silakan jalankan scraper.py terlebih dahulu atau pastikan path file benar.")
        return

    slang_dict = load_slang_dict(args.slang_file)

    print(f"[*] Membaca data dari: {args.input}")
    df = pd.read_csv(args.input)

    if "text" not in df.columns:
        print("[!] ERROR: Kolom 'text' tidak ditemukan dalam file CSV.")
        return

    print(f"[*] Memulai preprocessing {len(df)} baris data untuk profil '{args.profile}'...")

    if args.profile == "all":
        result = df.copy()
        result["cleaned_svm"] = [
            clean_text(
                text,
                slang_dict,
                profile="svm",
                lowercase=args.lowercase if args.lowercase else None,
                remove_punctuation=True if args.remove_punct else None,
            )
            for text in tqdm(result["text"], desc="Pembersihan SVM")
        ]
        result["cleaned_lstm"] = [
            clean_text(
                text,
                slang_dict,
                profile="lstm",
                lowercase=args.lowercase if args.lowercase else None,
                remove_punctuation=True if args.remove_punct else None,
            )
            for text in tqdm(result["text"], desc="Pembersihan LSTM")
        ]
        result["cleaned_indobert"] = [
            clean_text(
                text,
                slang_dict,
                profile="indobert",
                lowercase=args.lowercase if args.lowercase else None,
                remove_punctuation=args.remove_punct if args.remove_punct else None,
            )
            for text in tqdm(result["text"], desc="Pembersihan IndoBERT")
        ]
        result["cleaned_text"] = result["cleaned_indobert"]
    else:
        result = preprocess_dataframe(
            df,
            slang_dict=slang_dict,
            profile=args.profile,
            lowercase=args.lowercase if args.lowercase else None,
            remove_punctuation=args.remove_punct if args.remove_punct else None,
        )

    result.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[+] Pembersihan selesai. Data disimpan ke: {os.path.abspath(args.output)}")

    if args.test_tokenizer:
        sample_cleaned_col = "cleaned_indobert" if "cleaned_indobert" in result.columns else "cleaned_text"
        sample_row = None

        for _, row in result.iterrows():
            if str(row["text"]) != str(row[sample_cleaned_col]) and len(str(row["text"])) > 10:
                sample_row = row
                break

        if sample_row is None and len(result) > 0:
            sample_row = result.iloc[0]

        if sample_row is not None:
            test_indobert_tokenization(sample_row["text"], sample_row[sample_cleaned_col])
    elif len(result) > 0:
        print("\n[Tips] Jalankan dengan flag '--test_tokenizer' untuk melihat dampak preprocessing terhadap IndoBERT.")
        print("Contoh: python preprocessing.py --profile all --test_tokenizer")


if __name__ == "__main__":
    main()
