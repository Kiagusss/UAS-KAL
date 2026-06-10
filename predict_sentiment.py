import json
import argparse
import os

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_models import LSTMClassifier
from ml_common import encode_text, resolve_text_column, set_seed
from preprocessing import DEFAULT_SLANG_DICT, clean_text, load_slang_dict


def _batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _decode_label(prediction, label_classes):
    if label_classes and 0 <= int(prediction) < len(label_classes):
        return label_classes[int(prediction)]
    return str(prediction)


def _load_slang_dict(slang_file):
    if slang_file:
        return load_slang_dict(slang_file)
    return DEFAULT_SLANG_DICT


def _prepare_texts(texts, model_type, slang_dict, skip_cleaning=False):
    if skip_cleaning:
        return [str(text) if text is not None else "" for text in texts]

    profile = {
        "svm": "svm",
        "lstm": "lstm",
        "indobert": "indobert",
    }[model_type]
    return [clean_text(str(text), slang_dict, profile=profile) for text in texts]


def _load_text_source(args):
    if args.text is not None:
        return [args.text], None, None

    if not args.input:
        raise ValueError("Gunakan --text untuk satu kalimat atau --input untuk file CSV.")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    df = pd.read_csv(args.input)
    text_column = resolve_text_column(df, preferred=args.text_column)
    texts = df[text_column].astype(str).tolist()
    return texts, df, text_column


def _predict_svm(texts, model_path):
    artifact = joblib.load(model_path)
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    label_classes = artifact.get("label_classes")

    encoded = vectorizer.transform(texts)
    predictions = model.predict(encoded)
    labels = [_decode_label(prediction, label_classes) for prediction in predictions]
    return predictions.tolist(), labels


class _InferenceSequenceDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.texts = list(texts)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = encode_text(self.texts[index], self.vocab, self.max_len)
        return torch.tensor(encoded, dtype=torch.long)


def _predict_lstm(texts, model_path, batch_size):
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    max_len = checkpoint.get("max_len", 128)
    embedding_dim = checkpoint.get("embedding_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 128)
    dropout = checkpoint.get("dropout", 0.3)
    label_classes = checkpoint.get("label_classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=checkpoint["num_classes"],
        pad_idx=vocab["<pad>"],
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = _InferenceSequenceDataset(texts, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            lengths = (batch != vocab["<pad>"]).sum(dim=1)
            logits = model(batch, lengths)
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

    labels = [_decode_label(prediction, label_classes) for prediction in predictions]
    return predictions, labels


def _predict_indobert(texts, model_dir, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    metadata_path = os.path.join(model_dir, "metadata.json")
    label_classes = None
    max_len = 128

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        label_classes = metadata.get("label_classes")
        max_len = int(metadata.get("max_len", max_len))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch_texts in _batch_iter(texts, batch_size):
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

    labels = [_decode_label(prediction, label_classes) for prediction in predictions]
    return predictions, labels


def main():
    parser = argparse.ArgumentParser(description="Prediksi sentimen menggunakan model terlatih SVM, LSTM, atau IndoBERT")
    parser.add_argument("--model_type", choices=["svm", "lstm", "indobert"], required=True)
    parser.add_argument("--model_path", default="model_outputs", help="Folder output hasil training")
    parser.add_argument("--input", default=None, help="CSV input yang berisi teks")
    parser.add_argument("--text", default=None, help="Satu teks untuk diprediksi")
    parser.add_argument("--text_column", default=None, help="Kolom teks pada CSV input")
    parser.add_argument("--output", default="predictions.csv", help="CSV output untuk hasil prediksi batch")
    parser.add_argument("--batch_size", type=int, default=32, help="Ukuran batch inferensi")
    parser.add_argument("--slang_file", default=None, help="CSV kamus slang tambahan")
    parser.add_argument("--skip_cleaning", action="store_true", help="Lewati preprocessing sebelum prediksi")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.random_state)

    texts, df, text_column = _load_text_source(args)
    slang_dict = _load_slang_dict(args.slang_file)
    prepared_texts = _prepare_texts(texts, args.model_type, slang_dict, skip_cleaning=args.skip_cleaning)

    if args.model_type == "svm":
        model_file = os.path.join(args.model_path, "svm_pipeline.joblib")
        predictions, labels = _predict_svm(prepared_texts, model_file)
    elif args.model_type == "lstm":
        model_file = os.path.join(args.model_path, "lstm_model.pt")
        predictions, labels = _predict_lstm(prepared_texts, model_file, args.batch_size)
    else:
        model_dir = os.path.join(args.model_path, "indobert_model")
        predictions, labels = _predict_indobert(prepared_texts, model_dir, args.batch_size)

    if args.text is not None:
        print(f"Teks: {args.text}")
        print(f"Prediksi sentimen: {labels[0]} (kelas {predictions[0]})")
        return

    result = df.copy()
    result[text_column] = texts
    result[f"predicted_{args.model_type}"] = labels
    result[f"predicted_{args.model_type}_index"] = predictions
    result.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[+] Prediksi selesai. File disimpan ke: {os.path.abspath(args.output)}")
    print(result[f"predicted_{args.model_type}"].value_counts().to_string())


if __name__ == "__main__":
    main()