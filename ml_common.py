import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

TEXT_COLUMN_CANDIDATES = (
    "cleaned_indobert",
    "cleaned_lstm",
    "cleaned_svm",
    "cleaned_text",
    "text",
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_text_column(df, preferred=None, candidates=None):
    if preferred and preferred in df.columns:
        return preferred

    search_order = candidates or TEXT_COLUMN_CANDIDATES
    for candidate in search_order:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Kolom teks tidak ditemukan. Gunakan salah satu kolom: "
        + ", ".join(TEXT_COLUMN_CANDIDATES)
    )


def load_labeled_dataframe(filepath, text_column=None, label_column="label", text_candidates=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan. Tambahkan kolom label untuk benchmarking."
        )

    resolved_text_column = resolve_text_column(df, preferred=text_column, candidates=text_candidates)
    result = df[[resolved_text_column, label_column]].dropna().copy()
    result[resolved_text_column] = result[resolved_text_column].astype(str)
    result[label_column] = result[label_column].astype(str)
    result = result.rename(columns={resolved_text_column: "text", label_column: "label"})

    if result.empty:
        raise ValueError("Dataset kosong setelah pembersihan NA.")

    return result


def train_val_test_split(
    texts,
    labels,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
):
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    val_fraction = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_train if len(set(y_train)) > 1 else None,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def print_metrics(title, metrics):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro   : {metrics['recall_macro']:.4f}")
    print(f"F1 macro       : {metrics['f1_macro']:.4f}")


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def tokenize_basic(text):
    return re.findall(r"[\w']+|[.,!?;]", text.lower(), flags=re.UNICODE)


def build_vocab(texts, min_freq=2, max_vocab_size=30000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_basic(str(text)))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab_size:
            break

    return vocab


def encode_text(text, vocab, max_len=128):
    tokens = tokenize_basic(str(text))
    encoded = [vocab.get(token, vocab["<unk>"]) for token in tokens][:max_len]
    if len(encoded) < max_len:
        encoded.extend([vocab["<pad>"]] * (max_len - len(encoded)))
    return encoded


class SequenceTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_tensor = torch.tensor(encode_text(self.texts[index], self.vocab, self.max_len), dtype=torch.long)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return {
            "input_ids": text_tensor,
            "labels": label_tensor,
        }


class TransformerTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item
