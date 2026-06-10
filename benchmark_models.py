import argparse
import copy
import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ml_common import (
    SequenceTextDataset,
    TransformerTextDataset,
    build_vocab,
    compute_metrics,
    print_metrics,
    save_json,
    set_seed,
    train_val_test_split,
)


@dataclass
class BenchmarkResult:
    name: str
    metrics: dict


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        features = torch.cat((hidden_forward, hidden_backward), dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def _select_text_column(df, text_column, candidates):
    if text_column and text_column in df.columns:
        return text_column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError("Kolom teks tidak ditemukan.")


def run_svm(texts, labels, text_column_name, output_path=None, random_state=42, label_classes=None):
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(texts, labels, random_state=random_state)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    model = LinearSVC(class_weight="balanced", random_state=random_state)
    model.fit(x_train_vec, y_train)

    val_pred = model.predict(x_val_vec)
    test_pred = model.predict(x_test_vec)

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    print_metrics(f"SVM - Validation ({text_column_name})", val_metrics)
    print_metrics(f"SVM - Test ({text_column_name})", test_metrics)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": vectorizer,
                "model": model,
                "label_classes": list(label_classes) if label_classes is not None else None,
            },
            os.path.join(output_path, "svm_pipeline.joblib"),
        )

    return BenchmarkResult("SVM", test_metrics)


def _make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_lstm(
    texts,
    labels,
    num_classes,
    output_path=None,
    epochs=8,
    batch_size=32,
    max_len=128,
    random_state=42,
    label_classes=None,
):
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(texts, labels, random_state=random_state)

    vocab = build_vocab(x_train, min_freq=2, max_vocab_size=30000)
    train_dataset = SequenceTextDataset(x_train, y_train, vocab, max_len=max_len)
    val_dataset = SequenceTextDataset(x_val, y_val, vocab, max_len=max_len)
    test_dataset = SequenceTextDataset(x_test, y_test, vocab, max_len=max_len)

    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size=batch_size)
    test_loader = _make_loader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 128
    hidden_dim = 128
    dropout = 0.3
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=vocab["<pad>"],
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    patience = 2
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels_batch = batch["labels"].to(device)
            lengths = (input_ids != vocab["<pad>"]).sum(dim=1)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels_batch = batch["labels"].to(device)
                lengths = (input_ids != vocab["<pad>"]).sum(dim=1)
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels_batch)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                val_targets.extend(labels_batch.cpu().tolist())

        average_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_metrics = compute_metrics(val_targets, val_preds)
        print(f"[LSTM] Epoch {epoch + 1}/{epochs} - train_loss={train_loss / max(len(train_loader), 1):.4f} - val_loss={average_val_loss:.4f}")
        print_metrics("LSTM - Validation", val_metrics)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels_batch = batch["labels"].to(device)
            lengths = (input_ids != vocab["<pad>"]).sum(dim=1)
            logits = model(input_ids, lengths)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            test_targets.extend(labels_batch.cpu().tolist())

    test_metrics = compute_metrics(test_targets, test_preds)
    print_metrics("LSTM - Test", test_metrics)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "num_classes": num_classes,
                "max_len": max_len,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "label_classes": list(label_classes) if label_classes is not None else None,
            },
            os.path.join(output_path, "lstm_model.pt"),
        )

    return BenchmarkResult("LSTM", test_metrics)


def run_indobert(
    texts,
    labels,
    num_classes,
    output_path=None,
    epochs=3,
    batch_size=8,
    max_len=128,
    random_state=42,
    label_classes=None,
):
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(texts, labels, random_state=random_state)

    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    train_dataset = TransformerTextDataset(x_train, y_train, tokenizer, max_length=max_len)
    val_dataset = TransformerTextDataset(x_val, y_val, tokenizer, max_length=max_len)
    test_dataset = TransformerTextDataset(x_test, y_test, tokenizer, max_length=max_len)

    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size=batch_size)
    test_loader = _make_loader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        "indobenchmark/indobert-base-p1",
        num_labels=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(0.1 * total_steps), 1),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 2
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss if outputs.loss is not None else criterion(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss if outputs.loss is not None else criterion(outputs.logits, batch["labels"])
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
                val_targets.extend(batch["labels"].cpu().tolist())

        average_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_metrics = compute_metrics(val_targets, val_preds)
        print(f"[IndoBERT] Epoch {epoch + 1}/{epochs} - train_loss={train_loss / max(len(train_loader), 1):.4f} - val_loss={average_val_loss:.4f}")
        print_metrics("IndoBERT - Validation", val_metrics)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
            test_targets.extend(batch["labels"].cpu().tolist())

    test_metrics = compute_metrics(test_targets, test_preds)
    print_metrics("IndoBERT - Test", test_metrics)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(os.path.join(output_path, "indobert_model"))
        tokenizer.save_pretrained(os.path.join(output_path, "indobert_model"))
        save_json(
            {
                "num_classes": num_classes,
                "max_len": max_len,
                "label_classes": list(label_classes) if label_classes is not None else None,
            },
            os.path.join(output_path, "indobert_model", "metadata.json"),
        )

    return BenchmarkResult("IndoBERT", test_metrics)


def benchmark_models(results):
    print("\n" + "=" * 60)
    print("RINGKASAN BENCHMARK")
    print("=" * 60)
    for result in results:
        print(f"{result.name:<12} accuracy={result.metrics['accuracy']:.4f} | f1_macro={result.metrics['f1_macro']:.4f}")

    best = max(results, key=lambda item: item.metrics["accuracy"])
    print("\nModel paling akurat berdasarkan accuracy test: {} ({:.4f})".format(best.name, best.metrics["accuracy"]))
    return best


def main():
    parser = argparse.ArgumentParser(description="Benchmark SVM, LSTM, dan IndoBERT pada dataset komentar yang sudah dilabeli")
    parser.add_argument("--input", default="comments_cleaned.csv", help="CSV hasil preprocessing yang memiliki kolom label")
    parser.add_argument("--text_column", default=None, help="Kolom teks yang dipakai. Default: pilih otomatis")
    parser.add_argument("--label_column", default="label", help="Nama kolom label")
    parser.add_argument("--output_dir", default="model_outputs", help="Folder penyimpanan model")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs_lstm", type=int, default=8)
    parser.add_argument("--epochs_indobert", type=int, default=3)
    parser.add_argument("--batch_size_lstm", type=int, default=32)
    parser.add_argument("--batch_size_indobert", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--save_metrics", default="benchmark_metrics.json")

    args = parser.parse_args()

    set_seed(args.random_state)
    df = pd.read_csv(args.input)
    if args.label_column not in df.columns:
        raise ValueError(f"Kolom label '{args.label_column}' tidak ditemukan.")

    svm_text_column = _select_text_column(df, args.text_column, ("cleaned_svm", "cleaned_text", "text"))
    lstm_text_column = _select_text_column(df, args.text_column, ("cleaned_lstm", "cleaned_text", "text"))
    indobert_text_column = _select_text_column(df, args.text_column, ("cleaned_indobert", "cleaned_text", "text"))

    required_columns = [args.label_column, svm_text_column, lstm_text_column, indobert_text_column]
    df = df.dropna(subset=required_columns).copy()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[args.label_column].astype(str))
    label_classes = label_encoder.classes_.tolist()
    svm_texts = df[svm_text_column].astype(str).tolist()
    lstm_texts = df[lstm_text_column].astype(str).tolist()
    indobert_texts = df[indobert_text_column].astype(str).tolist()

    if len(label_encoder.classes_) < 2:
        raise ValueError("Benchmark membutuhkan setidaknya 2 kelas label.")

    print(f"[*] Dataset dimuat: {len(df)} baris, {len(label_encoder.classes_)} kelas label.")
    print(f"[*] Kelas label: {', '.join(label_encoder.classes_)}")

    results = []
    results.append(
        run_svm(
            svm_texts,
            encoded_labels,
            svm_text_column,
            output_path=args.output_dir,
            random_state=args.random_state,
            label_classes=label_classes,
        )
    )
    results.append(
        run_lstm(
            lstm_texts,
            encoded_labels,
            num_classes=len(label_encoder.classes_),
            output_path=args.output_dir,
            epochs=args.epochs_lstm,
            batch_size=args.batch_size_lstm,
            max_len=args.max_len,
            random_state=args.random_state,
            label_classes=label_classes,
        )
    )
    results.append(
        run_indobert(
            indobert_texts,
            encoded_labels,
            num_classes=len(label_encoder.classes_),
            output_path=args.output_dir,
            epochs=args.epochs_indobert,
            batch_size=args.batch_size_indobert,
            max_len=args.max_len,
            random_state=args.random_state,
            label_classes=label_classes,
        )
    )

    best = benchmark_models(results)
    metrics_payload = {
        result.name: result.metrics for result in results
    }
    metrics_payload["best_model"] = {
        "name": best.name,
        "accuracy": best.metrics["accuracy"],
        "f1_macro": best.metrics["f1_macro"],
    }

    save_json(metrics_payload, args.save_metrics)
    print(f"[+] Ringkasan benchmark disimpan ke {args.save_metrics}")


if __name__ == "__main__":
    main()