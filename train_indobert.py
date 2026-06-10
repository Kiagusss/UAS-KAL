import argparse

from sklearn.preprocessing import LabelEncoder

from benchmark_models import run_indobert
from ml_common import load_labeled_dataframe, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate IndoBERT on labeled YouTube comments")
    parser.add_argument("--input", default="comments_cleaned.csv")
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--output_dir", default="model_outputs")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.random_state)
    df = load_labeled_dataframe(
        args.input,
        text_column=args.text_column,
        label_column=args.label_column,
        text_candidates=("cleaned_indobert", "cleaned_text", "text"),
    )
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"])
    label_classes = label_encoder.classes_.tolist()

    run_indobert(
        df["text"].tolist(),
        labels,
        num_classes=len(label_encoder.classes_),
        output_path=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        random_state=args.random_state,
        label_classes=label_classes,
    )


if __name__ == "__main__":
    main()