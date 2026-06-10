import argparse

from sklearn.preprocessing import LabelEncoder

from benchmark_models import run_svm
from ml_common import load_labeled_dataframe, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SVM on labeled YouTube comments")
    parser.add_argument("--input", default="comments_cleaned.csv")
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--output_dir", default="model_outputs")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.random_state)
    df = load_labeled_dataframe(
        args.input,
        text_column=args.text_column,
        label_column=args.label_column,
        text_candidates=("cleaned_svm", "cleaned_text", "text"),
    )
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"])
    label_classes = label_encoder.classes_.tolist()

    run_svm(
        df["text"].tolist(),
        labels,
        args.text_column or "auto",
        output_path=args.output_dir,
        random_state=args.random_state,
        label_classes=label_classes,
    )


if __name__ == "__main__":
    main()