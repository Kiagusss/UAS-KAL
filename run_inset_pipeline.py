import argparse
import os
import subprocess
import sys


def run_command(command_args):
    print(f"[*] Menjalankan: {' '.join(command_args)}")
    completed = subprocess.run(command_args, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline lengkap: labeling InSet -> cleaning ulang -> benchmark SVM/LSTM/IndoBERT"
    )
    parser.add_argument("--input", default="comments_raw.csv", help="CSV input mentah")
    parser.add_argument("--labeled_output", default="comments_labeled_inset.csv", help="CSV hasil labeling InSet")
    parser.add_argument("--cleaned_output", default="comments_labeled_cleaned.csv", help="CSV hasil cleaning ulang")
    parser.add_argument("--benchmark_metrics", default="benchmark_metrics.json", help="File output metrik benchmark")
    parser.add_argument("--text_column", default=None, help="Kolom teks yang dipakai saat labeling")
    parser.add_argument("--profile", default="all", choices=["all", "svm", "lstm", "indobert"], help="Profil cleaning ulang")
    parser.add_argument("--label_column", default="label", help="Nama kolom label")
    parser.add_argument("--output_dir", default="model_outputs", help="Folder penyimpanan model")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs_lstm", type=int, default=8)
    parser.add_argument("--epochs_indobert", type=int, default=3)
    parser.add_argument("--batch_size_lstm", type=int, default=32)
    parser.add_argument("--batch_size_indobert", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--positive_lexicon", default="https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv")
    parser.add_argument("--negative_lexicon", default="https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    run_command([
        sys.executable,
        "label_with_inset.py",
        "--input",
        args.input,
        "--output",
        args.labeled_output,
        "--positive_lexicon",
        args.positive_lexicon,
        "--negative_lexicon",
        args.negative_lexicon,
    ] + (["--text_column", args.text_column] if args.text_column else []))

    run_command([
        sys.executable,
        "preprocessing.py",
        "--input",
        args.labeled_output,
        "--output",
        args.cleaned_output,
        "--profile",
        args.profile,
    ])

    run_command([
        sys.executable,
        "benchmark_models.py",
        "--input",
        args.cleaned_output,
        "--label_column",
        args.label_column,
        "--output_dir",
        args.output_dir,
        "--save_metrics",
        args.benchmark_metrics,
        "--random_state",
        str(args.random_state),
        "--epochs_lstm",
        str(args.epochs_lstm),
        "--epochs_indobert",
        str(args.epochs_indobert),
        "--batch_size_lstm",
        str(args.batch_size_lstm),
        "--batch_size_indobert",
        str(args.batch_size_indobert),
        "--max_len",
        str(args.max_len),
    ])

    print("[+] Pipeline selesai: labeling -> cleaning ulang -> benchmark")


if __name__ == "__main__":
    main()