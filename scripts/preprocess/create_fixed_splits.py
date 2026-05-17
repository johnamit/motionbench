import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/train_sequences_full.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_split_ratios(train_ratio, val_ratio, test_ratio):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0. Got {ratio_sum:.6f}")


def build_split_tables(sequence_table, train_ratio, val_ratio, test_ratio, seed):
    labels = sequence_table["exercise_label"]

    train_table, holdout_table = train_test_split(
        sequence_table,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    holdout_labels = holdout_table["exercise_label"]
    val_fraction_of_holdout = val_ratio / (val_ratio + test_ratio)
    val_table, test_table = train_test_split(
        holdout_table,
        test_size=(1.0 - val_fraction_of_holdout),
        random_state=seed,
        stratify=holdout_labels,
    )

    return train_table, val_table, test_table


def save_split_table(split_table, output_file_path):
    split_table.to_csv(output_file_path, index=False)


def build_split_info(train_table, val_table, test_table, args):
    split_info = {
        "input_file": str(Path(args.input_file)),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_samples": int(len(train_table)),
        "val_samples": int(len(val_table)),
        "test_internal_samples": int(len(test_table)),
        "class_counts": {
            "train": train_table["exercise_label"].value_counts().to_dict(),
            "val": val_table["exercise_label"].value_counts().to_dict(),
            "test_internal": test_table["exercise_label"].value_counts().to_dict(),
        },
    }
    return split_info


def save_split_info(split_info, output_file_path):
    info_rows = [
        {"key": "input_file", "value": split_info["input_file"]},
        {"key": "seed", "value": split_info["seed"]},
        {"key": "train_ratio", "value": split_info["train_ratio"]},
        {"key": "val_ratio", "value": split_info["val_ratio"]},
        {"key": "test_ratio", "value": split_info["test_ratio"]},
        {"key": "train_samples", "value": split_info["train_samples"]},
        {"key": "val_samples", "value": split_info["val_samples"]},
        {"key": "test_internal_samples", "value": split_info["test_internal_samples"]},
    ]
    for split_name, class_counts in split_info["class_counts"].items():
        for class_name, class_count in class_counts.items():
            info_rows.append({"key": f"class_counts.{split_name}.{class_name}", "value": class_count})
    pd.DataFrame(info_rows).to_csv(output_file_path, index=False)


def main():
    args = parse_args()

    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    input_file_path = Path(args.input_file)
    output_directory_path = Path(args.output_dir)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    sequence_table = pd.read_csv(input_file_path)
    if "exercise_label" not in sequence_table.columns:
        raise ValueError("Input file must include an 'exercise_label' column.")

    train_table, val_table, test_table = build_split_tables(
        sequence_table=sequence_table,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_output_path = output_directory_path / "train_sequences.csv"
    val_output_path = output_directory_path / "val_sequences.csv"
    test_output_path = output_directory_path / "test_internal_sequences.csv"
    info_output_path = output_directory_path / "split_info.csv"

    save_split_table(train_table, train_output_path)
    save_split_table(val_table, val_output_path)
    save_split_table(test_table, test_output_path)

    split_info = build_split_info(train_table, val_table, test_table, args)
    save_split_info(split_info, info_output_path)

    print(f"Saved: {train_output_path} ({len(train_table)} rows)")
    print(f"Saved: {val_output_path} ({len(val_table)} rows)")
    print(f"Saved: {test_output_path} ({len(test_table)} rows)")
    print(f"Saved: {info_output_path}")


if __name__ == "__main__":
    main()
