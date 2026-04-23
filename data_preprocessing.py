import argparse
import os

import numpy as np
import pandas as pd

from phishing_utils import DATASET_CSV_PATH, PROCESSED_DIR, prepare_email_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare phishing email dataset splits.")
    parser.add_argument(
        "--input-path",
        default=DATASET_CSV_PATH,
        help="Path to the MeAJOR phishing dataset CSV.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def load_and_prepare_dataset(input_path, max_samples=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded raw dataset with {len(df):,} rows.")

    if max_samples is not None and max_samples < len(df):
        sampled_parts = []
        for _, group in df.groupby("label"):
            take_n = max(1, int(round(max_samples * len(group) / len(df))))
            take_n = min(take_n, len(group))
            sampled_parts.append(group.sample(n=take_n, random_state=42))
        df = pd.concat(sampled_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Using capped sample size: {len(df):,} rows.")

    prepared_df = prepare_email_dataframe(df)
    prepared_df = prepared_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    prepared_df = prepared_df[prepared_df["text"].str.len() > 0].reset_index(drop=True)

    print("\nSelected columns:")
    print(", ".join(prepared_df.columns))
    print("\nLabel distribution:")
    print(prepared_df["label"].value_counts().sort_index())

    return prepared_df


def stratified_split(df, label_column, train_fraction, random_state):
    rng = np.random.default_rng(random_state)
    train_parts = []
    holdout_parts = []

    for _, group in df.groupby(label_column):
        shuffled_indices = rng.permutation(group.index.to_numpy())
        split_index = int(round(len(shuffled_indices) * train_fraction))

        if len(shuffled_indices) > 1:
            split_index = min(max(split_index, 1), len(shuffled_indices) - 1)
        else:
            split_index = len(shuffled_indices)

        train_indices = shuffled_indices[:split_index]
        holdout_indices = shuffled_indices[split_index:]

        train_parts.append(df.loc[train_indices])
        if len(holdout_indices) > 0:
            holdout_parts.append(df.loc[holdout_indices])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    holdout_df = pd.concat(holdout_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, holdout_df


def split_dataset(df, random_state):
    train_df, temp_df = stratified_split(df, "label", train_fraction=0.70, random_state=random_state)
    val_df, test_df = stratified_split(temp_df, "label", train_fraction=(2 / 3), random_state=random_state + 1)
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved processed splits:")
    print(f"Train ({len(train_df):,}) -> {train_path}")
    print(f"Validation ({len(val_df):,}) -> {val_path}")
    print(f"Test ({len(test_df):,}) -> {test_path}")


def main():
    args = parse_args()
    prepared_df = load_and_prepare_dataset(args.input_path, args.max_samples)
    train_df, val_df, test_df = split_dataset(prepared_df, args.random_state)
    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
