import sys
import os
import glob
import re
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Ensure project root is in sys.path for direct execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Constants and Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CATEGORICAL_COLS = ["Track", "Driver", "Team"]
ALL_CAT_COLS = ["Track", "Driver", "Team", "Year"]
COLUMNS_TO_DROP = [
    "Laps",
    "Total Time/Gap/Retirement",
    "Points",
    "Fastest Lap",
    "Position",
    "No",
    "Time/Retired",
    "+1 Pt",
    "Set Fastest Lap",
    "Fastest Lap Time",
]
RANDOM_SEED = 42
BATCH_SIZE = 32


def extract_year_from_filename(filename: str) -> int | None:
    """Extract 4-digit year from filename (e.g. formula1_2022_season_race_results.csv)."""
    match = re.search(r'(\d{4})', Path(filename).name)
    if match:
        return int(match.group(1))
    return None


def categorize_position(pos: Any) -> int:
    """
    Categorize race position:
    0 -> Podium (1-3)
    1 -> Points (4-10)
    2 -> No Points / DNF (11+)
    """
    try:
        pos_int = int(pos)
        if pos_int <= 3:
            return 0
        elif pos_int <= 10:
            return 1
        else:
            return 2
    except (ValueError, TypeError):
        return 2


def clean_grid(grid: Any) -> int:
    """Convert starting grid to int, defaulting to 20 on failure."""
    try:
        return int(grid)
    except (ValueError, TypeError):
        return 20


def load_raw_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load all CSV files from raw data directory and append 'Year' column."""
    file_pattern = os.path.join(str(raw_data_dir), '*.csv')
    file_list = sorted(glob.glob(file_pattern))

    df_list = []
    for file_path in file_list:
        year = extract_year_from_filename(file_path)
        if year is None:
            continue
        df = pd.read_csv(file_path)
        df['Year'] = year
        df_list.append(df)

    if not df_list:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    full_df = pd.concat(df_list, ignore_index=True)
    return full_df


def preprocess_data(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Clean data, categorize target, clean grid, and encode categorical columns."""
    df = full_df.copy()
    df['Target_Tier'] = df['Position'].apply(categorize_position)

    cols_to_drop_existing = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df.drop(columns=cols_to_drop_existing, inplace=True)

    df['Starting Grid'] = df['Starting Grid'].apply(clean_grid)

    encoders = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.30,
    val_ratio: float = 1 / 3,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split by Year into train (70%), val (20%), and test (10%)."""
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['Year'],
        random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_ratio,
        stratify=temp_df['Year'],
        random_state=random_state
    )
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    output_dir: Path = PROCESSED_DATA_DIR
) -> None:
    """Save processed datasets and encoders to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """Execute complete data preprocessing pipeline from raw CSVs to saved processed datasets."""
    print(f"Ham veriler okunuyor: {raw_dir}")
    full_df = load_raw_data(raw_dir)
    print(f"Toplam {len(full_df)} satır veri yüklendi.")

    processed_df, encoders = preprocess_data(full_df)
    print("Ön işleme ve kategorik kodlama tamamlandı.")

    train_df, val_df, test_df = split_data(processed_df)
    save_processed_data(train_df, val_df, test_df, encoders, processed_dir)

    print(f"İşlenmiş veriler kaydedildi: {processed_dir}")
    print(f"Train Seti (%70): {len(train_df)} satır")
    print(f"Validation Seti (%20): {len(val_df)} satır")
    print(f"Test Seti (%10): {len(test_df)} satır")

    return train_df, val_df, test_df, encoders


def prepare_tabular_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str] = ALL_CAT_COLS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Apply one-hot encoding across datasets and extract NumPy arrays for training."""
    all_df = pd.concat([train_df, val_df, test_df], keys=['train', 'val', 'test'])
    all_df = pd.get_dummies(all_df, columns=cat_cols)

    train_enc = all_df.xs('train')
    val_enc = all_df.xs('val')
    test_enc = all_df.xs('test')

    feature_columns = train_enc.drop('Target_Tier', axis=1).columns.tolist()

    X_train = train_enc.drop('Target_Tier', axis=1).values.astype(np.float32)
    y_train = train_enc['Target_Tier'].values.astype(np.int64)

    X_val = val_enc.drop('Target_Tier', axis=1).values.astype(np.float32)
    y_val = val_enc['Target_Tier'].values.astype(np.int64)

    X_test = test_enc.drop('Target_Tier', axis=1).values.astype(np.float32)
    y_test = test_enc['Target_Tier'].values.astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoader instances for train, validation, and test datasets."""
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    run_pipeline()
