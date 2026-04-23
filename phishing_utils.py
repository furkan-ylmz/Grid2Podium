import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("torch is required for dataset and model operations.")


DATASET_CSV_PATH = "datasets/phishing_email/meajor_cleaned_preprocessed.csv"
PROCESSED_DIR = "processed_data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

RAW_NUMERIC_COLUMNS = [
    "url_count",
    "url_length_max",
    "url_length_avg",
    "url_subdom_max",
    "url_subdom_avg",
    "attachment_count",
]

DERIVED_NUMERIC_COLUMNS = [
    "has_attachments",
    "has_html",
    "has_plaintext",
    "is_english",
    "text_char_count",
    "text_word_count",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "digit_ratio",
    "suspicious_term_count",
]

NUMERIC_FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS

DEFAULT_VOCAB_SIZE = 20000
DEFAULT_MAX_LENGTH = 256
DEFAULT_EMBED_DIM = 128

SUSPICIOUS_TERMS = [
    "account",
    "bank",
    "billing",
    "click",
    "confirm",
    "credential",
    "invoice",
    "limited",
    "login",
    "password",
    "pay",
    "payment",
    "secure",
    "security",
    "suspended",
    "unlock",
    "urgent",
    "verify",
    "wallet",
]
SUSPICIOUS_PATTERN = re.compile(r"\b(" + "|".join(SUSPICIOUS_TERMS) + r")\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\[[a-z_]+\]|[a-z0-9]+(?:'[a-z0-9]+)?")


def _to_string(value):
    if pd.isna(value):
        return ""
    return str(value)


def _to_binary_flag(value):
    if isinstance(value, bool):
        return int(value)
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    return int(text in {"1", "true", "yes", "y"})


def normalize_text(text):
    text = _to_string(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def combine_email_text(subject, body):
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"[SUBJECT] {subject} [BODY] {body}"
    if subject:
        return f"[SUBJECT] {subject}"
    if body:
        return f"[BODY] {body}"
    return ""


def _safe_divide(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_uppercase_ratio(text):
    alpha_chars = [char for char in text if char.isalpha()]
    if not alpha_chars:
        return 0.0
    uppercase_chars = [char for char in alpha_chars if char.isupper()]
    return _safe_divide(len(uppercase_chars), len(alpha_chars))


def compute_digit_ratio(text):
    if not text:
        return 0.0
    digit_count = sum(char.isdigit() for char in text)
    return _safe_divide(digit_count, len(text))


def count_suspicious_terms(text):
    return len(SUSPICIOUS_PATTERN.findall(text))


def prepare_email_dataframe(df):
    prepared = df.copy()

    for column in ["subject", "body", "content_types", "language"]:
        prepared[column] = prepared.get(column, "").fillna("").astype(str)

    for column in RAW_NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared.get(column, 0), errors="coerce").fillna(0.0)

    prepared["has_attachments"] = prepared.get("has_attachments", 0).apply(_to_binary_flag).astype(float)
    prepared["content_types"] = prepared["content_types"].str.lower()
    prepared["language"] = prepared["language"].str.lower().replace("", "unknown")
    prepared["text"] = prepared.apply(
        lambda row: combine_email_text(row.get("subject", ""), row.get("body", "")),
        axis=1,
    )

    prepared["has_html"] = prepared["content_types"].str.contains("html", regex=False).astype(float)
    prepared["has_plaintext"] = prepared["content_types"].str.contains("plain", regex=False).astype(float)
    prepared["is_english"] = prepared["language"].eq("en").astype(float)

    prepared["text_char_count"] = prepared["text"].str.len().astype(float)
    prepared["text_word_count"] = prepared["text"].str.split().map(len).astype(float)
    prepared["exclamation_count"] = prepared["text"].str.count("!").astype(float)
    prepared["question_count"] = prepared["text"].str.count(r"\?").astype(float)
    prepared["uppercase_ratio"] = prepared["text"].map(compute_uppercase_ratio).astype(float)
    prepared["digit_ratio"] = prepared["text"].map(compute_digit_ratio).astype(float)
    prepared["suspicious_term_count"] = prepared["text"].map(count_suspicious_terms).astype(float)

    if "label" in prepared.columns:
        prepared["label"] = pd.to_numeric(prepared["label"], errors="coerce").fillna(0).astype(int)

    return prepared


def build_vocabulary(texts, max_vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, frequency in counter.most_common():
        if frequency < min_frequency:
            continue
        if token in vocabulary:
            continue
        vocabulary[token] = len(vocabulary)
        if len(vocabulary) >= max_vocab_size:
            break
    return vocabulary


def tokenize_text(text):
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def encode_text(text, vocabulary, max_length=DEFAULT_MAX_LENGTH):
    tokens = tokenize_text(text)
    encoded = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokens[:max_length]]
    if len(encoded) < max_length:
        encoded.extend([vocabulary[PAD_TOKEN]] * (max_length - len(encoded)))
    return encoded


def fit_numeric_scaler(df):
    scaler = {"columns": NUMERIC_FEATURE_COLUMNS, "mean": {}, "std": {}}
    for column in NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
        mean = float(values.mean())
        std = float(values.std())
        scaler["mean"][column] = mean
        scaler["std"][column] = std if std > 1e-6 else 1.0
    return scaler


def transform_numeric_features(df, scaler):
    arrays = []
    for column in scaler["columns"]:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
        mean = scaler["mean"][column]
        std = scaler["std"][column]
        arrays.append(((values - mean) / std).to_numpy(dtype=np.float32))
    return np.stack(arrays, axis=1).astype(np.float32)


if TORCH_AVAILABLE:
    class EmailDataset(Dataset):
        def __init__(self, text_sequences, numeric_features, labels):
            self.text_sequences = torch.tensor(text_sequences, dtype=torch.long)
            self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
            self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.text_sequences)

        def __getitem__(self, index):
            item = {
                "text": self.text_sequences[index],
                "numeric": self.numeric_features[index],
            }
            if self.labels is not None:
                item["label"] = self.labels[index]
            return item


    class MeanEmbeddingMLP(nn.Module):
        def __init__(self, vocab_size, numeric_dim, embed_dim=DEFAULT_EMBED_DIM, output_dim=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim + numeric_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, output_dim),
            )

        def forward(self, text_ids, numeric_features):
            embeddings = self.embedding(text_ids)
            mask = (text_ids != 0).unsqueeze(-1)
            summed = (embeddings * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths
            features = torch.cat([pooled, numeric_features], dim=1)
            return self.classifier(features)


    class TextCNNClassifier(nn.Module):
        def __init__(self, vocab_size, numeric_dim, embed_dim=DEFAULT_EMBED_DIM, output_dim=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
                    nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2),
                    nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
                ]
            )
            self.dropout = nn.Dropout(0.35)
            self.classifier = nn.Sequential(
                nn.Linear(128 * len(self.convs) + numeric_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, output_dim),
            )

        def forward(self, text_ids, numeric_features):
            embeddings = self.embedding(text_ids).transpose(1, 2)
            pooled_outputs = []
            for conv in self.convs:
                features = torch.relu(conv(embeddings))
                pooled = torch.max(features, dim=2).values
                pooled_outputs.append(pooled)
            text_features = torch.cat(pooled_outputs, dim=1)
            combined = torch.cat([self.dropout(text_features), numeric_features], dim=1)
            return self.classifier(combined)


    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, numeric_dim, embed_dim=DEFAULT_EMBED_DIM, hidden_dim=128, output_dim=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2 + numeric_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, output_dim),
            )

        def forward(self, text_ids, numeric_features):
            embeddings = self.embedding(text_ids)
            outputs, _ = self.lstm(embeddings)
            text_features = outputs.mean(dim=1)
            combined = torch.cat([text_features, numeric_features], dim=1)
            return self.classifier(combined)
else:
    class EmailDataset(Dataset):
        pass


    class MeanEmbeddingMLP:
        pass


    class TextCNNClassifier:
        pass


    class BiLSTMClassifier:
        pass


def create_model(model_key, vocab_size, numeric_dim, output_dim=2):
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError("torch is required to create and train models.")
    if model_key == "mean_mlp":
        return MeanEmbeddingMLP(vocab_size=vocab_size, numeric_dim=numeric_dim, output_dim=output_dim)
    if model_key == "text_cnn":
        return TextCNNClassifier(vocab_size=vocab_size, numeric_dim=numeric_dim, output_dim=output_dim)
    if model_key == "bi_lstm":
        return BiLSTMClassifier(vocab_size=vocab_size, numeric_dim=numeric_dim, output_dim=output_dim)
    raise ValueError(f"Unknown model key: {model_key}")


def encode_dataframe(df, vocabulary, scaler, max_length=DEFAULT_MAX_LENGTH):
    text_sequences = np.asarray([encode_text(text, vocabulary, max_length) for text in df["text"]], dtype=np.int64)
    numeric_features = transform_numeric_features(df, scaler)
    labels = None if "label" not in df.columns else df["label"].to_numpy(dtype=np.int64)
    return text_sequences, numeric_features, labels


def save_pickle(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)
