import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phishing_utils import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_VOCAB_SIZE,
    MODELS_DIR,
    NUMERIC_FEATURE_COLUMNS,
    PROCESSED_DIR,
    RESULTS_DIR,
    EmailDataset,
    build_vocabulary,
    create_model,
    encode_dataframe,
    fit_numeric_scaler,
    save_pickle,
)


MODEL_SPECS = [
    ("mean_mlp", "Custom Mean Embedding + MLP"),
    ("text_cnn", "TextCNN + Metadata"),
    ("bi_lstm", "BiLSTM + Metadata"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train phishing email classifiers.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length.")
    parser.add_argument("--max-vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Maximum vocabulary size.")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience on validation F1.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def load_split(split_name):
    path = os.path.join(PROCESSED_DIR, f"{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed split not found: {path}. Run data_preprocessing.py first.")
    return pd.read_csv(path)


def limit_split_size(df, max_samples):
    if max_samples is None or max_samples >= len(df):
        return df
    sampled_parts = []
    for _, group in df.groupby("label"):
        take_n = max(1, int(round(max_samples * len(group) / len(df))))
        take_n = min(take_n, len(group))
        sampled_parts.append(group.sample(n=take_n, random_state=42))
    limited = pd.concat(sampled_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return limited


def build_dataloaders(train_df, val_df, test_df, vocabulary, scaler, max_length, batch_size):
    train_text, train_numeric, train_labels = encode_dataframe(train_df, vocabulary, scaler, max_length)
    val_text, val_numeric, val_labels = encode_dataframe(val_df, vocabulary, scaler, max_length)
    test_text, test_numeric, test_labels = encode_dataframe(test_df, vocabulary, scaler, max_length)

    train_dataset = EmailDataset(train_text, train_numeric, train_labels)
    val_dataset = EmailDataset(val_text, val_numeric, val_labels)
    test_dataset = EmailDataset(test_text, test_numeric, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        text_ids = batch["text"].to(device)
        numeric_features = batch["numeric"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(text_ids, numeric_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text"].to(device)
            numeric_features = batch["numeric"].to(device)
            labels = batch["label"].cpu().numpy()

            logits = model(text_ids, numeric_features)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            all_labels.extend(labels)
            all_predictions.extend(predictions)

    cm = binary_confusion_matrix(all_labels, all_predictions)
    accuracy, precision, recall, f1 = binary_classification_metrics(cm)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def binary_confusion_matrix(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))

    return np.array([[true_negative, false_positive], [false_negative, true_positive]], dtype=np.int64)


def binary_classification_metrics(confusion):
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


def train_single_model(model_key, model_name, train_loader, val_loader, device, args, class_weights):
    model = create_model(
        model_key=model_key,
        vocab_size=args.vocab_size,
        numeric_dim=args.numeric_dim,
        output_dim=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}
    best_state = None
    best_metrics = None
    best_f1 = -1.0
    stale_epochs = 0

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"{model_name} | epoch {epoch + 1}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, history, best_metrics


def plot_results(histories, metrics_by_model):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for model_name, history in histories.items():
        plt.plot(history["train_loss"], label=model_name)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_name, history in histories.items():
        plt.plot(history["val_accuracy"], label=model_name)
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "validation_accuracy.png"))
    plt.close()

    fig, axes = plt.subplots(1, len(metrics_by_model), figsize=(18, 5))
    if len(metrics_by_model) == 1:
        axes = [axes]

    for axis, (model_name, metrics) in zip(axes, metrics_by_model.items()):
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axis,
            xticklabels=["Safe", "Phishing"],
            yticklabels=["Safe", "Phishing"],
        )
        axis.set_title(f"{model_name}\nConfusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plt.close()


def save_metrics_summary(metrics_by_model):
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)


def main():
    args = parse_args()
    train_df = limit_split_size(load_split("train"), args.max_samples)
    val_df = limit_split_size(load_split("val"), None if args.max_samples is None else max(500, args.max_samples // 2))
    test_df = limit_split_size(load_split("test"), None if args.max_samples is None else max(500, args.max_samples // 2))

    vocabulary = build_vocabulary(train_df["text"], max_vocab_size=args.max_vocab_size)
    scaler = fit_numeric_scaler(train_df)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        vocabulary,
        scaler,
        args.max_length,
        args.batch_size,
    )

    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / max(1, label_counts.get(label, 1)) for label in range(2)],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Vocabulary size: {len(vocabulary):,}")

    args.vocab_size = len(vocabulary)
    args.numeric_dim = len(NUMERIC_FEATURE_COLUMNS)
    class_weights = class_weights.to(device)

    os.makedirs(MODELS_DIR, exist_ok=True)

    histories = {}
    metrics_by_model = {}
    best_bundle = None

    for model_key, model_name in MODEL_SPECS:
        print(f"\nTraining {model_name}...")
        model, history, best_val_metrics = train_single_model(
            model_key=model_key,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
            class_weights=class_weights,
        )

        test_metrics = evaluate_model(model, test_loader, device)
        histories[model_name] = history
        metrics_by_model[model_name] = test_metrics

        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_key}.pth"))
        print(
            f"{model_name} test metrics | "
            f"acc={test_metrics['accuracy']:.4f} "
            f"precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )

        if best_bundle is None or best_val_metrics["f1"] > best_bundle["val_f1"]:
            best_bundle = {
                "model_key": model_key,
                "model_name": model_name,
                "model_state": copy.deepcopy(model.state_dict()),
                "val_f1": best_val_metrics["f1"],
                "test_metrics": test_metrics,
            }

    torch.save(best_bundle["model_state"], os.path.join(MODELS_DIR, "best_phishing_model.pth"))
    save_pickle(
        os.path.join(MODELS_DIR, "phishing_assets.pkl"),
        {
            "vocabulary": vocabulary,
            "numeric_scaler": scaler,
            "max_length": args.max_length,
            "best_model_key": best_bundle["model_key"],
            "best_model_name": best_bundle["model_name"],
            "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
            "label_map": {0: "Safe Email", 1: "Phishing Email"},
            "model_metrics": metrics_by_model,
        },
    )

    plot_results(histories, metrics_by_model)
    save_metrics_summary(metrics_by_model)

    print("\nBest model saved:")
    print(f"Name: {best_bundle['model_name']}")
    print(f"Key : {best_bundle['model_key']}")
    print(f"Path: {os.path.join(MODELS_DIR, 'best_phishing_model.pth')}")
    print(f"Assets: {os.path.join(MODELS_DIR, 'phishing_assets.pkl')}")


if __name__ == "__main__":
    main()
