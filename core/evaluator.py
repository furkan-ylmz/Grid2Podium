from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    dataset_name: str = "Test",
    device: torch.device = torch.device('cpu')
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Evaluate PyTorch model on a dataloader.
    Returns:
        accuracy, precision, recall, f1_score, confusion_matrix
    """
    model.eval()
    model.to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_b, y_b in dataloader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y_b.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- Model Sonuçları ({dataset_name} Seti Üzerinde) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")

    return float(acc), float(prec), float(rec), float(f1), cm
