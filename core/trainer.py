import copy
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train model for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        outputs = model(X_b)
        loss = criterion(outputs, y_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_b.size(0)
        correct += (predicted == y_b).sum().item()

    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def validate_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model for one epoch and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += y_b.size(0)
            correct += (predicted == y_b).sum().item()

    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 0.001,
    device: torch.device = torch.device('cpu')
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train a neural network model with early stopping and learning rate scheduling.
    Returns:
        train_losses, val_losses, train_accuracies, val_accuracies
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_single_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early Stopping tetiklendi! En iyi epoch: {epoch - patience + 1}")
            break

    model.load_state_dict(best_model_wts)
    return train_losses, val_losses, train_accuracies, val_accuracies
