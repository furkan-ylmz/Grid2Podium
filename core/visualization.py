import os
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CLASS_NAMES = ["Podyum", "Puan", "Puansız"]


def plot_loss_curves(
    all_train_losses: List[List[float]],
    all_val_losses: List[List[float]],
    model_names: List[str],
    save_path: Path = RESULTS_DIR / "loss_curves.png"
) -> None:
    """Plot and save training and validation loss curves for all models."""
    os.makedirs(save_path.parent, exist_ok=True)
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        axes[i].plot(all_train_losses[i], label='Train Loss', color='#1f77b4', linewidth=2)
        axes[i].plot(all_val_losses[i], label='Validation Loss', color='#ff7f0e', linewidth=2)
        axes[i].set_title(f'{model_name}\nLoss', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accuracy_curves(
    all_train_accs: List[List[float]],
    all_val_accs: List[List[float]],
    model_names: List[str],
    save_path: Path = RESULTS_DIR / "accuracy_curves.png"
) -> None:
    """Plot and save training and validation accuracy curves for all models."""
    os.makedirs(save_path.parent, exist_ok=True)
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        axes[i].plot(all_train_accs[i], label='Train Accuracy', color='#2ca02c', linewidth=2)
        axes[i].plot(all_val_accs[i], label='Validation Accuracy', color='#d62728', linewidth=2)
        axes[i].set_title(f'{model_name}\nAccuracy', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Accuracy')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrices(
    all_cms: List[np.ndarray],
    model_names: List[str],
    title_suffix: str = "Val Confusion Matrix",
    save_path: Path = RESULTS_DIR / "validation_confusion_matrices.png"
) -> None:
    """Plot and save confusion matrices heatmaps."""
    os.makedirs(save_path.parent, exist_ok=True)
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        sns.heatmap(
            all_cms[i],
            annot=True,
            fmt='d',
            ax=ax,
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )
        ax.set_title(f'{model_name}\n{title_suffix}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tahmin Edilen')
        ax.set_ylabel('Gerçek')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metrics_table(
    metrics_data: List[Dict[str, Any]],
    save_path: Path = RESULTS_DIR / "model_evaluation_metrics.png"
) -> None:
    """Render and save a stylized comparison table of all model evaluation metrics."""
    os.makedirs(save_path.parent, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_data).set_index('Model').T
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Metrik / Model'}, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    the_table = ax.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        loc='center',
        cellLoc='center'
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1, 2)

    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        elif col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')

    plt.title('Modellerin Tüm Veri Setleri Üzerindeki Detaylı Performans Karşılaştırması', fontweight="bold", pad=10)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
