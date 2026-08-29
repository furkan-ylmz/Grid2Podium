import sys
import os
import pickle
from pathlib import Path

# Ensure project root is in sys.path for direct execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import torch

from core.preprocessing import prepare_tabular_features, get_dataloaders
from core.models import (
    CustomMLP,
    SimpleLSTM,
    ManualLSTM,
    CNN1D,
    TabularTransformer,
)
from core.trainer import train_model
from core.evaluator import evaluate_model
from core.visualization import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_confusion_matrices,
    plot_metrics_table,
)

# Paths and Constants
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

BEST_MODEL_PTH = MODELS_DIR / "best_model.pth"
BEST_MODEL_ARCH_PKL = MODELS_DIR / "best_model_arch.pkl"
FEATURE_COLUMNS_PKL = MODELS_DIR / "feature_columns.pkl"

NUM_CLASSES = 3
DEFAULT_EPOCHS = 100


def run_training_pipeline(epochs: int = DEFAULT_EPOCHS):
    """Train all 5 deep learning architectures, evaluate them, save the best model and generate charts."""
    print("İşlenmiş veriler yükleniyor...")
    if not TRAIN_CSV.exists() or not VAL_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            f"İşlenmiş veri setleri bulunamadı! Lütfen önce veri ön işlemeyi çalıştırın."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = prepare_tabular_features(
        train_df, val_df, test_df
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    input_dim = X_train.shape[1]
    output_dim = NUM_CLASSES

    print("Modeller tanımlanıyor...")
    model_mlp = CustomMLP(input_dim, output_dim)
    model_lstm_simple = SimpleLSTM(input_dim, hidden_dim=64, output_dim=output_dim)
    model_lstm_manual = ManualLSTM(input_dim, hidden_dim=64, output_dim=output_dim)
    model_cnn1d = CNN1D(input_dim, output_dim)
    model_transformer = TabularTransformer(input_dim, output_dim)

    models = [model_mlp, model_lstm_simple, model_lstm_manual, model_cnn1d, model_transformer]
    model_names = ['Özel MLP', 'Hazır LSTM', 'Manuel LSTM', '1D CNN', 'FT-Transformer']

    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    all_val_cms = []
    all_test_cms = []
    metrics_data = []

    best_acc = 0.0
    best_model = None
    best_model_name = ""
    best_arch = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    for model, name in zip(models, model_names):
        print(f"\n{name} eğitimi başlıyor...")
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, epochs=epochs, device=device
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        t_acc, t_prec, t_rec, t_f1, _ = evaluate_model(model, train_loader, dataset_name="Train", device=device)
        v_acc, v_prec, v_rec, v_f1, v_cm = evaluate_model(model, val_loader, dataset_name="Validation", device=device)
        ts_acc, ts_prec, ts_rec, ts_f1, ts_cm = evaluate_model(model, test_loader, dataset_name="Test", device=device)

        all_val_cms.append(v_cm)
        all_test_cms.append(ts_cm)

        metrics_data.append({
            'Model': name,
            'Train Accuracy': round(t_acc, 4),
            'Train Precision': round(t_prec, 4),
            'Train Recall': round(t_rec, 4),
            'Train F1-Score': round(t_f1, 4),
            'Val Accuracy': round(v_acc, 4),
            'Val Precision': round(v_prec, 4),
            'Val Recall': round(v_rec, 4),
            'Val F1-Score': round(v_f1, 4),
            'Test Accuracy': round(ts_acc, 4),
            'Test Precision': round(ts_prec, 4),
            'Test Recall': round(ts_rec, 4),
            'Test F1-Score': round(ts_f1, 4)
        })

        if ts_acc > best_acc:
            best_acc = ts_acc
            best_model = model
            best_model_name = name
            best_arch = model.__class__.__name__

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nEn başarılı model: {best_model_name} ({best_arch}) - Test Doğruluğu: {best_acc:.4f}")
    torch.save(best_model.state_dict(), BEST_MODEL_PTH)

    with open(FEATURE_COLUMNS_PKL, 'wb') as f:
        pickle.dump(feature_columns, f)
    with open(BEST_MODEL_ARCH_PKL, 'wb') as f:
        pickle.dump(best_arch, f)

    print("Sonuç grafikleri ve tablolar oluşturuluyor...")
    plot_loss_curves(all_train_losses, all_val_losses, model_names, RESULTS_DIR / "loss_curves.png")
    plot_accuracy_curves(all_train_accs, all_val_accs, model_names, RESULTS_DIR / "accuracy_curves.png")
    plot_confusion_matrices(all_val_cms, model_names, "Val Confusion Matrix", RESULTS_DIR / "validation_confusion_matrices.png")
    plot_confusion_matrices(all_test_cms, model_names, "Test Confusion Matrix", RESULTS_DIR / "test_confusion_matrices.png")
    plot_metrics_table(metrics_data, RESULTS_DIR / "model_evaluation_metrics.png")

    print(f"Modeller ve grafikler başarıyla kaydedildi!")


if __name__ == '__main__':
    run_training_pipeline()
