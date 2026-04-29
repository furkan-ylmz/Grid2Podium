import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

print("Veriler yükleniyor...")
train_df = pd.read_csv('processed_data/train.csv')
val_df = pd.read_csv('processed_data/val.csv')
test_df = pd.read_csv('processed_data/test.csv')

cat_cols = ['Track', 'Driver', 'Team', 'Year']

all_df = pd.concat([train_df, val_df, test_df], keys=['train', 'val', 'test'])
all_df = pd.get_dummies(all_df, columns=cat_cols)

train_df = all_df.xs('train')
val_df = all_df.xs('val')
test_df = all_df.xs('test')

X_train = train_df.drop('Target_Tier', axis=1).values.astype(np.float32)
y_train = train_df['Target_Tier'].values.astype(np.int64)

X_val = val_df.drop('Target_Tier', axis=1).values.astype(np.float32)
y_val = val_df['Target_Tier'].values.astype(np.int64)

X_test = test_df.drop('Target_Tier', axis=1).values.astype(np.float32)
y_test = test_df['Target_Tier'].values.astype(np.int64)

batch_size = 32
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

input_dim = X_train.shape[1]
output_dim = 3

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            dummy = self.pool2(self.conv2(self.pool1(self.conv1(dummy))))
            flat_dim = dummy.view(1, -1).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=3, d_model=64, nhead=4, num_layers=2):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

def train_model(model, name, epochs=40):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                outputs = model(X_b)
                _, predicted = torch.max(outputs.data, 1)
                total += y_b.size(0)
                correct += (predicted == y_b).sum().item()
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
    return train_losses, val_accuracies

def evaluate_model(model, name, dataloader, dataset_name="Test"):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_b, y_b in dataloader:
            outputs = model(X_b)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y_b.numpy())
            y_pred.extend(predicted.numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- {name} Sonuçları ({dataset_name} Seti Üzerinde) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
    
    return acc, prec, rec, f1, cm

def plot_results(all_train_losses, all_val_accs, all_cms, model_names):
    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(10, 5))
    for i, model_name in enumerate(model_names):
        plt.plot(all_train_losses[i], label=model_name)
    plt.title('Training Loss Karşılaştırması')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/training_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, model_name in enumerate(model_names):
        plt.plot(all_val_accs[i], label=model_name)
    plt.title('Validation Accuracy Karşılaştırması')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/validation_accuracy.png')
    plt.close()

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    for i, model_name in enumerate(model_names):
        ax = axes[i] if len(model_names) > 1 else axes
        sns.heatmap(all_cms[i], annot=True, fmt='d', ax=ax, cmap='Blues', 
                    xticklabels=['Podyum', 'Puan', 'Puansız'], 
                    yticklabels=['Podyum', 'Puan', 'Puansız'])
        ax.set_title(f'{model_name} \nConfusion Matrix')
        ax.set_xlabel('Tahmin Edilen')
        ax.set_ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()

def main():
    print("Modeller tanımlanıyor...")
    model_mlp = CustomMLP(input_dim, output_dim)
    model_lstm = SimpleLSTM(input_dim, hidden_dim=64, output_dim=output_dim)
    model_cnn1d = CNN1D(input_dim, output_dim)
    model_transformer = TabularTransformer(input_dim, output_dim)
    
    models = [model_mlp, model_lstm, model_cnn1d, model_transformer]
    model_names = ['Özel MLP', 'LSTM', '1D CNN', 'FT-Transformer']
    
    all_train_losses = []
    all_val_accs = []
    all_cms = []
    metrics_data = []
    
    best_acc = 0.0
    best_model = None
    best_model_name = ""
    best_arch = ""

    for model, name in zip(models, model_names):
        print(f"\n{name} eğitimi başlıyor...")
        train_losses, val_accs = train_model(model, name, epochs=30)
        all_train_losses.append(train_losses)
        all_val_accs.append(val_accs)

        val_acc, val_prec, val_rec, val_f1, _ = evaluate_model(model, name, val_loader, dataset_name="Validation")
        test_acc, test_prec, test_rec, test_f1, cm = evaluate_model(model, name, test_loader, dataset_name="Test")
        all_cms.append(cm)
        
        metrics_data.append({
            'Model': name,
            'Val_Accuracy': round(val_acc, 4),
            'Val_Precision': round(val_prec, 4),
            'Val_Recall': round(val_rec, 4),
            'Val_F1': round(val_f1, 4),
            'Test_Accuracy': round(test_acc, 4),
            'Test_Precision': round(test_prec, 4),
            'Test_Recall': round(test_rec, 4),
            'Test_F1': round(test_f1, 4)
        })

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_model_name = name
            
            if "MLP" in name:
                best_arch = "CustomMLP"
            elif "LSTM" in name:
                best_arch = "SimpleLSTM"
            elif "CNN" in name:
                best_arch = "CNN1D"
            elif "Transformer" in name:
                best_arch = "TabularTransformer"
            else:
                best_arch = "CustomMLP"

    os.makedirs('models', exist_ok=True)

    torch.save(best_model.state_dict(), 'models/best_model.pth')
    print(f"\nEn iyi model: {best_model_name} ({best_acc:.4f} accuracy) 'models/best_model.pth' olarak kaydedildi.")

    feature_cols = train_df.drop('Target_Tier', axis=1).columns.tolist()
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
        
    with open('models/best_model_arch.pkl', 'wb') as f:
        pickle.dump(best_arch, f)
        
    print("Özellik sütunları ve mimari bilgisi kaydedildi.")
        
    print("Sonuçlar grafiklere dökülüyor ve tablo olarak ('results' klasörüne) kaydediliyor...")
    
    metrics_df = pd.DataFrame(metrics_data)

    os.makedirs('results', exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 1 + len(metrics_df) * 0.4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4c72b0')
            cell.set_text_props(color='white')
            
    plt.title('Modellerin Validation ve Test Metrikleri Karşılaştırması', fontweight="bold", pad=10)
    plt.savefig('results/model_evaluation_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plot_results(all_train_losses, all_val_accs, all_cms, model_names)
    print("Tüm işlemler başarıyla tamamlandı!")

if __name__ == '__main__':
    main()