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

batch_size = 256
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

input_dim = X_train.shape[1]
output_dim = 3

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
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

class WideAndDeep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WideAndDeep, self).__init__()
        # Deep part
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.wide = nn.Linear(input_dim, output_dim)
        self.out = nn.Linear(64 + output_dim, output_dim)
        
    def forward(self, x):
        deep_out = self.deep(x)
        wide_out = self.wide(x)
        combined = torch.cat([deep_out, wide_out], dim=1)
        return self.out(combined)

def train_model(model, name, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
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

def evaluate_model(model, name):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            outputs = model(X_b)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y_b.numpy())
            y_pred.extend(predicted.numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- {name} Sonuçları (Test Seti Üzerinde) ---")
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, model_name in enumerate(model_names):
        sns.heatmap(all_cms[i], annot=True, fmt='d', ax=axes[i], cmap='Blues', 
                    xticklabels=['Podyum', 'Puan', 'Puansız'], 
                    yticklabels=['Podyum', 'Puan', 'Puansız'])
        axes[i].set_title(f'{model_name} \nConfusion Matrix')
        axes[i].set_xlabel('Tahmin Edilen')
        axes[i].set_ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()

def main():
    print("Modeller tanımlanıyor...")
    model_mlp = CustomMLP(input_dim, output_dim)
    model_lstm = SimpleLSTM(input_dim, hidden_dim=64, output_dim=output_dim)
    model_wide_deep = WideAndDeep(input_dim, output_dim)
    
    models = [model_mlp, model_lstm, model_wide_deep]
    model_names = ['Özel MLP (Model 1)', 'LSTM (Model 2)', 'Wide & Deep (Model 3)']
    
    all_train_losses = []
    all_val_accs = []
    all_cms = []
    
    best_acc = 0.0
    best_model = None
    best_model_name = ""

    for model, name in zip(models, model_names):
        print(f"\n{name} eğitimi başlıyor...")
        train_losses, val_accs = train_model(model, name, epochs=30)
        all_train_losses.append(train_losses)
        all_val_accs.append(val_accs)

        acc, prec, rec, f1, cm = evaluate_model(model, name)
        all_cms.append(cm)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name

    os.makedirs('models', exist_ok=True)

    torch.save(best_model.state_dict(), 'models/best_model.pth')
    
    if "MLP" in best_model_name:
        best_arch = "CustomMLP"
    elif "LSTM" in best_model_name:
        best_arch = "SimpleLSTM"
    else:
        best_arch = "WideAndDeep"

    import pickle
    feature_cols = train_df.drop('Target_Tier', axis=1).columns.tolist()
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
        
    with open('models/best_model_arch.pkl', 'wb') as f:
        pickle.dump(best_arch, f)
        
    print(f"\nEn iyi model: {best_model_name} ({best_acc:.4f} accuracy) 'models/best_model.pth' olarak kaydedildi.")
    print("Özellik sütunları ve mimari bilgisi kaydedildi.")
        
    print("Sonuçlar grafiklere dökülüyor ('results' klasörüne kaydedildi)...")
    plot_results(all_train_losses, all_val_accs, all_cms, model_names)
    print("Tüm işlemler başarıyla tamamlandı!")

if __name__ == '__main__':
    main()