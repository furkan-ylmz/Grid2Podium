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
import copy

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
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class ManualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(ManualLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.W_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        gates = torch.matmul(x, self.W_ih) + torch.matmul(h_t, self.W_hh) + self.bias

        i_t, f_t, g_t, o_t = gates.chunk(4, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return self.fc(h_t)

class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            dummy = self.pool2(self.bn2(self.conv2(self.pool1(self.bn1(self.conv1(dummy))))))
            flat_dim = dummy.view(1, -1).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return self.fc(x)

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=3, d_model=32, nhead=2, num_layers=1):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

def train_model(model, name, epochs=100, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_b.size(0)
            correct_train += (predicted == y_b).sum().item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_b, y_b in val_loader:
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_b.size(0)
                correct_val += (predicted == y_b).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early Stopping tetiklendi! En iyi epoch: {epoch - patience}")
            break

    model.load_state_dict(best_model_wts)
    return train_losses, val_losses, train_accuracies, val_accuracies

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

def plot_results(all_train_losses, all_val_losses, all_train_accs, all_val_accs, all_val_cms, all_test_cms, model_names):
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1: axes = [axes]
    for i, model_name in enumerate(model_names):
        axes[i].plot(all_train_losses[i], label='Train Loss')
        axes[i].plot(all_val_losses[i], label='Validation Loss')
        axes[i].set_title(f'{model_name} \nLoss')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('results/loss_curves.png')
    plt.close()

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1: axes = [axes]
    for i, model_name in enumerate(model_names):
        axes[i].plot(all_train_accs[i], label='Train Accuracy')
        axes[i].plot(all_val_accs[i], label='Validation Accuracy')
        axes[i].set_title(f'{model_name} \nAccuracy')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('results/accuracy_curves.png')
    plt.close()

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1: axes = [axes]
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        sns.heatmap(all_val_cms[i], annot=True, fmt='d', ax=ax, cmap='Blues', 
                    xticklabels=['Podyum', 'Puan', 'Puansız'], 
                    yticklabels=['Podyum', 'Puan', 'Puansız'])
        ax.set_title(f'{model_name} \nVal Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/validation_confusion_matrices.png')
    plt.close()

    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1: axes = [axes]
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        sns.heatmap(all_test_cms[i], annot=True, fmt='d', ax=ax, cmap='Blues', 
                    xticklabels=['Podyum', 'Puan', 'Puansız'], 
                    yticklabels=['Podyum', 'Puan', 'Puansız'])
        ax.set_title(f'{model_name} \nTest Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrices.png')
    plt.close()

def main():
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

    for model, name in zip(models, model_names):
        print(f"\n{name} eğitimi başlıyor...")
        train_losses, val_losses, train_accs, val_accs = train_model(model, name, epochs=100)
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        t_acc, t_prec, t_rec, t_f1, _ = evaluate_model(model, name, train_loader, dataset_name="Train")
        v_acc, v_prec, v_rec, v_f1, v_cm = evaluate_model(model, name, val_loader, dataset_name="Validation")
        ts_acc, ts_prec, ts_rec, ts_f1, ts_cm = evaluate_model(model, name, test_loader, dataset_name="Test")
        
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

    os.makedirs('models', exist_ok=True)
    torch.save(best_model.state_dict(), 'models/best_model.pth')
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(train_df.drop('Target_Tier', axis=1).columns.tolist(), f)
    with open('models/best_model_arch.pkl', 'wb') as f:
        pickle.dump(best_arch, f)

    metrics_df = pd.DataFrame(metrics_data).set_index('Model').T
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Metrik / Model'}, inplace=True)

    os.makedirs('results', exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    the_table = ax.table(cellText=metrics_df.values, 
                         colLabels=metrics_df.columns, 
                         loc='center', 
                         cellLoc='center')
    
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

    plt.title('Modellerin Tüm Veri Setleri Üzerindeki Detaylı Performans Karşılaştırması', fontweight="bold", pad=20)
    plt.savefig('results/model_evaluation_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plot_results(all_train_losses, all_val_losses, all_train_accs, all_val_accs, all_val_cms, all_test_cms, model_names)
    print("\nTüm işlemler başarıyla tamamlandı!")

if __name__ == '__main__':
    main()
