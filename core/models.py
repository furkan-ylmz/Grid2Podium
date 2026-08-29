import torch
import torch.nn as nn


class CustomMLP(nn.Module):
    """
    Multi-Layer Perceptron architecture with Batch Normalization, ReLU and Dropout.
    """
    def __init__(self, input_dim: int, output_dim: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleLSTM(nn.Module):
    """
    PyTorch built-in LSTM based model for tabular feature sequences.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class ManualLSTM(nn.Module):
    """
    Custom LSTM implementation without torch.nn.LSTM, detailing gates logic manually.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

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
    """
    1D Convolutional Neural Network for tabular feature representations.
    """
    def __init__(self, input_dim: int, output_dim: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return self.fc(x)


class TabularTransformer(nn.Module):
    """
    Tabular Transformer architecture utilizing self-attention mechanism.
    """
    def __init__(self, input_dim: int, output_dim: int = 3, d_model: int = 32, nhead: int = 2, num_layers: int = 1):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)


MODEL_REGISTRY = {
    "CustomMLP": CustomMLP,
    "SimpleLSTM": SimpleLSTM,
    "ManualLSTM": ManualLSTM,
    "CNN1D": CNN1D,
    "TabularTransformer": TabularTransformer,
}

MODEL_DISPLAY_NAMES = {
    "CustomMLP": "Özel MLP",
    "SimpleLSTM": "Hazır LSTM",
    "ManualLSTM": "Manuel LSTM",
    "CNN1D": "1D CNN",
    "TabularTransformer": "FT-Transformer",
}


def get_model(arch_name: str, input_dim: int, output_dim: int = 3, **kwargs) -> nn.Module:
    """
    Factory function to instantiate models by architecture name.
    """
    if arch_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch_name](input_dim=input_dim, output_dim=output_dim, **kwargs)
