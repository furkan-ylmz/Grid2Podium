import pickle
from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from core.models import get_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

BEST_MODEL_PTH = MODELS_DIR / "best_model.pth"
BEST_MODEL_ARCH_PKL = MODELS_DIR / "best_model_arch.pkl"
FEATURE_COLUMNS_PKL = MODELS_DIR / "feature_columns.pkl"
ENCODERS_PKL = PROCESSED_DATA_DIR / "encoders.pkl"

CLASS_DETAILS = {
    0: {"label": "Podyum! (İlk 3)", "icon": "🏆", "color": "#FFD700"},
    1: {"label": "Puan Alır (İlk 10)", "icon": "✅", "color": "#1E90FF"},
    2: {"label": "Puan Alamaz / Bitiremez (11+)", "icon": "❌", "color": "#A9A9A9"},
}


class RacePredictor:
    """
    Inference helper for Formula 1 race outcome prediction.
    """
    def __init__(
        self,
        model_path: Path = BEST_MODEL_PTH,
        arch_path: Path = BEST_MODEL_ARCH_PKL,
        features_path: Path = FEATURE_COLUMNS_PKL,
        encoders_path: Path = ENCODERS_PKL,
        device: torch.device = torch.device('cpu')
    ):
        self.model_path = model_path
        self.arch_path = arch_path
        self.features_path = features_path
        self.encoders_path = encoders_path
        self.device = device

        self.encoders = None
        self.feature_cols = None
        self.best_arch = None
        self.model = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load encoder dictionary, feature columns list, architecture, and model weights."""
        if not self.encoders_path.exists():
            raise FileNotFoundError(f"Encoders file not found at: {self.encoders_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Feature columns file not found at: {self.features_path}")
        if not self.arch_path.exists():
            raise FileNotFoundError(f"Best model arch file not found at: {self.arch_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Best model weights not found at: {self.model_path}")

        with open(self.encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)

        with open(self.features_path, 'rb') as f:
            self.feature_cols = pickle.load(f)

        with open(self.arch_path, 'rb') as f:
            self.best_arch = pickle.load(f)

        input_dim = len(self.feature_cols)
        self.model = get_model(self.best_arch, input_dim=input_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        track: str,
        driver: str,
        team: str,
        starting_grid: int,
        year: int
    ) -> Dict[str, Any]:
        """
        Run inference on a single race entry.
        Returns:
            Dictionary containing predicted_class, class_label, icon, color, probabilities, and model architecture.
        """
        track_encoded = self.encoders['Track'].transform([track])[0]
        driver_encoded = self.encoders['Driver'].transform([driver])[0]
        team_encoded = self.encoders['Team'].transform([team])[0]

        input_data = {
            'Track': track_encoded,
            'Driver': driver_encoded,
            'Team': team_encoded,
            'Starting Grid': starting_grid,
            'Year': year
        }

        input_df = pd.DataFrame([input_data])
        cat_cols = ['Track', 'Driver', 'Team', 'Year']
        input_df = pd.get_dummies(input_df, columns=cat_cols)

        final_input = pd.DataFrame(columns=self.feature_cols)
        for col in self.feature_cols:
            if col in input_df.columns:
                final_input[col] = input_df[col]
            else:
                final_input[col] = 0

        X_numpy = final_input.values.astype(np.float32)
        X_tensor = torch.tensor(X_numpy, device=self.device)

        with torch.no_grad():
            output = self.model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_class = predicted.item()
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        details = CLASS_DETAILS.get(predicted_class, {"label": "Bilinmeyen", "icon": "❓", "color": "#FFFFFF"})

        return {
            "predicted_class": predicted_class,
            "label": details["label"],
            "icon": details["icon"],
            "color": details["color"],
            "probabilities": {
                "podium": float(probabilities[0]),
                "points": float(probabilities[1]),
                "no_points": float(probabilities[2]),
            },
            "best_arch": self.best_arch,
        }
