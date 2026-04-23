# Phishing Email Detection System

## Project Overview
This project detects whether an email is safe or phishing by using the MeAJOR dataset and three deep learning models trained on the same data. The repository keeps the same workflow as the original project:

1. preprocess the raw dataset
2. train and compare three models
3. launch a Streamlit demo for inference

## Dataset
- Source: MeAJOR (Merged email Assets from Joint Open-source Repositories)
- File location expected by the project:
  - `datasets/phishing_email/meajor_cleaned_preprocessed.csv`
- Target label:
  - `0` -> Safe Email
  - `1` -> Phishing Email

## Features Used
The project does not need every column from the dataset. It focuses on the fields that are most useful for phishing detection:

- Text features:
  - `subject`
  - `body`
- Structural features:
  - `url_count`
  - `url_length_max`
  - `url_length_avg`
  - `url_subdom_max`
  - `url_subdom_avg`
  - `attachment_count`
  - `has_attachments`
  - `content_types`
  - `language`
- Derived features:
  - text length
  - word count
  - suspicious keyword count
  - uppercase ratio
  - digit ratio
  - punctuation counts

## Models
The training script compares three models on the same dataset:

1. `Custom Mean Embedding + MLP`
2. `TextCNN + Metadata`
3. `BiLSTM + Metadata`

The best model on validation F1-score is saved for the Streamlit app.

## Project Structure
- `data_preprocessing.py`: reads the MeAJOR CSV, cleans the data, derives features, and writes train/validation/test splits
- `train_models.py`: trains three deep learning models, compares metrics, and saves the best model
- `app.py`: Streamlit interface for phishing email prediction
- `phishing_utils.py`: shared preprocessing, encoding, and model definitions
- `datasets/phishing_email/`: raw dataset location
- `processed_data/`: generated train/validation/test CSV files
- `models/`: saved model weights and preprocessing assets
- `results/`: plots and metric summaries

## Installation
Install the main dependencies:

```bash
pip install torch pandas numpy matplotlib seaborn streamlit
```

## Usage
1. Prepare the dataset:

```bash
python data_preprocessing.py
```

2. Train the models:

```bash
python train_models.py
```

3. Start the demo:

```bash
streamlit run app.py
```

## Quick Smoke Test
For a faster local check before full training:

```bash
python data_preprocessing.py --max-samples 5000
python train_models.py --max-samples 5000 --epochs 2
```

## Outputs
After training, the project produces:

- `results/training_loss.png`
- `results/validation_accuracy.png`
- `results/confusion_matrices.png`
- `results/metrics_summary.csv`
- `models/best_phishing_model.pth`
- `models/phishing_assets.pkl`
