# Titanic Survival Prediction

A machine learning project for predicting passenger survival on the Titanic dataset using scikit-learn.

## Project Overview

This project implements a classification pipeline to predict whether a passenger survived the Titanic disaster based on features such as age, class, fare, and other attributes.

## Project Structure

```
titanic-project/
├── data/
│   ├── raw/                          # Raw dataset (not versioned)
│   │   └── titanic-dataset.csv       # Original Titanic dataset from Kaggle
│   ├── interim/                      # Intermediate processed data
│   │   └── cleaned-titanic-dataset.csv
│   └── processed/                    # Final train/test splits (generated)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/                           # Trained models (not versioned)
│   └── best_model.joblib             # Best trained model
├── notebook/                         # Jupyter notebooks for experimentation
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── data-cleaning.ipynb           # Data cleaning process
│   ├── models-normal-sampling.ipynb  # Model training experiments
│   └── models-stratified-sampling.ipynb
├── src/                              # Source code modules
│   ├── data_loader.py                # Data loading and splitting utilities
│   ├── evaluate.py                   # Model evaluation utilities
│   ├── models.py                     # Model definitions
│   ├── params.py                     # Hyperparameters and configuration
│   ├── paths.py                      # Project path definitions
│   ├── preprocess.py                 # Data preprocessing pipeline
│   └── train.py                      # Main training script
├── .gitignore
├── .pylintrc
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Dataset Setup

Before running the training script, you need to obtain the Titanic dataset from Kaggle:

1. Download the dataset from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
2. Place the downloaded file in `data/raw/` with the name `titanic-dataset.csv`

```bash
mkdir -p data/raw
# Place your titanic-dataset.csv here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Run the training script from the project root:

```bash
python src/train.py
```

This script will:
- Load and preprocess the data from `data/interim/cleaned-titanic-dataset.csv`
- Split data using stratified sampling (80/20)
- Train a voting classifier ensemble
- Save the trained model to `models/best_model.joblib`
- Evaluate the model and display metrics

## Features

- **Modular Architecture**: Separated concerns with dedicated modules for data loading, preprocessing, training, and evaluation
- **Data Preprocessing Pipeline**: Automated feature engineering with median imputation and robust scaling
- **Stratified Sampling**: Balanced train-test split to preserve class distribution
- **Ensemble Model**: Voting Classifier (hard voting) combining DecisionTreeClassifier, SVC, and LogisticRegression
- **Jupyter Notebooks**: Interactive exploration and model experimentation

## Project Modules

- **`src/data_cleaning.py`**: Data cleaning and preprocessing of raw data
- **`src/data_loader.py`**: Handles data loading and stratified train-test splitting
- **`src/preprocess.py`**: Data preprocessing pipeline with imputation and scaling
- **`src/models.py`**: Model definitions and configurations
- **`src/train.py`**: Main training script
- **`src/evaluate.py`**: Model evaluation and metrics
- **`src/paths.py`**: Centralized path management
- **`src/params.py`**: Hyperparameters and configuration settings

## Dependencies

- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- jupyter >= 1.0.0
- joblib >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Results

Models are evaluated using standard classification metrics (accuracy, precision, recall, F1-score).
