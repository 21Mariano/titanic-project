# Titanic Survival Prediction

A machine learning project for predicting passenger survival on the Titanic dataset using scikit-learn.

## Project Overview

This project implements a classification pipeline to predict whether a passenger survived the Titanic disaster based on features such as age, class, fare, and other attributes.

## Project Structure

```
titanic-project/
├── data/                             # Dataset directory
│   ├── titanic-dataset.csv           # Original raw dataset
│   └── cleaned-titanic-dataset.csv   # Preprocessed dataset
├── models/                           # Trained model artifacts
│   ├── voting_clf_normal_sampling.joblib
│   ├── voting_clf_stratified_sampling.joblib
│   └── rf_clf_normal_sampling.joblib
├── best-models/                      # Best model notebooks and results
│   ├── best-voting-clf.ipynb
│   └── best-voting-clf-stratified.ipynb
├── models-tested/                    # Initial model exploration notebooks
│   ├── models-normal-sampling.ipynb
│   └── models-stratified-sampling.ipynb
├── data-cleaning.ipynb               # EDA and data cleaning
├── preprocess.py                     # Data preprocessing utility
├── requirements.txt                  # Project dependencies
└── README.md
```

## Features

- **Data Preprocessing**: Automated feature engineering with missing value imputation and scaling
- **Two Sampling Strategies**: 
  - Normal random train-test split
  - Stratified sampling for balanced distribution
- **Ensemble Model**: Voting classifier combining multiple algorithms
- **Jupyter Notebooks**: Interactive exploration and model evaluation

## Data Preprocessing Pipeline

The `preprocess.py` module provides:

- **Numeric Features**: Median imputation + RobustScaler
- **Categorical Features**: OneHotEncoding
- **Data Splitting**: Stratified or random split (80/20)

```python
from preprocess import PreprocessData

# Load your data
preprocessor = PreprocessData(dataset, stratified=True)
X_train, X_test, y_train, y_test = preprocessor.fit_transform()
```

## Model Training

Two models are trained and saved:
- `voting_clf_normal_sampling.joblib` - Using random train-test split
- `voting_clf_stratified_sampling.joblib` - Using stratified sampling

Both models use a voting ensemble combining multiple classifiers.

## Dependencies

- scikit-learn
- pandas
- numpy
- jupyter
- joblib

## Notebooks

1. **data-cleaning.ipynb**: Data exploration, cleaning, and visualization
2. **models-tested/models-normal-sampling.ipynb**: Model training with standard split
3. **models-tested/models-stratified-sampling.ipynb**: Model training with stratified approach
4. **best-models/best-voting-clf.ipynb**: Final voting classifier (normal sampling)
5. **best-models/best-voting-clf-stratified.ipynb**: Final voting classifier (stratified sampling)

## Usage

1. Run data cleaning: `jupyter notebook data-cleaning.ipynb`
2. Train models: `jupyter notebook models-normal-sampling.ipynb`
3. Compare strategies: `jupyter notebook models-stratified-sampling.ipynb`

## Results

Models are evaluated using standard classification metrics (accuracy, precision, recall, F1-score).
