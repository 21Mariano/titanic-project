# Titanic Project

## Run
python src/train.py

## Prerequisites
- `data/raw/titanic-dataset.csv` from Kaggle Titanic competition

## Output
- `models/best_model.joblib` - trained model

## Pipeline
Voting Classifier (hard voting) with:
- DecisionTreeClassifier (entropy, max_depth=7, max_features=sqrt)
- SVC (rbf kernel, C=1)
- LogisticRegression (l2 penalty, C=0.01, liblinear solver)

## Scripts
- `src/train.py` - Main training script
- `src/evaluate.py` - Model evaluation
- `src/data_cleaning.py` - Data cleaning
- `src/preprocess.py` - Preprocessing pipeline
- `src/models.py` - Model definitions
- `src/data_loader.py` - Data loading and splitting
- `src/params.py` - Model hyperparameters
- `src/paths.py` - Path configurations