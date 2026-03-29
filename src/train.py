import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import pandas as pd
from src.preprocess import PreprocessData
from src.paths import DATA_INTERIM_PATH, MODELS_PATH
from src.models import model
from src.data_loader import get_split_data
from src.evaluate import EvaluateModel
from sklearn.pipeline import Pipeline
import joblib

dataset = pd.read_csv(DATA_INTERIM_PATH)
X_train, X_test, y_train, y_test = get_split_data(dataset, save=True)

print("Initializing the model...")
full_pipeline = Pipeline(steps=[
  ("preprocessor", PreprocessData(X_train).get_preprocessor()),
  ("voting_clf", model())
])
print("Training the model...")
full_pipeline.fit(X_train, y_train)

print("Saving the model...")
try:
  if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
  joblib.dump(full_pipeline, os.path.join(MODELS_PATH, "best_model.joblib"))
  print("Model saved successfully.")
except Exception as e:
  print(f"Error saving the model: {e}")

print("Evaluating the model...")
evaluator = EvaluateModel()
evaluator.evaluate(X_test, y_test)
