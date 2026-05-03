import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

from src.paths import MODELS_PATH, DATA_PROCESSED_PATH
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib

class EvaluateModel:
  def __init__(self, model_path=os.path.join(MODELS_PATH, "best_model.joblib")):
    self.model_path = model_path
    self.model = None

  def _load_model(self):
    try:
      self.model = joblib.load(self.model_path)
      print("Model loaded successfully.")
    except Exception as e:
      print(f"Error loading the model: {e}")
      self.model = None

  def _report(self, X_test, y_test):
    y_pred = self.model.predict(X_test)

    return {
      "classification_report": classification_report(y_test, y_pred, output_dict=True),
      "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

  def evaluate(self, X_test=None, y_test=None):
    if self.model is None:
      self._load_model()
    if X_test is None or y_test is None:
      try:
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "y_test.csv"))
      except Exception as e:
        print(f"Error loading test data: {e}")
        return None

    if self.model is not None:
      report = self._report(X_test, y_test)
      print("\n" + "="*30)
      print(" MODEL EVALUATION REPORT ")
      print("="*30 + "\n")
      print("Classification Report: ")
      print(pd.DataFrame(report["classification_report"]).transpose().round(2))
      print("\n" + "Confusion Matrix: ")
      print(report["confusion_matrix"])
    else:
      print("Model evaluation failed due to loading issues.")
      return None
    
if __name__ == "__main__":
  evaluation = EvaluateModel()
  evaluation.evaluate()