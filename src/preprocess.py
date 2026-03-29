from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np


class PreprocessData:
  def __init__(self, X_train):
    self.numeric_features = X_train.select_dtypes(include=np.number).columns
    self.categorical_features = X_train.select_dtypes(include="object").columns
  
  def get_preprocessor(self):
    categorical_transformer = Pipeline(steps=[
      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric_transformer = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", RobustScaler())
    ])

    preprocessor = ColumnTransformer(
      transformers=[
        ("numeric", numeric_transformer, self.numeric_features),
        ("categorical", categorical_transformer, self.categorical_features)
      ]
    )

    return preprocessor