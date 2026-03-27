from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np


class PreprocessData:
  def __init__(self, dataset, stratified=False):
    self.dataset = dataset
    self.stratified = stratified

  def _stratified_split(self):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(self.dataset, self.dataset["survived"]):
      X_train = self.dataset.loc[train_index].drop(columns=["survived"])
      y_train = self.dataset.loc[train_index]["survived"]
      X_test = self.dataset.loc[test_index].drop(columns=["survived"])
      y_test = self.dataset.loc[test_index]["survived"]
      return X_train, X_test, y_train, y_test
    raise RuntimeError("No split was created by StratifiedShuffleSplit")
  def _train_test_split(self):
    X = self.dataset.drop(columns=["survived"])
    y = self.dataset["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
  
  def fit_transform(self):
    if self.stratified:
      X_train, X_test, y_train, y_test = self._stratified_split()
    else:
      X_train, X_test, y_train, y_test = self._train_test_split()

    numeric_features = X_train.select_dtypes(include=np.number).columns
    categorical_features = X_train.select_dtypes(include="object").columns

    categorical_transformer = Pipeline(steps=[
      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric_transformer = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", RobustScaler())
    ])

    preprocessor = ColumnTransformer(
      transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
      ]
    )

    preprocessor.fit(X_train)
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test