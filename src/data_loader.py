import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

from sklearn.model_selection import StratifiedShuffleSplit
from src.params import SEED
from src.paths import DATA_PROCESSED_PATH

def get_split_data(dataset, save=False):
  """Carga el dataset y realiza la partición estratificada."""
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
  
  for train_index, test_index in split.split(dataset, dataset["survived"]):
    X_train = dataset.loc[train_index].drop(columns=["survived"])
    y_train = dataset.loc[train_index]["survived"]
    X_test = dataset.loc[test_index].drop(columns=["survived"])
    y_test = dataset.loc[test_index]["survived"]

  if save:
    if not os.path.exists(DATA_PROCESSED_PATH):
      os.makedirs(DATA_PROCESSED_PATH)
    X_train.to_csv(os.path.join(DATA_PROCESSED_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_PROCESSED_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_PROCESSED_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_PROCESSED_PATH, "y_test.csv"), index=False)
    print(f"Datasets saved in {DATA_PROCESSED_PATH}")

  return X_train, X_test, y_train, y_test