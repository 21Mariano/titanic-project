import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import pandas as pd
from src.paths import DATA_RAW_PATH, DATA_INTERIM_PATH


def run_data_cleaning():
  if not os.path.exists(DATA_RAW_PATH):
    raise FileNotFoundError(f"Raw dataset not found: {DATA_RAW_PATH}")

  print(f"Loading raw data from {DATA_RAW_PATH}")
  dataset = pd.read_csv(DATA_RAW_PATH)

  print("Cleaning data...")
  dataset.columns = [col.lower().replace(' ', '_') for col in dataset.columns]

  dataset = dataset.drop(columns=['passengerid', 'name', 'ticket', 'cabin'])

  dataset = dataset.dropna(subset=['embarked']).reset_index(drop=True)

  os.makedirs(os.path.dirname(DATA_INTERIM_PATH), exist_ok=True)
  dataset.to_csv(DATA_INTERIM_PATH, index=False)
  print(f"Cleaned data saved to {DATA_INTERIM_PATH}")