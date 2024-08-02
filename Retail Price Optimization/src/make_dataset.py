import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import DATA_DIR

# from config import config

warnings.filterwarnings("ignore")


class Ingestor:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from data/raw folder"""
        file_path = Path(DATA_DIR, self.file_name)
        df = pd.read_csv(file_path)
        return df


class LabelEncoder(object):
    def __init__(self, df: pd.DataFrame, cat_cols: list):
        self.df = df
        self.cat_cols = cat_cols

    def fit(self):
        self.cat_dict = {}
        for col in self.cat_cols:
            self.cat_dict[col] = {k: v for v, k in enumerate(self.df[col].unique(), 0)}
        return self

    def transform(self):
        df = self.df.copy()
        for col in self.cat_cols:
            df[col] = df[col].map(self.cat_dict[col])
        return df

    def fit_transform(self):
        return self.fit().transform()

    def inverse_transform(self, df: pd.DataFrame):
        for col in self.cat_cols:
            df[col] = df[col].map({v: k for k, v in self.cat_dict[col].items()})
        return df

    def get_feature_names(self):
        return self.cat_cols

    def get_params(self, deep=True):
        return {"cat_cols": self.cat_cols}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class ProcessData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_null_values(self):
        df = self.df.copy()
        df = df.dropna()
        return df


if __name__ == "__main__":
    pass
