import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Split dataset into train and test"""
    X = df.drop("sales", axis=1)
    y = df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
