import logging

import pandas as pd
import statsmodels.api as sm

from make_dataset import Ingestor


class ModelBuilder:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def build_model(self, summary: bool = True):
        """Build model using statsmodels"""
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X.astype(float)).fit()
        if summary:
            logging.info(model.summary())
        return model


if __name__ == "__main__":
    ingestor = Ingestor(file_name="data.csv")
    df = ingestor.load_dataset()
    X = df.drop("sales", axis=1)
    y = df["sales"]
    print(X.head())
    print(y.head())
    builder = ModelBuilder(X=X, y=y)
    model = builder.build_model()
    
