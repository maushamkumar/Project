import pandas as pd


class BuildFeatures:
    """
    BuildFeatures takes a dataframe and adds some features to it.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_features(self):
        """
        It takes a dataframe, adds columns to it, and returns the dataframe
        """
        self.df["revenue"] = self.df["price"] * self.df["sales"]
        self.df["profit"] = self.df["revenue"] - self.df["cost"]
        self.df["profit_margin"] = self.df["profit"] / self.df["revenue"]
        self.df["sales_change"] = self.df["sales"].diff()
        return self.df
