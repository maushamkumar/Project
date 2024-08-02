import pandas as pd
import statsmodels.api as sm


class Predict:
    def __init__(self, input_data: pd.DataFrame, model: sm.OLS) -> None:
        self.input_data = input_data
        self.model = model

    def predict(self) -> pd.DataFrame:
        """Predict the input data"""
        self.input_data = sm.add_constant(self.input_data)
        y_pred = self.model.predict(self.input_data)
        return y_pred


if __name__ == "__main__":
    pass
