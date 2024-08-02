import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Evaluate:
    def __init__(self, model, x, y) -> None:
        self.model = model
        self.x = x
        self.y = y

    def evaluate(self) -> None:
        """Evaluate the model"""
        self.x = sm.add_constant(self.x)
        y_pred = self.model.predict(self.x)
        print(f"R2 score: {r2_score(self.y, y_pred)}")
        print(f"Mean absolute error: {mean_absolute_error(self.y, y_pred)}")
        print(f"Mean squared error: {mean_squared_error(self.y, y_pred)}")
        print(f"Root mean squared error: {np.sqrt(mean_squared_error(self.y, y_pred))}")
        print(
            f"Mean absolute percentage error: {np.mean(np.abs((self.y - y_pred) / self.y)) * 100}"
        )
        print(f"Explained variance score: {explained_variance_score(self.y, y_pred)}")
        print(f"Max error: {max_error(self.y, y_pred)}")
