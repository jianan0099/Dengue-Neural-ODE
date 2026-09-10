# Seasonal naive
import numpy as np
from .baseline_utils import SEASON


class SeasonalNaive:

    name = "seasonal_naive"

    def __init__(self, season=SEASON):
        self.season = season
        self.cases = None

    def fit(self, history):
        self.cases = history["cases"]
        return self

    def predict(self, future_interval_indices):
        return self.cases.loc[np.asarray(future_interval_indices) - self.season].to_numpy(dtype=float)
