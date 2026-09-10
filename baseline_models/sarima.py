# SARIMA
import functools
import warnings
import numpy as np
from itertools import product


CANDIDATE_ORDERS = [
    (
        (p, d, 0),
        (P, D, 0, 12),
    )
    for p, d, P, D in product(
        [1, 2, 3],
        [0, 1],
        [0, 1, 2, 3],
        [0, 1],
    )
]


def build_sarimax(y_log, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    trend = "c" if order[1] == 0 and seasonal_order[1] == 0 else None
    return SARIMAX(y_log, order=order, seasonal_order=seasonal_order, trend=trend)


def check_history(model, n):
    if n <= model.k_states:
        raise ValueError("history is too short")


def fit_sarimax(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.fit(disp=0)
        except Exception:
            return None


def order_tag(order, seasonal_order):
    return f"{tuple(order)}x{tuple(seasonal_order)}".replace(" ", "")


def select_order(whole_interval_frame, max_horizon=6, trace=None):
    from .baseline_utils import refit_roll, select_by_rolling_error

    return select_by_rolling_error(
        CANDIDATE_ORDERS,
        refit_roll(lambda c: functools.partial(SARIMA, order=c[0], seasonal_order=c[1]),
                   whole_interval_frame, max_horizon),
        lambda c: order_tag(c[0], c[1]),
        whole_interval_frame, max_horizon, what="SARIMA", trace=trace)


class SARIMA:

    name = "sarima"

    def __init__(self, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), point="median"):
        self.order, self.seasonal_order, self.point = order, seasonal_order, point
        self.res = None

    def fit(self, history):
        y_log = np.log1p(history["cases"].to_numpy(dtype=float))
        model = build_sarimax(y_log, self.order, self.seasonal_order)
        check_history(model, int(np.isfinite(y_log).sum()))
        self.res = fit_sarimax(model)
        if self.res is None:
            raise RuntimeError(f"model failed")
        return self

    def predict(self, future_interval_indices):
        forecast = self.res.get_forecast(steps=len(future_interval_indices))

        # ------- predicted results calculation -------------------
        m = np.asarray(forecast.predicted_mean, dtype=float)
        v = np.clip(np.asarray(forecast.var_pred_mean, dtype=float), 0, None)

        if self.point == "median":
            centre = np.exp(m)
        elif self.point == "mean":
            centre = np.exp(m + v / 2)
        else:
            raise ValueError(f"point must be 'median' or 'mean', but got {self.point}")
        return np.clip(centre - 1.0, 1e-8, None)
