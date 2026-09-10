# Climate-based GLM
import functools
import itertools
import warnings
from collections import namedtuple
import numpy as np
import pandas as pd
from dataloader.loc_core import CLIMATE_CSV, CLIMATE_FEATURES, FUTURE_SOURCES, read_clean
from dataloader.location_data import future_climate


# ##################### Build forms ######################################
WINDOWS = (0, 1, 2, 3, 4, 5, 6)
GlmForm = namedtuple("GlmForm", "feats window")


def feature_subsets(feats=CLIMATE_FEATURES):
    """Every non-zero subset: ("T2M",), ..., ("T2M", "PRECTOTCORR_SUM", "QV2M")."""
    return [tuple(c) for k in range(len(feats) + 1) for c in itertools.combinations(feats, k) if len(c) > 0]


def glm_forms(feats=CLIMATE_FEATURES):
    """Every form the GLM chooses between"""
    return [GlmForm(subset, window)
            for subset in feature_subsets(feats)
            for window in WINDOWS]


def form_tag(form):
    """e.g., GlmForm(("T2M", "QV2M"), 3) -> "T2M+QV2M@0-3" """
    span = "0" if form.window == 0 else f"0-{form.window}"
    return f"{'+'.join(form.feats)}@{span}"


# ##################### Select the best form based on training data ######################################
def select_form(whole_interval_frame, stats, location, max_horizon=6,
                feats=CLIMATE_FEATURES, trace=None):
    from .baseline_utils import refit_roll, select_by_rolling_error

    return select_by_rolling_error(
        glm_forms(feats),
        refit_roll(lambda form: functools.partial(NBGLM, form=form,
                                                  whole_interval_frame=whole_interval_frame,
                                                  stats=stats, location=location),
                   whole_interval_frame, max_horizon, predict_kwargs={"scenario": "observed"}),
        form_tag,
        whole_interval_frame, max_horizon, what="GLM", trace=trace)


# ##################### Collect future climates info ######################################
def check_scenario(scenario):
    """observed | climatology | forecast."""
    if scenario not in FUTURE_SOURCES:
        raise ValueError(f"unknown climate scenario '{scenario}'; known: {list(FUTURE_SOURCES)}")
    return scenario


@functools.lru_cache(maxsize=None)
def _future_raw(location, scenario, origin_date, steps, feats):
    """future_climate, get the base climate info for

        origin_date+1, ..., origin_date+steps

    for three scenarios: observed | climatology | forecast
    """
    return future_climate(location, scenario, origin_date, steps, base_feats=list(feats))


# ##################### Collect historical climates info ######################################
def pre_record_climate(location, first_date, feats, months):
    if months <= 0:
        return {f: pd.Series(dtype=float) for f in feats}

    dates = pd.date_range(end=pd.Timestamp(first_date) - pd.DateOffset(months=1), periods=months, freq="MS")
    src = read_clean(location, CLIMATE_CSV)
    missing = [f for f in feats if f not in src.columns]
    if missing:
        raise ValueError(f"[{location}] {CLIMATE_CSV} does not carry {missing}")

    src = src.assign(date=pd.to_datetime(src["date"])).set_index("date").reindex(dates)
    return {f: pd.Series(src[f].to_numpy(dtype=float), index=range(-months, 0)) for f in feats}


def observed_climate(whole_interval_frame, feats, origin, location, window=0):
    history_climate_info = whole_interval_frame.iloc[:origin + 1]
    seen = {f: pd.Series(history_climate_info[f].to_numpy(dtype=float),
                         index=history_climate_info.index)
            for f in feats}
    if window <= 0:
        return seen

    before = pre_record_climate(location, whole_interval_frame["date"].iloc[0], feats, window)
    return {f: pd.concat([before[f], seen[f]]) for f in feats}


# ##################### Combine both historical & future climates ######################################
def climate_through_horizon(whole_interval_frame, feats, origin, location, scenario, steps, window=0):
    out = observed_climate(whole_interval_frame, feats, origin, location, window)
    if not steps:
        return out

    future_climate_info = _future_raw(location, scenario,
                                      str(whole_interval_frame["date"].iloc[origin]),
                                      steps, tuple(feats))
    for f in feats:
        ahead = pd.Series(future_climate_info[f].to_numpy(dtype=float),
                          index=range(origin + 1, origin + 1 + len(future_climate_info)))
        out[f] = pd.concat([out[f], ahead])
    return out


# ##################### Design X ######################################
def design_matrix(interval_row_indices, feats, window, climate, stats):
    """Regressors for these interval rows"""
    X = pd.DataFrame({"const": 1.0}, index=interval_row_indices)
    for f in feats:
        mean, std = stats[f]
        for k in range(window + 1):
            lagged = climate[f].reindex(interval_row_indices - k).to_numpy(dtype=float)
            X[f if k == 0 else f"{f}_lag{k}"] = (lagged - mean) / std
    return X


def fit_design(history, form, stats, whole_interval_frame, location):
    origin = history.index[-1]
    climate = observed_climate(whole_interval_frame, form.feats, origin, location, form.window)
    X = design_matrix(history.index, form.feats, form.window, climate, stats)
    return X, X.notna().all(axis=1) & history["cases"].notna()


# ##################### FIT MODEL ######################################
def fit_nb(y, X):
    """MLE fit of a NB2 GLM; Falls back to Poisson when the NB likelihood not converge"""
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0)
            alpha = float(np.asarray(res.params)[-1])
            if res.mle_retvals.get("converged") and np.isfinite(alpha) and alpha > 0:
                return res
            print(f"  NB GLM did not converge (alpha={alpha:.3g}); using Poisson")
        except Exception as exc:
            print(f"  NB GLM did not fit ({type(exc).__name__}: {exc}); using Poisson")

        return sm.GLM(y, X, family=sm.families.Poisson()).fit()


# ##################### Main class ######################################
class NBGLM:
    def __init__(self, form, whole_interval_frame, stats, location):
        self.form = form
        self.whole_interval_frame, self.stats = whole_interval_frame, stats
        self.location = location
        self.name = f"glm_{form_tag(form)}"
        self.res, self.origin = None, None

    def fit(self, history):
        self.origin = history.index[-1]
        X, keep = fit_design(history, self.form, self.stats, self.whole_interval_frame, self.location)
        if int(keep.sum()) <= X.shape[1]:
            raise ValueError(f"the history is too short.")
        self.res = fit_nb(history.loc[keep, "cases"].to_numpy(dtype=float), X.loc[keep])
        return self

    def predict(self, future_interval_indices, scenario="observed"):
        climate = climate_through_horizon(self.whole_interval_frame, self.form.feats, self.origin,
                                          self.location, scenario, len(future_interval_indices), self.form.window)
        X = design_matrix(future_interval_indices, self.form.feats, self.form.window, climate, self.stats)
        return np.asarray(self.res.predict(X), dtype=float)
