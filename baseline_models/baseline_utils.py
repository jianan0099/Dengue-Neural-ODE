import numpy as np
import pandas as pd

SEASON = 12
MIN_SELECT_ORIGINS = 6


# ########################## Data preparation #########################################
def interval_frame(record, feats=()):
    """anchor row dropped: record row 1 -> frame row 0"""
    frame = record.iloc[1:].reset_index(drop=True)
    missing = [f for f in feats if f not in frame.columns]
    if missing:
        raise KeyError(f"feature(s) {missing} not in the data record")

    return pd.DataFrame({
        "date": frame["date"].to_numpy(),
        "cases": frame["cases"].to_numpy(dtype=float),
        "pop": frame["pop"].to_numpy(dtype=float),
        "month": pd.to_datetime(frame["date"]).dt.month.to_numpy(),
        "split": frame["split"].to_numpy(),
        **{f: frame[f].to_numpy(dtype=float) for f in feats},
    })


def train_end(whole_interval_frame):
    """e.g., 48 train + 60 test -> 47"""
    test_rows = np.flatnonzero((whole_interval_frame["split"] == "test").to_numpy())
    return int(test_rows[0]) - 1


def test_origins(whole_interval_frame):
    """e.g., 48 train + 60 test -> origins 47..106"""
    first = train_end(whole_interval_frame)
    if first < SEASON:
        raise ValueError(f"no enough history data")
    return list(range(first, len(whole_interval_frame) - 1))


# ########################## Model selection window set #########################################
def selection_min_history(n_train):
    min_history = max(SEASON, min(3 * SEASON, n_train - MIN_SELECT_ORIGINS))
    n_origins = n_train - min_history
    if n_origins < MIN_SELECT_ORIGINS:
        raise ValueError("not eligible")
    return min_history


def model_selection_window(whole_interval_frame):
    """e.g., 48 train + 60 test -> origins 35..46, last_row 47"""
    last_train_row = train_end(whole_interval_frame)
    min_history = selection_min_history(last_train_row + 1)
    origins = list(range(min_history - 1, last_train_row))
    return origins, last_train_row


# ########################## Model selection #########################################
def select_by_rolling_error(candidates, roll, label, whole_interval_frame, max_horizon, what="model", trace=None):
    origins, last_row = model_selection_window(whole_interval_frame)
    candidates = list(candidates)
    kept = roll_candidates(candidates, roll, label, origins, last_row, what, trace)
    return rank_candidates(kept, max_horizon, len(origins), what, trace)


def roll_candidates(candidates, roll, label, origins, last_row, what="model", trace=None):
    def gone(tag, reason):
        if trace is not None:
            trace.drop(tag, reason)

    kept = []
    for candidate in candidates:
        tag = label(candidate)
        try:
            table = roll(candidate, origins, last_row)
        except (ValueError, RuntimeError) as exc:
            gone(tag, str(exc))
            continue

        fitted = table["unfittable"].eq("")
        owed = fitted & table["actual"].notna()
        diverged = int((owed & ~np.isfinite(table["pred"])).sum())
        if diverged:
            gone(tag, f"{diverged} non-finite forecast(s)")
            continue

        err = table.dropna(subset=["abs_err"])
        reached = int(err["origin_date"].nunique())
        if reached < MIN_SELECT_ORIGINS:
            gone(tag, "not eligible")
            continue

        if trace is not None:
            trace.keep(tag, table)
        kept.append((candidate, tag, table, reached))

    if not kept:
        raise RuntimeError(f"no {what} candidate could be fitted on this history.")
    return kept


def rank_candidates(kept, max_horizon, n_origins, what="model", trace=None):
    finished = []
    for candidate, tag, table, _ in kept:
        whole = table[table["horizon"] == max_horizon].dropna(subset=["cum_sq_err"])
        if whole.empty:
            if trace is not None:
                trace.drop(tag, "no complete window")
            continue
        finished.append((candidate, tag, whole))

    common = set.intersection(*(set(w["origin_date"]) for _, _, w in finished)) if finished else set()
    if not common:
        raise RuntimeError(f"{what}: no shared complete window")

    scored = [(float(np.sqrt(w[w["origin_date"].isin(common)]["cum_sq_err"].mean())), c, t, len(w))
              for c, t, w in finished]
    value, candidate, tag, own = min(scored, key=lambda t: t[0])
    if trace is not None:
        trace.select(tag)
    return candidate


# ########################## Rolling forecast ######################################
def refit_roll(factory, whole_interval_frame, max_horizon, predict_kwargs=None):
    return lambda candidate, origins, last_row: rolling_forecast(
        factory(candidate), whole_interval_frame, max_horizon,
        origins=origins, last_row=last_row,
        predict_kwargs=predict_kwargs, skip_unfittable=True)


def rolling_forecast(model_factory, whole_interval_frame, max_horizon,
                     origins=None, last_row=None, predict_kwargs=None, skip_unfittable=False):
    origins = test_origins(whole_interval_frame) if origins is None else origins
    last_row = len(whole_interval_frame) - 1 if last_row is None else last_row

    rows = []
    for origin in origins:
        last_future_interval_index = min(origin + max_horizon, last_row)
        history = whole_interval_frame.iloc[:origin + 1]
        future = whole_interval_frame.iloc[origin + 1:last_future_interval_index + 1]

        model = model_factory()
        unfittable = ""
        try:
            model.fit(history)
            mu = np.asarray(model.predict(future.index, **(predict_kwargs or {})), dtype=float)
        except (ValueError, RuntimeError) as exc:
            if not skip_unfittable:
                raise
            unfittable = f"{type(exc).__name__}: {exc}"
            mu = np.full(len(future), np.nan)

        rows += forecast_rows(whole_interval_frame, origin, future, mu, unfittable)

    return forecast_table(rows)


def forecast_rows(whole_interval_frame, origin, future, mu, unfittable=""):
    """one row per horizon"""
    rows = []
    for h, (_, target) in enumerate(future.iterrows(), start=1):
        obs, pred = target["cases"], float(mu[h - 1])
        scored = bool(np.isfinite(obs) and np.isfinite(pred))
        err = np.float64(pred) - np.float64(obs)
        rows.append({
            "origin_date": whole_interval_frame["date"].iloc[origin],
            "horizon": h,
            "target_date": target["date"],
            "origin_split": whole_interval_frame["split"].iloc[origin],
            "target_split": target["split"],
            "unfittable": unfittable,
            "actual": float(obs),
            "pred": pred,
            "abs_err": abs(err) if scored else np.nan,
            "sq_err": err ** 2 if scored else np.nan,
        })
    return rows


def forecast_table(rows):
    df = pd.DataFrame(rows)
    return df if df.empty else add_cumulative(df)


def add_cumulative(df):
    df = df.sort_values(["origin_date", "horizon"]).reset_index(drop=True)
    voided = df["sq_err"].isna().groupby(df["origin_date"]).cummax().astype(bool)
    df["cum_sq_err"] = df.groupby("origin_date")["sq_err"].cumsum()
    df.loc[voided, "cum_sq_err"] = np.nan
    return df
