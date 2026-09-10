"""Location config"""
from pathlib import Path
import numpy as np
import pandas as pd

LOC_DATA_ROOT = Path(__file__).resolve().parent / "data_by_loc"

LOCATIONS = {
    "Goa": {"dir": "Goa", "epi_base": "Goa_v1"},
    "Bengaluru": {"dir": "Bengaluru", "epi_base": "Bengaluru_v1"},
    "SanJuan": {"dir": "SanJuan", "epi_base": "SanJuan_v1"},
}

CLIMATE_FEATURES = ["T2M", "PRECTOTCORR_SUM", "QV2M"]

CLIMATE_CSV, CLIMATOLOGY_CSV, FORECAST_CSV = "climate.csv", "climate_climatology.csv", "climate_forecast.csv"

_REQUIRED_COLS = {
    CLIMATE_CSV: ("date",),
    CLIMATOLOGY_CSV: ("date", "window_years"),
    FORECAST_CSV: ("issue_date", "date"),
}
FUTURE_SOURCES = {"observed": CLIMATE_CSV, "climatology": CLIMATOLOGY_CSV, "forecast": FORECAST_CSV}
ISO_DATE = "%Y-%m-%d"
_NON_FEATURE_COLS = {"date", "step", "issue_date", "member", "window_years", "T2MDEW", "PSFC", "MSL"}

_CLEAN_CACHE = {}


def _stat_key(path):
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def read_clean(location, name):
    path = LOC_DATA_ROOT / LOCATIONS[location]["dir"] / name
    key = (location, name)

    stat_key = _stat_key(path)
    cached = _CLEAN_CACHE.get(key)
    if cached is not None and (stat_key is None or cached[0] == stat_key):
        return cached[1].copy()
    if stat_key is None:
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLS.get(name, ()) if c not in df.columns]
    if missing:
        raise ValueError(f"[{location}] {name} missing {missing}")
    _CLEAN_CACHE[key] = (stat_key, df)
    return df.copy()


def _period_ords(dates, where=""):
    try:
        dt = pd.to_datetime(dates, format=ISO_DATE)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{where}dates must be {ISO_DATE}") from e
    return pd.PeriodIndex(dt, freq="M").astype(np.int64).to_numpy()


def _periods_after(origin, n):
    return pd.DatetimeIndex([pd.Timestamp(origin) + pd.DateOffset(months=k) for k in range(1, n + 1)])
