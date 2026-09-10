import numpy as np
import pandas as pd
import torch

from dataloader.loc_core import (
    LOCATIONS, LOC_DATA_ROOT, read_clean, _period_ords,
    FUTURE_SOURCES, _NON_FEATURE_COLS, _periods_after, ISO_DATE,
)


# ##################### For observed features ######################################
def train_stats(record, base_feats):
    """{feat: (mean, std)} over the train rows"""
    tr = record[record["split"] == "train"]
    if tr.empty:
        raise ValueError("record has no train rows")
    stats = {}
    for b in base_feats:
        mean, std = float(tr[b].mean()), float(tr[b].std())
        if not std > 0:
            raise ValueError("zero variance")
        stats[b] = (mean, std)
    return stats


def load_data_record(location, test_steps, start_date=None, master_name="master.csv"):
    master = pd.read_csv(LOC_DATA_ROOT / LOCATIONS[location]["dir"] / master_name)

    if start_date is not None:
        pos = master.index[master["date"] == start_date].tolist()
        if (not pos) or (pos[0] < 1):
            raise ValueError("start_date is out of range")
        master = master.iloc[pos[0] - 1:].reset_index(drop=True)

    T = len(master)  # points
    if not 1 <= test_steps <= T - 2:
        raise ValueError(f"test_steps={test_steps} out of range")
    train_L = (T - 1) - test_steps  # incidence intervals in train

    all_feats = [c for c in master.columns if c not in ("date", "step", "cases", "pop")]
    data_df = pd.DataFrame({"date": master["date"].to_numpy()})
    for f in all_feats:
        data_df[f] = master[f].astype(float).to_numpy()
    data_df["cases"] = master["cases"].to_numpy()
    data_df["pop"] = master["pop"].to_numpy()
    data_df["split"] = ["train" if i <= train_L else "test" for i in range(T)]

    # '_z' columns
    stats = train_stats(data_df, {c.partition("_lag")[0] for c in all_feats})
    for f in all_feats:
        mean, std = stats[f.partition("_lag")[0]]
        data_df[f + "_z"] = (data_df[f] - mean) / std

    from utils.general_utils import save_data_record, save_train_stats
    start_token = start_date if start_date is not None else str(data_df["date"].iloc[0])
    _, data_rel_path = save_data_record(location, start_token, test_steps, data_df)
    save_train_stats(location, start_token, test_steps, stats)
    return data_df, data_rel_path, stats


def tensors_from_record(data_df, feat_cols):
    miss = [f for f in feat_cols if f + "_z" not in data_df.columns]
    if miss:
        raise ValueError(f"feat_cols not in data record: {miss}")

    T = len(data_df)
    Xz = data_df[[f + "_z" for f in feat_cols]].to_numpy()
    X = torch.tensor(Xz, dtype=torch.float32).unsqueeze(0)
    POP = torch.tensor(data_df["pop"].to_numpy(), dtype=torch.float32).reshape(T, 1)
    Y = torch.tensor(data_df["cases"].to_numpy(dtype=float)[1:], dtype=torch.float32).unsqueeze(0)

    train_L = int((data_df["split"] == "train").sum()) - 1
    train = (X[:, :train_L + 1, :], Y[:, :train_L], POP[:train_L + 1, :])
    full = (X, Y, POP)
    return train, full, data_df["date"].tolist()


# ##################### For future features ######################################
def _require_base_feats(base_feats, caller):
    lagged = [f for f in (base_feats or ()) if "_lag" in f]
    if lagged:
        raise ValueError(f"{caller} takes base features, got lagged {lagged}")


def future_climate(location, scenario, origin, horizon, base_feats=None):
    if (scenario not in FUTURE_SOURCES) or (location not in LOCATIONS) or (horizon < 1):
        raise ValueError("error setting")
    _require_base_feats(base_feats, "future_climate")

    name = FUTURE_SOURCES[scenario]
    src = read_clean(location, name)
    where = f"[{location}] {name}: "

    targets = _periods_after(origin, horizon)

    if scenario == "forecast":
        issued_at = pd.to_datetime(src["issue_date"], format=ISO_DATE) == pd.Timestamp(origin)
        if not issued_at.any():
            raise ValueError(f"{where}no issue at origin {origin}")
        src = src[issued_at]

    src = src.assign(date=pd.to_datetime(src["date"], format=ISO_DATE)).set_index("date")
    if src.index.has_duplicates:
        raise ValueError(f"{where}duplicate dates")

    available = [c for c in src.columns if c not in _NON_FEATURE_COLS]
    if base_feats is None:
        base_feats = available
    missing_feats = [f for f in base_feats if f not in available]
    if missing_feats:
        raise ValueError(f"{where}missing {missing_feats}")

    out = src.reindex(targets)
    gaps = targets[out[base_feats].isna().any(axis=1).to_numpy()]
    if gaps.size:
        raise ValueError(f"{where}no data for {gaps.size} of {horizon} step(s) after {origin}")

    return pd.DataFrame({"date": targets.strftime(ISO_DATE),
                         **{f: out[f].to_numpy(dtype=float) for f in base_feats}})


def future_coverage(location, scenario, origin, base_feats=None):
    if scenario not in FUTURE_SOURCES:
        raise ValueError(f"unknown scenario {scenario}")
    _require_base_feats(base_feats, "future_coverage")
    name = FUTURE_SOURCES[scenario]
    src = read_clean(location, name)

    if scenario == "forecast":
        issued_at = pd.to_datetime(src["issue_date"], format=ISO_DATE) == pd.Timestamp(origin)
        if not issued_at.any():
            return 0
        src = src[issued_at]

    have = set(pd.to_datetime(src["date"], format=ISO_DATE))
    if base_feats is not None:
        available = [c for c in src.columns if c not in _NON_FEATURE_COLS]
        if any(f not in available for f in base_feats):
            return 0

    present = [d in have for d in _periods_after(origin, len(src))]
    return len(present) if all(present) else present.index(False)


def scenario_provider(location, observed_df, feat_cols, scenario, stats, max_horizon):
    """-> provider(origin_date) -> X (1, L, d)"""
    if (scenario not in FUTURE_SOURCES) or (not 1 <= max_horizon <= 12):
        raise ValueError("setting error")

    bases = sorted({f.partition("_lag")[0] for f in feat_cols})
    max_lag = max((int(f.partition("_lag")[2]) for f in feat_cols if "_lag" in f), default=0)
    pos_of = {d: i for i, d in enumerate(observed_df["date"])}
    T = len(observed_df)

    def provider(origin_date):
        pos = pos_of.get(origin_date)
        if pos is None:
            return None
        end = pos + 1  # first projected point
        if not max(1, max_lag) <= end < T:
            return None

        horizon = min(future_coverage(location, scenario, origin_date, base_feats=bases),
                      (T - 1) - end + 1,  # capped at the record's end
                      max_horizon)
        if horizon < 1:
            return None
        future_df = future_climate(location, scenario, origin_date, horizon, base_feats=bases)
        X, _, _ = future_tensor(location, observed_df, future_df, feat_cols, stats)
        return X

    return provider


def future_tensor(location, observed_df, future_df, feat_cols, stats):
    if (stats is None) or (location not in LOCATIONS):
        raise ValueError("setting error")

    missing = [f for f in feat_cols if f not in observed_df.columns]
    if missing:
        raise ValueError(f"feat_cols not in data record: {missing}")

    obs_ord = _period_ords(observed_df["date"], where="[record] ")
    fut_ord = _period_ords(future_df["date"], where="[future] ")
    if np.any(np.diff(fut_ord) != 1):
        raise ValueError("future dates are not contiguous.")

    history_df = observed_df[obs_ord < fut_ord[0]]
    hist_ord = obs_ord[obs_ord < fut_ord[0]]
    if not len(hist_ord) or hist_ord[-1] != fut_ord[0] - 1:
        raise ValueError("future does not start right after the record")

    def split_lag(feat):
        """"T2M_lag2" -> ("T2M", 2); "T2M" -> ("T2M", 0)."""
        head, _, tail = feat.partition("_lag")
        return (head, int(tail)) if tail else (feat, 0)

    needed_bases = {split_lag(f)[0] for f in feat_cols}
    missing_bases = [b for b in needed_bases if b not in future_df.columns]
    if missing_bases:
        raise ValueError(f"future lacks {sorted(missing_bases)}")

    combined_series = {
        b: pd.concat([pd.Series(history_df[b].to_numpy(dtype=float), index=hist_ord),
                      pd.Series(future_df[b].to_numpy(dtype=float), index=fut_ord)])
        for b in needed_bases
    }

    missing_stats = [b for b in needed_bases if b not in stats]
    if missing_stats:
        raise ValueError(f"stats lack {sorted(missing_stats)}")

    cols = []
    for f in feat_cols:
        feat_base, lag = split_lag(f)
        values = combined_series[feat_base].reindex(fut_ord - lag)
        if values.isna().any():
            raise ValueError(f"'{f}' reaches before the record starts")
        mean, std = stats[feat_base]
        cols.append((values.to_numpy() - mean) / std)

    X = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32).unsqueeze(0)
    POP = torch.full((len(fut_ord), 1), float(history_df["pop"].iloc[-1]), dtype=torch.float32)
    return X, POP, future_df["date"].tolist()
