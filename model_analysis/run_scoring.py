from pathlib import Path
import numpy as np
import pandas as pd
import torch

from utils.general_utils import CKPT_NAME, DIAG_NAME, resolve_run_data_path

MAIN_SHEET = "incidence"
MAX_HORIZON = 6


def load_run(run_dir):
    run_dir = Path(run_dir)
    diag_path, ckpt_path = run_dir / DIAG_NAME, run_dir / CKPT_NAME
    for p in (diag_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")

    book = pd.ExcelFile(diag_path)
    inc_sheets = {name: book.parse(name, index_col=0)
                  for name in book.sheet_names if name.startswith(MAIN_SHEET)}
    rho = book.parse("rho", index_col=0)
    beta = book.parse("beta", index_col=0)
    x0 = book.parse("x0", index_col=0)

    ckpt = torch.load(ckpt_path, weights_only=False)
    inc = inc_sheets[MAIN_SHEET]
    actual, splits = _load_record(ckpt, inc.index)

    return {
        "run_dir": run_dir, "ckpt": ckpt,
        "inc": inc, "inc_sheets": inc_sheets,
        "rho": rho, "beta": beta, "x0": x0,
        "history": ckpt["extra"]["train_hist"],
        "segment_len": ckpt["fit_config"]["segment_len"],
        "actual": actual,
        "splits": splits,
        "n_train_intervals": int((splits == "train").sum()),
        "n_segments": inc.shape[1],
        "stopped_at": ckpt["extra"]["segment_id"],
    }


def _load_record(ckpt, interval_index):
    data_df = pd.read_csv(resolve_run_data_path(ckpt["extra"]["data_rel_path"]))
    idx = pd.Index(data_df["date"].tolist()[1:], name="date")
    actual = pd.Series(data_df["cases"].to_numpy()[1:], index=idx)
    splits = pd.Series(data_df["split"].to_numpy()[1:], index=idx)
    return actual.reindex(interval_index), splits.reindex(interval_index)


def segment_starts(run):
    point_dates = [None] + list(run["inc"].index)       # point 0 has no interval row
    pos = {d: i for i, d in enumerate(point_dates) if d is not None}
    starts = {}
    for seg_id, row in run["x0"].iterrows():
        d = row["start_date"]
        starts[int(seg_id)] = pos.get(d, 0) if d in pos else 0
    return starts


def horizon_table(run, max_horizon, sheet=MAIN_SHEET):
    inc = run["inc_sheets"].get(sheet)
    if inc is None:
        raise KeyError(f"sheet '{sheet}' not in this run.")

    actual, splits, seg_len = run["actual"], run["splits"], run["segment_len"]
    starts = segment_starts(run)
    rows = []

    for col in inc.columns:
        seg_id = int(col.split("_")[1])
        last_fitted = starts[seg_id] + seg_len - 1
        rho = float(run["rho"][col].dropna().iloc[0])

        for h in range(1, max_horizon + 1):
            r = last_fitted + h
            if r >= len(inc):
                break
            pred, obs = inc[col].iloc[r], actual.iloc[r]
            if not np.isfinite(pred) or not np.isfinite(obs):
                continue
            rows.append({
                "segment_id": seg_id, "horizon": h,
                "origin_date": inc.index[last_fitted], "target_date": inc.index[r],
                "origin_split": splits.iloc[last_fitted],
                "target_split": splits.iloc[r],
                "actual": float(obs), "pred": float(pred), "rho": rho,
                "abs_err": abs(float(pred) - float(obs)),
                "sq_err": (float(pred) - float(obs)) ** 2,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["segment_id", "horizon"])
    grp = df.groupby("segment_id")
    cum_cols = []
    for col in ("abs_err", "sq_err"):
        df[f"cum_{col}"] = grp[col].cumsum()
        cum_cols.append(f"cum_{col}")

    holed = df["horizon"].to_numpy() != grp.cumcount().to_numpy() + 1
    incomplete = pd.Series(holed, index=df.index).groupby(df["segment_id"]).cummax()
    df.loc[incomplete, cum_cols] = np.nan
    return df
