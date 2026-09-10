"""baseline_results/<data_tag>/test/

    forecasts_h6.xlsx   one sheet per model, one pred_<scenario> column per climate source
    summary_h6.xlsx     per model and lead time: n_origins, rmse, mae
"""
import numpy as np
import pandas as pd
from dataloader.loc_core import FUTURE_SOURCES
from utils.general_utils import read_book, write_book
from .baseline_utils import add_cumulative

SCENARIOS = tuple(FUTURE_SOURCES)
KEYS = ["origin_date", "horizon", "target_date"]
SHEET_COLS = ["model", "form"] + KEYS + ["origin_split", "target_split", "unfittable", "actual", "epi_tag"]


# ##################### paths ######################################
def test_dir(out_dir):
    return out_dir / "test"


def forecasts_path(out_dir, max_horizon):
    return test_dir(out_dir) / f"forecasts_h{max_horizon}.xlsx"


def summary_path(out_dir, max_horizon):
    return test_dir(out_dir) / f"summary_h{max_horizon}.xlsx"


# ##################### read roll ######################################
def scored(table):
    err = table["pred"].to_numpy(dtype=float) - table["actual"].to_numpy(dtype=float)
    if not np.isfinite(err).all():
        raise ValueError("pred error!")
    return add_cumulative(table.assign(abs_err=np.abs(err), sq_err=err ** 2))


def one_roll(sheet, scenario):
    column = f"pred_{scenario}"
    if column not in sheet.columns:
        return None

    table = sheet.drop(columns=[c for c in sheet.columns if c.startswith("pred_")])
    return scored(table.assign(pred=sheet[column].to_numpy(dtype=float),
                               actual=table["actual"].astype(float),
                               unfittable=table["unfittable"].fillna("")))


def cached_rolls(out_dir, max_horizon, scenario, names=None, epi_tag=None):
    book = read_book(forecasts_path(out_dir, max_horizon))
    rolls = {name: one_roll(sheet, scenario) for name, sheet in book.items()
             if names is None or name in names}
    return {name: table for name, table in rolls.items()
            if table is not None and (epi_tag is None or not name.startswith("seirs")
                                      or set(table.get("epi_tag", [])) == {epi_tag})}


# ##################### write roll ######################################
def sheet_identity(frame):
    """(form, epi_tag or None)"""
    tag = frame["epi_tag"].iloc[0] if "epi_tag" in frame.columns else None
    return frame["form"].iloc[0], None if pd.isna(tag) else tag


def merge_scenario(sheet, part, scenario):
    base = part[[c for c in SHEET_COLS if c in part.columns]].reset_index(drop=True)
    if "epi_tag" in base.columns and base["epi_tag"].isna().all():
        base = base.drop(columns="epi_tag")
    if sheet is None or sheet_identity(sheet) != sheet_identity(base):   # a different model now
        sheet = base

    sheet = sheet.copy()
    sheet[f"pred_{scenario}"] = part.set_index(KEYS)["pred"].reindex(
        pd.MultiIndex.from_frame(sheet[KEYS])).to_numpy()

    preds = [f"pred_{s}" for s in SCENARIOS if f"pred_{s}" in sheet.columns]
    return sheet[[c for c in sheet.columns if c not in preds] + preds]


def save_rolls(out_dir, fresh, max_horizon, scenario):
    path = forecasts_path(out_dir, max_horizon)
    sheets = read_book(path)
    for name, part in fresh.groupby("model"):
        sheets[name] = merge_scenario(sheets.get(name), part, scenario)
    write_book(path, sheets)


# ##################### error metrics ######################################
def horizon_scores(sheet, max_horizon):
    rows = sheet[sheet["target_split"].eq("test") & sheet["horizon"].le(max_horizon)]
    preds = [c for c in rows.columns if c.startswith("pred_")]

    scorable = rows["actual"].notna() & np.isfinite(rows[preds]).all(axis=1)
    if not scorable.all():
        raise ValueError("check it, not all scorable")

    out = pd.DataFrame({"n_origins": rows.groupby("horizon")["origin_date"].nunique()})
    for col in preds:
        err = rows[col] - rows["actual"]
        out[f"rmse_{col[5:]}"] = np.sqrt((err ** 2).groupby(rows["horizon"]).mean())
        out[f"mae_{col[5:]}"] = err.abs().groupby(rows["horizon"]).mean()

    return out.reset_index().assign(model=sheet["model"].iloc[0], form=sheet["form"].iloc[0])


def write_summary(out_dir, max_horizon):
    sheets = read_book(forecasts_path(out_dir, max_horizon))
    if not sheets:
        return
    scores = pd.concat([horizon_scores(sheet, max_horizon) for sheet in sheets.values()], ignore_index=True)
    lead = ["model", "form", "horizon"]
    write_book(summary_path(out_dir, max_horizon),
               {"summary": scores[lead + [c for c in scores.columns if c not in lead]]})
