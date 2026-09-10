import numpy as np
import pandas as pd
from utils.general_utils import read_book, write_book


class SelectionTrace:

    def __init__(self, name):
        self.name = name
        self.tables = {}
        self.status = {}

    def keep(self, tag, table):
        self.tables[tag] = table
        self.status[tag] = "kept"

    def drop(self, tag, reason):
        self.status[tag] = f"dropped: {reason}"

    def select(self, tag):
        if tag not in self.status:
            raise KeyError(f"{self.name}: {tag!r} was never rolled")
        self.status[tag] = "selected"


def form_summary(trace, max_horizon):
    whole = {tag: t[t["horizon"].eq(max_horizon) & t["cum_sq_err"].notna()] for tag, t in trace.tables.items()}
    whole = {tag: w for tag, w in whole.items() if len(w)}
    common = set.intersection(*(set(w["origin_date"]) for w in whole.values())) if whole else set()

    rows = []
    for tag, status in trace.status.items():
        w = whole.get(tag)
        s = None if w is None else w[w["origin_date"].isin(common)]
        rows.append({"model": trace.name, "form": tag,
                     "window_rmse_common": np.nan if w is None else np.sqrt(s["cum_sq_err"].mean()),
                     "status": status})
    return pd.DataFrame(rows).sort_values("window_rmse_common", kind="stable", na_position="last",
                                          ignore_index=True)


def summary_path(out_dir, max_horizon):
    return out_dir / "selection" / f"form_summary_h{max_horizon}.xlsx"


def write_selection(traces, out_dir, max_horizon):
    if not traces:
        return
    path = summary_path(out_dir, max_horizon)
    sheets = read_book(path)
    sheets.update({name: form_summary(trace, max_horizon) for name, trace in traces.items()})
    write_book(path, sheets)
