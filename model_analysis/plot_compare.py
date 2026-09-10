import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baseline_models import BASELINE_NAMES
from model_analysis.ensemble import MAX_HORIZON, SHEET_KEYS
from utils.general_utils import LOCAL_BASELINE_RESULTS, LOCAL_NEURAL_RESULTS

NEURAL = "neural ODE"
LEADS = (1, 3, 6)
COLORS = dict(zip((NEURAL, *BASELINE_NAMES), ("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a")))


def neural_alias(data_tag):
    hits = [p.parent.name for p in LOCAL_NEURAL_RESULTS.glob("*/identity.json")
            if json.loads(p.read_text())["data_tag"] == data_tag]
    if len(hits) != 1:
        raise ValueError(f"{data_tag}: expected one exported alias, found {hits}")
    return hits[0]


def load_lines(data_tag, scenario="observed", max_horizon=MAX_HORIZON):
    book = pd.read_excel(LOCAL_NEURAL_RESULTS / neural_alias(data_tag) / f"{scenario}.xlsx", sheet_name=None)
    parts = [book[f"h{h}"].assign(horizon=h, pred=book[f"h{h}"].drop(columns=list(SHEET_KEYS)).mean(axis=1))
             for h in range(1, max_horizon + 1)]
    lines = {NEURAL: pd.concat(parts)}

    book = pd.read_excel(LOCAL_BASELINE_RESULTS / data_tag / "test" / f"forecasts_h{max_horizon}.xlsx",
                         sheet_name=None)
    for name in BASELINE_NAMES:
        if name in book:
            lines[name] = book[name].assign(pred=book[name][f"pred_{scenario}"])
    return {name: t[t["target_split"] == "test"] for name, t in lines.items()}


def scores(table):
    err = table["pred"] - table["actual"]
    out = table.assign(rmse=err ** 2, mae=err.abs()).groupby("horizon")[["rmse", "mae"]].mean()
    return out.assign(rmse=np.sqrt(out["rmse"]))


def plot_compare(data_tag, out_path, scenario="observed", leads=LEADS, max_horizon=MAX_HORIZON):
    # scenario: "observed" | "climatology" | "forecast"
    lines = load_lines(data_tag, scenario, max_horizon)
    style = {name: dict(color=COLORS[name], ls="-" if name == NEURAL else "--", lw=1.5) for name in lines}

    fig = plt.figure(figsize=(11, 2.3 * len(leads)))
    grid = fig.add_gridspec(2 * len(leads), 2, width_ratios=[3, 1], hspace=0.7, wspace=0.3)
    left = [fig.add_subplot(grid[0:2, 0])]
    left += [fig.add_subplot(grid[2 * i:2 * i + 2, 0], sharex=left[0]) for i in range(1, len(leads))]
    right = [fig.add_subplot(grid[:len(leads), 1]), fig.add_subplot(grid[len(leads):, 1])]

    for ax, lead in zip(left, leads):
        ref = lines[NEURAL][lines[NEURAL]["horizon"] == lead].sort_values("target_date")
        ax.plot(pd.to_datetime(ref["target_date"]), ref["actual"], "ko", ms=3, label="reported")
        for name, table in lines.items():
            part = table[table["horizon"] == lead].sort_values("target_date")
            ax.plot(pd.to_datetime(part["target_date"]), part["pred"], label=name, **style[name])
        ax.set_title(f"{lead} month{'s' * (lead > 1)} ahead", loc="right", fontsize=10)
    left[len(leads) // 2].set_ylabel("reported cases")
    left[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    left[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for ax in left[:-1]:
        ax.tick_params(labelbottom=False)

    for ax, metric in zip(right, ("rmse", "mae")):
        for name, table in lines.items():
            s = scores(table)
            ax.plot(s.index, s[metric], marker="o", ms=4, **style[name])
        ax.set_ylabel(metric.upper())
        ax.set_xticks(range(1, max_horizon + 1))
    right[-1].set_xlabel("lead time (months)")

    fig.legend(*left[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{data_tag} | {scenario} climate")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
