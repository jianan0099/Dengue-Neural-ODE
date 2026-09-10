import argparse
import functools
import pandas as pd
from dataloader.loc_core import CLIMATE_FEATURES, FUTURE_SOURCES
from utils.general_utils import baseline_results_dir, load_run_stats, location_of, resolve_run_data_path
from . import BASELINE_NAMES
from .baseline_utils import interval_frame, rolling_forecast
from .glm_nb import NBGLM, check_scenario, form_tag, select_form
from .sarima import SARIMA, order_tag, select_order
from .seasonal_naive import SeasonalNaive
from .seirs_season import fit_windows, form_tag as seirs_form_tag, seirs_forecast, select_form as select_seirs_form
from .selection_log import SelectionTrace, write_selection
from .test_report import cached_rolls, save_rolls, summary_path, write_summary


def load_or_forecast(data_rel_path, names=BASELINE_NAMES, max_horizon=6, climate_feats=CLIMATE_FEATURES,
                     refit=False, climate_scenario="observed", epi_tag=None):
    """cached under baseline_results/<data_tag>/"""
    climate_scenario = check_scenario(climate_scenario)

    out_dir = baseline_results_dir(data_rel_path)

    saved = ({} if refit else
             cached_rolls(out_dir, max_horizon, climate_scenario, names, epi_tag))
    todo = [n for n in names if n not in saved]
    for name in names:
        if name in saved:
            print(f"  reusing {name}")

    rolled = False
    if todo:
        print(f"  selecting {todo} on the training period, then refitting at each test origin")

        record = pd.read_csv(resolve_run_data_path(data_rel_path))
        traces = {}
        fresh = baseline_forecasts(record, load_run_stats(data_rel_path),
                                   location_of(data_rel_path), climate_scenario,
                                   todo, max_horizon, climate_feats, traces=traces, epi_tag=epi_tag)
        if not fresh.empty:
            save_rolls(out_dir, fresh, max_horizon, climate_scenario)
            write_selection(traces, out_dir, max_horizon)
            rolled = True

    if rolled or not summary_path(out_dir, max_horizon).exists():
        write_summary(out_dir, max_horizon)

    tables = cached_rolls(out_dir, max_horizon, climate_scenario, names, epi_tag)
    ordered = [tables[n] for n in names if n in tables]
    return pd.concat(ordered, ignore_index=True) if ordered else pd.DataFrame()


# ##################### forecasts on test period based on the built baselines ##############################
def baseline_forecasts(record, stats, location, scenario="observed", names=BASELINE_NAMES,
                       max_horizon=6, climate_feats=CLIMATE_FEATURES, traces=None, epi_tag=None):
    tables = score_baselines(record, names=names, stats=stats, location=location, scenario=scenario,
                             max_horizon=max_horizon, climate_feats=climate_feats, traces=traces, epi_tag=epi_tag)
    if not tables:
        return pd.DataFrame()
    out = pd.concat([t.assign(model=name) for name, t in tables.items() if not t.empty], ignore_index=True)
    lead = ["model", "form"]
    return out[lead + [c for c in out.columns if c not in lead]]


def score_baselines(record, names=BASELINE_NAMES, stats=None, location=None, scenario="observed",
                    max_horizon=6, climate_feats=CLIMATE_FEATURES, traces=None, epi_tag=None):
    whole_interval_frame = interval_frame(record, feats=list(climate_feats))

    tables = {}
    for name in names:
        print(f"---------  scoring {name} ---------------")
        trace = None
        if traces is not None and name != "seasonal_naive":
            trace = traces[name] = SelectionTrace(name)

        roll, form = build_baseline(name, whole_interval_frame, stats=stats, location=location,
                                    max_horizon=max_horizon, record=record, scenario=scenario,
                                    trace=trace, epi_tag=epi_tag)

        tables[name] = roll().assign(form=form)
        if name.startswith("seirs_season"):
            tables[name] = tables[name].assign(epi_tag=epi_tag)
    return tables


# ##################### build baselines based on training data ##############################
def build_baseline(name, whole_interval_frame, stats=None, location=None, max_horizon=6,
                   record=None, scenario="observed", trace=None, epi_tag=None):
    def refit(factory, predict_kwargs=None):
        return lambda: rolling_forecast(factory, whole_interval_frame, max_horizon, predict_kwargs=predict_kwargs)

    if name == "glm":
        if stats is None or location is None:
            raise ValueError("the GLM needs train stats and a location to be built.")
        form = select_form(whole_interval_frame, stats, location, max_horizon, trace=trace)
        return (refit(functools.partial(NBGLM, form=form,
                                        whole_interval_frame=whole_interval_frame,
                                        stats=stats, location=location),
                      {"scenario": scenario}),
                form_tag(form))
    if name.startswith("seirs_season"):
        if location is None or record is None or epi_tag is None:
            raise ValueError("the seirs_season baseline needs a location, the record, and epi_tag to be built.")
        form = select_seirs_form(whole_interval_frame, location,
                                 fit_windows(whole_interval_frame),
                                 record, epi_tag, max_horizon, trace=trace)
        return (lambda: seirs_forecast(form, whole_interval_frame, location, record, max_horizon, epi_tag),
                seirs_form_tag(form))
    if name == "sarima":
        order, seasonal = select_order(whole_interval_frame, max_horizon, trace=trace)
        return (refit(functools.partial(SARIMA, order=order, seasonal_order=seasonal)),
                order_tag(order, seasonal))
    if name == "seasonal_naive":
        return refit(SeasonalNaive), "-"
    raise KeyError(f"unknown baseline '{name}'; known: {list(BASELINE_NAMES)}")


def get_args():
    parser = argparse.ArgumentParser(description="Select each baseline's form on the training period, "
                                                 "then refit it at every test origin and roll forward.")
    parser.add_argument("--data_rel_path", required=True, type=str, help="e.g. Goa_2015-01-01_t60/run_data.csv")
    parser.add_argument("--epi_tag", required=True, type=str,
                        help="The neural run's epi tag, e.g. epi_v1; the SEIRS baseline fits under that box.")
    parser.add_argument("--names", nargs="+", default=list(BASELINE_NAMES), metavar="NAME")
    parser.add_argument("--scenarios", nargs="+", default=list(FUTURE_SOURCES), choices=list(FUTURE_SOURCES),
                        metavar="NAME")
    parser.add_argument("--max_horizon", default=6, type=int)
    parser.add_argument("--refit", action="store_true", help="Ignore cached rolls and refit.")
    return parser.parse_args()


if __name__ == "__main__":
    a = get_args()
    for scenario in a.scenarios:
        print(f"scenario: {scenario}")
        load_or_forecast(a.data_rel_path, names=a.names, max_horizon=a.max_horizon, refit=a.refit,
                         climate_scenario=scenario, epi_tag=a.epi_tag)
