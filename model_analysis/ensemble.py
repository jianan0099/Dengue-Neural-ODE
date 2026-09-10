import json
from pathlib import Path
import pandas as pd
from model_analysis.run_scoring import load_run, horizon_table
from utils.general_utils import LOCAL_NEURAL_RESULTS, LOCAL_RUNS_ROOT, run_identity

MAX_HORIZON = 6
N_SEEDS = 10             # seeds per ensemble
SCENARIOS = ("observed", "climatology", "forecast")
SHEET_KEYS = ("target_date", "actual", "target_split")


def cell_runs(data_tag, epi_tag, spec_tag, seed_start=0, n_seeds=N_SEEDS, mode="test"):
    cell = LOCAL_RUNS_ROOT / data_tag / epi_tag / spec_tag / mode
    if not cell.is_dir():
        raise FileNotFoundError(f"no such cell: {cell}")

    wanted = range(seed_start, seed_start + n_seeds)
    found = {int(p.name.split("_")[1]): p for p in cell.glob("se_*")
             if p.name.split("_")[1].isdigit()}
    runs = [found[s] for s in wanted if s in found]
    if not runs:
        raise FileNotFoundError(f"{cell} holds no se_<seed> run for seed(s) "
                                f"{wanted.start}..{wanted.stop - 1}.")

    missing = [s for s in wanted if s not in found]
    if missing:
        print("missing seeds!")
    return runs


def save_identity(out_dir, identity):
    path = Path(out_dir) / "identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2, default=str))
    return path


def export_ensembles(alias, data_tag, epi_tag, spec_tag, seed_start=0,
                     n_seeds=100, block=N_SEEDS, max_horizon=MAX_HORIZON, scenarios=SCENARIOS,
                     mode="test"):
    runs = cell_runs(data_tag, epi_tag, spec_tag, seed_start, n_seeds, mode)
    out_dir = LOCAL_NEURAL_RESULTS / alias

    reference = load_run(runs[0])
    record = record_frame(runs[0])

    written, per_scenario = {}, {}
    for scenario in scenarios:
        print(f"  --- {scenario} ---")
        try:
            long = seed_forecasts(runs, max_horizon, scenario)
        except (ValueError, KeyError):
            continue
        blocks = ensemble_blocks(long["seed"].unique(), block)
        written[scenario] = str(write_scenario_book(out_dir, scenario, long, max_horizon, blocks, record))
        per_scenario[scenario] = {
            "seeds": sorted(int(s) for s in long["seed"].unique()),
            "ensembles": {k: [int(s) for s in v] for k, v in blocks.items()},
            "short_ensembles": {k: len(v) for k, v in blocks.items() if len(v) < block},
        }

    save_identity(out_dir, {
        "alias": alias,
        "data_tag": data_tag, "epi_tag": epi_tag, "spec_tag": spec_tag,
        "mode": mode,
        "data_rel_path": reference["ckpt"]["extra"]["data_rel_path"],
        "seed_start": seed_start, "n_seeds": n_seeds, "ensemble_size": block,
        "max_horizon": max_horizon,
        "runs_root": str(runs[0].parents[4]), "n_runs_found": len(runs),
        "workbooks": written, "scenarios": per_scenario,
    })
    return out_dir


def export_from_root(run_root, alias, n_seeds=N_SEEDS, block=N_SEEDS, seed_start=0, mode="test"):
    """run_root = checkpoints_local/<data_tag>/<epi_tag>/<spec_tag>"""
    data_tag, epi_tag, spec_tag = Path(run_root).parts[-3:]
    return export_ensembles(alias, data_tag, epi_tag, spec_tag, seed_start, n_seeds, block, mode=mode)


def write_scenario_book(out_dir, scenario, long, max_horizon, blocks, record=None):
    path = Path(out_dir) / f"{scenario}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        if record is not None:
            record.to_excel(writer, sheet_name="record", index=False)
        long.to_excel(writer, sheet_name="forecasts", index=False)
        for horizon in range(1, max_horizon + 1):
            sheet = horizon_sheet(long, horizon, blocks)
            if sheet.empty:
                continue
            sheet.to_excel(writer, sheet_name=f"h{horizon}", index=False)
    return path


def record_frame(run_dir):
    run = load_run(run_dir)
    return pd.DataFrame({
        "date": run["actual"].index,
        "actual": run["actual"].to_numpy(dtype=float),
        "split": run["splits"].to_numpy(),
    })


def ensemble_blocks(seeds, block=N_SEEDS):
    """range(100) -> {"se0-9": [0..9], "se10-19": [10..19], ...}"""
    seeds = sorted(seeds)
    out = {}
    for seed in seeds:
        lo = (seed // block) * block
        out.setdefault(f"se{lo}-{lo + block - 1}", []).append(seed)

    short = {label: len(members) for label, members in out.items() if len(members) < block}
    if short:
        print(f"  WARNING: ensemble(s) smaller than {block} seeds: {short}")
    return out


def seed_forecasts(runs, max_horizon=MAX_HORIZON, scenario="observed"):
    sheet = scenario_sheet(scenario)
    parts = []
    for run_dir in runs:
        seed = run_identity(run_dir)["seed"]
        try:
            table = horizon_table(load_run(run_dir), max_horizon, sheet=sheet)
        except (FileNotFoundError, KeyError) as exc:
            continue
        if table.empty:
            continue
        parts.append(table.assign(seed=seed))

    if not parts:
        raise ValueError(f"no seed has a '{sheet}' forecast to export.")
    out = pd.concat(parts, ignore_index=True)
    out = out[["seed"] + [c for c in out.columns if c != "seed"]]
    return out.sort_values(["seed", "origin_date", "horizon"]).reset_index(drop=True)


def horizon_sheet(long, horizon, blocks):
    part = long[long["horizon"] == horizon]
    out = part[list(SHEET_KEYS)].drop_duplicates(subset="target_date").set_index("target_date")

    for label, members in blocks.items():
        rows = part[part["seed"].isin(members)]
        out[label] = rows.groupby("target_date")["pred"].mean()
    return out.sort_index().reset_index()


def scenario_sheet(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown climate scenario '{scenario}'; "
                         f"known: {list(SCENARIOS)}")
    return "incidence" if scenario == "observed" else f"incidence_{scenario}"

