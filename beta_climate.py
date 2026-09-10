from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dataloader.loc_core import _period_ords
from utils.general_utils import CKPT_NAME, DIAG_NAME, SEIR_Re_cal, resolve_run_data_path


N_SEEDS = 10
MODES = ("train", "test")
STATE_NAMES = ("S", "E", "I", "R", "CUM_INC")


def run_paths(run_root, out_dir):
    """run_root: .../<data_tag>/<epi_tag>/<spec_tag>"""
    root = Path(run_root)
    return {"run_dirs": {mode: root / mode for mode in MODES},
            "table": f"{out_dir}/{root.parts[-3]}_beta_climate.csv"}


def latest_fitted_beta(seed_dir):
    """one beta per month: the most recent window's"""
    diag_path = Path(seed_dir) / DIAG_NAME
    beta = pd.read_excel(diag_path, sheet_name="beta").set_index("date")
    values = beta.to_numpy(dtype=float)
    n_points, n_segments = values.shape

    window = n_points - n_segments + 1
    taken = np.minimum(np.arange(n_points), n_segments - 1)
    fitted = values[np.arange(n_points), taken]
    if np.isnan(fitted).any() or (np.arange(n_points) >= taken + window).any():
        raise ValueError("error!")

    ckpt = torch.load(Path(seed_dir) / CKPT_NAME, weights_only=False)
    extra = ckpt["extra"]
    source = {"feats": list(extra["beta_input_feats"]),
              "data_rel_path": str(extra["data_rel_path"]),
              "static_epi_paras": dict(ckpt["model_config"]["static_epi_paras"])}
    return (pd.Series(fitted, index=beta.index),
            pd.Series(taken + 1, index=beta.index),
            source)


def run_betas(run_dir, n_seeds):
    betas, segments, source = {}, None, None
    for seed in range(n_seeds):
        seed_dir = Path(run_dir) / f"se_{seed}"
        beta, segment, seed_source = latest_fitted_beta(seed_dir)
        if segments is None:
            segments, source = segment, seed_source
        elif not (segment.equals(segments) and seed_source == source):
            raise ValueError("error!")
        betas[f"beta_se{seed}"] = beta

    frame = pd.DataFrame(betas)
    frame.insert(0, "fit_segment", segments)
    return frame, source


def run_root_and_modes(run_dirs):
    roots = {Path(run_dir).parent for run_dir in run_dirs.values()}
    if len(roots) != 1:
        raise ValueError("error!")
    return roots.pop(), tuple(Path(run_dir).name for run_dir in run_dirs.values())


def seed_states(seed_dir, with_roll):
    starts = pd.read_excel(Path(seed_dir) / DIAG_NAME, sheet_name="x0")
    starts = starts.assign(date=starts["start_date"].astype(str)).set_index("date")[list(STATE_NAMES)]
    if not with_roll:
        return starts
    rolled = pd.read_excel(Path(seed_dir) / DIAG_NAME, sheet_name="trajectory", index_col=0)[list(STATE_NAMES)]
    rolled.index = rolled.index.astype(str)
    if rolled.index[0] != starts.index[-1]:
        raise ValueError("error!")
    return pd.concat([starts.iloc[:-1], rolled])


def run_states_wide(run_root, run_modes, n_seeds):
    frames = []
    for mode in run_modes:
        per_seed = [seed_states(Path(run_root) / mode / f"se_{seed}", with_roll=(mode == run_modes[-1]))
                    .add_suffix(f"_se{seed}") for seed in range(n_seeds)]
        frames.append(pd.concat(per_seed, axis=1))
    wide = pd.concat(frames)
    wide = wide[~wide.index.duplicated(keep="last")].sort_index()
    return wide[[f"{state}_se{seed}" for state in STATE_NAMES for seed in range(n_seeds)]]


def check_consecutive(table, sources):
    ords = _period_ords(table.index, where="[beta table] ")
    breaks = np.flatnonzero(np.diff(ords) != 1)
    if breaks.size:
        raise ValueError("months are not consecutive")

    seen = table["fit_source"].tolist()
    blocks = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    if blocks != [s for s in sources if s in set(seen)]:
        raise ValueError("error!")


def add_seed_summary(table, prefix, n_seeds):
    values = table[[f"{prefix}_se{seed}" for seed in range(n_seeds)]].to_numpy(dtype=float)
    table[f"{prefix}_mean"] = values.mean(axis=1)
    table[f"{prefix}_sd"] = values.std(axis=1, ddof=1)


def beta_climate_table(run_dirs, table_path, n_seeds=N_SEEDS):
    frames, source = [], None
    for label, run_dir in run_dirs.items():
        frame, run_source = run_betas(run_dir, n_seeds)
        if source is None:
            source = run_source
        elif run_source != source:
            raise ValueError("error!")
        frame.insert(0, "fit_source", label)
        frames.append(frame)

    table = pd.concat(frames)
    table = table[~table.index.duplicated(keep="last")].sort_index()
    check_consecutive(table, list(run_dirs))

    run_root, run_modes = run_root_and_modes(run_dirs)
    states = run_states_wide(run_root, run_modes, n_seeds)
    states = states.reindex(table.index)
    if states.isna().any().any():
        raise ValueError("error!")
    table = table.join(states)

    data = pd.read_csv(resolve_run_data_path(source["data_rel_path"])).set_index("date")
    table = table.join(data[["cases", "pop", *source["feats"]]], how="left")
    if table[source["feats"]].isna().any().any() or table["pop"].isna().any():
        raise ValueError(f"{source['data_rel_path']} info missing")
    table = table.rename(columns={"pop": "N"})

    rates = source["static_epi_paras"]
    for seed in range(n_seeds):
        _, re = SEIR_Re_cal(table[f"beta_se{seed}"], rates["sigma"], rates["gamma"],
                            rates["mu_pop"], table[f"S_se{seed}"] * table["N"], table["N"])
        table[f"Re_se{seed}"] = re

    table = table[[f"{p}_se{s}" for p in ("beta", "S", "Re") for s in range(n_seeds)]].copy()
    add_seed_summary(table, "Re", n_seeds)

    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index_label="date")
    return table


def save_beta_climate_table(run_root, out_dir, n_seeds=N_SEEDS):
    """run_root: the checkpoint folder holding train/ and test/ -> <out_dir>/<data_tag>_beta_climate.csv"""
    paths = run_paths(run_root, out_dir)
    return beta_climate_table(paths["run_dirs"], paths["table"], n_seeds)
