import numpy as np
import torch
import pandas as pd
import random
import os
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_CACHE = PROJECT_ROOT / "data_cache"
LOCAL_RUNS_ROOT = PROJECT_ROOT / "checkpoints_local"
LOCAL_BASELINE_RESULTS = PROJECT_ROOT / "baseline_results"
LOCAL_NEURAL_RESULTS = PROJECT_ROOT / "neural_epi_results"

RUN_META_NAME = "run_meta.json"
CKPT_NAME = "final_fitter.pt"
DIAG_NAME = "segment_diagnostics.xlsx"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------
# tags
# --------------------
def summarize_feats(feat_list):
    """["T2M", "T2M_lag1", "QV2M"] -> "Q l0-TMl0l1" """
    climate_feat_short = {'PRECTOTCORR_SUM': 'PS', 'QV2M': 'Q', 'T2M': 'TM'}
    groups = defaultdict(list)
    for f in feat_list:
        if '_lag' in f:
            raw_feat, lag = f.split("_lag")[0], f.split("_lag")[1]
        else:
            raw_feat, lag = f, '0'
        groups[climate_feat_short.get(raw_feat, raw_feat)].append(lag)
    return "-".join(
        k + "".join(f"l{lag}" for lag in sorted(v, key=int))
        for k, v in sorted(groups.items())
    )


def loc_data_tag(location, start_date, test_steps):
    return f"{location}_{start_date}_t{test_steps}"


def model_spec_tag(feats, hidden_dim, pen_lam, segment_len):
    return f"{summarize_feats(feats)}_h{hidden_dim}_p{pen_lam}_w{segment_len}"


EPI_RANGE_FIELDS = (("beta", "beta_prior_min", "beta_prior_max"),
                    ("rho", "rho_prior_min", "rho_prior_max"))
EPI_INIT_FIELDS = (("initbeta", "init_beta"), ("initrho", "init_rho"))


def epi_para_tag(model_cfg, epi_base, version):
    """epi_<version> plus only what departs from the base, e.g. epi_v1_sigma6_beta1.5-25"""
    from models.neural_epi_modules import ModelConfig
    d = ModelConfig()
    ref = {**vars(d), **epi_base}
    ref_static = {**d.static_epi_paras, **epi_base.get("static_epi_paras", {})}

    parts = sorted(f"{k}{v:g}" for k, v in model_cfg.static_epi_paras.items()
                   if v != ref_static.get(k))
    parts += [f"{name}{getattr(model_cfg, lo):g}-{getattr(model_cfg, hi):g}"
              for name, lo, hi in EPI_RANGE_FIELDS
              if (getattr(model_cfg, lo), getattr(model_cfg, hi)) != (ref[lo], ref[hi])]
    parts += [f"{name}{getattr(model_cfg, field):g}"
              for name, field in EPI_INIT_FIELDS if getattr(model_cfg, field) != ref[field]]

    return "_".join([f"epi_{version}"] + parts)


# --------------------
# run tree: <data_tag>/<epi_tag>/<spec_tag>/<mode>/se_<seed>
# --------------------
def run_cell_dir(args, data_rel_path, mode):
    ldt = Path(data_rel_path).parent.name
    mst = model_spec_tag(args.beta_input_feats, args.beta_model_hidden_dim,
                         args.beta_pen_lam, args.segment_len)
    return LOCAL_RUNS_ROOT / ldt / args.epi_tag / mst / mode


def run_provenance():
    return {
        "stamp": datetime.now().strftime("%y%m%d_%H%M%S"),
        "host": os.uname().nodename,
        "pid": os.getpid(),
    }


def make_run_dir(args, data_rel_path, mode):
    run_dir = run_cell_dir(args, data_rel_path, mode) / f"se_{args.seed}"
    meta_path = run_dir / RUN_META_NAME

    if run_dir.exists():
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if meta is None:
            print(f"WARNING: {run_dir} exists with no {RUN_META_NAME}; overwriting it.")
        elif meta.get("status") == "running":
            raise RuntimeError(f"{run_dir} is claimed by a run still marked 'running' ")
        else:
            print(f"overwriting the previous run of this cell")
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True)
    meta_path.write_text(json.dumps({"status": "running", **run_provenance()}, indent=2))
    return run_dir


def finish_run_dir(run_dir):
    meta_path = Path(run_dir) / RUN_META_NAME
    meta = json.loads(meta_path.read_text())
    meta["status"] = "done"
    meta["finished"] = datetime.now().strftime("%y%m%d_%H%M%S")
    meta_path.write_text(json.dumps(meta, indent=2))


def run_identity(run_dir):
    """.../<data_tag>/<epi_tag>/<spec_tag>/<mode>/se_3 -> {data_tag, epi_tag, spec_tag, mode, seed: 3}"""
    data_tag, epi_tag, spec_tag, mode, run = Path(run_dir).parts[-5:]
    seed = re.match(r"se_(\d+)$", run)
    return {"data_tag": data_tag, "epi_tag": epi_tag, "spec_tag": spec_tag, "mode": mode,
            "seed": int(seed.group(1)) if seed else np.nan}


# --------------------
# data cache: <data_tag>/run_data.csv + train_stats.json
# --------------------
def run_data_subpath(location, start_date, test_steps):
    return Path(loc_data_tag(location, start_date, test_steps)) / "run_data.csv"


def resolve_run_data_path(rel_path):
    return LOCAL_DATA_CACHE / rel_path


def save_data_record(location, start_date, test_steps, data_df):
    rel_path = run_data_subpath(location, start_date, test_steps)
    data_path = LOCAL_DATA_CACHE / rel_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(data_path, index=False)
    print(f"Run data saved to: {data_path}")
    return data_path, rel_path


def baseline_results_dir(data_rel_path):
    return LOCAL_BASELINE_RESULTS / Path(data_rel_path).parent


def location_of(data_rel_path):
    """Goa_2015-01-01_t60/run_data.csv -> "Goa" """
    from dataloader.loc_core import LOCATIONS

    tag = Path(data_rel_path).parent.name
    for loc in sorted(LOCATIONS, key=len, reverse=True):
        if tag == loc or tag.startswith(f"{loc}_"):
            return loc
    raise KeyError(f"no known location at the head of '{tag}'")


def save_train_stats(location, start_date, test_steps, stats_by_feat):
    """{"T2M": {"mean", "std"}, ...} beside run_data.csv"""
    rel_path = Path(run_data_subpath(location, start_date, test_steps)).with_name("train_stats.json")
    stats_path = LOCAL_DATA_CACHE / rel_path

    payload = {f: {"mean": float(m), "std": float(s)} for f, (m, s) in stats_by_feat.items()}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Train stats saved to: {stats_path}")
    return stats_path, rel_path


def load_run_stats(data_rel_path):
    """-> {feat: (mean, std)}"""
    stats_path = resolve_run_data_path(Path(data_rel_path).with_name("train_stats.json"))
    if not stats_path.exists():
        raise FileNotFoundError(f"no train stats at {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    return {f: (v["mean"], v["std"]) for f, v in payload.items()}


# --------------------
# workbooks
# --------------------
def save_dfs_atomic(file_path, dfs, sheet_names, indexes):
    file_path = Path(file_path)
    tmp_path = file_path.with_name(f"{file_path.stem}.tmp.{os.getpid()}{file_path.suffix}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            for df, sheet_name, index in zip(dfs, sheet_names, indexes):
                df.to_excel(writer, sheet_name=sheet_name, index=index)
        os.replace(tmp_path, file_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_book(file_path):
    file_path = Path(file_path)
    return pd.read_excel(file_path, sheet_name=None) if file_path.exists() else {}


def write_book(file_path, sheets):
    save_dfs_atomic(file_path, list(sheets.values()), list(sheets), [False] * len(sheets))


def SEIR_Re_cal(beta, sigma, gamma, mu, S, N):
    R0 = beta * sigma / ((sigma + mu) * (gamma + mu))
    return R0, R0 * S / N
