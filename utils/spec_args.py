import argparse
from dataclasses import fields

from models.neural_epi_fitter import FitConfig
from models.neural_epi_modules import ModelConfig, MECHANISMS
from utils.epi_base import static_epi_keys

MODEL_FIELDS = {f.name for f in fields(ModelConfig)}
FIT_FIELDS = {f.name for f in fields(FitConfig)}
SPEC_FIELD_TYPES = {f.name: f.type for f in (*fields(ModelConfig), *fields(FitConfig))}
SETTABLE_KEYS = {k for k, t in SPEC_FIELD_TYPES.items() if t in (int, float, str)} - {"beta_input_dim"}

STATIC_EPI_KEYS = {k for m in MECHANISMS.values() for k in m.required_rates}
SPEC_FIELD_TYPES.update({k: float for k in STATIC_EPI_KEYS})
SETTABLE_KEYS |= STATIC_EPI_KEYS

NON_SCALAR_KEYS = ("init_ode_state", "fixed_init_state_list")


def get_args():
    parser = argparse.ArgumentParser(description="Fit one model spec on one location")
    parser.add_argument("--location", default="Goa", type=str)
    parser.add_argument("--test_steps", default=60, type=int, help="#Incidence intervals held out as test.")
    parser.add_argument("--start_date", default=None, type=str,
                        help="First incidence date, e.g. 2015-01-01; default uses all.")
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"],
                        help="train holds out the last --test_steps intervals; "
                             "test continues the train run across the held-out period (the train run must exist).")
    parser.add_argument("--feats", default="T2M,PRECTOTCORR_SUM", type=str,
                        help="Beta inputs, comma-separated, e.g. T2M,PRECTOTCORR_SUM,QV2M.")
    parser.add_argument("--set", action="append", default=None, dest="set_fields", metavar="KEY=VAL",
                        help="Override a ModelConfig / FitConfig field or a static epi rate, "
                             "e.g. --set init_beta=3.0 --set segment_len=18. Repeatable.")
    parser.add_argument("--init_ode_state", nargs="+", type=float, default=None, metavar="V",
                        help="e.g. --init_ode_state 0.9898 0.0001 0.0001 0.01")
    parser.add_argument("--fixed_init_state_list", nargs="*", type=int, default=None, metavar="IDX",
                        help="Indices of initial-state parts held fixed, e.g. --fixed_init_state_list 2 3.")
    parser.add_argument("--seed_start", default=0, type=int)
    parser.add_argument("--n_inits", default=10, type=int, help="Seeds --seed_start .. --seed_start+n_inits-1.")
    parser.add_argument("--alt_scenarios", nargs="*", default=None, choices=["climatology", "forecast"],
                        metavar="NAME", help="Extra climate sources each segment is re-rolled under.")
    parser.add_argument("--alt_horizon", default=6, type=int, metavar="STEPS",
                        help="How many steps past each segment an --alt_scenarios roll may reach; 1..12.")

    args = parser.parse_args()
    args.alt_scenarios = args.alt_scenarios or []
    return args


def parse_kv(item):
    key, sep, raw = item.partition("=")
    key, raw = key.strip(), raw.strip()
    if not sep or key not in SETTABLE_KEYS:
        raise ValueError(f"--set '{item}' must be KEY=VAL with KEY in {sorted(SETTABLE_KEYS)}")
    cast = SPEC_FIELD_TYPES[key]
    try:
        return key, cast(raw)
    except ValueError as exc:
        raise ValueError(f"--set '{key}' expects {cast.__name__}; got '{raw}'") from exc


def build_spec(args):
    spec = {"beta_input_feats": args.feats.split(",")}
    for key in NON_SCALAR_KEYS:
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    for item in (args.set_fields or []):
        key, value = parse_kv(item)
        if key in spec:
            raise ValueError(f"key '{key}' given more than once")
        spec[key] = value
    return spec


def build_cfgs(spec, epi_base):
    """spec + the location's epi base -> (feats, model_cfg, fit_cfg), unnamed fields keep their dataclass defaults"""
    feats = spec["beta_input_feats"]

    # the location's baseline goes in first and the spec overrides it
    model_overrides = dict(epi_base)
    model_overrides.update({k: v for k, v in spec.items() if k in MODEL_FIELDS})
    model_overrides.setdefault("beta_input_dim", len(feats))

    rates = {**model_overrides.get("static_epi_paras", {}), **{k: v for k, v in spec.items() if k in STATIC_EPI_KEYS}}
    required = static_epi_keys(model_overrides.get("neural_epi_name", ModelConfig.neural_epi_name))
    missing = [k for k in required if k not in rates]
    if missing:
        raise ValueError(f"epi base is missing static_epi_paras {missing}")
    model_overrides["static_epi_paras"] = rates
    model_cfg = ModelConfig(**model_overrides)

    fit_cfg = FitConfig(**{k: v for k, v in spec.items() if k in FIT_FIELDS})
    return feats, model_cfg, fit_cfg
