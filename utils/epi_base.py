import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "epi_bases"

BASE_SCALAR_FIELDS = ("init_beta", "beta_prior_min", "beta_prior_max",
                      "init_rho", "rho_prior_min", "rho_prior_max")


def static_epi_keys(neural_epi_name):
    from models.neural_epi_modules import MECHANISMS
    if neural_epi_name not in MECHANISMS:
        raise NotImplementedError(f"Epi mechanism '{neural_epi_name}' not implemented")
    return MECHANISMS[neural_epi_name].required_rates


def load_epi_base(location, locations):
    cfg = locations.get(location)
    if cfg is None:
        raise KeyError(f"unknown location '{location}' ")

    name = cfg.get("epi_base")
    if not name:
        raise KeyError("no epi_base")

    path = BASE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"epi base '{name}' for '{location}' not found at {path}")

    raw = json.loads(path.read_text())
    version = raw.get("version")
    if not version:
        raise ValueError(f"{path} has no 'version'")

    overrides = {k: raw[k] for k in BASE_SCALAR_FIELDS if k in raw}
    overrides["static_epi_paras"] = dict(raw.get("static_epi_paras") or {})
    return version, overrides


