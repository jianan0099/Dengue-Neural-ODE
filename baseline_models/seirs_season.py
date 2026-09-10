# Seasonal SEIRS
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from dataloader.loc_core import LOCATIONS
from utils.epi_base import load_epi_base

# ##################### seirs mechanism ######################################
SUB_STEPS = 4


def seirs_rhs(y, beta, rates):
    """[S, E, I, R, CUM_INC]"""
    s, e, i, r, _ = y
    sigma, gamma, omega, mu = rates["sigma"], rates["gamma"], rates["omega"], rates["mu_pop"]

    ds = mu - beta * s * i + omega * r - mu * s
    de = beta * s * i - sigma * e - mu * e
    di = sigma * e - gamma * i - mu * i
    dr = gamma * i - omega * r - mu * r
    return np.array([ds, de, di, dr, sigma * e])


def integrate_states(y0, beta_grid, win):
    h = 1.0 / win.n_sub
    rates, y = win.setup.rates, y0
    out = [y]
    for step in range(win.n_sub * (win.n_points - 1)):
        k = 2 * step
        b0, bh, b1 = beta_grid[k], beta_grid[k + 1], beta_grid[k + 2]
        k1 = seirs_rhs(y, b0, rates)
        k2 = seirs_rhs(y + 0.5 * h * k1, bh, rates)
        k3 = seirs_rhs(y + 0.5 * h * k2, bh, rates)
        k4 = seirs_rhs(y + h * k3, b1, rates)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (step + 1) % win.n_sub == 0:
            out.append(y)
    return np.array(out)


# ##################### diff ######################################
def predicted_incidence(theta, win):
    x0, rho, coef = unpack(theta, win)
    beta = beta_curve(coef, win.X, win.setup.beta_min, win.setup.beta_max)
    cum_INC = integrate_states(x0, beta, win)[:, 4] * win.pop
    return rho * (cum_INC[1:] - cum_INC[:-1])


# ##################### epi model set up ######################################
@dataclass
class EpiSetup:
    rates: dict
    beta_min: float
    beta_max: float
    init_beta: float
    rho_min: float
    rho_max: float
    init_rho: float


def epi_departures(epi_tag, location):
    from models.neural_epi_modules import ModelConfig
    from utils.general_utils import EPI_INIT_FIELDS, EPI_RANGE_FIELDS, epi_para_tag

    version, _, tail = epi_tag.partition("_")[2].partition("_")
    ranges = {name: (lo, hi) for name, lo, hi in EPI_RANGE_FIELDS}
    inits = dict(EPI_INIT_FIELDS)
    statics = tuple(ModelConfig().static_epi_paras)

    names = sorted([*ranges, *inits, *statics], key=len, reverse=True)
    number = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    token = re.compile(rf"(?:^|_)({'|'.join(map(re.escape, names))})({number})(?:-({number}))?(?=_|$)")

    found, seen = {}, 0
    for match in token.finditer(tail):
        name, low, high = match.group(1), float(match.group(2)), match.group(3)
        seen += match.end() - match.start()
        if name in ranges and high is not None:
            found[ranges[name][0]], found[ranges[name][1]] = low, float(high)
        elif name in inits and high is None:
            found[inits[name]] = low
        elif name in statics and high is None:
            found[name] = low
        else:
            raise ValueError("Not matched!")

    if tail and seen != len(tail):
        raise ValueError("Not matched!")

    base_version, base = load_epi_base(location, LOCATIONS)
    rebuilt = epi_para_tag(ModelConfig(**_over_base(base, found)), base, base_version)
    if rebuilt != epi_tag:
        raise ValueError("Not matched!")
    return found


def _over_base(base, departures):
    """epi base + departures -> ModelConfig kwargs"""
    overrides = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for field, value in departures.items():
        if field in overrides["static_epi_paras"]:
            overrides["static_epi_paras"][field] = value
        else:
            overrides[field] = value
    return overrides


def epi_setup(location, epi_tag):
    from models.neural_epi_modules import ModelConfig

    _, base = load_epi_base(location, LOCATIONS)
    cfg = ModelConfig(**_over_base(base, epi_departures(epi_tag, location)))
    return EpiSetup(rates=dict(cfg.static_epi_paras),
                    beta_min=cfg.beta_prior_min, beta_max=cfg.beta_prior_max, init_beta=cfg.init_beta,
                    rho_min=cfg.rho_prior_min, rho_max=cfg.rho_prior_max, init_rho=cfg.init_rho)


# ##################### beta = clip(w_cos * cos + w_sin * sin + b) ######################################
def form_tag(window):
    return f"w{window}"


def beta_curve(coef, X, beta_min, beta_max):
    return np.clip(X @ coef[:2] + coef[2], beta_min, beta_max)


def season_matrix(months):
    angle = 2.0 * np.pi * np.asarray(months, dtype=float) / 12.0
    if not np.isfinite(angle).all():
        raise ValueError("a month is missing")
    return np.stack([np.cos(angle), np.sin(angle)], axis=1)


def interp_points(X, m):
    T = X.shape[0]
    t = np.linspace(0.0, T - 1, m * (T - 1) + 1)
    i0 = np.clip(np.floor(t).astype(int), 0, T - 2)
    w = (t - i0)[:, None]
    return (1.0 - w) * X[i0] + w * X[i0 + 1]


# ##################### the initial state ######################################
INIT_I, INIT_R = 0.0001, 0.01
FREE_MASS = 1.0 - INIT_I - INIT_R
INIT_E0 = 0.0001


def initial_state(E0):
    """[S, E, I, R, CUM_INC]"""
    return np.array([FREE_MASS - E0, E0, INIT_I, INIT_R, 0.0])


# ##################### unknowns: one window, one theta ######################################
@dataclass
class Window:
    X: np.ndarray  # (G, 2) [cos, sin]
    pop: np.ndarray  # (T,)
    cases: np.ndarray  # (T-1,)
    n_points: int  # T
    setup: EpiSetup
    init_state: np.ndarray = None  # (5,) warm start; None means this window fits x0
    n_sub: int = SUB_STEPS

    @property
    def n_x0(self):
        return 0 if self.init_state is not None else 1


def unpack(theta, win):
    x0 = (np.asarray(win.init_state, dtype=float) if win.init_state is not None
          else initial_state(theta[0]))
    return x0, theta[win.n_x0], theta[win.n_x0 + 1:]


def initial_theta(win):
    E0 = [INIT_E0] if win.n_x0 else []
    return np.array([*E0, win.setup.init_rho, 0.0, 0.0, win.setup.init_beta])


def anchored_frame(whole_interval_frame, record):
    row = record.iloc[0]
    anchor = {c: (row[c] if c in record.columns else np.nan) for c in whole_interval_frame.columns}
    anchor.update(month=pd.Timestamp(row["date"]).month, cases=np.nan)

    below_index = pd.DataFrame([anchor], index=[-1])
    return pd.concat([below_index, whole_interval_frame]).astype(whole_interval_frame.dtypes.to_dict())


def build_window(anchored, point_rows, setup, init_state=None, n_sub=SUB_STEPS):
    X = season_matrix(anchored["month"].reindex(point_rows).to_numpy())
    return Window(
        X=interp_points(X, 2 * n_sub),
        pop=anchored["pop"].reindex(point_rows).to_numpy(dtype=float),
        cases=anchored["cases"].reindex(point_rows[1:]).to_numpy(dtype=float),
        n_points=len(point_rows),
        setup=setup, init_state=init_state, n_sub=n_sub,
    )


def trajectory(theta, win):
    x0, _, coef = unpack(theta, win)
    beta = beta_curve(coef, win.X, win.setup.beta_min, win.setup.beta_max)
    return integrate_states(x0, beta, win)


# ##################### solving one window ######################################
def fit_window(win, theta0=None):
    theta = theta0 if theta0 is not None else initial_theta(win)

    lo = np.full(len(theta), -np.inf)
    hi = np.full(len(theta), np.inf)
    lo[:win.n_x0], hi[:win.n_x0] = 0.0, FREE_MASS
    lo[win.n_x0], hi[win.n_x0] = win.setup.rho_min, win.setup.rho_max
    lo[-1], hi[-1] = win.setup.beta_min, win.setup.beta_max

    res = least_squares(lambda t: predicted_incidence(t, win) - win.cases,
                        np.clip(theta, lo, hi), bounds=(lo, hi))
    return res.x, res


# ##################### the model ######################################
class SEIRSIndex:
    def __init__(self, window, anchored, location, epi_tag):
        self.window = window
        self.anchored = anchored
        self.location = location
        self.epi_tag = epi_tag
        self.setup = epi_setup(location, epi_tag)
        self.name = f"seirs_{form_tag(window)}"
        self.snapshots = {}  # origin -> (theta, state at the origin)
        self.walked = None

    def fit(self, last_origin):
        first = int(self.window) - 1
        if last_origin < first:
            raise ValueError("cannot be fitted here")
        state = None  # cold start: the first window fits x0
        for origin in range(first, last_origin + 1):
            state, _ = self._fit_at(origin, state, None)
        self.walked = last_origin
        return self

    def _fit_at(self, origin, init_state, theta0):
        start = origin - self.window + 1
        point_rows = np.arange(start - 1, origin + 1)
        win = build_window(self.anchored, point_rows, self.setup, init_state=init_state)

        theta, _ = fit_window(win, theta0)
        states = trajectory(theta, win)
        self.snapshots[origin] = (theta, states[-1])

        # the next window slides one month on
        return states[1], theta[self._n_x0(theta):]

    def _n_x0(self, theta):
        return len(theta) - 4

    def predict(self, origin, future_interval_indices):
        """e.g., predict(58, [59, 60, 61]), nothing is refitted"""
        if origin not in self.snapshots:
            raise ValueError(f"{self.name} has no window ending at origin {origin}.")
        theta, end_state = self.snapshots[origin]

        rows = np.asarray(future_interval_indices, dtype=int)
        if rows[0] != origin + 1 or np.any(np.diff(rows) != 1):
            raise ValueError(f"index wrong")
        point_rows = np.concatenate([[origin], rows])

        win = build_window(self.anchored, point_rows, self.setup, init_state=end_state)
        return predicted_incidence(theta[self._n_x0(theta):], win)


# ##################### form selection ######################################
LADDER = (10, 12, 14, 16, 18, 36, 42)


def fit_windows(whole_interval_frame, ladder=LADDER):
    from .baseline_utils import MIN_SELECT_ORIGINS, train_end

    n_train = train_end(whole_interval_frame) + 1
    cap = n_train - MIN_SELECT_ORIGINS
    kept = tuple(w for w in ladder if w <= cap)
    if not kept:
        raise ValueError(f"no window in {ladder} is eligible")
    return kept


def seirs_roll(whole_interval_frame, location, record, max_horizon, epi_tag):
    """rolling forecast for specific window"""
    return lambda window, origins, last_row: seirs_forecast(
        window, whole_interval_frame, location, record, max_horizon, epi_tag,
        origins=origins, last_row=last_row)


def seirs_forecast(window, whole_interval_frame, location, record, max_horizon, epi_tag,
                   origins=None, last_row=None):
    from .baseline_utils import forecast_rows, forecast_table, test_origins

    origins = test_origins(whole_interval_frame) if origins is None else origins
    last_row = len(whole_interval_frame) - 1 if last_row is None else last_row

    anchored = anchored_frame(whole_interval_frame, record)
    model = SEIRSIndex(window, anchored, location, epi_tag).fit(max(origins))

    rows = []
    for origin in origins:
        future = whole_interval_frame.iloc[origin + 1:min(origin + max_horizon, last_row) + 1]
        if origin in model.snapshots:
            mu, unfittable = np.asarray(model.predict(origin, future.index), dtype=float), ""
        else:
            mu = np.full(len(future), np.nan)
            unfittable = f"{window} interval(s) of history are needed before the first origin"
        rows += forecast_rows(whole_interval_frame, origin, future, mu, unfittable)

    return forecast_table(rows)


def select_form(whole_interval_frame, location, windows, record, epi_tag, max_horizon=6, trace=None):
    from .baseline_utils import select_by_rolling_error

    return select_by_rolling_error(
        windows,
        seirs_roll(whole_interval_frame, location, record, max_horizon, epi_tag),
        form_tag,
        whole_interval_frame, max_horizon, what="SEIRS", trace=trace)
