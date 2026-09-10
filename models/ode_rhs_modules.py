import torch
from torch import Tensor
import torch.nn as nn
import math


# ---------------------------------------------------------------------
# Beta network
# ---------------------------------------------------------------------
class BetaNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, hidden_layers: int, max_beta: float, min_beta: float,
                 init_beta: float):
        super().__init__()
        self.register_buffer('max_beta', torch.tensor(float(max_beta), dtype=torch.get_default_dtype()))
        self.register_buffer('min_beta', torch.tensor(float(min_beta), dtype=torch.get_default_dtype()))
        self.register_buffer('init_beta', torch.tensor(float(init_beta), dtype=torch.get_default_dtype()))

        # --------- define model ----------
        layers = []
        for l_i in range(hidden_layers):
            if l_i == 0:
                layers += [nn.Linear(in_dim, hidden_dim), nn.Softplus()]
            else:
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Softplus()]
        layers += [nn.Linear(hidden_dim if hidden_layers else in_dim, 1)]
        self.net = nn.Sequential(*layers)

        # --------- init layers ------------
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # then adjust only the last layer
        p = (init_beta - min_beta) / (max_beta - min_beta)
        z0 = math.log(p / (1.0 - p))

        with torch.no_grad():
            last_linear: nn.Linear = self.net[-1]
            last_linear.weight.mul_(1e-3)
            last_linear.bias.fill_(z0)

    def forward(self, x: Tensor) -> Tensor:
        z = self.net(x)
        beta = self.min_beta + (self.max_beta - self.min_beta) * torch.sigmoid(z)
        return beta.squeeze(-1)


# ---------------------------------------------------------------------
#  Generic ODE mechanism
# ---------------------------------------------------------------------
class BaseMechanismDynBeta(nn.Module):
    state_names = ()
    required_rates = ()

    def __init__(self, state_dim: int):
        super().__init__()
        if not self.state_names:
            raise NotImplementedError(f"{type(self).__name__} must declare state_names")
        self.state_dim = state_dim

    # --------- shared utility ----------
    @staticmethod
    def _interp_time_series(t: Tensor, series: Tensor) -> Tensor:
        assert t.ndim == 0, "Expected scalar t for ODE solver."
        assert series.ndim == 3, "Expected series of shape (B, T, d)"

        B, T, d = series.shape

        idx0 = torch.floor(t).clamp(0, T - 2).to(dtype=torch.long)
        idx1 = (idx0 + 1).clamp(max=T - 1)

        w = t - idx0

        x0 = series[:, idx0, :]  # (B, d)
        x1 = series[:, idx1, :]  # (B, d)

        return (1.0 - w) * x0 + w * x1  # (B, d)

    @staticmethod
    def interp_time_series_vector(beta_lag_feats_all_steps: torch.Tensor, inter_points: int):
        B, T, d = beta_lag_feats_all_steps.shape
        t_steps_dense = torch.linspace(0., float(T - 1), (T - 1) * (inter_points - 1) + T)
        Td = t_steps_dense.numel()

        idx0 = torch.floor(t_steps_dense).clamp(0, T - 2).long()
        idx1 = (idx0 + 1).clamp(max=T - 1)
        w = (t_steps_dense - idx0).view(1, Td, 1)
        idx0_g = idx0.view(1, Td, 1).expand(B, Td, d)
        idx1_g = idx1.view(1, Td, 1).expand(B, Td, d)
        x0 = beta_lag_feats_all_steps.gather(dim=1, index=idx0_g)
        x1 = beta_lag_feats_all_steps.gather(dim=1, index=idx1_g)

        X_dense = (1.0 - w) * x0 + w * x1
        return X_dense, Td

    def set_context(self, beta_lag_feats_all_steps: Tensor) -> None:
        raise NotImplementedError

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        """RHS dy/dt"""
        raise NotImplementedError


# ---------------------------------------------------------------------
#  Specific ODE mechanisms
# ---------------------------------------------------------------------
class DynBetaSEIRSrhs(BaseMechanismDynBeta):
    state_names = ("S", "E", "I", "R", "CUM_INC")
    required_rates = ("sigma", "gamma", "omega", "mu_pop")

    def __init__(
            self,
            part_state_dim: int,
            # for beta training
            beta_model_in_dim: int,
            beta_model_hidden_dim: int,
            beta_model_hidden_layers: int,
            beta_setting: tuple,   # (init_beta, min_beta, max_beta)
            # fixed rates
            sigma: float, gamma: float, omega: float, mu_pop: float,
    ):
        super().__init__(state_dim=part_state_dim + 1)  # see state_names

        init_beta, min_beta, max_beta = beta_setting
        self.beta_model = BetaNN(
            in_dim=beta_model_in_dim,
            hidden_dim=beta_model_hidden_dim,
            hidden_layers=beta_model_hidden_layers,
            min_beta=min_beta, max_beta=max_beta, init_beta=init_beta)
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.register_buffer("omega", torch.tensor(float(omega)))
        self.register_buffer("mu_pop", torch.tensor(float(mu_pop)))

        # per-batch context
        self.beta_lag_feats_all_steps = None  # (B, T, d)

    def set_context(self, beta_lag_feats_all_steps: Tensor) -> None:
        self.beta_lag_feats_all_steps = beta_lag_feats_all_steps

    def forward(self,
                t: Tensor,
                y: Tensor) -> Tensor:
        """
        RHS: dy/dt for SEIRS.
        """
        assert self.beta_lag_feats_all_steps is not None, "Call set_context first."

        # Interpolate features at time t
        beta_feats_t = self._interp_time_series(t, self.beta_lag_feats_all_steps)  # (B, d)
        beta_t = self.beta_model(beta_feats_t)  # (B,)

        sigma = self.sigma
        gamma = self.gamma
        omega = self.omega
        mu_pop = self.mu_pop

        s, e, i, r, cum_inc = y.unbind(-1)  # each (B,)

        # elementwise SEIRS RHS
        ds = mu_pop - beta_t * s * i + omega * r - mu_pop * s
        de = beta_t * s * i - sigma * e - mu_pop * e
        di = sigma * e - gamma * i - mu_pop * i
        dr = gamma * i - omega * r - mu_pop * r
        d_cum_inc = sigma * e

        dy = torch.stack([ds, de, di, dr, d_cum_inc], dim=-1)  # (B, 5)
        return dy
