import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Tuple, Union
from dataclasses import dataclass, field
from .ode_rhs_modules import BaseMechanismDynBeta, DynBetaSEIRSrhs


class DynBetaOde(nn.Module):
    def __init__(
            self,
            mechanism: BaseMechanismDynBeta,
            # ------ for init state training -------
            init_state_parts: list,
            fixed_init_state_list: list,
            # ------ for generating observed incidence -------
            rho_setting: tuple,
            # ------ solver --------------------
            solver: str = "rk4"
    ):
        super().__init__()

        self.mechanism = mechanism
        self.state_dim = mechanism.state_dim

        # -------- Initial states -------------
        self._init_x0_train(init_state_parts, fixed_init_state_list)

        # -------- Reporting rate -------------
        init_rho, min_rho, max_rho = rho_setting
        self.sigmoid_theta_rho = nn.Parameter(self._logit(torch.tensor((init_rho - min_rho) / (max_rho - min_rho))))
        self.register_buffer("min_rho", torch.tensor(float(min_rho)))
        self.register_buffer("max_rho", torch.tensor(float(max_rho)))

        # -------- solver config -----------------
        self.solver = solver

    def _init_x0_train(
            self,
            init_state_parts: list[float],
            fixed_init_state_list: list[int] | None,
    ):
        init_parts = torch.as_tensor(init_state_parts)
        parts_dim = self.state_dim - 1

        if init_parts.numel() != parts_dim:
            raise ValueError(f"init_state_parts must have length {parts_dim}, got {init_parts.numel()}")

        self.fixed_idx = sorted(set(int(i) for i in (fixed_init_state_list or [])))
        self.free_idx = [i for i in range(parts_dim) if i not in self.fixed_idx]

        # store fixed values (not trainable)
        fixed_vals = init_parts[self.fixed_idx] if self.fixed_idx else init_parts.new_zeros((0,))
        self.register_buffer("x0_fixed_vals", fixed_vals)

        fixed_sum = float(fixed_vals.sum().item()) if self.fixed_idx else 0.0
        if fixed_sum >= 1.0:
            print(f"Sum of fixed init parts is {fixed_sum} >= 1.0")

        if self.free_idx:
            free_init = init_parts[self.free_idx].clamp_min(1e-12)
            free_init = free_init / free_init.sum()
            self.softmax_theta_free_x0 = nn.Parameter(torch.log(free_init))
        else:
            self.softmax_theta_free_x0 = None

    @staticmethod
    def _min_max_sigmoid(_min, _max, _free_v):
        return _min + (_max - _min) * torch.sigmoid(_free_v)

    @staticmethod
    def _logit(p: Tensor, eps: float = 1e-12) -> Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _constrained(self) -> Tuple[Tensor, Tensor]:
        part_x0 = torch.zeros(self.state_dim - 1)

        fixed_sum = 0.0
        if self.fixed_idx:  # some values need to be fixed
            part_x0[self.fixed_idx] = self.x0_fixed_vals
            fixed_sum = float(self.x0_fixed_vals.sum().item())

        rem = max(1.0 - fixed_sum, 0.0)

        if self.free_idx:   # some values are set free
            free_probs = F.softmax(self.softmax_theta_free_x0, dim=0) * rem
            part_x0[self.free_idx] = free_probs

        x0 = torch.cat([part_x0, torch.zeros(())[None]], dim=0)  # (state_dim,) add cumulative incidence

        rho = self._min_max_sigmoid(self.min_rho, self.max_rho, self.sigmoid_theta_rho)
        return x0, rho

    def forward_with_init(
            self,
            beta_lag_feats_all_steps: Tensor,
            y0: Union[Tensor, None],
            *,
            with_beta: bool = True,
            inter_points: int = 30,
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, None]]:
        B, T, d = beta_lag_feats_all_steps.shape

        x0_learned, rho = self._constrained()

        t_steps = torch.linspace(0., float(T - 1), T)

        self.mechanism.set_context(beta_lag_feats_all_steps)

        if y0 is None:
            y0B = x0_learned.expand(B, self.state_dim).contiguous()
            x0_return = x0_learned
        else:
            y0B = y0.contiguous().detach()
            x0_return = y0B[0].clone()

        traj = odeint(
            func=self.mechanism,
            y0=y0B,
            t=t_steps,
            method=self.solver
        )

        beta_ts = None
        if with_beta:
            beta_feats_X, Td = self.mechanism.interp_time_series_vector(
                beta_lag_feats_all_steps, inter_points=inter_points)
            beta_ts = self.mechanism.beta_model(beta_feats_X.reshape(B * Td, d)).reshape(B, Td)

        return traj, rho, x0_return, beta_ts

    def forward(
            self,
            beta_lag_feats_all_steps: Tensor,
            *,
            with_beta: bool = True,
            inter_points: int = 30,
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, None]]:
        return self.forward_with_init(
            beta_lag_feats_all_steps, y0=None, with_beta=with_beta, inter_points=inter_points)

    @staticmethod
    def obs_INC_cal(traj: torch.Tensor, rho: torch.Tensor, pop_size,):
        cum_inc = traj[..., -1]
        cum_INC = cum_inc * pop_size
        INC = cum_INC[1:, :] - cum_INC[:-1, :]
        if rho.ndim == 0:
            obs_INC_TB = INC * rho
        else:
            rho_TB = rho.transpose(0, 1)
            obs_INC_TB = INC * rho_TB
        return obs_INC_TB.transpose(0, 1)


# ---------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------
@dataclass
class ModelConfig:
    # ----- epidemic model mechanism & initial states -----
    neural_epi_name: str = "SEIRS_FixKnSigGamOme"
    init_ode_state: list = field(default_factory=lambda: [0.9898, 0.0001, 0.0001, 0.01])
    fixed_init_state_list: list = field(default_factory=lambda: [2, 3])
    # ----- betaNN architecture -----
    beta_input_dim: int = 2
    beta_model_hidden_dim: int = 3
    beta_model_hidden_layers: int = 1
    # ----- beta -----
    init_beta: float = 5.0
    beta_prior_min: float = 1.5
    beta_prior_max: float = 20.0
    # ----- rho -----
    init_rho: float = 3.5e-3
    rho_prior_min: float = 1e-8
    rho_prior_max: float = 0.5
    # ----- other constant epidemiological parameters -----
    static_epi_paras: dict = field(
        default_factory=lambda: {"sigma": 30 / 5.9, "gamma": 30 / 7, "omega": 1 / 4.0, "mu_pop": 1 / (70.7 * 12)})
    # ----- ode solver ----------
    ode_solver: str = "dopri5"


# neural_epi_name -> RHS class
MECHANISMS = {"SEIRS_FixKnSigGamOme": DynBetaSEIRSrhs}


def build_neural_epi(model_cfg: ModelConfig) -> DynBetaOde:
    beta_setting = (float(model_cfg.init_beta),
                    float(model_cfg.beta_prior_min), float(model_cfg.beta_prior_max))
    rho_setting = (float(model_cfg.init_rho),
                   float(model_cfg.rho_prior_min), float(model_cfg.rho_prior_max))

    # ------- mechanism (right-hand side) ----------
    if model_cfg.neural_epi_name not in MECHANISMS:
        raise NotImplementedError(f"Epi mechanism '{model_cfg.neural_epi_name}' not implemented")
    rhs_cls = MECHANISMS[model_cfg.neural_epi_name]
    epi_rhs = rhs_cls(
        part_state_dim=len(rhs_cls.state_names) - 1,
        beta_model_in_dim=model_cfg.beta_input_dim,
        beta_model_hidden_dim=model_cfg.beta_model_hidden_dim,
        beta_model_hidden_layers=model_cfg.beta_model_hidden_layers,
        beta_setting=beta_setting,
        **{k: model_cfg.static_epi_paras[k] for k in rhs_cls.required_rates})

    # ------- neural ODE wrapping the mechanism ----------
    neural_epi = DynBetaOde(
        mechanism=epi_rhs,
        init_state_parts=model_cfg.init_ode_state,
        fixed_init_state_list=model_cfg.fixed_init_state_list,
        rho_setting=rho_setting,
        solver=model_cfg.ode_solver,
    )
    neural_epi.model_cfg = model_cfg
    return neural_epi
