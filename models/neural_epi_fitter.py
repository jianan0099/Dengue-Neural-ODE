import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange
from typing import Any, Optional
from dataclasses import dataclass
import math
import os
import sys
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR
from .neural_epi_modules import DynBetaOde
from .losses import poisson_deviance
from utils.general_utils import save_dfs_atomic


use_tqdm = sys.stdout.isatty()


@dataclass
class FitConfig:
    segment_len: int = 18            # incidence intervals per fitting window
    beta_pen_lam: float = 0.1
    first_seg_epochs: int = 2000
    later_seg_epochs: int = 250
    lr: float = 1.5e-3
    warmup_frac: float = 0.05
    explore_epochs_frac: float = 0.03
    final_lr: float = 5e-4


@dataclass
class FitResult:
    n_segments: int
    segment_spans: list
    history: list
    segment_diagnostics: list


def point_date(dates, i):
    if dates is None or not 0 <= i < len(dates):
        return None
    return dates[i]


class NeuralEpiFitter:
    def __init__(self, neural_epi: DynBetaOde, fit_cfg: FitConfig):
        self.neural_epi = neural_epi
        self.model_cfg = neural_epi.model_cfg
        self.fit_cfg = fit_cfg

        self.optimizer = self._define_optimizer()
        self.scheduler = None
        self.warmup_epochs = max(0, int(self.fit_cfg.warmup_frac * self.fit_cfg.first_seg_epochs))

    # --------------------
    # freezing helpers
    # --------------------
    @staticmethod
    def _set_requires_grad(module_or_param, flag: bool):
        if module_or_param is None:
            return
        if isinstance(module_or_param, torch.nn.Parameter):
            module_or_param.requires_grad_(flag)
        else:
            for p in module_or_param.parameters():
                p.requires_grad_(flag)

    def _maybe_freeze_warmup_params(self, epoch: int):
        if self.warmup_epochs <= 0:
            return

        if epoch == 0:
            print('freeze beta, rho')
            self._set_requires_grad(self.neural_epi.mechanism.beta_model, False)
            self._set_requires_grad(self.neural_epi.sigmoid_theta_rho, False)

        if epoch == self.warmup_epochs:
            print('unfreeze beta')
            self._set_requires_grad(self.neural_epi.mechanism.beta_model, True)

        if epoch == 2 * self.warmup_epochs:
            print('unfreeze rho')
            self._set_requires_grad(self.neural_epi.sigmoid_theta_rho, True)

    # --------------------
    # optimizer
    # --------------------
    def _define_optimizer(self):
        base_lr = float(self.fit_cfg.lr)

        params = [p for p in self.neural_epi.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters (all requires_grad=False). Check warmup freezing.")

        return torch.optim.Adam(params, lr=base_lr)

    def _define_scheduler(self, total_epochs, explore_epochs_frac, final_lr):
        explore_epochs = math.floor(total_epochs * explore_epochs_frac)
        print(f"explore epochs: {explore_epochs}")
        scheduler1 = ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=explore_epochs
        )

        scheduler2 = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - explore_epochs,
            eta_min=final_lr
        )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[explore_epochs]
        )
        return scheduler

    def _reset_optimizer_lrs(self):
        base_lr = float(self.fit_cfg.lr)
        for g in self.optimizer.param_groups:
            g["lr"] = base_lr

    # --------------------
    # loss helpers
    # --------------------
    def beta_l2_penalty(self):
        lam = self.fit_cfg.beta_pen_lam
        penalty = 0.0
        for module in self.neural_epi.mechanism.beta_model.modules():
            if isinstance(module, torch.nn.Linear):
                penalty += module.weight.pow(2).sum()
        return lam * penalty

    def _compute_obs_INC_based_losses(self, true_obs_INC, pred_obs_INC):
        data_loss = poisson_deviance(true_obs_INC, pred_obs_INC)
        beta_pen = self.beta_l2_penalty()
        loss = data_loss + beta_pen

        return loss, data_loss

    # --------------------
    # train
    # --------------------
    def _optimizer_step(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            POP: torch.Tensor,
            init_state: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)

        pred_traj, pred_rho, pred_x0, _ = self.forward_segment(X, init_state, with_beta=False)
        pred_obs_INC = self.neural_epi.obs_INC_cal(pred_traj, pred_rho, POP)

        loss, data_loss = self._compute_obs_INC_based_losses(Y, pred_obs_INC)

        loss.backward()
        self.optimizer.step()

        return {
            "pred_rho": pred_rho,
            "pred_x0": pred_x0,
            "loss": loss,
            "data_loss": data_loss,
        }

    # ---------------------------------------------------------------------
    # Segments helper
    # ---------------------------------------------------------------------
    @staticmethod
    def _make_segments(X, Y, POP, seg_len: int):
        B, T, d = X.shape
        assert Y.shape == (B, T - 1), f"Y must be (B,T-1). Got {Y.shape} vs {(B, T - 1)}"
        assert POP.shape == (T, B), f"POP must be (T,B). Got {POP.shape} vs {(T, B)}"

        seg_len = int(seg_len)
        if seg_len < 1:
            raise ValueError(f"segment_len must be >= 1 incidence interval. Got {seg_len}")

        seg_points = seg_len + 1
        if T < seg_points:
            raise ValueError("Data is shorter than the fitting window")

        segs = []
        for start in range(0, T - seg_len):
            end = start + seg_points
            segs.append((start, end, X[:, start:end, :], Y[:, start:end - 1], POP[start:end, :]))

        return segs

    def forward_segment(self, Xs: torch.Tensor, init_state: Optional[torch.Tensor], with_beta: bool = True):
        if init_state is None:
            return self.neural_epi(Xs, with_beta=with_beta)
        else:
            if init_state.shape != (Xs.shape[0], self.neural_epi.state_dim):
                raise ValueError('boundary shape wrong!')
        return self.neural_epi.forward_with_init(Xs, init_state, with_beta=with_beta)

    @torch.no_grad()
    def _get_boundary_state(self, Xs: torch.Tensor, init_state: Optional[torch.Tensor]) -> torch.Tensor:
        self.neural_epi.eval()
        pred_traj, *_ = self.forward_segment(Xs, init_state, with_beta=False)
        return pred_traj[1].detach().clone()  # (B, state_dim)

    @torch.no_grad()
    def _segment_diagnostics(
            self,
            *,
            seg_id: int,
            start: int,
            end: int,
            X: torch.Tensor,
            POP: torch.Tensor,
            init_state: Optional[torch.Tensor],
            alt_covariates: Optional[dict] = None,
            dates: Optional[list] = None,
    ) -> dict[str, Any]:
        self.neural_epi.eval()
        T = X.shape[1]

        X_span = X[:, start:, :]    # (B, T-start, d)
        POP_span = POP[start:, :]   # (T-start, B)

        traj, rho, x0, _ = self.forward_segment(X_span, init_state, with_beta=False)
        pred_inc = self.neural_epi.obs_INC_cal(traj, rho, POP_span)  # (B, T-start-1)

        B, T_span, d = X_span.shape
        beta_pts = self.neural_epi.mechanism.beta_model(
            X_span.reshape(B * T_span, d)
        ).reshape(B, T_span)  # (B, T-start)

        alt_pred_inc = {}
        if alt_covariates is not None:
            origin_date = point_date(dates, end - 1)
            for name, provider in alt_covariates.items():
                X_future = provider(origin_date) if origin_date is not None else None
                if X_future is None:
                    continue
                L = X_future.shape[1]
                X_alt = torch.cat([X[:, start:end, :], X_future], dim=1)   # (B, end-start+L, d)
                traj_alt, rho_alt, _, _ = self.forward_segment(X_alt, init_state, with_beta=False)
                assert torch.equal(rho_alt, rho), f"[{name}] alt roll changed rho {rho.item()} -> {rho_alt.item()}"
                alt_pred_inc[name] = {
                    "span": (start, end + L),
                    "future_start": end,
                    "X_future": X_future[0].detach().clone(),
                    "pred_inc": self.neural_epi.obs_INC_cal(traj_alt, rho_alt, POP[start:end + L, :]).detach().clone(),
                }

        return {
            "segment_id": seg_id,
            "segment_start": start,
            "segment_end": end,
            "span": (start, T),

            "init_state": x0.detach().clone(),
            "rho": float(rho.item()),
            "beta": beta_pts.detach().clone(),
            "pred_inc": pred_inc.detach().clone(),
            "traj": traj.detach().clone(),
            "alt_pred_inc": alt_pred_inc,
        }

    # ---------------------------------------------------------------------
    # fit()
    # ---------------------------------------------------------------------
    def fit(self, X, Y, POP, save_path,
            data_rel_path=None, feats=None, alt_covariates=None, dates=None,
            on_segment=None, init_state=None) -> FitResult:
        save_path = Path(save_path)

        if alt_covariates and dates is None:
            raise ValueError("alt_covariates needs `dates` to name each segment's origin.")
        if dates is not None and len(dates) < X.shape[1]:
            raise ValueError(f"dates has {len(dates)} entries, fewer than X's {X.shape[1]} points.")

        if X.shape[-1] != self.model_cfg.beta_input_dim:
            raise ValueError(
                f"X has {X.shape[-1]} beta features but the model was built for "
                f"beta_input_dim={self.model_cfg.beta_input_dim}."
            )

        if feats is None or len(feats) != X.shape[-1]:
            raise ValueError(f"feats must name all {X.shape[-1]} beta features; got {feats}.")

        if init_state is not None and init_state.shape != (X.shape[0], self.neural_epi.state_dim):
            raise ValueError(f"init_state must be (B, state_dim) = "
                             f"{(X.shape[0], self.neural_epi.state_dim)}; got {tuple(init_state.shape)}.")

        seg_len = int(self.fit_cfg.segment_len)
        segments = self._make_segments(X, Y, POP, seg_len=seg_len)
        n_segs = len(segments)

        continued = init_state is not None
        global_ep = 0
        next_init_state = init_state
        history = []
        diagnostics = []

        for seg_id, (start, end, Xs, Ys, POPs) in enumerate(segments, start=1):
            seg_init_state = next_init_state
            cold = seg_id == 1 and not continued
            seg_epochs = self.fit_cfg.first_seg_epochs if cold else self.fit_cfg.later_seg_epochs

            self._reset_optimizer_lrs()
            self.scheduler = self._define_scheduler(seg_epochs, self.fit_cfg.explore_epochs_frac, self.fit_cfg.final_lr)

            print(
                f"\n=== Segment {seg_id}/{n_segs}  [t={start}:{end}]  "
                f"(points={end - start}, intervals={end - start - 1}) ===")

            hist = _make_hist_dict()

            pbar = trange(seg_epochs, disable=not use_tqdm, file=sys.stdout)
            for local_ep in pbar:
                if not continued:
                    self._maybe_freeze_warmup_params(global_ep)

                self.neural_epi.train()

                cache = self._optimizer_step(Xs, Ys, POPs, init_state=seg_init_state)

                self.scheduler.step()
                global_ep += 1
                _append_cache_to_hist(hist, cache)
                self._log_beta_summary(Xs, hist)

                postfix = {k[:3]: f"{hist[k][-1]:.4g}" for k in LOSS_KEYS}

                if local_ep % 50 == 0:
                    msg = (
                        f"Ep {local_ep} | LR {self.scheduler.get_last_lr()[0]:.2e} | "
                        f"beta_info {hist['beta_mean'][-1]:.2g} | {hist['beta_min'][-1]:.2g}"
                    )
                    print(msg, flush=True)
                    if use_tqdm:
                        pbar.set_postfix(postfix)

            history.append(hist)

            diagnostics.append(self._segment_diagnostics(
                seg_id=seg_id, start=start, end=end, X=X, POP=POP, init_state=seg_init_state,
                alt_covariates=alt_covariates, dates=dates,
            ))

            self.save_atomic(save_path, extra={
                "segment_id": seg_id,
                "segment_start": start,
                "segment_end": end,
                "init_state": seg_init_state,
                "train_hist": history,
                "data_rel_path": data_rel_path,
                "beta_input_feats": list(feats),
            })

            if on_segment is not None:
                on_segment(diagnostics)

            if seg_id < n_segs:
                next_init_state = self._get_boundary_state(Xs, init_state=seg_init_state)

        print(f"\nSegmented training done! Checkpoint: {save_path}")
        return FitResult(
            n_segments=n_segs,
            segment_spans=[(s, e) for s, e, *_ in segments],
            history=history,
            segment_diagnostics=diagnostics,
        )

    def _log_beta_summary(self, X, hist_dict):
        with torch.no_grad():
            B, T, d = X.shape
            beta_ts = self.neural_epi.mechanism.beta_model(X.reshape(B * T, d)).reshape(B, T)

            hist_dict['beta_mean'].append(float(beta_ts.mean().item()))
            hist_dict['beta_min'].append(float(beta_ts.min().item()))
            hist_dict['beta_max'].append(float(beta_ts.max().item()))

    # --------------------
    # saving / loading
    # --------------------
    def _checkpoint(self, extra=None):
        checkpoint = {
            "model_config": vars(self.model_cfg),
            "model_state_dict": self.neural_epi.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
            "fit_config": vars(self.fit_cfg),
        }
        if extra is not None:
            checkpoint["extra"] = extra
        return checkpoint

    def save_atomic(self, path, extra=None):
        path = Path(path)
        tmp_path = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self._checkpoint(extra), tmp_path)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)


def load_fitter(path, fit_cfg=None):
    from .neural_epi_modules import ModelConfig, build_neural_epi

    ckpt = torch.load(Path(path), weights_only=False)
    neural_epi = build_neural_epi(ModelConfig(**ckpt["model_config"]))
    neural_epi.load_state_dict(ckpt["model_state_dict"])

    fitter = NeuralEpiFitter(neural_epi, fit_cfg or FitConfig(**ckpt["fit_config"]))
    fitter.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return fitter, ckpt.get("extra", {})


# ------ Log & Diagnosis Funcs -------------
LOSS_KEYS = ("loss", "data_loss")


def _make_hist_dict() -> dict[str, Any]:
    return {
        "loss": [],
        "data_loss": [],

        "rho": [],

        "x0": [],
        "beta_mean": [],
        "beta_min": [],
        "beta_max": [],
    }


def _append_cache_to_hist(hist: dict[str, Any], cache: dict[str, Any]):
    hist["loss"].append(float(cache["loss"].item()))
    hist["data_loss"].append(float(cache["data_loss"].item()))
    hist["rho"].append(float(cache["pred_rho"].mean().item()))
    hist["x0"].append(cache["pred_x0"].detach().numpy().tolist())


def segment_diagnostics_to_excel(segment_diagnostics, dates, state_names, file_path, feats=None):
    dates = list(dates)
    idx_pts = pd.Index(dates, name="date")        # points
    idx_itv = pd.Index(dates[1:], name="date")    # intervals

    cols = [f"seg_{s['segment_id']}" for s in segment_diagnostics]
    inc = pd.DataFrame(index=idx_itv, columns=cols, dtype=float)
    beta = pd.DataFrame(index=idx_pts, columns=cols, dtype=float)
    rho = pd.DataFrame(index=idx_pts, columns=cols, dtype=float)

    for s in segment_diagnostics:
        c = f"seg_{s['segment_id']}"
        start = int(s["segment_start"])

        inc.iloc[start:, inc.columns.get_loc(c)] = np.asarray(s["pred_inc"][0])
        beta.iloc[start:, beta.columns.get_loc(c)] = np.asarray(s["beta"][0])
        rho.iloc[start:, rho.columns.get_loc(c)] = float(s["rho"])

    alt_names = sorted({n for s in segment_diagnostics for n in s.get("alt_pred_inc", {})})
    alt_incs = {}
    for name in alt_names:
        frame = pd.DataFrame(index=idx_itv, columns=cols, dtype=float)
        for s in segment_diagnostics:
            got = s.get("alt_pred_inc", {}).get(name)
            if got is None:
                continue
            start = int(s["segment_start"])
            values = np.asarray(got["pred_inc"][0])
            assert start + len(values) == got["span"][1] - 1, \
                f"seg {s['segment_id']} {name}: span/pred_inc disagree"
            frame.iloc[start:start + len(values), frame.columns.get_loc(f"seg_{s['segment_id']}")] = values
        alt_incs[f"incidence_{name}"] = frame

    alt_X = pd.DataFrame(
        [
            {"segment_id": s["segment_id"], "scenario": name,
             "date": dates[got["future_start"] + i],
             **{f: float(v) for f, v in zip(feats or [], row)}}
            for s in segment_diagnostics
            for name, got in s.get("alt_pred_inc", {}).items()
            for i, row in enumerate(np.asarray(got["X_future"]))
        ],
        columns=["segment_id", "scenario", "date", *(feats or [])],
    )

    x0 = pd.DataFrame(
        [
            {"segment_id": s["segment_id"],
             "start_date": dates[int(s["segment_start"])],
             **{name: float(s["init_state"][k]) for k, name in enumerate(state_names)}}
            for s in segment_diagnostics
        ]
    ).set_index("segment_id")

    last = segment_diagnostics[-1]
    trajectory = pd.DataFrame(np.asarray(last["traj"][:, 0, :]), columns=list(state_names),
                              index=idx_pts[int(last["segment_start"]):])

    sheets = {"incidence": inc, **alt_incs, "beta": beta, "rho": rho, "x0": x0, "trajectory": trajectory,
              **({"alt_covariates": alt_X} if not alt_X.empty else {})}
    save_dfs_atomic(
        file_path,
        dfs=list(sheets.values()),
        sheet_names=list(sheets),
        indexes=[name != "alt_covariates" for name in sheets],
    )
    return sheets
