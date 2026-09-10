from typing import Literal
import torch


def poisson_deviance(
        obs, pred_mean,
        eps: float = 1e-8,
        reduction: Literal["sum", "mean", "none"] = "mean"):
    mu = pred_mean.clamp_min(eps)
    lead = torch.where(obs > 0, obs * (torch.log(obs.clamp_min(eps)) - torch.log(mu)),
                       torch.zeros_like(mu))
    dev = 2.0 * (lead - (obs - mu))

    if reduction == "sum":
        return dev.sum()
    if reduction == "mean":
        return dev.mean()
    else:  # "none"
        return dev
