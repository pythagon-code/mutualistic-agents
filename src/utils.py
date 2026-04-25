import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal, TanhTransform, TransformedDistribution
from torch.nn.functional import logsigmoid
from typing import TypeVar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def polyak_update(target: nn.Module, online: nn.Module, polyak_factor: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(online_param.data * polyak_factor + target_param.data * (1 - polyak_factor))


def flatten_2d_module_list(module_list: nn.ModuleList) -> list[nn.Module]:
    return [module for sublist in module_list for module in sublist]


def get_bradley_terry_loss(
        q1: Tensor,
        q2: Tensor,
        log_prob1: Tensor,
        log_prob2: Tensor,
        c: float = 1e-4,
        T: float = 1.0,
    ) -> tuple[Tensor, float]:
    score1 = log_prob1.nan_to_num(nan = 0.0).clamp(min = -10, max = 10)
    score2 = log_prob2.nan_to_num(nan = 0.0).clamp(min = -10, max = 10)
    with torch.no_grad():
        weight = abs(q1 - q2)
        weight = weight / weight.mean() + c
    score_diff = score1 - score2
    score_contrast = torch.where(q1 > q2, score_diff, -score_diff)
    loss = -(T * weight * logsigmoid(score_contrast)).mean()
    return loss, score_contrast.mean().item()


def get_multivariate_normal_size(action_dim: int) -> int:
    return action_dim + action_dim * (action_dim + 1) // 2


def get_tanh_multivariate_normal(params: Tensor, action_dim: int) -> TransformedDistribution:
    loc = params[:, : action_dim]
    scale = params[:, action_dim :]
    L = torch.zeros(params.shape[0], action_dim, action_dim, device = params.device)
    tril_indices = torch.tril_indices(row = action_dim, col = action_dim)
    L[:, tril_indices[0], tril_indices[1]] = scale
    L_diag_idx = torch.arange(action_dim, device = params.device)
    L[:, L_diag_idx, L_diag_idx] = torch.exp(L[:, L_diag_idx, L_diag_idx].clamp(min = -8, max = 2))
    return TransformedDistribution(
        MultivariateNormal(loc, scale_tril = L),
        TanhTransform(cache_size = 1),
    )