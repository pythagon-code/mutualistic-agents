import torch
from torch import Tensor, nn
from typing import TypeVar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = TypeVar("T", float, Tensor)


def get_ema(ema: T, data: T, factor: float) -> T:
    return data * factor + ema * (1 - factor)


def get_ema_and_emv(ema: T, emv: T, data: T, factor: float) -> tuple[T, T]:
    var = (data - ema) ** 2
    return get_ema(ema, data, factor), get_ema(emv, var, factor)


def polyak_update(target: nn.Module, online: nn.Module, polyak_factor: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(get_ema(target_param.data, online_param.data, polyak_factor))


def flatten_module_list(module_list: nn.ModuleList) -> list[nn.Module]:
    return [module for sublist in module_list for module in sublist]