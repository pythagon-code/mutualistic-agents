import torch
from torch import Tensor, nn
from typing import TypeVar

class View(nn.Module):
    def __init__(self, shape: tuple[int]) -> None:
        super().__init__()
        self._shape = shape

    
    def forward(self, x: Tensor) -> Tensor:
        return x.view(self._shape)


class RangedTanh(nn.Module):
    def __init__(self, min_value: float, max_value: float) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value


    def forward(self, x: Tensor) -> Tensor:
        tanh_0_to_1 = (torch.tanh(x) + 1) / 2
        return self.min_value + tanh_0_to_1 * (self.max_value - self.min_value)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = TypeVar("T", float, Tensor)


def get_ema(ema: T, data: T, factor: float) -> T:
    return data * factor + ema * (1 - factor)


# def get_ema_and_emv(ema: T, emv: T, data: T, factor: float) -> tuple[T, T]:
#     var = (data - ema) ** 2
#     return get_ema(ema, data, factor), get_ema(emv, var, factor)


def polyak_update(target: nn.Module, online: nn.Module, polyak_factor: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(get_ema(target_param.data, online_param.data, polyak_factor))


def flatten_2d_module_list(module_list: nn.ModuleList) -> list[nn.Module]:
    return [module for sublist in module_list for module in sublist]