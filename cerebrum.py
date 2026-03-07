import torch
from torch import Tensor, nn

class Cerebrum(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_qkv_layers: int,
        num_heads: int,
        num_levels: int,
        num_inputs_per_level: int,
    ) -> None:
        super().__init__()
        