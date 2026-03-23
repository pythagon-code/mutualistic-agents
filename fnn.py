from torch import Tensor, nn


class FNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
        output_activation: nn.Module = nn.Identity(),
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self._net = nn.Sequential(*layers)
        self._output_activation = output_activation


    def forward(self, x: Tensor) -> Tensor:
        return self._output_activation(self._net(x))