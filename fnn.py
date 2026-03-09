from torch import Tensor, nn


class FNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self._net = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)