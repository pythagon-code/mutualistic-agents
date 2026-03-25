from torch import Tensor, nn

class FNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
        output_activation: nn.Module = nn.Identity(),
        layer_norm: bool = True,
        spectral_norm: bool = False,
        dropout_rate: float = .0,
    ) -> None:
        super().__init__()
        assert not spectral_norm or dropout_rate == .0
        if spectral_norm:
            layer_norm = False
        layers = []
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layer = nn.Linear(in_dim, hidden_size)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.append(layer)
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
            if dropout_rate > .0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
        layer = nn.Linear(hidden_size, output_size)
        if spectral_norm:
            layer = nn.utils.spectral_norm(layer)
        layers.append(layer)
        layers.append(output_activation)
        self._net = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)