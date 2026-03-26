from math import sqrt
from torch import Tensor, nn

from .fnn import FNN
from .utils import View

class EncoderCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_channels: int,
        num_conv_t_3s: int,
        num_upscales: int,
        num_conv_5s: int,
        num_conv_3s: int,
        batch_norm: bool = True,
        dropout_rate: float = .0,
    ) -> None:
        super().__init__()
        layers = []
        layers.append(FNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            output_size = hidden_size,
            layer_norm = True,
            dropout_rate = dropout_rate,
        ))
        initial_dim = int(sqrt(hidden_size / num_channels))
        assert initial_dim ** 2 * num_channels == hidden_size
        layers.append(View((-1, num_channels, initial_dim, initial_dim)))
        for _ in range(num_upscales):
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.ConvTranspose2d(
                num_channels,
                num_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ))
        for _ in range(num_upscales):
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.ConvTranspose2d(
                num_channels,
                num_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1,
            ))
        for _ in range(num_conv_5s):
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ))
        for _ in range(num_conv_3s):
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ))
        self._net = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)