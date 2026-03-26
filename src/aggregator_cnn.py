import torch
from torch import Tensor, nn

class AggregatorCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_conv_5s: int,
        num_conv_3s: int,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        assert num_conv_3s >= 1
        layers = []
        for _ in range(num_conv_5s):
            if batch_norm:
                layers.append(nn.BatchNorm2d(input_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ))
        for i in range(num_conv_3s):
            if batch_norm:
                layers.append(nn.BatchNorm2d(input_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Conv2d(
                input_channels,
                input_channels if i < num_conv_3s - 1 else output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ))
        self._net = nn.Sequential(*layers)


    def forward(self, xs: list[Tensor]) -> Tensor:
        agg_in = torch.cat(xs, dim = 1)
        return self._net(agg_in)