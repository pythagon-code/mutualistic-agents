import numpy as np
from torch import Tensor, nn

from .aggregator_cnn import AggregatorCNN
from .encoder_cnn import EncoderCNN
from .utils import flatten_2d_module_list, polyak_update

class ModularCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_channels: int,
        num_upscales: int,
        num_conv_5s: int,
        num_conv_3s: int,
        num_encoders: int,
        tree_depth: int,
        fan_in: int,
        rng: np.random.Generator,
        batch_norm: bool = True,
        dropout_rate: float = .0,
    ) -> None:
        super().__init__()
        assert fan_in ** (tree_depth - 1) == num_encoders
        self._fan_in = fan_in
        self._rng = rng
        self._encoders = nn.ModuleList([
            EncoderCNN(
                input_size        = input_size,
                hidden_size       = hidden_size,
                num_hidden_layers = num_hidden_layers,
                num_channels      = num_channels,
                num_upscales      = num_upscales,
                num_conv_5s       = num_conv_5s,
                num_conv_3s       = num_conv_3s,
                batch_norm        = batch_norm,
                dropout_rate      = dropout_rate,
            )
            for _ in range(num_encoders)
        ])
        self._aggregators = nn.ModuleList()
        for i in range(tree_depth - 2, -1, -1):
            level = nn.ModuleList([
                AggregatorCNN(
                    input_channels  = num_channels * fan_in,
                    output_channels = 1 if i == 0 else num_channels,
                    num_conv_5s     = num_conv_5s,
                    num_conv_3s     = num_conv_3s,
                    batch_norm      = batch_norm,
                )
                for _ in range(fan_in ** i)
            ])
            self._aggregators.append(level)

        self._target_encoders = nn.ModuleList([
            EncoderCNN(
                input_size        = input_size,
                hidden_size       = hidden_size,
                num_hidden_layers = num_hidden_layers,
                num_channels      = num_channels,
                num_upscales      = num_upscales,
                num_conv_5s       = num_conv_5s,
                num_conv_3s       = num_conv_3s,
                batch_norm        = batch_norm,
                dropout_rate      = dropout_rate,
            )
            for _ in range(num_encoders)
        ])
        self._target_aggregators = nn.ModuleList()
        for i in range(tree_depth - 2, -1, -1):
            level = nn.ModuleList([
                AggregatorCNN(
                    input_channels  = num_channels * fan_in,
                    output_channels = 1 if i == 0 else num_channels,
                    num_conv_5s     = num_conv_5s,
                    num_conv_3s     = num_conv_3s,
                    batch_norm      = batch_norm,
                )
                for _ in range(fan_in ** i)
            ])
            self._target_aggregators.append(level)

        self._online_modules = tuple((
            *self._encoders,
            *flatten_2d_module_list(self._aggregators),
        ))
        self._target_modules = tuple((
            *self._target_encoders,
            *flatten_2d_module_list(self._target_aggregators),
        ))
        assert len(self._online_modules) == len(self._target_modules)
        for online, target in zip(self._online_modules, self._target_modules):
            target.load_state_dict(online.state_dict())
            for param in target.parameters():
                param.requires_grad_(False)


    def forward(
        self,
        x: Tensor,
        online_module_id: int = -1,
        live_encoder_id: int = -1,
    ) -> Tensor:
        assert online_module_id == -1 or live_encoder_id == -1

        prev_embeds = []
        i = 0
        for enc, target_enc in zip(self._encoders, self._target_encoders):
            if i == online_module_id:
                prev_embeds.append(enc(x))
            else:
                prev_embeds.append(target_enc(x))
            i += 1
        if live_encoder_id != -1:
            for idx in range(len(prev_embeds)):
                if idx != live_encoder_id:
                    prev_embeds[idx] = prev_embeds[idx].detach()

        for lvl, target_lvl in zip(self._aggregators, self._target_aggregators):
            current_embeds = []
            for j, (agg, target_agg) in enumerate(zip(lvl, target_lvl)):
                start_idx = j * self._fan_in
                end_idx = start_idx + self._fan_in
                if i == online_module_id:
                    embed = agg(prev_embeds[start_idx : end_idx])
                else:
                    embed = target_agg(prev_embeds[start_idx : end_idx])
                current_embeds.append(embed)
                i += 1
            prev_embeds = current_embeds

        assert len(prev_embeds) == 1
        return prev_embeds[0].squeeze(dim = 1)


    def polyak_update(self, module_id: int, polyak_factor: float) -> None:
        polyak_update(
            target = self._target_modules[module_id],
            online = self._online_modules[module_id],
            polyak_factor = polyak_factor,
        )


    def get_random_module_id(self) -> int:
        return self._rng.integers(0, len(self._online_modules))


    def get_random_encoder_id(self) -> int:
        return self._rng.integers(0, len(self._encoders))