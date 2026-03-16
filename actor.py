from config import Config
from critic import Critic
from fnn import FNN
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from utils import get_ema, get_ema_and_emv, polyak_update

class Actor(nn.Module):
    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        super().__init__()
        self._config = config
        self._rng = rng

        num_encoders = config.num_encoders
        state_dim = config.state_dim
        embed_dim = config.embed_dim
        num_hidden_layers = config.num_hidden_layers
        self._encoders = nn.ModuleList([
            FNN(
                input_size = state_dim,
                hidden_size = embed_dim,
                num_hidden_layers = num_hidden_layers,
                output_size = embed_dim,
            )
            for _ in range(num_encoders)
        ])
        self._target_encoders = nn.ModuleList([
            FNN(
                input_size = state_dim,
                hidden_size = embed_dim,
                num_hidden_layers = num_hidden_layers,
                output_size = embed_dim,
            )
            for _ in range(num_encoders)
        ])
        
        tree_depth = config.tree_depth
        self._fan_in = int(round(num_encoders ** (1 / (tree_depth - 1))))
        assert self._fan_in ** (tree_depth - 1) == num_encoders
        combined_dim = embed_dim * self._fan_in
        self._aggregators = nn.ModuleList()
        for i in range(tree_depth - 1):
            level = nn.ModuleList([
                FNN(
                    input_size = combined_dim,
                    hidden_size = combined_dim,
                    num_hidden_layers = num_hidden_layers,
                    output_size = embed_dim,
                )
                for _ in range(self._fan_in ** i)
            ])
            self._aggregators.append(level)
        self._target_aggregators = nn.ModuleList()
        for i in range(tree_depth - 1):
            level = nn.ModuleList([
                FNN(
                    input_size = combined_dim,
                    hidden_size = combined_dim,
                    num_hidden_layers = num_hidden_layers,
                    output_size = embed_dim,
                )
            ])
            self._target_aggregators.append(level)
        
        action_dim = config.action_dim
        self._head = FNN(
            input_size = embed_dim,
            hidden_size = embed_dim,
            num_hidden_layers = num_hidden_layers,
            output_size = action_dim,
        )
        self._target_head = FNN(
            input_size = embed_dim,
            hidden_size = embed_dim,
            num_hidden_layers = num_hidden_layers,
            output_size = action_dim,
        )

        online_models = [*self._encoders, *self._aggregators, self._head]
        target_models = [*self._target_encoders, *self._target_aggregators, self._target_head]
        for online, target in zip(online_models, target_models):
            target.load_state_dict(online.state_dict())
            for param in target.parameters():
                param.requires_grad_(False)

    
    def forward(self, state: Tensor, module_idx: int = -1) -> tuple[Tensor, Tensor]:
        i = 0
        prev_embeds = []
        for encoder, target_encoder in zip(self._encoders, self._target_encoders):
            if i == module_idx:
                prev_embeds.append(encoder(state))
            else:
                prev_embeds.append(target_encoder(state))
            i += 1
        all_embeds = prev_embeds.copy()
        
        for level, target_level in zip(self._aggregators, self._target_aggregators):
            current_embeds = []
            for j, (aggregator, target_aggregator) in enumerate(zip(level, target_level)):
                start_idx = j * self._fan_in
                end_idx = start_idx + self._fan_in
                aggregator_in = torch.cat(prev_embeds[start_idx : end_idx], dim = -1)
                if i == module_idx:
                    embed = aggregator(aggregator_in)
                else:
                    embed = target_aggregator(aggregator_in)
                current_embeds.append(embed)
                i += 1
            prev_embeds = current_embeds
            all_embeds += prev_embeds

        if i == module_idx:
            action = self._head(prev_embeds[0])
        else:
            action = self._target_head(prev_embeds[0])
        all_embeds = torch.stack(all_embeds)
        return all_embeds, action


    def train(self, critic: "Critic", states: Tensor) -> None:
        module_idx