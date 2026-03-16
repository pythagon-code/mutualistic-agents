from calendar import c
from config import Config
from fnn import FNN
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from utils import get_ema, get_ema_and_emv, polyak_update

class Critic(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

        num_encoders = config.num_encoders
        state_dim = config.state_dim
        embed_dim = config.embed_dim
        combined_dim = state_dim + embed_dim
        num_hidden_layers = config.num_hidden_layers
        self._encoders = nn.ModuleList([
            FNN(
                input_size = combined_dim,
                hidden_size = combined_dim,
                num_hidden_layers = num_hidden_layers,
                output_size = embed_dim,
            )
            for _ in range(num_encoders)
        ])
        self._target_encoders = nn.ModuleList([
            FNN(
                input_size = combined_dim,
                hidden_size = combined_dim,
                num_hidden_layers = num_hidden_layers,
                output_size = embed_dim,
            )
            for _ in range(num_encoders)
        ])
        
        tree_depth = config.tree_depth
        self._fan_in = int(round(self._num_encoders ** (1 / (tree_depth - 1))))
        combined_dim = embed_dim * self._fan_in + embed_dim
        self._aggregators = nn.ModuleList()
        for i in range(tree_depth - 2):
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
        for i in range(tree_depth - 2):
            level = nn.ModuleList([
                FNN(
                    input_size = combined_dim,
                    hidden_size = combined_dim,
                    num_hidden_layers = num_hidden_layers,
                    output_size = embed_dim,
                )
                for _ in range(self._fan_in ** i)
            ])
            self._target_aggregators.append(level)

        action_dim = config.action_dim
        combined_dim = embed_dim + action_dim
        self._head = FNN(
            input_size = combined_dim,
            hidden_size = combined_dim,
            num_hidden_layers = num_hidden_layers,
            output_size = 1,
        )
        self._target_head = FNN(
            input_size = combined_dim,
            hidden_size = combined_dim,
            num_hidden_layers = num_hidden_layers,
            output_size = 1,
        )

        online_models = [*self._encoders, *self._aggregators, self._head]
        target_models = [*self._target_encoders, *self._target_aggregators, self._target_head]
        for online, target in zip(online_models, target_models):
            target.load_state_dict(online.state_dict())
            for param in target.parameters():
                param.requires_grad_(False)


    def forward(
        self,
        state: Tensor,
        actor_embeds: list[Tensor],
        action: Tensor,
        module_idx: int = -1,
    ) -> Tensor:
        prev_embeds = []
        i = 0
        for encoder, target_encoder in zip(self._encoders, self._target_encoders):
            encoder_in = torch.cat([state, actor_embeds[i]], dim = -1)
            if i == module_idx:
                prev_embeds.append(encoder(encoder_in))
            else:
                prev_embeds.append(target_encoder(encoder_in))
            i += 1

        for level, target_level in zip(self._aggregators, self._target_aggregators):
            current_embeds = []
            for j, (aggregator, target_aggregator) in enumerate(zip(level, target_level)):
                start_idx = j * self._fan_in
                end_idx = start_idx + self._fan_in
                aggregator_in = torch.cat(prev_embeds[start_idx : end_idx] + [actor_embeds[:, i, :]], dim = -1)
                if i == module_idx:
                    embed = aggregator(aggregator_in)
                else:
                    embed = target_aggregator(aggregator_in)
                current_embeds.append(embed)
                i += 1
            prev_embeds = current_embeds

        head_in = torch.cat([prev_embeds[0], action], dim = -1)
        if i == module_idx:
            q_value = self._head(head_in).squeeze(-1)
        else:
            q_value = self._target_head(head_in).squeeze(-1)
        return q_value
        