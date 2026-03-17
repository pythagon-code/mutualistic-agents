from config import Config
from fnn import FNN
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from typing import TYPE_CHECKING
from utils import get_ema, get_ema_and_emv, polyak_update

if TYPE_CHECKING:
    from actor import Actor

class Critic(nn.Module):
    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        super().__init__()
        self._rng = rng
        self._discount_rate = config.discount_rate
        self._polyak_factor = config.polyak_factor

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
        self._fan_in = int(round(num_encoders ** (1 / (tree_depth - 1))))
        combined_dim = embed_dim * self._fan_in + embed_dim
        self._aggregators = nn.ModuleList()
        for i in range(tree_depth - 2, -1, -1):
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
        for i in range(tree_depth - 2, -1, -1):
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
        self._heads = nn.ModuleList([
            FNN(
                input_size = combined_dim,
                hidden_size = combined_dim,
                num_hidden_layers = num_hidden_layers,
                output_size = 1,
            )
            for _ in range(2)
        ])
        self._target_heads = nn.ModuleList([
            FNN(
                input_size = combined_dim,
                hidden_size = combined_dim,
                num_hidden_layers = num_hidden_layers,
                output_size = 1,
            )
            for _ in range(2)
        ])
        self._online_models = [*self._encoders, *self._aggregators, *self._heads]
        self._target_models = [*self._target_encoders, *self._target_aggregators, *self._target_heads]
        for online, target in zip(self._online_models, self._target_models):
            target.load_state_dict(online.state_dict())
            for param in target.parameters():
                param.requires_grad_(False)
        
        lr = config.learning_rate
        self._optimizers = [optim.Adam(model.parameters(), lr = lr) for model in self._online_models]


    def forward(
        self,
        state: Tensor,
        actor_embeds: Tensor,
        action: Tensor,
        module_idx: int = -1,
    ) -> tuple[Tensor, Tensor]:
        prev_embeds = []
        i = 0
        for encoder, target_encoder in zip(self._encoders, self._target_encoders):
            encoder_in = torch.cat([state, actor_embeds[:, i, :]], dim = -1)
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
            q_value0 = self._heads[0](head_in)
        else:
            q_value0 = self._target_heads[0](head_in)
        if i + 1 == module_idx:
            q_value1 = self._heads[1](head_in)
        else:
            q_value1 = self._target_heads[1](head_in)
        return q_value0, q_value1


    def train(
        self,
        actor: "Actor",
        state: Tensor,
        actor_embeds: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        next_state: Tensor,
        module_idx: int = -1,
    ) -> tuple[float, float]:
        module_idx = self._rng.integers(0, len(self._online_models))
        q_values = self.forward(state, actor_embeds, action, module_idx = module_idx)
        next_actor_embeds, next_action = actor.forward(next_state)
        next_q_values = self.forward(next_state, next_actor_embeds, next_action)
        next_q_value = torch.min(next_q_values[0], next_q_values[1])
        target_q_value = reward + (1 - done) * self._discount_rate * next_q_value
        assert not target_q_value.requires_grad
        critic_loss = mse_loss(q_values[0], target_q_value) + mse_loss(q_values[1], target_q_value)
        critic_loss.backward()
        clip_grad_norm_(self._online_models[module_idx].parameters(), max_norm = 1.0)
        self._optimizers[module_idx].step()
        self._optimizers[module_idx].zero_grad(set_to_none = True)
        polyak_update(self._target_models[module_idx], self._online_models[module_idx], self._polyak_factor)
        return critic_loss.item(), target_q_value.mean().item()