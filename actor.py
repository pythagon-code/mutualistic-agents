from config import Config
from critic import Critic
from fnn import FNN
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.utils import clip_grad_norm_
from utils import flatten_module_list, polyak_update


class Actor(nn.Module):
    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        super().__init__()
        self._rng = rng
        self._polyak_factor = config.actor_polyak_factor

        num_encoders = config.num_encoders
        state_dim = config.state_dim
        embed_dim = config.embed_dim
        num_hidden_layers = config.actor_num_hidden_layers
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
        for i in range(tree_depth - 2, 0, -1):
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
        for i in range(tree_depth - 2, 0, -1):
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
        action_activation = nn.Tanh() if config.action_tanh else nn.Identity()
        self._final_aggregator = FNN(
            input_size = combined_dim,
            hidden_size = combined_dim,
            num_hidden_layers = num_hidden_layers,
            output_size = action_dim,
            output_activation = action_activation,
        )
        self._target_final_aggregator = FNN(
            input_size = combined_dim,
            hidden_size = combined_dim,
            num_hidden_layers = num_hidden_layers,
            output_size = action_dim,
            output_activation = action_activation,
        )

        self._online_models = [
            *self._encoders,
            *flatten_module_list(self._aggregators),
            self._final_aggregator,
        ]
        self._target_models = [
            *self._target_encoders,
            *flatten_module_list(self._target_aggregators),
            self._target_final_aggregator,
        ]
        assert len(self._online_models) == len(self._target_models)
        for online, target in zip(self._online_models, self._target_models):
            target.load_state_dict(online.state_dict())
            for param in target.parameters():
                param.requires_grad_(False)

        lr = config.learning_rate
        self._optimizers = [optim.Adam(model.parameters(), lr = lr) for model in self._online_models]

    
    def forward(self, state: Tensor, noise_std: float = 0.0, module_idx: int = -1, not_detach_idx: int = -1) -> tuple[Tensor, Tensor]:
        i = 0
        prev_embeds = []
        for encoder, target_encoder in zip(self._encoders, self._target_encoders):
            if i == module_idx:
                prev_embeds.append(encoder(state))
            else:
                prev_embeds.append(target_encoder(state))
            i += 1
        all_embeds = prev_embeds.copy()
        if not_detach_idx != -1:
            for idx in range(len(all_embeds)):
                if idx != not_detach_idx:
                    all_embeds[idx] = all_embeds[idx].detach()
        
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

        aggregator_in = torch.cat(prev_embeds[0 : self._fan_in], dim = -1)
        if i == module_idx:
            action = self._final_aggregator(aggregator_in)
        else:
            action = self._target_final_aggregator(aggregator_in)
        action += torch.randn_like(action) * noise_std
        all_embeds = torch.stack(all_embeds, dim = 1)
        return all_embeds, action


    def train(self, critic: "Critic", state: Tensor) -> float:
        module_idx = self._rng.integers(0, len(self._online_models))
        all_embeds, action = self.forward(state, module_idx = module_idx)
        q_values = critic.forward(state, all_embeds, action, module_idx = module_idx)
        q_value = torch.min(q_values[0], q_values[1])
        actor_loss = -q_value.mean()
        actor_loss.backward()
        clip_grad_norm_(self._online_models[module_idx].parameters(), max_norm = 1.0)
        self._optimizers[module_idx].step()
        self._optimizers[module_idx].zero_grad(set_to_none = True)
        polyak_update(self._target_models[module_idx], self._online_models[module_idx], self._polyak_factor)
        return actor_loss.item()
