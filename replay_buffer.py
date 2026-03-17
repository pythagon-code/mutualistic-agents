from config import Config
import numpy as np
import torch
from torch import Tensor
from utils import device

class ReplayBuffer:
    def __init__(self, filename: str, config: Config, rng: np.random.Generator) -> None:
        self._state_dim = config.state_dim
        self._action_dim = config.action_dim
        self._embed_dim = config.embed_dim
        num_encoders = config.num_encoders
        tree_depth = config.tree_depth
        fan_in = int(round(num_encoders ** (1 / (tree_depth - 1))))
        self._num_modules = sum((fan_in ** i for i in range(tree_depth)))
        self._item_size = (
            self._state_dim
            + self._action_dim
            + 1
            + 1
            + self._state_dim
            + self._embed_dim * self._num_modules
        )
        
        self._capacity = config.replay_buffer_size
        self._batch_size = config.batch_size
        self._array = np.memmap(filename, dtype=np.float32, mode="w+", shape=(self._capacity, self._item_size))
        self._head = 0
        self._size = 0
        self._rng = rng


    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        next_state: Tensor,
        actor_embeds: Tensor,
    ) -> None:
        item = torch.cat(
            [
                state.flatten(),
                action.flatten(),
                reward.flatten(),
                done.flatten(),
                next_state.flatten(),
                actor_embeds.flatten(),
            ]
        )
        item = item.cpu().numpy()
        assert item.shape == (self._item_size,)
        if self._size < self._capacity:
            self._array[self._size] = item
            self._size += 1
        else:
            self._array[self._head] = item
            self._head = (self._head + 1) % self._capacity
    
    
    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_size = min(self._batch_size, self._size)
        indices = self._rng.integers(0, self._size, size = sample_size)
        sample = self._array[indices]

        start_idx = 0
        state = sample[:, start_idx : start_idx + self._state_dim]
        start_idx += self._state_dim
        action = sample[:, start_idx : start_idx + self._action_dim]
        start_idx += self._action_dim
        reward = sample[:, start_idx : start_idx + 1]
        start_idx += 1
        done = sample[:, start_idx : start_idx + 1]
        start_idx += 1
        next_state = sample[:, start_idx : start_idx + self._state_dim]
        start_idx += self._state_dim
        embeds = sample[:, start_idx :]
        embeds = embeds.reshape(-1, self._num_modules, self._embed_dim)

        state = torch.tensor(state, device = device)
        action = torch.tensor(action, device = device)
        reward = torch.tensor(reward, device = device)
        done = torch.tensor(done, device = device)
        next_state = torch.tensor(next_state, device = device)
        actor_embeds = torch.tensor(embeds, device = device)
        return state, action, reward, done, next_state, actor_embeds


    def flush(self) -> None:
        self._array.flush()
