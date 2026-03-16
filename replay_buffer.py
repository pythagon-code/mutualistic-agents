from config import Config
import numpy as np
import torch
from torch import Tensor
from utils import device

class ReplayBuffer:
    def __init__(
        self,
        filename: str,
        config: Config,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        self._state_dim = config.state_dim
        self._action_dim = config.action_dim
        self._embed_dim = config.embed_dim
        num_encoders = config.num_encoders
        tree_depth = config.tree_depth
        fan_in = int(round(num_encoders ** (1 / (tree_depth - 1))))
        self._num_modules = sum((fan_in ** i for i in range(tree_depth)))
        self._item_size = self._state_dim + self._action_dim + self._embed_dim * self._num_modules
        
        self._capacity = config.replay_buffer_size
        self._batch_size = batch_size
        self._array = np.memmap(filename, dtype=np.float32, mode="w+", shape=(self._capacity, self._item_size))
        self._head = 0
        self._size = 0
        self._rng = rng


    def add(self, state: Tensor, action: Tensor, embeds: Tensor) -> None:
        item = torch.cat([state.flatten(), action.flatten(), embeds.flatten()])
        item = item.cpu().numpy()
        assert item.shape == (self._item_size,)
        if self._size < self._capacity:
            self._array[self._size] = item
            self._size += 1
        else:
            self._array[self._head] = item
            self._head = (self._head + 1) % self._capacity
    
    
    def sample(self) -> tuple[Tensor, Tensor, Tensor]:
        sample_size = min(self._batch_size, self._size)
        indices = self._rng.integers(0, self._size, size = sample_size)
        sample = self._array[indices]
        states = sample[:, : self._state_dim]
        actions = sample[:, self._state_dim : self._state_dim + self._action_dim]
        embeds = sample[:, self._state_dim + self._action_dim :]
        embeds = embeds.reshape(-1, self._num_modules, self._embed_dim)
        states = torch.tensor(states, device = device)
        actions = torch.tensor(actions, device = device)
        embeds = torch.tensor(embeds, device = device)
        return states, actions, embeds


    def flush(self) -> None:
        self._array.flush()
