from collections import deque
import numpy as np
import torch
from torch import Tensor

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        device: str,
        rng: np.random.Generator,
        unsqueezed: bool = True,
    ) -> None:
        self._buffer = deque(maxlen = capacity)
        self._batch_size = batch_size
        self._rng = rng
        self._device = device
        self._unsqueezed = unsqueezed


    def add(self, items: tuple[Tensor, ...]) -> None:
        self._buffer.append(items)
        

    def ready(self) -> bool:
        return len(self._buffer) >= self._batch_size

    
    def sample(self) -> tuple[Tensor, ...]:
        indices = self._rng.integers(0, len(self._buffer), size = self._batch_size)
        batch = [self._buffer[i] for i in indices]
        batch = zip(*batch)
        if self._unsqueezed:
            return tuple((torch.cat(item).to(self._device) for item in batch))
        else:
            return tuple((torch.stack(item).to(self._device) for item in batch))