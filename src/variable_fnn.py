import numpy as np
import torch
from torch import nn, Tensor

class VariableFNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
        rng: np.random.Generator,
        device: str = "cpu",
        sparsity: float = .0,
    ) -> None:
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._output_size = output_size
        self._input_layer_size = input_size * self._hidden_size
        self._hidden_layer_size = self._hidden_size * self._hidden_size
        self._output_layer_size = self._hidden_size * self._output_size
        self._lrelu = nn.LeakyReLU()
        self._model_layers = []
        self._masks = []
        for _ in range(self._num_hidden_layers - 1):
            mask = torch.ones(self._hidden_layer_size, device = device)
            indices = rng.choice(
                self._hidden_layer_size,
                size = int(self._hidden_layer_size * sparsity),
                replace = False,
            )
            mask[indices] = 0
            mask = mask.reshape(self._hidden_size, self._hidden_size)
            self._masks.append(mask)


    def get_model_state_size(self) -> int:
        return (
            self._input_layer_size
            + self._hidden_layer_size * (self._num_hidden_layers - 1)
            + self._output_layer_size
        )
    

    def set_model(self, model_state: Tensor) -> None:
        self._model_layers.clear()
        i = 0
        block = model_state[i : i + self._input_layer_size]
        block = block.reshape(self._hidden_size, self._input_size)
        self._model_layers.append(block)
        i += self._input_layer_size
        for mask in self._masks:
            block = model_state[i : i + self._hidden_layer_size]
            block = block.reshape(self._hidden_size, self._hidden_size)
            block *= mask
            self._model_layers.append(block)
            i += self._hidden_layer_size
        block = model_state[i : i + self._output_layer_size]
        block = block.reshape(self._output_size, self._hidden_size)
        self._model_layers.append(block)
        i += self._output_layer_size


    def forward(self, x: Tensor) -> Tensor:
        assert len(self._model_layers) != 0
        for layer in self._model_layers[: -1]:
            x = layer @ x
            x = self._lrelu(x)
        return self._model_layers[-1] @ x