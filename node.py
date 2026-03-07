from collections import deque
from mmap_object import MMapObject
from torch import nn, optim
from typing import TypeVar

ModelType = TypeVar("ModelType", nn.Module)

class Node[T]:
    def __init__(
        self,
        filename: str,
        main_containers: deque[tuple[ModelType, optim.Optimizer]],
        target_container: deque[ModelType],
    ) -> None:
        super().__init__()
        default_model, default_optimizer = main_containers[0]
        self._main_mmap = MMapObject[tuple[dict, dict]](
            filename + "-main.pth",
            (
                default_model.state_dict(),
                default_optimizer.state_dict(),
            ),
        )
        self._target_mmap = MMapObject[dict](
            filename + "-target.pth",
            default_model.state_dict(),
        )
        self._main_containers = main_containers
        self._target_containers = target_container


    def load_main(self) -> tuple[ModelType, optim.Optimizer]:
        model, optimizer = self._main_containers.popleft()
        model_state, optimizer_state = self._main_mmap.load()
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        return model, optimizer


    def load_target(self) -> ModelType:
        model = self._target_containers.popleft()
        model.load_state_dict(self._target_mmap.load())
        return model


    def save_main(self, model: ModelType, optimizer: optim.Optimizer) -> None:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        self._main_mmap.save((model_state, optimizer_state))


    def save_target(self, model: ModelType) -> None:
        self._target_mmap.save(model.state_dict())


    def flush(self) -> None:
        self._main_mmap.flush()
        self._target_mmap.flush()


    def close(self) -> None:
        self._main_mmap.close()
        self._target_mmap.close()