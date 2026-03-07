from mmap import mmap
from pathlib import Path
import pickle
from typing import TypeVar

T = TypeVar("T")

class MMapObject[T]:
    def __init__(self, filename: str, default_obj: T, resize_factor: float = 1.25) -> None:
        self._filename = filename
        file_path = Path(filename)
        file_path.parent.mkdir(parents = True, exist_ok = True)
        file_existed = file_path.exists()

        self._file = open(file_path, "rb+" if file_existed else "wb+")
        if not file_existed:
            self._file.truncate(8)

        self._mmap = mmap(self._file.fileno(), length = 0)
        self._resize_factor = resize_factor
        if not file_existed:
            self.save(default_obj)


    def load(self) -> T:
        obj_size = int.from_bytes(self._mmap[:8])
        end = 8 + obj_size
        return pickle.loads(self._mmap[8:end])


    def save(self, obj: T) -> None:
        data = pickle.dumps(obj)
        self._mmap[:8] = len(data).to_bytes(length = 8)
        end = 8 + len(data)

        if self._mmap.size() < end:
            new_size = max(end, int(self._mmap.size() * self._resize_factor))
            self._file.truncate(new_size)
            self._mmap.resize(new_size)

        self._mmap[8:end] = data


    def flush(self) -> None:
        self._mmap.flush()


    def close(self) -> None:
        self.flush()
        self._file.close()
        self._mmap.close()


    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_file"]
        del state["_mmap"]
        return state


    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        file_path = Path(self._filename)
        self._file = open(file_path, "rb+")
        self._mmap = mmap(self._file.fileno(), length = 0)