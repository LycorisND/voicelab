from __future__ import annotations
import threading
from typing import Any, Callable


class ModelRegistry:
    _instance: ModelRegistry | None = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, loader: Callable[[], Any]) -> None:
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        if name not in self._cache:
            with self._lock:
                if name not in self._cache:
                    if name not in self._loaders:
                        raise KeyError(f"Model '{name}' not registered")
                    self._cache[name] = self._loaders[name]()
        return self._cache[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
