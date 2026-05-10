from __future__ import annotations
from typing import Callable, Generator, Iterator
import numpy as np

CHUNK_SIZE = 512
HOP_SIZE = 256
TARGET_SR = 16000


class StreamEngine:
    def __init__(self, chunk_size: int = CHUNK_SIZE, hop_size: int = HOP_SIZE) -> None:
        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def _iter_chunks(self, audio: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Yield overlapping chunks of fixed size. Last chunk is zero-padded."""
        start = 0
        while start + self.chunk_size <= len(audio):
            yield audio[start : start + self.chunk_size].copy()
            start += self.hop_size
        remaining = audio[start:]
        if len(remaining) > 0:
            padded = np.zeros(self.chunk_size, dtype=audio.dtype)
            padded[: len(remaining)] = remaining
            yield padded

    def process_file(
        self,
        audio: np.ndarray,
        sr: int,
        processor: Callable[[np.ndarray, int, float], object],
    ) -> list:
        results = []
        for i, chunk in enumerate(self._iter_chunks(audio)):
            timestamp = (i * self.hop_size) / sr
            results.append(processor(chunk, sr, timestamp))
        return results

    def process_stream(
        self,
        source: Iterator[np.ndarray],
        sr: int,
        processor: Callable[[np.ndarray, int, float], object],
    ) -> Generator[object, None, None]:
        """Consume an iterator of raw audio chunks, apply processor per chunk."""
        buffer = np.zeros(self.chunk_size, dtype=np.float32)
        sample_count = 0
        for incoming in source:
            n = len(incoming)
            buffer = np.roll(buffer, -n)
            buffer[-n:] = incoming
            sample_count += n
            timestamp = sample_count / sr
            yield processor(buffer.copy(), sr, timestamp)
