from __future__ import annotations
import queue
import threading
from typing import Generator, Iterator
import numpy as np
from voicelab.core.stream_engine import StreamEngine
from voicelab.schema import Config, FrameResult

TARGET_SR = 16000


class VoiceStream:
    def __init__(self, config: Config | None = None, sr: int = TARGET_SR) -> None:
        self.config = config or Config()
        self.sr = sr
        self._engine = StreamEngine(
            chunk_size=self.config.chunk_size, hop_size=self.config.hop_size
        )

    def _frame_processor(
        self, chunk: np.ndarray, sr: int, timestamp: float
    ) -> FrameResult:
        import librosa
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        is_voiced = energy > 0.01
        mfcc = librosa.feature.mfcc(
            y=chunk, sr=sr, n_mfcc=self.config.n_mfcc
        ).mean(axis=1).astype(np.float32)
        zcr = librosa.feature.zero_crossing_rate(chunk)[0]
        f0_approx = float(sr * np.mean(zcr) / 2) if is_voiced else float("nan")
        return FrameResult(
            timestamp=timestamp,
            pitch=f0_approx,
            energy=energy,
            is_voiced=is_voiced,
            mfcc=mfcc,
        )

    def _process_source(self, source: Iterator[np.ndarray]) -> Generator[FrameResult, None, None]:
        yield from self._engine.process_stream(source, self.sr, self._frame_processor)

    def __enter__(self):
        import sounddevice as sd
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()

        def callback(indata, frames, time_info, status):
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=self.config.hop_size,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_):
        self._stream.stop()
        self._stream.close()

    def __iter__(self) -> Generator[FrameResult, None, None]:
        def _mic_source():
            while not self._stop.is_set():
                try:
                    yield self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
        yield from self._process_source(_mic_source())

    def stop(self):
        self._stop.set()
