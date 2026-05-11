# voicelab/emotion/stream.py
from __future__ import annotations
import queue
from typing import Generator, Iterator
import numpy as np
from voicelab.analysis.voice_stream import VoiceStream
from voicelab.schema import Config, EmotionConfig, EmotionFrame, FrameResult

TARGET_SR = 16000


class EmotionStream:
    def __init__(self, config: EmotionConfig | None = None, sr: int = TARGET_SR) -> None:
        self.config = config or EmotionConfig()
        self.sr = sr
        self._voice_stream = VoiceStream(Config(neural=False), sr=sr)

    def _process_source(
        self, source: Iterator[np.ndarray]
    ) -> Generator[EmotionFrame, None, None]:
        from voicelab.emotion.fusion import _run_fusion_lite
        for frame in self._voice_stream._process_source(source):
            yield _run_fusion_lite(frame, self.config)

    def __enter__(self):
        self._voice_stream.__enter__()
        return self

    def __exit__(self, *args):
        self._voice_stream.__exit__(*args)

    def __iter__(self) -> Generator[EmotionFrame, None, None]:
        def _mic_source():
            while not self._voice_stream._stop.is_set():
                try:
                    yield self._voice_stream._q.get(timeout=0.1)
                except queue.Empty:
                    continue
        yield from self._process_source(_mic_source())

    def stop(self):
        self._voice_stream.stop()
