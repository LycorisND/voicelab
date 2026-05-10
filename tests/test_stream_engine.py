import numpy as np
import pytest
from voicelab.core.stream_engine import StreamEngine

SR = 16000


def _dummy_processor(chunk: np.ndarray, sr: int, timestamp: float) -> dict:
    return {"ts": timestamp, "rms": float(np.sqrt(np.mean(chunk ** 2)))}


def test_iter_chunks_covers_full_audio():
    engine = StreamEngine(chunk_size=512, hop_size=256)
    audio = np.ones(3000, dtype=np.float32)
    chunks = list(engine._iter_chunks(audio))
    assert len(chunks) > 0
    assert all(len(c) == 512 for c in chunks)


def test_process_file_returns_one_result_per_chunk():
    engine = StreamEngine(chunk_size=512, hop_size=512)  # no overlap
    audio = np.zeros(2048, dtype=np.float32)
    results = engine.process_file(audio, SR, _dummy_processor)
    assert len(results) == 4  # 2048 / 512 = 4


def test_timestamps_increase():
    engine = StreamEngine(chunk_size=512, hop_size=256)
    audio = np.zeros(2000, dtype=np.float32)
    results = engine.process_file(audio, SR, _dummy_processor)
    timestamps = [r["ts"] for r in results]
    assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))


def test_process_stream_yields_frames():
    engine = StreamEngine(chunk_size=512, hop_size=256)
    audio = np.zeros(4000, dtype=np.float32)
    chunks = [audio[i:i+256] for i in range(0, 4000, 256)]
    frames = list(engine.process_stream(iter(chunks), SR, _dummy_processor))
    assert len(frames) > 0
