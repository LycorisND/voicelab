from __future__ import annotations
import json
import csv
import numpy as np
from pathlib import Path
from voicelab.schema import AnalysisResult


def to_dict(result: AnalysisResult) -> dict:
    """Convert AnalysisResult to a JSON-serialisable dict (arrays → lists)."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
        return obj

    return _convert(result)


def to_json(result: AnalysisResult, path: str) -> None:
    Path(path).write_text(json.dumps(to_dict(result), indent=2))


def to_csv(result: AnalysisResult, path: str) -> None:
    """Write scalar features to CSV (one row). Arrays are skipped."""
    flat: dict[str, object] = {}
    d = to_dict(result)

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}.")
        elif not isinstance(obj, list):
            flat[prefix.rstrip(".")] = obj

    _flatten(d)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def to_hdf5(result: AnalysisResult, path: str) -> None:
    """Write all features including arrays to HDF5."""
    import h5py

    def _write(group, obj):
        if hasattr(obj, "__dataclass_fields__"):
            for k in obj.__dataclass_fields__:
                v = getattr(obj, k)
                if isinstance(v, np.ndarray):
                    group.create_dataset(k, data=v, compression="gzip")
                elif hasattr(v, "__dataclass_fields__"):
                    sub = group.require_group(k)
                    _write(sub, v)
                else:
                    try:
                        group.attrs[k] = v
                    except Exception:
                        pass

    with h5py.File(path, "w") as f:
        _write(f, result)
