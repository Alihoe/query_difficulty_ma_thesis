from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ArtifactStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        location = self.root_dir.joinpath(*parts)
        location.parent.mkdir(parents=True, exist_ok=True)
        return location

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        try:
            frame.to_parquet(temp_path, index=False)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def write_joblib(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    def read_joblib(self, path: Path) -> Any:
        return joblib.load(path)

    def write_numpy(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    def read_numpy(self, path: Path) -> np.ndarray:
        return np.load(path)

    def write_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp_path, path)

    def exists(self, path: Path) -> bool:
        return path.exists()
