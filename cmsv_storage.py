"""Shared helpers for npy-based feature-store discovery, IO, and resume handling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


FEATURE_STORE_SUFFIX = ".npy"


def canonical_contig_name(contig: str) -> str:
    name = str(contig)
    return name if name.startswith("chr") else f"chr{name}"


def region_stem(contig: str, start: int, end: int) -> str:
    return f"{canonical_contig_name(contig)}_{int(start)}_{int(end)}"


def region_store_path(output_dir: str | Path, contig: str, start: int, end: int) -> Path:
    return Path(output_dir) / f"{region_stem(contig, start, end)}{FEATURE_STORE_SUFFIX}"


def legacy_feature_path(output_dir: str | Path, contig: str, start: int, end: int) -> Path:
    return region_store_path(output_dir, contig, start, end)


def legacy_index_path(output_dir: str | Path, contig: str, start: int, end: int) -> Path:
    return Path(output_dir) / f"{region_stem(contig, start, end)}_index.npy"


def feature_store_path(
    output_dir: str | Path,
    contig: str,
    start: int,
    end: int,
    *,
    storage_format: str = "npy",
) -> Path:
    if str(storage_format).lower() != "npy":
        raise ValueError(f"Unsupported storage_format={storage_format!r}; only 'npy' is supported")
    return legacy_feature_path(output_dir, contig, start, end)


def resolve_feature_store_path(data_dir: str | Path, contig: str, start: int, end: int) -> Optional[Path]:
    npy_path = legacy_feature_path(data_dir, contig, start, end)
    if npy_path.exists():
        return npy_path
    return None


def iter_feature_store_paths(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    return sorted(
        path for path in root.rglob("*.npy")
        if not path.name.endswith(("_index.npy", "_label.npy", "_predict.npy"))
    )


def save_npy_feature_store(
    store_path: str | Path,
    data: np.ndarray,
    index: np.ndarray,
    *,
    overwrite: bool = True,
) -> Path:
    path = Path(store_path)
    index_path = path.with_name(f"{path.stem}_index.npy")
    if not overwrite and path.exists() and index_path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.asarray(data))
    np.save(str(index_path), np.asarray(index, dtype=np.int32))
    return path


def feature_store_complete(path: str | Path) -> bool:
    store_path = Path(path)
    if store_path.suffix != ".npy" or store_path.name.endswith(("_index.npy", "_label.npy", "_predict.npy")):
        return False
    index_path = store_path.with_name(f"{store_path.stem}_index.npy")
    return store_path.exists() and index_path.exists()


def delete_feature_store(path: str | Path) -> None:
    store_path = Path(path)
    if store_path.name.endswith(("_index.npy", "_label.npy", "_predict.npy")):
        store_path.unlink(missing_ok=True)
        return
    store_path.unlink(missing_ok=True)
    store_path.with_name(f"{store_path.stem}_index.npy").unlink(missing_ok=True)
    store_path.with_name(f"{store_path.stem}_label.npy").unlink(missing_ok=True)
    store_path.with_name(f"{store_path.stem}_predict.npy").unlink(missing_ok=True)


def load_feature_data(path: str | Path, mmap_mode: Optional[str] = "r"):
    return np.load(str(Path(path)), mmap_mode=mmap_mode)


def load_feature_index(path: str | Path) -> np.ndarray:
    store_path = Path(path)
    if store_path.name.endswith("_index.npy"):
        return np.load(str(store_path))
    return np.load(str(store_path.with_name(f"{store_path.stem}_index.npy")))


def feature_store_has_labels(path: str | Path) -> bool:
    store_path = Path(path)
    if store_path.name.endswith("_label.npy"):
        return store_path.exists()
    return store_path.with_name(f"{store_path.stem}_label.npy").exists()


def load_feature_labels(path: str | Path) -> np.ndarray:
    store_path = Path(path)
    if store_path.name.endswith("_label.npy"):
        return np.load(str(store_path))
    return np.load(str(store_path.with_name(f"{store_path.stem}_label.npy")))


def save_feature_labels(path: str | Path, labels: np.ndarray, overwrite: bool = True) -> None:
    store_path = Path(path)
    label_path = store_path.with_name(f"{store_path.stem}_label.npy")
    if label_path.exists() and not overwrite:
        return
    np.save(str(label_path), np.asarray(labels, dtype=np.uint8))


def feature_store_sample_count(path: str | Path) -> int:
    data = load_feature_data(path)
    shape = getattr(data, "shape", ())
    if not shape:
        return 0
    return int(shape[0])
