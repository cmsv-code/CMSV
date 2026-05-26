from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from cmsv_storage import iter_feature_store_paths, load_feature_data


DEFAULT_WINDOW_SIZE = 2000
LEGACY_FEATURE_VERSION = "legacy20"
DEFAULT_FEATURE_VERSION = LEGACY_FEATURE_VERSION

LEGACY_CHANNEL_NAMES = [
    "del_cigar_rev",
    "ins_cigar_rev",
    "del_split_rev",
    "ins_split_rev",
    "inv_split_rev",
    "dup_split_rev",
    "bnd_split_rev",
    "depth_rev",
    "clip_sm_rev",
    "clip_ms_rev",
    "del_cigar_fwd",
    "ins_cigar_fwd",
    "del_split_fwd",
    "ins_split_fwd",
    "inv_split_fwd",
    "dup_split_fwd",
    "bnd_split_fwd",
    "depth_fwd",
    "clip_sm_fwd",
    "clip_ms_fwd",
]

FEATURE_SPECS = {
    LEGACY_FEATURE_VERSION: {
        "feature_version": LEGACY_FEATURE_VERSION,
        "feature_dim": len(LEGACY_CHANNEL_NAMES),
        "window_size": DEFAULT_WINDOW_SIZE,
        "channel_names": LEGACY_CHANNEL_NAMES,
        "normalization": "zscore",
    },
}


def get_feature_spec(feature_version: str) -> Dict[str, Any]:
    try:
        return dict(FEATURE_SPECS[str(feature_version)])
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_SPECS))
        raise ValueError(f"Unknown feature_version={feature_version!r}. Valid values: {valid}") from exc


def infer_feature_dim_from_array(feature_array: np.ndarray, window_size: int = DEFAULT_WINDOW_SIZE) -> int:
    arr = np.asarray(feature_array)
    if arr.ndim == 3:
        if int(arr.shape[1]) != int(window_size):
            raise ValueError(f"Expected window_size={window_size}, got array shape {arr.shape}")
        return int(arr.shape[2])
    if arr.ndim == 2:
        flat_dim = int(arr.shape[1])
        if flat_dim % int(window_size) != 0:
            raise ValueError(f"Flat feature length {flat_dim} is not divisible by window_size={window_size}")
        return flat_dim // int(window_size)
    if arr.ndim == 1:
        flat_dim = int(arr.shape[0])
        if flat_dim % int(window_size) != 0:
            raise ValueError(f"Flat feature length {flat_dim} is not divisible by window_size={window_size}")
        return flat_dim // int(window_size)
    raise ValueError(f"Unsupported feature array shape: {arr.shape}")


def infer_feature_config_from_sample(
    sample_file: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Any]:
    sample_arr = load_feature_data(sample_file)
    shape = getattr(sample_arr, "shape", ())
    if not shape:
        raise ValueError(f"Unable to infer feature shape from {sample_file}")
    if int(shape[0]) > 0:
        probe = np.asarray(sample_arr[0])
    else:
        probe = np.asarray(sample_arr)
    feature_dim = infer_feature_dim_from_array(probe, window_size=window_size)
    for spec in FEATURE_SPECS.values():
        if spec["feature_dim"] == feature_dim and int(spec["window_size"]) == int(window_size):
            inferred = dict(spec)
            inferred["source"] = str(sample_file)
            return inferred
    return {
        "feature_version": f"inferred_{feature_dim}d",
        "feature_dim": int(feature_dim),
        "window_size": int(window_size),
        "channel_names": [f"feature_{idx}" for idx in range(int(feature_dim))],
        "normalization": "unknown",
        "source": str(sample_file),
    }


def _normalize_feature_config(config: Dict[str, Any], source: str) -> Dict[str, Any]:
    feature_dim = int(config["feature_dim"])
    window_size = int(config.get("window_size", DEFAULT_WINDOW_SIZE))
    channel_names = list(config.get("channel_names") or [f"feature_{idx}" for idx in range(feature_dim)])
    if len(channel_names) != feature_dim:
        raise ValueError(f"channel_names length {len(channel_names)} != feature_dim {feature_dim} in {source}")
    return {
        "feature_version": str(config.get("feature_version", f"inferred_{feature_dim}d")),
        "feature_dim": feature_dim,
        "window_size": window_size,
        "channel_names": channel_names,
        "normalization": str(config.get("normalization", "unknown")),
        "storage_format": str(config.get("storage_format", "unknown")),
        "storage_dtype": str(config.get("storage_dtype", "unknown")),
        "source": source,
    }


def load_feature_config_from_dir(
    data_dir: str,
    default_window_size: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Any]:
    data_path = Path(data_dir)
    config_path = data_path / "feature_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return _normalize_feature_config(config, str(config_path))

    nested_config_paths = sorted(path for path in data_path.rglob("feature_config.json") if path != config_path)
    if nested_config_paths:
        configs = []
        for path in nested_config_paths:
            with open(path, "r", encoding="utf-8") as handle:
                configs.append(_normalize_feature_config(json.load(handle), str(path)))
        reference = configs[0]
        for candidate in configs[1:]:
            if (
                int(candidate["feature_dim"]) != int(reference["feature_dim"])
                or int(candidate["window_size"]) != int(reference["window_size"])
                or str(candidate["feature_version"]) != str(reference["feature_version"])
                or str(candidate["normalization"]) != str(reference["normalization"])
                or list(candidate["channel_names"]) != list(reference["channel_names"])
            ):
                raise ValueError(
                    "Inconsistent feature configuration found under "
                    f"{data_dir}: {reference['source']} vs {candidate['source']}"
                )
        merged = dict(reference)
        merged["source"] = f"{reference['source']} (+{len(configs) - 1} matched subdirs)"
        return merged

    sample_files = [str(path) for path in iter_feature_store_paths(data_path)]
    if not sample_files:
        raise FileNotFoundError(f"No feature stores found under {data_dir}")
    for sample_file in sample_files:
        try:
            if getattr(load_feature_data(sample_file), "shape", (0,))[0] == 0:
                continue
            return infer_feature_config_from_sample(sample_file, window_size=default_window_size)
        except Exception:
            continue
    raise FileNotFoundError(f"No non-empty feature stores found under {data_dir}")


def load_feature_config_from_weights(weights_path: str) -> Optional[Dict[str, Any]]:
    meta_path = Path(weights_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    feature_config = meta.get("feature_config")
    if not feature_config:
        return None
    return _normalize_feature_config(feature_config, str(meta_path))


def resolve_feature_config(
    data_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
    default_version: str = LEGACY_FEATURE_VERSION,
) -> Dict[str, Any]:
    data_config = load_feature_config_from_dir(data_dir) if data_dir else None
    weight_config = load_feature_config_from_weights(weights_path) if weights_path else None

    if data_config and weight_config:
        if int(data_config["feature_dim"]) != int(weight_config["feature_dim"]):
            raise ValueError(
                f"Feature-dimension mismatch between data ({data_config['feature_dim']}) and weights "
                f"({weight_config['feature_dim']})."
            )
        if int(data_config["window_size"]) != int(weight_config["window_size"]):
            raise ValueError(
                f"Window-size mismatch between data ({data_config['window_size']}) and weights "
                f"({weight_config['window_size']})."
            )
        merged = dict(weight_config)
        merged.update(data_config)
        merged["source"] = f"{data_config['source']} + {weight_config['source']}"
        return merged

    if data_config:
        return data_config
    if weight_config:
        return weight_config

    fallback = get_feature_spec(default_version)
    fallback["source"] = f"default:{default_version}"
    return fallback


def write_feature_config(
    output_dir: str,
    feature_version: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    storage_format: str = "npy",
    storage_dtype: str = "float16",
) -> Dict[str, Any]:
    spec = get_feature_spec(feature_version)
    spec["window_size"] = int(window_size)
    spec["storage_format"] = str(storage_format)
    spec["storage_dtype"] = str(storage_dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "feature_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, ensure_ascii=False)
    spec["source"] = str(config_path)
    return spec
