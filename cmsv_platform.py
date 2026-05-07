"""Platform helpers shared by training and inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


PLATFORM_NAMES = ("ccs", "clr", "ont")
PLATFORM_TO_ID = {name: idx for idx, name in enumerate(PLATFORM_NAMES)}
PLATFORM_ID_TO_NAME = {idx: name for name, idx in PLATFORM_TO_ID.items()}
PLATFORM_ALIASES = {
    "ccs": ("ccs", "hifi"),
    "clr": ("clr",),
    "ont": ("ont", "nanopore"),
}
NUM_PLATFORMS = len(PLATFORM_NAMES)


def normalize_platform_name(name: str) -> str:
    value = str(name).strip().lower()
    if not value:
        raise ValueError("platform name is empty")
    for canonical, aliases in PLATFORM_ALIASES.items():
        if value == canonical or value in aliases:
            return canonical
    raise ValueError(f"Unsupported platform={name!r}. Valid values: {', '.join(PLATFORM_NAMES)}")


def platform_id_from_name(name: str) -> int:
    return PLATFORM_TO_ID[normalize_platform_name(name)]


def platform_name_from_id(platform_id: int) -> str:
    pid = int(platform_id)
    try:
        return PLATFORM_ID_TO_NAME[pid]
    except KeyError as exc:
        raise ValueError(f"Unsupported platform_id={platform_id!r}") from exc


def _detect_platform_in_text(text: str) -> Optional[str]:
    lowered = str(text).strip().lower()
    if not lowered:
        return None
    for canonical, aliases in PLATFORM_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(^|[^a-z0-9]){re.escape(alias)}([^a-z0-9]|$)", lowered):
                return canonical
    return None


def infer_platform_name_from_path(path: str | Path) -> Optional[str]:
    path_obj = Path(path)
    for candidate in [path_obj.name, *reversed(path_obj.parts)]:
        platform = _detect_platform_in_text(candidate)
        if platform is not None:
            return platform
    return None


def resolve_platform_name(explicit: Optional[str] = None, *candidates: object) -> str:
    if explicit:
        return normalize_platform_name(explicit)

    matches: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (str, Path)):
            platform = infer_platform_name_from_path(candidate)
        else:
            platform = _detect_platform_in_text(str(candidate))
        if platform is not None:
            matches.append(platform)

    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ValueError(f"Conflicting platform hints found: {unique}")
    raise ValueError(
        "Unable to infer platform from the provided paths. "
        f"Please specify one of: {', '.join(PLATFORM_NAMES)}"
    )


def resolve_platform_id(explicit: Optional[str] = None, *candidates: object) -> int:
    return platform_id_from_name(resolve_platform_name(explicit, *candidates))


def describe_platform_mapping() -> str:
    return ", ".join(f"{name}={pid}" for name, pid in PLATFORM_TO_ID.items())


def unique_platform_names(values: Iterable[int]) -> list[str]:
    return [platform_name_from_id(pid) for pid in sorted(set(int(v) for v in values))]
