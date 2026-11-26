"""Remove rebuildable files under the selected runtime's cache directory."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import stat


class CacheConflict(RuntimeError):
    pass


def _is_reparse_point(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _protected_entries(runtime: Path, cache: Path) -> set[Path]:
    storage = [runtime / "models", runtime / "library"]
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE",
                     "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        if value := os.environ.get(variable):
            storage.append(Path(os.path.abspath(os.path.expanduser(value))))
    protected = set()
    for path in storage:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            resolved = path
        for location in (path, resolved):
            if location == cache:
                raise CacheConflict("The cache directory is configured for model or history storage.")
            if location.is_relative_to(cache):
                protected.add(cache / location.relative_to(cache).parts[0])
    return protected


def _remove_entry(path: Path, cache: Path, device: int) -> None:
    if not path.parent.is_relative_to(cache) or path.parent.resolve(strict=True) != path.parent:
        raise OSError(f"Cache entry has an unsafe parent: {path}")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        path.unlink()
    elif _is_reparse_point(info):
        if not path.is_junction():
            raise OSError(f"Unsupported cache reparse point: {path}")
        path.rmdir()
    elif stat.S_ISDIR(info.st_mode):
        if info.st_dev != device or path.resolve(strict=True) != path:
            raise OSError(f"Cache directory has an unsafe location: {path}")
        for child in path.iterdir():
            _remove_entry(child, cache, device)
        path.rmdir()
    else:
        path.unlink()


def clear_runtime_cache(runtime: Path) -> tuple[int, int]:
    cache = runtime / "cache"
    try:
        info = cache.lstat()
    except FileNotFoundError:
        return 0, 0
    if (not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info)
            or cache.resolve(strict=True) != cache or info.st_dev != runtime.stat().st_dev):
        raise CacheConflict("Cache cleanup requires a regular cache directory inside the runtime.")

    protected = _protected_entries(runtime, cache)
    removed = failed = 0
    for path in cache.iterdir():
        if path in protected:
            logging.warning("Skipped cache entry used for model or history storage: %s", path)
            failed += 1
            continue
        try:
            _remove_entry(path, cache, info.st_dev)
        except (OSError, RuntimeError):
            logging.exception("Could not remove cache entry %s", path)
            failed += 1
        else:
            removed += 1
    return removed, failed
