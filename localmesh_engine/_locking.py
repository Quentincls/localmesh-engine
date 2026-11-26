"""Advisory process locks on Windows and Linux."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


class LockBusy(RuntimeError):
    pass


@contextmanager
def process_lock(path: Path, *, wait: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            from msvcrt import LK_LOCK, LK_NBLCK, LK_UNLCK, locking

            handle.seek(0)
            try:
                locking(handle.fileno(), LK_LOCK if wait else LK_NBLCK, 1)
            except OSError as exc:
                raise LockBusy from exc
            try:
                yield
            finally:
                handle.seek(0)
                locking(handle.fileno(), LK_UNLCK, 1)
        else:
            from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock

            try:
                flock(handle.fileno(), LOCK_EX if wait else LOCK_EX | LOCK_NB)
            except BlockingIOError as exc:
                raise LockBusy from exc
            try:
                yield
            finally:
                flock(handle.fileno(), LOCK_UN)
