"""Small, dependency-free IO primitives for flaky Kaggle disks: atomic writes, tolerant reads, bounded retries."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Writes `data` to a sibling temp file, fsyncs, then `os.replace`s it: readers never see a partial file."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return dst


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=False, default=str))


def read_json(path: str | Path, default: Any = None) -> Any:
    """Parsed JSON, or `default` when the file is missing, empty, or corrupt (a half-written file never raises)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else default
    except (OSError, ValueError) as exc:
        if Path(path).exists():
            logger.warning("Ignoring unreadable JSON %s (%s)", path, exc)
        return default


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (OSError,),
    what: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Calls `fn` up to `attempts` times with exponential backoff; the last failure propagates."""
    attempts = max(1, int(attempts))
    wait = float(delay)
    for i in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:
            if i == attempts:
                raise
            logger.warning("%s failed (%s: %s); retry %d/%d in %.1fs", what, type(exc).__name__, exc, i, attempts - 1, wait)
            sleep(wait)
            wait *= backoff
    raise AssertionError("unreachable")  # pragma: no cover


def existing_parent(path: str | Path) -> Path:
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_gib(path: str | Path) -> float:
    """Free space (GiB) on the filesystem holding `path` (or its nearest existing parent)."""
    try:
        return shutil.disk_usage(existing_parent(path)).free / 1024**3
    except OSError:
        return float("inf")  # unknown: do not block work on a failed probe


def dir_size_bytes(path: str | Path) -> int:
    total = 0
    for f in Path(path).rglob("*"):
        with contextlib.suppress(OSError):
            if f.is_file():
                total += f.stat().st_size
    return total


def safe_rmtree(path: str | Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def publish_dir(tmp_dir: str | Path, final_dir: str | Path) -> Path:
    """Moves a fully written directory into place: the final directory only ever exists in complete form."""
    tmp, final = Path(tmp_dir), Path(final_dir)
    if final.exists():
        old = final.with_name(f".old_{final.name}")
        safe_rmtree(old)
        final.rename(old)
        try:
            tmp.rename(final)
        except BaseException:
            old.rename(final)  # put the previous good copy back
            raise
        safe_rmtree(old)
    else:
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(final)
    return final
