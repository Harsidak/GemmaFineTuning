"""Task-level progress ledger: resume by *which problems were trained on*, not by a position in a shuffled list.

The sampler's resume offset only means something while the dataset is byte-identical. The ledger stores the keys of
the examples already consumed in the current epoch, so after the dataset grew (10,000 problems collected, 4,173 done,
more mined since) the next session skips exactly those and continues with the rest.

Example key = `<task_id>:<8-byte digest of the token ids>`: human readable, unique per distinct trajectory, stable
across sessions and independent of row order.
"""

from __future__ import annotations

import array
import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fsio import atomic_write_json, read_json

LEDGER_VERSION = 1
LEDGER_FILE = "progress.json"


def example_key(row: dict[str, Any]) -> str:
    """Stable key for one tokenized training row."""
    ids = row.get("input_ids") or []
    digest = hashlib.blake2b(array.array("i", [int(t) for t in ids]).tobytes(), digest_size=8).hexdigest()
    return f"{row.get('task_id', '?')}:{digest}"


def example_keys(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [example_key(r) for r in rows]


@dataclass
class ResumePlan:
    """How the next session should consume the data."""

    mode: str  # "fresh" | "continue" | "next_epoch" | "restart"
    epoch: int = 0
    exclude_indices: set[int] = field(default_factory=set)
    done_in_epoch: int = 0
    total: int = 0
    unseen: int = 0
    stale_keys: int = 0  # consumed keys that no longer exist in the data (e.g. dedupe / filter changes)
    note: str = ""


class ProgressLedger:
    def __init__(self, adapter: str) -> None:
        self.adapter = adapter
        self.epoch = 0
        self.consumed: set[str] = set()
        self.data_sha: str | None = None
        self.data_count = 0
        self.examples_seen_total = 0
        self.sessions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LEDGER_VERSION,
            "adapter": self.adapter,
            "epoch": self.epoch,
            "consumed": sorted(self.consumed),
            "data_sha": self.data_sha,
            "data_count": self.data_count,
            "examples_seen_total": self.examples_seen_total,
            "sessions": self.sessions[-50:],
            "saved_at_ts": time.time(),
        }

    @classmethod
    def from_dict(cls, adapter: str, d: dict[str, Any] | None) -> ProgressLedger:
        led = cls(adapter)
        if not isinstance(d, dict):
            return led
        led.epoch = max(0, int(d.get("epoch", 0)))
        led.consumed = {str(k) for k in d.get("consumed", [])}
        led.data_sha = d.get("data_sha")
        led.data_count = int(d.get("data_count", 0))
        led.examples_seen_total = int(d.get("examples_seen_total", 0))
        led.sessions = list(d.get("sessions", []))
        return led

    def save(self, directory: str | Path) -> Path:
        return atomic_write_json(Path(directory) / LEDGER_FILE, self.to_dict(), indent=0)

    @classmethod
    def load(cls, adapter: str, directory: str | Path) -> ProgressLedger | None:
        d = read_json(Path(directory) / LEDGER_FILE)
        return cls.from_dict(adapter, d) if isinstance(d, dict) else None

    # ------------------------------------------------------------------ planning
    def plan(self, keys: Sequence[str], *, on_data_change: str = "continue_unseen") -> ResumePlan:
        """Decides which rows the resumed epoch skips.

        `on_data_change`: "continue_unseen" (default) skips what was consumed; "restart_epoch" ignores the ledger's
        consumption (warm start on the new data from the top of epoch 0).
        """
        total = len(keys)
        if not self.consumed and self.epoch == 0:
            return ResumePlan("fresh", 0, total=total, unseen=total, note="no previous progress")
        if on_data_change == "restart_epoch":
            return ResumePlan("restart", 0, total=total, unseen=total, note="ledger ignored (resume.on_data_change=restart_epoch)")

        present = set(keys)
        skip = {i for i, k in enumerate(keys) if k in self.consumed}
        stale = len(self.consumed - present)
        if total and len(skip) >= total:  # everything was consumed: the epoch is complete
            return ResumePlan(
                "next_epoch", self.epoch + 1, set(), total, total, total,
                stale, f"epoch {self.epoch} complete ({len(skip)}/{total} examples)",
            )
        return ResumePlan(
            "continue", self.epoch, skip, len(skip), total, total - len(skip), stale,
            f"epoch {self.epoch}: {len(skip)}/{total} examples already trained, {total - len(skip)} left",
        )

    # ------------------------------------------------------------------ updates
    def snapshot(self, *, epoch: int, consumed_keys: Iterable[str], data_sha: str | None, data_count: int) -> None:
        """Records progress at a checkpoint: `consumed_keys` are everything trained on in `epoch` so far."""
        new = set(consumed_keys)
        if epoch != self.epoch:
            self.examples_seen_total += len(self.consumed)
            self.epoch = int(epoch)
        self.consumed = new
        self.data_sha, self.data_count = data_sha, int(data_count)

    def end_session(self, **info: Any) -> None:
        self.sessions.append({"ended_ts": time.time(), **info})
