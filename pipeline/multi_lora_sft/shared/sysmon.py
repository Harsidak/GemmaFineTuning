"""Host / GPU / TPU utilisation sampling for live run feedback (no training code depends on it).

`SystemMonitor` runs in the orchestrator process on a daemon thread, so a slow or hanging probe can never delay a
phase deadline. Every `every_sec` it appends one flat JSON row to `metrics/system.jsonl`; `latest()` returns the
newest row (the heartbeat line prints it) and `read_latest(path)` lets a training process merge it into its W&B log
as `sys/*` without touching the devices itself.

Probes, each optional and failure-isolated:
  host  psutil (CPU %, RAM, process count) else /proc/meminfo + loadavg; disk free of the output root
  GPU   NVML (nvidia-ml-py: utilisation, memory, power, temperature) else `nvidia-smi --query-gpu`
  TPU   tpu-info (libtpu's runtime metrics service on localhost:8431: HBM in use, TensorCore duty cycle). It talks to
        the libtpu inside the training process over gRPC, so it never opens the TPU and cannot take the device lock.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GIB = 1024**3


class SystemSampler:
    """One `sample()` = one flat dict of numbers. Probes that cannot exist here (no module, no device) switch off."""

    def __init__(self, disk_path: str | Path = "/kaggle/working") -> None:
        self.disk_path = Path(disk_path)
        self._psutil: Any = None
        self._nvml: Any = None
        self._nvml_handles: list[Any] = []
        self._tpu: tuple[Any, Any, Any] | None = None
        self.enabled = {"host": True, "gpu": True, "tpu": True}
        self.errors: dict[str, str] = {}
        try:
            import psutil

            self._psutil = psutil
            psutil.cpu_percent(None)  # prime: the first call always returns 0.0
        except Exception:  # noqa: BLE001
            self._psutil = None

    # ------------------------------------------------------------------ probes
    def _host(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._psutil is not None:
            vm = self._psutil.virtual_memory()
            out.update(cpu_pct=float(self._psutil.cpu_percent(None)), ram_used_gib=(vm.total - vm.available) / GIB,
                       ram_total_gib=vm.total / GIB)
        else:
            info = {}
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    k, _, v = line.partition(":")
                    info[k] = float(v.split()[0]) * 1024
            out.update(ram_used_gib=(info["MemTotal"] - info.get("MemAvailable", 0.0)) / GIB, ram_total_gib=info["MemTotal"] / GIB)
        if hasattr(os, "getloadavg"):
            out["load1"] = float(os.getloadavg()[0])
        probe = self.disk_path if self.disk_path.exists() else next((p for p in self.disk_path.parents if p.exists()), Path("/"))
        out["disk_free_gib"] = shutil.disk_usage(probe).free / GIB
        return out

    def _gpu(self) -> dict[str, float]:
        rows = self._gpu_nvml() if self._nvml is not False else None
        if rows is None:
            rows = self._gpu_smi()
        out: dict[str, float] = {}
        for i, r in enumerate(rows):
            for k, v in r.items():
                out[f"gpu{i}_{k}"] = v
        if rows:
            out["gpu_util_mean"] = sum(r["util_pct"] for r in rows) / len(rows)
            out["gpu_mem_max_gib"] = max(r["mem_gib"] for r in rows)
            out["gpu_mem_total_gib"] = max(r["mem_total_gib"] for r in rows)
        return out

    def _gpu_nvml(self) -> list[dict[str, float]] | None:
        try:
            if self._nvml is None:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            nv = self._nvml
            rows = []
            for h in self._nvml_handles:
                mem = nv.nvmlDeviceGetMemoryInfo(h)
                row = {"util_pct": float(nv.nvmlDeviceGetUtilizationRates(h).gpu), "mem_gib": mem.used / GIB, "mem_total_gib": mem.total / GIB}
                try:
                    row["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                    row["temp_c"] = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
                except Exception:  # noqa: BLE001, S110 - not every board reports power/temperature
                    pass
                rows.append(row)
            return rows
        except Exception:  # noqa: BLE001 - no NVML library / driver: fall back to nvidia-smi
            self._nvml = False
            return None

    def _gpu_smi(self) -> list[dict[str, float]]:
        if shutil.which("nvidia-smi") is None:
            self.enabled["gpu"] = False
            return []
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False,
        )
        rows = []
        for line in r.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 3:
                continue
            row = {"util_pct": float(f[0]), "mem_gib": float(f[1]) / 1024, "mem_total_gib": float(f[2]) / 1024}
            for key, val in zip(("power_w", "temp_c"), f[3:5]):
                try:
                    row[key] = float(val)
                except ValueError:  # "[N/A]"
                    pass
            rows.append(row)
        if not rows:
            self.enabled["gpu"] = False
        return rows

    def _tpu_probe(self) -> dict[str, float]:
        if self._tpu is None:
            from tpu_info import device, metrics

            chip_type, count = device.get_local_chips()
            if not chip_type:
                self.enabled["tpu"] = False
                return {}
            self._tpu = (metrics, chip_type, count)
        metrics, chip_type, _ = self._tpu
        try:
            usage = metrics.get_chip_usage(chip_type)
        except Exception:  # noqa: BLE001 - nothing holds the TPU yet (setup/preflight): the metrics server is not up
            return {}
        out: dict[str, float] = {}
        for u in usage:
            out[f"tpu{u.device_id}_hbm_gib"] = u.memory_usage / GIB
            out[f"tpu{u.device_id}_duty_pct"] = float(u.duty_cycle_pct)
        if usage:
            out["tpu_duty_mean"] = sum(float(u.duty_cycle_pct) for u in usage) / len(usage)
            out["tpu_hbm_max_gib"] = max(u.memory_usage for u in usage) / GIB
            out["tpu_hbm_total_gib"] = max(u.total_memory for u in usage) / GIB
        return out

    # ------------------------------------------------------------------ public
    def sample(self) -> dict[str, Any]:
        row: dict[str, Any] = {"ts": round(time.time(), 1)}
        for name, fn in (("host", self._host), ("gpu", self._gpu), ("tpu", self._tpu_probe)):
            if not self.enabled[name]:
                continue
            try:
                row.update({k: round(v, 3) if isinstance(v, float) else v for k, v in fn().items()})
            except ImportError as exc:
                self.enabled[name] = False
                self.errors[name] = f"ImportError: {exc}"
            except Exception as exc:  # noqa: BLE001 - one bad probe must not hide the others
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        return row


class SystemMonitor:
    """Background sampler: `start()`, `latest()`, `stop()`. Every method is safe to call in any state."""

    def __init__(self, out_path: str | Path, every_sec: float = 30.0, disk_path: str | Path = "/kaggle/working") -> None:
        self.out_path = Path(out_path)
        self.every_sec = float(every_sec)
        self.sampler = SystemSampler(disk_path)
        self._latest: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> SystemMonitor:
        if self.every_sec <= 0 or self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="system-monitor", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = self.sampler.sample()
                with self._lock:
                    self._latest = row
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            except Exception as exc:  # noqa: BLE001 - monitoring is best-effort
                logger.debug("system monitor sample failed: %s", exc)
            self._stop.wait(self.every_sec)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def tail_line(path: str | Path, max_bytes: int = 8192) -> str:
    """Last non-empty line of a (possibly huge, still growing) text file; '' when missing."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - max_bytes))
            lines = fh.read().decode("utf-8", errors="replace").strip().splitlines()
        return lines[-1].strip() if lines else ""
    except OSError:
        return ""


def read_latest(path: str | Path, max_age_sec: float | None = 300.0) -> dict[str, Any]:
    """Newest JSON row of a .jsonl (reads only the file's tail); {} when missing, unreadable, or older than
    `max_age_sec` by its `ts` field (None = no age check)."""
    try:
        row = json.loads(tail_line(path) or "{}")
    except ValueError:
        return {}
    if not isinstance(row, dict):
        return {}
    if max_age_sec is not None and time.time() - float(row.get("ts", 0)) > max_age_sec:
        return {}
    return row


def format_system(row: dict[str, Any]) -> str:
    """One short human line, e.g. 'TPU duty 91% HBM 13.2/15.7 GiB | CPU 40% RAM 120/400 GiB | disk 55 GiB free'."""
    parts = []
    if "tpu_duty_mean" in row:
        parts.append(f"TPU duty {row['tpu_duty_mean']:.0f}% HBM {row.get('tpu_hbm_max_gib', 0):.1f}/{row.get('tpu_hbm_total_gib', 0):.1f} GiB")
    if "gpu_util_mean" in row:
        parts.append(f"GPU util {row['gpu_util_mean']:.0f}% mem {row.get('gpu_mem_max_gib', 0):.1f}/{row.get('gpu_mem_total_gib', 0):.1f} GiB")
    host = []
    if "cpu_pct" in row:
        host.append(f"CPU {row['cpu_pct']:.0f}%")
    if "ram_used_gib" in row:
        host.append(f"RAM {row['ram_used_gib']:.0f}/{row.get('ram_total_gib', 0):.0f} GiB")
    if host:
        parts.append(" ".join(host))
    if "disk_free_gib" in row:
        parts.append(f"disk {row['disk_free_gib']:.0f} GiB free")
    return " | ".join(parts)


def wandb_system_metrics(row: dict[str, Any]) -> dict[str, float]:
    """`sys/<key>` for W&B, aggregate keys only (per-device keys stay in system.jsonl)."""
    keep = ("cpu_pct", "ram_used_gib", "disk_free_gib", "load1", "gpu_util_mean", "gpu_mem_max_gib", "tpu_duty_mean", "tpu_hbm_max_gib")
    return {f"sys/{k}": row[k] for k in keep if isinstance(row.get(k), (int, float))}


def summarize(path: str | Path) -> dict[str, Any]:
    """Peaks / means over a whole system.jsonl for the run report."""
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        return {}
    if not rows:
        return {}

    def col(k: str) -> list[float]:
        return [float(r[k]) for r in rows if isinstance(r.get(k), (int, float))]

    out: dict[str, Any] = {"samples": len(rows), "span_min": round((rows[-1]["ts"] - rows[0]["ts"]) / 60.0, 1)}
    for k, how in (("ram_used_gib", max), ("disk_free_gib", min), ("gpu_mem_max_gib", max), ("tpu_hbm_max_gib", max)):
        if v := col(k):
            out[f"{k}_{how.__name__}"] = round(how(v), 2)
    for k in ("cpu_pct", "gpu_util_mean", "tpu_duty_mean"):
        if v := col(k):
            out[f"{k}_avg"] = round(sum(v) / len(v), 1)
    for k in ("ram_total_gib", "gpu_mem_total_gib", "tpu_hbm_total_gib"):
        if v := col(k):
            out[k] = round(max(v), 2)
    return out
