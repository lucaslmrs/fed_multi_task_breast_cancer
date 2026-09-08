"""Operational helpers shared by every training entrypoint.

Numerical precision is part of the scientific protocol and lives under ``training.precision``.
The top-level ``runtime`` block is observational/operational: loader workers, Ray resources,
inference batching, and telemetry may change wall-clock time but not the training budget.
"""

from __future__ import annotations

import csv
import logging
import math
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch


PRECISIONS = {"fp32", "bf16"}


def precision_name(config: dict) -> str:
    """Resolve the declared numerical protocol; old snapshots remain FP32."""
    value = str(config.get("training", {}).get("precision", "fp32")).lower()
    if value not in PRECISIONS:
        raise ValueError(f"training.precision must be one of {sorted(PRECISIONS)}, got {value!r}")
    return value


class PrecisionPolicy:
    """Validated autocast policy for one resolved device."""

    def __init__(self, precision: str, device: str | torch.device):
        self.name = str(precision).lower()
        if self.name not in PRECISIONS:
            raise ValueError(
                f"training.precision must be one of {sorted(PRECISIONS)}, got {precision!r}"
            )
        self.device = torch.device(device)
        if self.name == "bf16":
            if self.device.type != "cuda":
                raise RuntimeError(
                    "training.precision=bf16 requires a CUDA device; use fp32 for CPU runs"
                )
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "training.precision=bf16 requires a CUDA GPU with native BF16 support"
                )

    @property
    def enabled(self) -> bool:
        return self.name == "bf16"

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def move(self, tensor: torch.Tensor) -> torch.Tensor:
        return move_to_device(tensor, self.device)

    def ensure_finite(self, loss: torch.Tensor, context: str) -> None:
        if not bool(torch.isfinite(loss).all().item()):
            raise FloatingPointError(f"Non-finite loss under {self.name} ({context})")


def precision_policy(config: dict, device: str | torch.device) -> PrecisionPolicy:
    return PrecisionPolicy(precision_name(config), device)


def move_to_device(tensor: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Use asynchronous host-to-device copies whenever CUDA and pinned memory allow it."""
    resolved = torch.device(device)
    return tensor.to(resolved, non_blocking=resolved.type == "cuda")


def runtime_config(config: dict) -> dict:
    """Resolve runtime-only defaults without mutating the supplied config."""
    configured = config.get("runtime", {}) or {}
    loader = dict(configured.get("dataloader", {}) or {})
    loader.setdefault("num_workers", 4)
    loader.setdefault("federated_num_workers", 0)
    loader.setdefault("pin_memory", True)
    loader.setdefault("persistent_workers", True)
    loader.setdefault("prefetch_factor", 2)

    federated = dict(configured.get("federated", {}) or {})
    legacy_fed = config.get("federated", {}) or {}
    federated.setdefault("ray_num_cpus", legacy_fed.get("ray_num_cpus", 2))
    federated.setdefault(
        "client_resources",
        legacy_fed.get("client_resources", {"num_cpus": 1, "num_gpus": 0.0}),
    )
    federated["client_resources"] = dict(federated["client_resources"])

    telemetry = dict(configured.get("telemetry", {}) or {})
    telemetry.setdefault("enabled", True)
    telemetry.setdefault("gpu_interval_seconds", 1.0)

    return {
        "inference_batch_size": int(configured.get("inference_batch_size", 32)),
        "dataloader": loader,
        "federated": federated,
        "telemetry": telemetry,
    }


def validate_runtime_config(config: dict) -> None:
    precision_name(config)
    runtime = runtime_config(config)
    if runtime["inference_batch_size"] < 1:
        raise ValueError("runtime.inference_batch_size must be a positive integer")
    loader = runtime["dataloader"]
    for field in ("num_workers", "federated_num_workers"):
        value = loader[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"runtime.dataloader.{field} must be a non-negative integer")
    if not isinstance(loader["pin_memory"], bool):
        raise ValueError("runtime.dataloader.pin_memory must be a boolean")
    if not isinstance(loader["persistent_workers"], bool):
        raise ValueError("runtime.dataloader.persistent_workers must be a boolean")
    if int(loader["prefetch_factor"]) < 1:
        raise ValueError("runtime.dataloader.prefetch_factor must be positive")
    telemetry = runtime["telemetry"]
    if not isinstance(telemetry["enabled"], bool):
        raise ValueError("runtime.telemetry.enabled must be a boolean")
    if float(telemetry["gpu_interval_seconds"]) <= 0:
        raise ValueError("runtime.telemetry.gpu_interval_seconds must be positive")


def dataloader_kwargs(config: dict, *, federated: bool = False) -> dict:
    """Build kwargs accepted by ``torch.utils.data.DataLoader``."""
    loader = runtime_config(config)["dataloader"]
    workers = int(
        loader["federated_num_workers"] if federated else loader["num_workers"]
    )
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(loader["pin_memory"]),
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=bool(loader["persistent_workers"]),
            prefetch_factor=int(loader["prefetch_factor"]),
        )
    return kwargs


def configure_worker_threads(config: dict) -> None:
    """Prevent Ray actors from each claiming every host CPU thread."""
    resources = runtime_config(config)["federated"]["client_resources"]
    threads = max(1, int(math.ceil(float(resources.get("num_cpus", 1)))))
    torch.set_num_threads(threads)


_EVENT_FIELDS = [
    "timestamp_utc",
    "phase",
    "setup",
    "fold",
    "client_id",
    "round",
    "epoch",
    "duration_seconds",
    "examples",
    "batches",
    "examples_per_second",
    "batches_per_second",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
]


class RuntimeEvents:
    """Buffered phase-level performance events."""

    def __init__(self, output_path: str | Path, device: str | torch.device):
        self.output_path = Path(output_path)
        self.device = torch.device(device)
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def measure(self, phase: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        payload: dict[str, Any] = {}
        yield payload
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        duration = time.perf_counter() - started
        examples = int(payload.get("examples", 0))
        batches = int(payload.get("batches", 0))
        row = {field: "" for field in _EVENT_FIELDS}
        row.update(metadata)
        row.update(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            phase=phase,
            duration_seconds=duration,
            examples=examples,
            batches=batches,
            examples_per_second=(examples / duration if examples else ""),
            batches_per_second=(batches / duration if batches else ""),
        )
        if self.device.type == "cuda":
            row["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
            row["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / 2**20
        self.rows.append(row)

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.output_path)


class GpuTelemetry:
    """Low-frequency aggregate GPU sampler backed by NVIDIA NVML."""

    _FIELDS = [
        "timestamp_utc",
        "elapsed_seconds",
        "gpu_index",
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "memory_used_mb",
        "memory_total_mb",
        "power_watts",
        "temperature_c",
    ]

    def __init__(self, output_path: str | Path, *, enabled: bool, interval_seconds: float):
        self.output_path = Path(output_path)
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.interval = float(interval_seconds)
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._nvml = None
        self._handle = None

    def start(self) -> "GpuTelemetry":
        if not self.enabled:
            return self
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # telemetry must never invalidate a scientific run
            logging.warning("GPU telemetry disabled because NVML initialization failed: %s", exc)
            self.enabled = False
            return self
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            nvml, handle = self._nvml, self._handle
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temperature = nvml.nvmlDeviceGetTemperature(
                    handle, nvml.NVML_TEMPERATURE_GPU
                )
                self.rows.append(
                    {
                        "timestamp_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "elapsed_seconds": time.perf_counter() - self._started,
                        "gpu_index": 0,
                        "utilization_gpu_percent": util.gpu,
                        "utilization_memory_percent": util.memory,
                        "memory_used_mb": memory.used / 2**20,
                        "memory_total_mb": memory.total / 2**20,
                        "power_watts": power,
                        "temperature_c": temperature,
                    }
                )
            except Exception as exc:
                logging.warning("GPU telemetry sampling stopped: %s", exc)
                break
            self._stop.wait(self.interval)

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2))
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.output_path)


def start_gpu_telemetry(config: dict, run_path: str | Path) -> GpuTelemetry:
    telemetry = runtime_config(config)["telemetry"]
    return GpuTelemetry(
        Path(run_path) / "gpu_telemetry.csv",
        enabled=telemetry["enabled"],
        interval_seconds=telemetry["gpu_interval_seconds"],
    ).start()
