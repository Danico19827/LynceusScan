# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Device discovery and optional numerical acceleration for the pipeline.

The CPU path is always available. GPU libraries are optional and are detected
without importing them eagerly. The device inventory is deliberately broader
than the currently implemented array adapter (CUDA-only in the tiler) so the
Preferences selector can show all hardware without changing the controller API.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

VALID_MODES = {"auto", "on", "off"}
VALID_BACKENDS = {"auto", "cpu", "cuda", "opencl", "directml"}


@dataclass(frozen=True)
class AccelerationInfo:
    mode: str
    device: str
    backend: str
    available: bool
    reason: str
    device_id: str = ""
    usable: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AccelerationDevice:
    """Hardware/backend entry shown in the Preferences selector."""

    device_id: str
    name: str
    backend: str
    available: bool
    usable: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _normalise_mode(mode: str | None) -> str:
    value = (mode or os.environ.get("LYNCEUS_ACCELERATION", "auto")).strip().lower()
    return value if value in VALID_MODES else "auto"


def _normalise_backend(backend: str | None) -> str:
    value = (backend or os.environ.get("LYNCEUS_ACCELERATION_BACKEND", "auto")).strip().lower()
    return value if value in VALID_BACKENDS else "auto"


def _cuda_devices() -> list[AccelerationDevice]:
    try:
        import cupy as cp

        devices = []
        count = int(cp.cuda.runtime.getDeviceCount())
        for index in range(count):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            devices.append(
                AccelerationDevice(
                    f"cuda:{index}", str(name), "cuda", True, True
                )
            )
        return devices
    except Exception as exc:
        return [AccelerationDevice("cuda", "CUDA", "cuda", False, False, str(exc))]


def _opencl_devices() -> list[AccelerationDevice]:
    try:
        import pyopencl as cl

        devices = []
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                device_id = f"opencl:{platform.name}:{device.name}"
                devices.append(
                    AccelerationDevice(
                        device_id,
                        f"{device.name} ({platform.name})",
                        "opencl",
                        True,
                        False,
                        "Detected; array adapter pending",
                    )
                )
        return devices
    except Exception as exc:
        return [AccelerationDevice("opencl", "OpenCL", "opencl", False, False, str(exc))]


def _directml_devices() -> list[AccelerationDevice]:
    try:
        import torch_directml

        return [
            AccelerationDevice(
                "directml:0",
                str(torch_directml.device_name(0)),
                "directml",
                True,
                False,
                "Detected; tensor adapter pending",
            )
        ]
    except Exception as exc:
        return [AccelerationDevice("directml", "DirectML", "directml", False, False, str(exc))]


@functools.lru_cache(maxsize=1)
def _cached_device_probe() -> tuple:
    """Device inventory, probed once (imports + driver calls are slow)."""
    return (
        tuple(_cuda_devices()),
        tuple(_opencl_devices()),
        tuple(_directml_devices()),
    )


def available_acceleration_devices() -> list[AccelerationDevice]:
    """Return CPU plus detected optional backends for the UI selector."""
    cuda, opencl, directml = _cached_device_probe()
    return [
        AccelerationDevice("cpu", "CPU", "cpu", True, True),
        *cuda,
        *opencl,
        *directml,
    ]


def resolve_acceleration(
    mode: str | None = None,
    backend: str | None = None,
    device_id: str | None = None,
) -> AccelerationInfo:
    """Resolve policy and selected device, falling back to the CPU safely."""
    selected = _normalise_mode(mode)
    requested_backend = _normalise_backend(backend)
    if selected == "off":
        return AccelerationInfo(selected, "cpu", "none", False, "disabled", "cpu", False)
    if requested_backend == "cpu":
        return AccelerationInfo(selected, "CPU", "cpu", True, "available", "cpu", True)

    devices = available_acceleration_devices()
    candidates = [d for d in devices if d.usable and d.backend != "cpu"]
    if requested_backend != "auto":
        candidates = [d for d in candidates if d.backend == requested_backend]
    if device_id:
        candidates = [d for d in candidates if d.device_id == device_id]
    if candidates:
        device = candidates[0]
        return AccelerationInfo(
            selected, device.name, device.backend, True, "available",
            device.device_id, True,
        )

    detected = [d for d in devices if d.available and d.backend != "cpu"]
    if requested_backend != "auto":
        detected = [d for d in detected if d.backend == requested_backend]
    reason = (
        detected[0].reason if detected else
        f"No usable acceleration backend found for '{requested_backend}'"
    )
    if selected == "on":
        logger.warning("Acceleration requested but unavailable; using CPU: %s", reason)
    return AccelerationInfo(selected, "cpu", "none", False, reason, "cpu", False)


def gpu_array_module(info: AccelerationInfo):
    """Return the active array module, or ``None`` for the CPU path."""
    if not info.usable or info.backend != "cuda":
        return None
    try:
        import cupy as cp
    except ImportError:
        return None

    return cp
