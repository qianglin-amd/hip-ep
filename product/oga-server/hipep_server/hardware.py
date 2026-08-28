# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""AMD GPU detection and device profiles.

Detection mirrors build.py exactly: the driver-provided HIP runtime on Windows and the
KFD topology on Linux. Neither needs ROCm or TheRock installed, so the server can
report what hardware it is on before anything has been built for it.

The device profile matters to a user in a way the raw gfx string does not: it decides
which certified models will actually fit, and it is what the console and the docs talk
about ("Strix Halo", not "gfx1151").
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")

# gfx arch -> (marketing name, profile id, typical addressable memory in GB).
# memory_hint is only a hint used for ranking: the real value is read from the device
# when the HIP runtime is available.
_ARCH_PROFILES: dict[str, tuple[str, str, int]] = {
    "gfx1151": ("Strix Halo (Ryzen AI Max)", "strix-halo", 48),
    "gfx1150": ("Strix Point (Ryzen AI 300)", "strix", 16),
    "gfx1152": ("Krackan Point (Ryzen AI 7/5)", "krackan", 16),
    "gfx1153": ("Krackan Point 2", "krackan", 16),
    "gfx1100": ("Radeon RX 7900 XTX / XT", "radeon-dgpu", 24),
    "gfx1101": ("Radeon RX 7800 XT", "radeon-dgpu", 16),
    "gfx1102": ("Radeon RX 7700 / 7600", "radeon-dgpu", 12),
    "gfx1200": ("Radeon RX 9060", "radeon-dgpu", 16),
    "gfx1201": ("Radeon RX 9070 XT", "radeon-dgpu", 16),
    "gfx1030": ("Radeon RX 6900 / 6800", "radeon-dgpu", 16),
    "gfx942": ("Instinct MI300X", "instinct", 192),
    "gfx950": ("Instinct MI350X", "instinct", 288),
}


@dataclass
class GpuInfo:
    index: int
    name: str
    gfx_arch: str | None
    memory_gb: float | None
    profile: str
    profile_name: str


@dataclass
class HostInfo:
    hostname: str
    os: str
    os_version: str
    arch: str
    cpu: str | None
    cpu_cores: int | None
    memory_gb: float | None
    python: str
    gpus: list[GpuInfo] = field(default_factory=list)

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def has_amd_gpu(self) -> bool:
        return any(g.gfx_arch for g in self.gpus)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_amd_gpu"] = self.has_amd_gpu
        return data


def _profile_for(arch: str | None) -> tuple[str, str, int | None]:
    if not arch:
        return "unknown", "Unknown device", None
    if arch in _ARCH_PROFILES:
        name, profile, mem = _ARCH_PROFILES[arch]
        return profile, name, mem
    # Unknown but well-formed gfx string: classify by family so a brand-new part still
    # gets sensible treatment instead of being rejected.
    if arch.startswith("gfx11") or arch.startswith("gfx12"):
        return "radeon-dgpu", f"AMD RDNA GPU ({arch})", None
    if arch.startswith("gfx9"):
        return "instinct", f"AMD Instinct ({arch})", None
    return "unknown", f"AMD GPU ({arch})", None


def _detect_windows() -> list[GpuInfo]:
    """Read device properties from the driver-shipped HIP runtime via ctypes.

    The gcnArchName offset matches HIP's R0600 (6.x and later) hipDeviceProp_t layout,
    the same fixed-offset trick build.py and LLVM's offload-arch use.
    """
    import ctypes

    # Leading fields are declared explicitly so ctypes applies the same alignment the
    # C header does; gcnArchName then lands at its fixed offset 1160.
    class _PropHead(ctypes.Structure):
        _fields_ = [
            ("name", ctypes.c_char * 256),
            ("uuid", ctypes.c_char * 16),
            ("luid", ctypes.c_char * 8),
            ("luidDeviceNodeMask", ctypes.c_uint),
            ("totalGlobalMem", ctypes.c_size_t),
        ]

    _GCN_ARCH_OFFSET = 1160
    _head_size = ctypes.sizeof(_PropHead)
    if _head_size > _GCN_ARCH_OFFSET:  # layout changed under us; give up cleanly
        return []

    class _PropR0600(ctypes.Structure):
        _fields_ = [
            ("head", _PropHead),
            ("_pad", ctypes.c_char * (_GCN_ARCH_OFFSET - _head_size)),
            ("gcnArchName", ctypes.c_char * 256),
            ("_pad2", ctypes.c_char * 512),
        ]

    for dll in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"):
        try:
            lib = ctypes.WinDLL(dll)
        except OSError:
            continue
        try:
            get_count = lib.hipGetDeviceCount
            get_props = lib.hipGetDevicePropertiesR0600
        except AttributeError:
            continue
        count = ctypes.c_int(0)
        if get_count(ctypes.byref(count)) != 0 or count.value <= 0:
            continue
        gpus: list[GpuInfo] = []
        for i in range(count.value):
            prop = _PropR0600()
            if get_props(ctypes.byref(prop), i) != 0:
                continue
            arch = prop.gcnArchName.decode("ascii", "ignore").strip().split(":", 1)[0]
            name = prop.head.name.decode("ascii", "ignore").strip()
            profile, profile_name, mem_hint = _profile_for(arch or None)
            total = prop.head.totalGlobalMem
            mem = total / (1024**3) if total else mem_hint
            gpus.append(
                GpuInfo(
                    index=i,
                    name=name or profile_name,
                    gfx_arch=arch or None,
                    memory_gb=round(mem, 1) if mem else None,
                    profile=profile,
                    profile_name=profile_name,
                )
            )
        if gpus:
            return gpus
    return []


def _detect_linux() -> list[GpuInfo]:
    """Read gfx_target_version and memory from /sys/class/kfd topology.

    gfx_target_version encodes the arch as major*10000 + minor*100 + step, so 110501
    means gfx1151. The CPU node reports 0 and is skipped.
    """
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    if not nodes.is_dir():
        return []
    gpus: list[GpuInfo] = []
    for node in sorted(nodes.glob("*"), key=lambda p: p.name):
        props = node / "properties"
        if not props.is_file():
            continue
        try:
            text = props.read_text()
        except OSError:
            continue
        fields = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                fields[parts[0]] = parts[1]
        ver = int(fields.get("gfx_target_version", 0) or 0)
        if not ver:
            continue
        major, minor, step = ver // 10000, (ver // 100) % 100, ver % 100
        arch = f"gfx{major}{minor:x}{step:x}"
        profile, profile_name, mem_hint = _profile_for(arch)
        mem = mem_hint
        mem_banks = node / "mem_banks"
        if mem_banks.is_dir():
            total = 0
            for bank in mem_banks.glob("*/properties"):
                try:
                    for line in bank.read_text().splitlines():
                        if line.startswith("size_in_bytes "):
                            total += int(line.split()[1])
                except (OSError, ValueError):
                    continue
            if total:
                mem = total / (1024**3)
        name = (node / "name").read_text().strip() if (node / "name").is_file() else ""
        gpus.append(
            GpuInfo(
                index=len(gpus),
                name=name or profile_name,
                gfx_arch=arch,
                memory_gb=round(mem, 1) if mem else None,
                profile=profile,
                profile_name=profile_name,
            )
        )
    return gpus


def detect_gpus() -> list[GpuInfo]:
    try:
        return _detect_windows() if IS_WINDOWS else _detect_linux()
    except Exception:  # noqa: BLE001 - detection must never take the server down
        return []


def _cpu_info() -> tuple[str | None, int | None, float | None]:
    cpu = platform.processor() or None
    cores = None
    memory = None
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        memory = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001
        pass
    return cpu, cores, memory


def detect_host() -> HostInfo:
    cpu, cores, memory = _cpu_info()
    return HostInfo(
        hostname=platform.node(),
        os="Windows" if IS_WINDOWS else platform.system(),
        os_version=platform.version(),
        arch=platform.machine(),
        cpu=cpu,
        cpu_cores=cores,
        memory_gb=memory,
        python=platform.python_version(),
        gpus=detect_gpus(),
    )


def fits_on_device(min_vram_gb: float | None, host: HostInfo) -> bool | None:
    """Whether a model's minimum memory requirement fits this host.

    Returns None when we cannot tell, which the UI renders as "unknown" rather than
    guessing. On APUs the GPU shares system memory, so we compare against whichever
    number we have rather than assuming dedicated VRAM.
    """
    if min_vram_gb is None:
        return None
    gpu = host.primary_gpu
    available = None
    if gpu and gpu.memory_gb:
        available = gpu.memory_gb
        if gpu.profile in {"strix-halo", "strix", "krackan"} and host.memory_gb:
            # Unified memory: the GPU can address well beyond its reported carve-out.
            available = max(available, host.memory_gb * 0.75)
    elif host.memory_gb:
        available = host.memory_gb * 0.75
    if available is None:
        return None
    return available >= min_vram_gb
