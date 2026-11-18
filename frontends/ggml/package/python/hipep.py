#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Use the HIP EP in this package from ONNX Runtime and ONNX Runtime GenAI.

import hipep
import onnxruntime as ort

so = hipep.session_options()          # HIP EP with CPU fallback disabled
sess = ort.InferenceSession("model.onnx", sess_options=so)

import onnxruntime_genai as og

model = og.Model(hipep.genai_config("path/to/genai_model_dir"))
"""

import os

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin"))
EP_LIBRARY = os.path.join(BIN_DIR, "hipgpu.dll")

# Registration names. ONNX Runtime reports the EP under the name it was
# registered with; GenAI configs written for this EP name it "AMDGPU".
ORT_EP_NAME = "hipgpu"
GENAI_EP_NAME = "AMDGPU"

_registered = set()


def _prepare_path():
    # The EP loads hipblaslt.dll and its kernel library by name at run time,
    # so the package's bin directory must be on PATH.
    path = os.environ.get("PATH", "")
    if BIN_DIR.lower() not in (p.lower() for p in path.split(os.pathsep)):
        os.environ["PATH"] = BIN_DIR + os.pathsep + path
    if not os.path.isfile(EP_LIBRARY):
        raise FileNotFoundError(f"HIP EP library not found: {EP_LIBRARY}")


def register_ort(name=ORT_EP_NAME):
    """Registers the HIP EP with ONNX Runtime and returns its devices."""
    import onnxruntime as ort

    _prepare_path()
    if ("ort", name) not in _registered:
        ort.register_execution_provider_library(name, EP_LIBRARY)
        _registered.add(("ort", name))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not devices:
        raise RuntimeError(f"no {name} device found; check the GPU and driver")
    return devices


def session_options(provider_options=None, allow_cpu_fallback=False, name=ORT_EP_NAME):
    """SessionOptions that place the model on the HIP EP.

    With allow_cpu_fallback=False (the default) session creation fails when a
    node cannot run on the GPU, so a successful session is proof of GPU
    execution. Pass True for models with ops the EP does not support.
    """
    import onnxruntime as ort

    devices = register_ort(name)
    so = ort.SessionOptions()
    if not allow_cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(devices, dict(provider_options or {}))
    return so


def register_genai(name=GENAI_EP_NAME):
    """Registers the HIP EP with ONNX Runtime GenAI."""
    import onnxruntime_genai as og

    _prepare_path()
    if ("genai", name) not in _registered:
        og.register_execution_provider_library(name, EP_LIBRARY)
        _registered.add(("genai", name))


def genai_config(model_dir, name=GENAI_EP_NAME):
    """An og.Config for model_dir that runs on the HIP EP, whatever provider
    the model's genai_config.json names."""
    import onnxruntime_genai as og

    register_genai(name)
    config = og.Config(model_dir)
    config.clear_providers()
    config.append_provider(name)
    return config
