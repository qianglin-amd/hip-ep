#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run an ONNX model on the HIP EP with random inputs.

python run_onnx.py model.onnx [--compare-cpu] [--runs N] [--dim name=value ...]
"""

import argparse
import time

import numpy as np
import onnxruntime as ort

import hipep

_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def random_input(spec, dims, rng):
    dtype = _NP_DTYPE.get(spec.type, np.float32)
    shape = []
    for d in spec.shape:
        if isinstance(d, int) and d > 0:
            shape.append(d)
        else:
            shape.append(dims.get(d, 1))
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    return rng.integers(0, 2, size=shape).astype(dtype)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=3, help="timed runs (default 3)")
    ap.add_argument(
        "--compare-cpu",
        action="store_true",
        help="also run on the CPU EP and compare outputs",
    )
    ap.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="let unsupported nodes run on the CPU",
    )
    ap.add_argument(
        "--dim",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="value for a symbolic input dimension",
    )
    args = ap.parse_args()

    dims = {}
    for item in args.dim:
        key, _, value = item.partition("=")
        dims[key] = int(value)

    t0 = time.perf_counter()
    so = hipep.session_options(allow_cpu_fallback=args.allow_cpu_fallback)
    sess = ort.InferenceSession(args.model, sess_options=so)
    print(
        f"session created in {(time.perf_counter() - t0) * 1000:.0f} ms, providers: {sess.get_providers()}"
    )

    rng = np.random.default_rng(42)
    feeds = {i.name: random_input(i, dims, rng) for i in sess.get_inputs()}
    outputs = None
    for r in range(max(args.runs, 1)):
        t0 = time.perf_counter()
        outputs = sess.run(None, feeds)
        print(f"run {r}: {(time.perf_counter() - t0) * 1000:.2f} ms")
    for spec, value in zip(sess.get_outputs(), outputs):
        print(f"output {spec.name}: shape={value.shape} dtype={value.dtype}")

    if args.compare_cpu:
        cpu = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        for spec, gpu_value, cpu_value in zip(
            sess.get_outputs(), outputs, cpu.run(None, feeds)
        ):
            g = gpu_value.astype(np.float64)
            c = cpu_value.astype(np.float64)
            err = np.abs(g - c).max() if g.size else 0.0
            ref = np.abs(c).max() if c.size else 0.0
            print(f"compare {spec.name}: max abs diff {err:.4g} (max |cpu| {ref:.4g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
