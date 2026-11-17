#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Greedy-decode a GGUF model on llama.cpp's CPU backend and on HIPEP0.

The HIP EP run loads ggml-hipep through GGML_BACKEND_PATH and offloads every
layer, so a matching continuation shows the graphs ran through HIP EP rather
than falling back to the CPU. Example:

  python compare_llama_gguf.py \
      --llama-bin D:/llama.cpp/build/bin \
      --backend D:/hip-ep/build/bin/ggml-hipep.dll \
      --model Llama-3.2-1B-Instruct-f16.gguf
"""

import argparse
import os
import subprocess
import sys
import time


def run(args, device):
    env = os.environ.copy()
    env["GGML_BACKEND_PATH"] = args.backend
    extra = [os.path.dirname(os.path.abspath(args.backend))] + args.path
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    exe = os.path.join(args.llama_bin, "llama-completion")
    cmd = [
        exe,
        "-m",
        args.model,
        "-p",
        args.prompt,
        "-n",
        str(args.tokens),
        "--temp",
        "0",
        "--top-k",
        "1",
        "-s",
        "1",
        "-c",
        str(args.ctx),
        "-fa",
        "off",
        "--simple-io",
        "--no-warmup",
    ]
    if device == "cpu":
        cmd += ["-dev", "none", "-ngl", "0"]
    else:
        cmd += ["-dev", "HIPEP0", "-ngl", "99"]
    start = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{device} run failed with exit code {proc.returncode}")
    return proc.stdout, proc.stderr, elapsed


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--llama-bin", required=True)
    p.add_argument("--backend", required=True, help="path to ggml-hipep.dll/.so")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument(
        "--path",
        action="append",
        default=[],
        help="extra directory for runtime DLLs (repeatable)",
    )
    args = p.parse_args()

    cpu, _, cpu_s = run(args, "cpu")
    gpu, gpu_log, gpu_s = run(args, "hipep")
    print(f"CPU   ({cpu_s:.1f}s): {cpu!r}")
    print(f"HIPEP ({gpu_s:.1f}s): {gpu!r}")
    if "loaded HIPEP backend" not in gpu_log:
        print(
            "warning: the llama.cpp log does not show ggml-hipep loading",
            file=sys.stderr,
        )

    common = 0
    for a, b in zip(cpu, gpu):
        if a != b:
            break
        common += 1
    if cpu == gpu:
        print("PASS: identical greedy continuation")
        return 0
    print(f"DIFF: outputs match for the first {common} characters")
    return 1


if __name__ == "__main__":
    sys.exit(main())
