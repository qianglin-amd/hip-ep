---
title: Install on Linux
subtitle: Native or in a container
order: 2
---

Linux support is real but younger than Windows: the compiler, the server, the console and
the client examples all work, and the certified benchmark numbers in this release were
measured on Windows. Expect Linux numbers to appear as a separate device column rather
than being assumed equal.

## Before you start

| Requirement | Notes |
|---|---|
| AMD GPU | Ryzen AI, Radeon RX 6000 or newer, or Instinct |
| ROCm | A release matching your kernel. `rocminfo` must run. |
| Kernel with amdgpu | Distribution kernel is fine on Ubuntu 22.04/24.04 |
| Python 3.10 or newer | `python3 --version` |
| Group membership | Your user must be in `render` and `video` |

```bash
rocminfo | grep -m1 gfx          # the architecture the driver reports
groups | tr ' ' '\n' | grep -E 'render|video'
```

If the second command prints nothing:

```bash
sudo usermod -aG render,video "$USER"   # then log out and back in
```

## Install

```bash
./product/install.sh          # add --dev for the test suite
```

Ubuntu 24.04 and Fedora 40 both refuse a pip install outside a virtualenv, so the script
creates one at `product/.venv`, installs into it, builds this documentation and finishes
by running the preflight below.

There is no release on PyPI yet, so installing from a checkout is the only path today.
When one is published this page will carry the `pip install` line instead.

## Add the GPU stack

The step above gives you a working server that *replays* published timings. To run models
on the GPU it needs the AMD execution provider wheels, and those are not on PyPI: each is
built for one ROCm version, one GPU architecture and one Python version, so `pip install
onnxruntime-genai` reaches a CPU build instead. They ship as a bundle:

```bash
./product/install.sh --gpu-wheels <path-to-the-wheel-bundle>
```

The Python version in the wheel names has to match the virtualenv's. `hipep doctor` says
`engine  onnxruntime-genai available` once this lands, and the server prints a real GPU
name instead of `replay` at startup.

## Add the porting toolchain, if you want to bring your own model

Only needed to turn a Hugging Face checkpoint into a HIP EP artifact; serving certified
models does not use it. The exporter needs Python 3.11–3.13 and pulls in torch, so it gets
its own environment rather than being forced into the server's:

```bash
./product/install-port-toolchain.sh
```

It finds a suitable interpreter, builds the environment where the server looks for it by
convention, and proves it can import the exporter before reporting success. Nothing needs
configuring afterwards — `hipep doctor` will show `porting toolchain  ok`.

## Check the machine first

```bash
hipep doctor
```

Each line is one precondition — driver, wheels, GPU, store, catalog, port — so a problem
names the layer that owns it. On Linux the check that most often needs attention is group
membership: without `render` and `video`, ROCm initialises and then reports no agents,
which shows up here as `gpu: no AMD GPU detected` rather than as a slow model later.

Warnings are not failures. A machine with no GPU serves the replay engine and runs the
full test suite. Add `--json` to consume the result from a script, `--no-server` to skip
the check for a running server.

## Start the server

```bash
hipep serve
```

The startup block reports the GPU, the architecture and the engine. As on Windows, if
`engine` says `replay` the server did not find a usable GPU stack and the reason is printed
beside it.

```bash
hipep pull llama3.1:8b
xdg-open http://127.0.0.1:11434
```

## Running as a service

```ini
# /etc/systemd/system/hipep.service
[Unit]
Description=HIP EP oga-server
After=network-online.target

[Service]
User=%i
Environment=HIPEP_HOME=/var/lib/hipep
ExecStart=/usr/local/bin/hipep serve --host 127.0.0.1 --port 11434
Restart=on-failure
# Model loads are long; do not let a slow first load look like a hang.
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now hipep@$USER
journalctl -u hipep@$USER -f
```

Keep `--host 127.0.0.1` unless the machine is on a network you control. The server does not
authenticate.

## In a container

```bash
docker compose -f product/docker/compose.yaml up --build
```

Without compose, the device passthrough has to be spelled out, and it is the whole trick:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -p 127.0.0.1:11434:11434 \
  -v hipep-home:/var/lib/hipep \
  hipep/oga-server:latest
```

All four of `/dev/kfd`, `/dev/dri`, `video` and `render` are needed. Miss any one and the
container starts, ROCm finds no agents, and the server falls back to the replay engine.
`docker exec -it <container> hipep doctor` says which.

The image does not carry `onnxruntime-genai`, because that wheel is built for one ROCm
version and one GPU architecture and would fail at the first kernel launch on a different
machine. Supply yours with `OGA_WHEEL=/path/to/wheel docker compose ... up --build`. See
[product/docker/README.md](https://github.com/ROCm/hip-ep/tree/main/product/docker).

For building the compiler in a container, this repository has `./docker/run.sh build` at
the root, which creates the image, detects the host architecture, bind-mounts the
workspace and runs `build.py` inside a compile-only container. It is the authoritative
build path on the project's Linux host.

## Running the tests

The product's own test suite runs against the replay engine, so it needs no GPU, no
`onnxruntime-genai` and no network. It is the same suite, and the same result, as on
Windows.

```bash
cd product
pip install -e ".[dev,site]"
pytest
```

About thirty seconds. It covers the catalog, both API surfaces, admission control, the
job pipeline, artifact sharing, the benchmark harness, the CLI and the documentation
build. `pytest -m "not slow"` skips the one test that runs at the certified replay
pace.

## Building from source

```bash
# Detect the architecture; do not guess it
"$THEROCK_DIST/lib/llvm/bin/amdgpu-arch"

python3 build.py                  # native
./docker/run.sh build             # containerised, recommended
```

A kernel built for the wrong architecture builds cleanly and then fails at launch, so let
detection do its job unless you have a reason to override it with `--hip_arch`.

## Differences from Windows worth knowing

- **Shared-memory GPUs report differently.** On APUs, VRAM is carved from system memory,
  and how much is available to the GPU depends on the BIOS setting and the kernel. The
  console shows what the driver reports, which can differ from the BIOS number.
- **Benchmarks are not cross-comparable.** A number measured on Linux belongs in a Linux
  column. Do not compare it against the Windows certified numbers on this site.
- **The default model directory follows `$HIPEP_HOME`**, falling back to `~/.hipep`.
