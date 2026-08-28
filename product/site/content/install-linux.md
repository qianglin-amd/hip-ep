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
python3 -m pip install --upgrade pip
python3 -m pip install hipep-server
```

Or from this repository:

```bash
cd product/oga-server
python3 -m pip install -e .
```

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

The device files and the two groups are the whole trick:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -p 11434:11434 \
  -v "$HOME/.hipep:/root/.hipep" \
  hipep/oga-server:latest
```

Mounting `~/.hipep` keeps downloaded models outside the container, which matters when a
model is 12 GB and the container is disposable.

For building the compiler in a container, this repository has `./docker/run.sh build`,
which creates the image, detects the host architecture, bind-mounts the workspace and runs
`build.py` inside a compile-only container. It is the authoritative build path on the
project's Linux host.

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
