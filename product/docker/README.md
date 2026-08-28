<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# oga-server in a container

For Linux hosts. Windows runs the server natively; see
[install-windows.md](../site/content/install-windows.md).

## Build and run

From the repository root:

```bash
docker compose -f product/docker/compose.yaml up --build
```

Then <http://127.0.0.1:11434>.

Without compose, the device passthrough has to be spelled out:

```bash
docker build -t hipep/oga-server:latest -f product/docker/Dockerfile product

docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -p 127.0.0.1:11434:11434 \
  -v hipep-home:/var/lib/hipep \
  hipep/oga-server:latest
```

All four of `/dev/kfd`, `/dev/dri`, `video` and `render` are needed. With any of them
missing, ROCm initialises and then reports no agents, which surfaces as the server
quietly choosing the replay engine.

## Checking what it chose

```bash
docker exec -it <container> hipep doctor
```

The `engine` line says which engine is live and, when it is the replay engine, which
precondition was not met. `gpu 0` should name your architecture; if it is absent, the
device passthrough is the thing to fix, not the image.

## The inference stack is not in the image

`onnxruntime-genai` is built against a specific ROCm version and GPU architecture. A
wheel baked into the image would work on the machine it was built for and fail at the
first kernel launch elsewhere, which is the failure mode this project's build rules
exist to prevent. So the image ships without one and runs the replay engine until you
supply it:

```bash
OGA_WHEEL=/path/to/onnxruntime_genai-<version>-cp312-linux_x86_64.whl \
  docker compose -f product/docker/compose.yaml up --build
```

`OGA_WHEEL` accepts anything `pip install` does, including an index URL.

## Models

The `hipep-home` volume holds downloaded and ported models. It is a named volume so a
`docker compose down` does not delete 30 GB of weights. To share a model directory with
the host instead:

```bash
-v "$HOME/.hipep:/var/lib/hipep"
```

## Compiling the compiler

This image serves models; it does not build hip-ep. For that the repository has
`./docker/run.sh build` at the root, which detects the host architecture, bind-mounts
the workspace and runs `build.py` in a compile-only container.
