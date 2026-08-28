##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# oga-server for Linux hosts with an AMD GPU.
#
# The image carries the server, the console, the CLI and the catalog. It deliberately
# does not carry onnxruntime-genai: that wheel is built against a ROCm version and a GPU
# architecture, so baking one in would produce an image that runs on one machine and
# crashes at the first kernel launch on the next. Supply it at build time with
# --build-arg OGA_WHEEL=..., or mount it, or run without it and get the replay engine
# with the reason printed at startup.

ARG ROCM_BASE=rocm/dev-ubuntu-24.04:6.4-complete
FROM ${ROCM_BASE}

LABEL org.opencontainers.image.title="oga-server"
LABEL org.opencontainers.image.description="OpenAI- and Ollama-compatible server for HIP EP on AMD GPUs"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HIPEP_HOME=/var/lib/hipep \
    HIPEP_HOST=0.0.0.0 \
    HIPEP_PORT=11434

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# A venv rather than --break-system-packages: the ROCm base ships Python packages of its
# own, and letting pip resolve over the top of them is how a working image stops working
# after an unrelated rebuild.
ENV VIRTUAL_ENV=/opt/hipep
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/hipep-src

# Dependencies resolve from the project metadata alone, so this layer survives an edit
# to any source file.
COPY pyproject.toml README.md ./
RUN mkdir -p oga-server/hipep_server toolkit/hipep_toolkit \
    && touch oga-server/hipep_server/__init__.py toolkit/hipep_toolkit/__init__.py \
    && pip install --upgrade pip \
    && pip install .

COPY oga-server ./oga-server
COPY toolkit ./toolkit
COPY modelzoo ./modelzoo
COPY site ./site

# --no-deps: the dependency layer above is already correct, and resolving again here
# would defeat it.
RUN pip install --no-deps -e . \
    && pip install "Markdown>=3.6" \
    && python site/build.py

ARG OGA_WHEEL=""
RUN if [ -n "$OGA_WHEEL" ]; then pip install "$OGA_WHEEL"; fi

RUN mkdir -p "$HIPEP_HOME"
VOLUME ["/var/lib/hipep"]
EXPOSE 11434

# The GPU stack is only reachable through /dev/kfd and /dev/dri, which a build cannot
# see, so the health check asks the server what it decided at startup instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${HIPEP_PORT}/health" || exit 1

ENTRYPOINT ["hipep"]
CMD ["serve"]
