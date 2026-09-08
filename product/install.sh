#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Install oga-server and the hipep CLI on Linux, then report what the machine can do.

set -euo pipefail

PRODUCT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${PRODUCT_DIR}/.venv"
EXTRAS=("site")
RUN_TESTS=0
QUANT=0
GPU_WHEELS=""

usage() {
    cat <<'EOF'
Install oga-server and the hipep CLI, then report what this machine can do.

  ./product/install.sh                  server, CLI, docs
  ./product/install.sh --dev            also install and run the test suite
  ./product/install.sh --quant          also the toolchain that ports models
  ./product/install.sh --gpu-wheels DIR also the AMD GPU stack, from a wheel bundle
  ./product/install.sh --venv PATH      install into a specific virtualenv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)   EXTRAS+=("dev"); RUN_TESTS=1; shift ;;
        # No extra here: the exporter caps Python below 3.14 and pulls in torch, so it
        # goes into its own environment further down rather than this one.
        --quant) QUANT=1; shift ;;
        --gpu-wheels) GPU_WHEELS="$2"; shift 2 ;;
        --venv)  VENV="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# The GPU wheels are built per Python version and the version is in the file name, so
# when a bundle is given it decides which interpreter the virtualenv needs. Any other
# choice ends in "not a supported wheel on this platform", which explains nothing.
INTERPRETER=python3
if [[ -n "$GPU_WHEELS" && -d "$GPU_WHEELS" ]]; then
    for wheel in "$GPU_WHEELS"/*.whl; do
        [[ -e "$wheel" ]] || continue
        if [[ "$(basename "$wheel")" =~ -cp3([0-9]+)- ]]; then
            wanted="3.${BASH_REMATCH[1]}"
            echo "==> the GPU wheels are built for Python $wanted"
            if command -v "python$wanted" >/dev/null 2>&1; then
                INTERPRETER="python$wanted"
            else
                echo "python$wanted was not found; install it so the wheels can be used" >&2
                exit 1
            fi
            break
        fi
    done
fi

# Ubuntu 24.04 and Fedora 40 both refuse a pip install outside a virtualenv, and
# --break-system-packages on a machine that also runs ROCm's own Python packages is how
# a working install stops working after an unrelated system update.
"$INTERPRETER" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "Python 3.10 or newer is required; found $("$INTERPRETER" --version)" >&2
    exit 1
}

if [[ ! -d "$VENV" ]]; then
    echo "==> creating virtualenv at $VENV"
    "$INTERPRETER" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> installing hipep-product[$(IFS=,; echo "${EXTRAS[*]}")]"
python -m pip install --upgrade pip >/dev/null
python -m pip install -e "${PRODUCT_DIR}[$(IFS=,; echo "${EXTRAS[*]}")]"

if [[ -n "$GPU_WHEELS" ]]; then
    # These are not on PyPI: each is built for one ROCm version, one GPU architecture
    # and one Python version, so they ship as a bundle rather than being resolved.
    shopt -s nullglob
    wheels=("$GPU_WHEELS"/*.whl)
    shopt -u nullglob
    if (( ${#wheels[@]} == 0 )); then
        echo "no .whl files in $GPU_WHEELS" >&2
        exit 1
    fi
    echo "==> installing the GPU stack (${#wheels[@]} wheels)"
    python -m pip install --force-reinstall "${wheels[@]}"
fi

if (( QUANT )); then
    echo "==> building the porting toolchain"
    "${PRODUCT_DIR}/install-port-toolchain.sh"
fi

echo "==> building the documentation site"
python "${PRODUCT_DIR}/site/build.py"

if (( RUN_TESTS )); then
    echo "==> running the test suite"
    ( cd "$PRODUCT_DIR" && python -m pytest -q )
fi

echo
hipep doctor || true

cat <<EOF

Activate this environment with:
  source ${VENV}/bin/activate

Then:
  hipep serve                     start the server and console on :11434
  hipep models                    the certified model zoo
  hipep pull llama3.1:8b          download a model

The GPU stack ships as a wheel bundle rather than on PyPI, because each wheel is built
for one ROCm version, one GPU architecture and one Python version. Point the installer
at it to run models on the GPU:

  ./product/install.sh --gpu-wheels <path-to-wheels>

Until it is present the server replays published timings and says so. To port your own
models as well, re-run with --quant.
EOF
