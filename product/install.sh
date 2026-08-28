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

usage() {
    cat <<'EOF'
Install oga-server and the hipep CLI, then report what this machine can do.

  ./product/install.sh              server, CLI, docs
  ./product/install.sh --dev        also install and run the test suite
  ./product/install.sh --quant      also the porting toolchain (Quark, torch)
  ./product/install.sh --venv PATH  install into a specific virtualenv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)   EXTRAS+=("dev"); RUN_TESTS=1; shift ;;
        --quant) EXTRAS+=("quant"); shift ;;
        --venv)  VENV="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Ubuntu 24.04 and Fedora 40 both refuse a pip install outside a virtualenv, and
# --break-system-packages on a machine that also runs ROCm's own Python packages is how
# a working install stops working after an unrelated system update.
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "Python 3.10 or newer is required; found $(python3 --version)" >&2
    exit 1
}

if [[ ! -d "$VENV" ]]; then
    echo "==> creating virtualenv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> installing hipep-product[$(IFS=,; echo "${EXTRAS[*]}")]"
python -m pip install --upgrade pip >/dev/null
python -m pip install -e "${PRODUCT_DIR}[$(IFS=,; echo "${EXTRAS[*]}")]"

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

The GPU stack (onnxruntime-genai) is installed separately because the wheel is built
for one ROCm version and one GPU architecture. Until it is present the server replays
published timings and says so.
EOF
