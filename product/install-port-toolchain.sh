#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Build the environment that ports a model.
#
# Serving certified models needs nothing from this script. It is only for turning a
# Hugging Face checkpoint into a HIP EP artifact, which needs the ONNX model builder and
# AMD Quark -- and those cap Python below 3.14 and pull in torch, so they get their own
# environment rather than being forced into the server's.
#
# The environment is built at a location the server looks in by convention, so nothing
# has to be configured afterwards.

set -euo pipefail

python_override=""
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_override="$2"; shift 2 ;;
        --force) force=1; shift ;;
        -h|--help)
            echo "usage: $0 [--python /path/to/python3.12] [--force]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

target="${HIPEP_HOME:-$HOME/.hipep}/toolchain"
python_exe="$target/bin/python3"

find_interpreter() {
    # AMD Quark publishes no wheel for 3.14 or later, and the name on PyPI is taken by
    # an unrelated 0.1.0 package that installs cleanly and ships no modules -- so a too
    # new interpreter does not fail here, it fails much later inside a port.
    if [[ -n "$python_override" ]]; then
        echo "$python_override"
        return
    fi
    for candidate in python3.12 python3.11 python3.13; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    echo ""
}

if [[ -x "$python_exe" && $force -eq 0 ]]; then
    echo "porting toolchain already at $target"
    echo "re-run with --force to rebuild it"
    exit 0
fi

interpreter="$(find_interpreter)"
if [[ -z "$interpreter" ]]; then
    cat >&2 <<'EOF'
No Python 3.11-3.13 found.

The exporter needs one; AMD Quark has no release for 3.14 or later.
Install one and run this again, or point this script at an existing one:

    ./product/install-port-toolchain.sh --python /usr/bin/python3.12
EOF
    exit 1
fi

version="$("$interpreter" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "building the porting toolchain with Python $version"
echo "  interpreter  $interpreter"
echo "  target       $target"

if [[ $force -eq 1 ]]; then
    rm -rf "$target"
fi

"$interpreter" -m venv "$target"
"$python_exe" -m pip install --upgrade pip --quiet

echo "installing the exporter (this downloads torch and takes a few minutes)"
"$python_exe" -m pip install \
    onnxruntime-genai onnx torch transformers huggingface_hub "amd-quark>=0.6"

# Prove the environment can do the one thing it exists for, rather than trusting that a
# clean install means a working one.
"$python_exe" -c "import onnxruntime_genai.models.builder, quark"

echo
echo "porting toolchain ready"
echo "  $python_exe"
echo
echo "The server finds it by itself. Check with: hipep doctor"
