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

[CmdletBinding()]
param(
    # Reuse an interpreter instead of searching for one.
    [string] $Python = "",
    # Rebuild even if the environment is already there.
    [switch] $Force
)

$ErrorActionPreference = "Stop"

$home_dir = if ($env:HIPEP_HOME) { $env:HIPEP_HOME } else { Join-Path $HOME ".hipep" }
$target = Join-Path $home_dir "toolchain"
$python_exe = Join-Path $target "Scripts\python.exe"

function Find-Interpreter {
    # AMD Quark publishes no wheel for 3.14 or later, and the name on PyPI is taken by
    # an unrelated 0.1.0 package that installs cleanly and ships no modules -- so a too
    # new interpreter does not fail here, it fails much later inside a port.
    if ($script:Python) {
        return $script:Python
    }
    foreach ($version in @("3.12", "3.11", "3.13")) {
        $found = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) {
            return $found.Trim()
        }
    }
    return ""
}

if ((Test-Path $python_exe) -and -not $Force) {
    Write-Host "porting toolchain already at $target" -ForegroundColor Green
    Write-Host "re-run with -Force to rebuild it"
    exit 0
}

$interpreter = Find-Interpreter
if (-not $interpreter) {
    Write-Host "No Python 3.11-3.13 found." -ForegroundColor Red
    Write-Host ""
    Write-Host "The exporter needs one; AMD Quark has no release for 3.14 or later."
    Write-Host "Install one from https://www.python.org/downloads/ and run this again,"
    Write-Host "or point this script at an existing one:"
    Write-Host ""
    Write-Host "    .\product\install-port-toolchain.ps1 -Python C:\Path\To\python.exe"
    exit 1
}

$version = (& $interpreter -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
Write-Host "building the porting toolchain with Python $version" -ForegroundColor Cyan
Write-Host "  interpreter  $interpreter"
Write-Host "  target       $target"

if ($Force -and (Test-Path $target)) {
    Remove-Item $target -Recurse -Force
}

& $interpreter -m venv $target
if ($LASTEXITCODE -ne 0) { throw "could not create the environment at $target" }

& $python_exe -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "could not upgrade pip" }

Write-Host "installing the exporter (this downloads torch and takes a few minutes)"
& $python_exe -m pip install `
    "onnxruntime-genai" "onnx" "torch" "transformers" "huggingface_hub" "amd-quark>=0.6"
if ($LASTEXITCODE -ne 0) { throw "could not install the porting toolchain" }

# Prove the environment can do the one thing it exists for, rather than trusting that a
# clean install means a working one.
& $python_exe -c "import onnxruntime_genai.models.builder, quark"
if ($LASTEXITCODE -ne 0) { throw "the toolchain installed but cannot import the builder" }

Write-Host ""
Write-Host "porting toolchain ready" -ForegroundColor Green
Write-Host "  $python_exe"
Write-Host ""
Write-Host "The server finds it by itself. Check with: hipep doctor"
