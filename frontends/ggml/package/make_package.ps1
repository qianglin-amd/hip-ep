##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

<#
.SYNOPSIS
Builds a self-contained Windows package that runs GGUF models through llama.cpp
on the HIP EP ggml backend (ggml-hipep), and ONNX models through ONNX Runtime
and ONNX Runtime GenAI on the HIP EP plugin (hipgpu), on a machine without a
ROCm or Visual Studio install.

.EXAMPLE
.\make_package.ps1 -LlamaBin D:\llama.cpp\build-hipep-pkg\bin `
    -HipEpBin D:\build\hip-ep\bin -RocmBin C:\TheRock\build\bin -Zip
#>
param(
    # llama.cpp build (BUILD_SHARED_LIBS=ON) whose ggml-base the backend was built against.
    [Parameter(Mandatory)] [string] $LlamaBin,
    # hip-ep build output: ggml-hipep.dll, hipgpu.dll, onnxruntime.dll,
    # hip-onnx-runner.exe and custom_kernels_<arch>.dll.
    [Parameter(Mandatory)] [string] $HipEpBin,
    # TheRock ROCm SDK bin directory (amdhip64_7.dll, hipblaslt.dll, hipblaslt\library).
    [Parameter(Mandatory)] [string] $RocmBin,
    [string] $Arch = "gfx1151",
    [string] $OutDir = (Join-Path (Get-Location) "dist"),
    [string] $Name = "",
    [switch] $Zip
)

$ErrorActionPreference = "Stop"

function Copy-Required([string] $Path, [string] $Dest) {
    if (-not (Test-Path $Path)) { throw "missing: $Path" }
    Copy-Item $Path $Dest -Recurse -Force
}

if (-not $Name) { $Name = "hipep-win-x64-$Arch-$(Get-Date -Format yyyyMMdd)" }
$root = Join-Path $OutDir $Name
$bin = Join-Path $root "bin"
if (Test-Path $root) { Remove-Item $root -Recurse -Force }
New-Item -ItemType Directory -Force $bin | Out-Null

# llama.cpp executables and every DLL of that build (ggml, llama, mtmd, the
# per-tool *-impl libraries).
$tools = "llama-cli", "llama-server", "llama-completion", "llama-bench",
         "llama-perplexity", "test-backend-ops"
foreach ($t in $tools) { Copy-Required (Join-Path $LlamaBin "$t.exe") $bin }
Get-ChildItem $LlamaBin -Filter *.dll | ForEach-Object { Copy-Item $_.FullName $bin }

# The ggml backend, the ONNX Runtime plugin EP with the ONNX Runtime it was
# built against and its runner, and the kernel library they share. The JIT
# loads custom_kernels_<arch>.dll from the directory of the loading DLL.
foreach ($f in "ggml-hipep.dll", "hipgpu.dll", "onnxruntime.dll",
               "hip-onnx-runner.exe", "custom_kernels_$Arch.dll") {
    Copy-Required (Join-Path $HipEpBin $f) $bin
}

# Python helpers for ONNX Runtime and ONNX Runtime GenAI.
Copy-Required (Join-Path $PSScriptRoot "python") $root

# ROCm runtime: the HIP runtime and its code-object manager, and hipBLASLt with
# the Tensile kernel data for the target architecture, which hipBLASLt finds in
# hipblaslt\library next to its DLL.
foreach ($dll in "amdhip64_7.dll", "hipblaslt.dll") {
    Copy-Required (Join-Path $RocmBin $dll) $bin
}
Get-ChildItem $RocmBin -Filter "amd_comgr*.dll" | ForEach-Object { Copy-Item $_.FullName $bin }
$tensile = Join-Path $RocmBin "hipblaslt\library\$Arch"
if (-not (Test-Path $tensile)) { throw "no hipBLASLt Tensile data for $Arch at $tensile" }
New-Item -ItemType Directory -Force (Join-Path $bin "hipblaslt\library") | Out-Null
Copy-Item $tensile (Join-Path $bin "hipblaslt\library\$Arch") -Recurse

# MSVC runtime, deployed app-locally so the target needs no redistributable.
$vsRoot = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -products * -property installationPath
$redist = Get-ChildItem (Join-Path $vsRoot "VC\Redist\MSVC") -Directory |
    Where-Object { $_.Name -match '^\d' } | Sort-Object Name -Descending | Select-Object -First 1
if (-not $redist) { throw "no VC redistributable directory under $vsRoot" }
foreach ($sub in "Microsoft.VC*.CRT", "Microsoft.VC*.OpenMP") {
    $dir = Get-ChildItem (Join-Path $redist.FullName "x64") -Directory -Filter $sub | Select-Object -First 1
    if (-not $dir) { throw "missing $sub in $($redist.FullName)\x64" }
    Get-ChildItem $dir.FullName -Filter *.dll | ForEach-Object { Copy-Item $_.FullName $bin }
}

# Launchers: point ggml at the backend, put bin\ first on PATH, and offload to
# HIPEP0. Options given on the command line come later and override these.
$envCmd = @"
@echo off
rem Sets up this shell to run the llama.cpp tools in bin\ on the HIP EP backend.
set "HIPEP_ROOT=%~dp0"
set "PATH=%HIPEP_ROOT%bin;%PATH%"
set "GGML_BACKEND_PATH=%HIPEP_ROOT%bin\ggml-hipep.dll"
"@
Set-Content -Path (Join-Path $root "hipep-env.cmd") -Value $envCmd -Encoding Ascii

$defaults = @{
    "llama-cli"        = "-dev HIPEP0 -ngl 99 -fa on"
    "llama-server"     = "-dev HIPEP0 -ngl 99 -fa on"
    "llama-completion" = "-dev HIPEP0 -ngl 99 -fa on"
    "llama-perplexity" = "-dev HIPEP0 -ngl 99 -fa on"
    "llama-bench"      = "-dev HIPEP0 -ngl 99 -fa 1"
}
foreach ($t in $defaults.Keys) {
    $cmd = @"
@echo off
setlocal
call "%~dp0hipep-env.cmd"
"%HIPEP_ROOT%bin\$t.exe" $($defaults[$t]) %*
"@
    Set-Content -Path (Join-Path $root "$t.cmd") -Value $cmd -Encoding Ascii
}

$onnxRun = @"
@echo off
setlocal
call "%~dp0hipep-env.cmd"
"%HIPEP_ROOT%bin\hip-onnx-runner.exe" %*
"@
Set-Content -Path (Join-Path $root "onnx-run.cmd") -Value $onnxRun -Encoding Ascii

$check = @"
@echo off
setlocal
call "%~dp0hipep-env.cmd"
"%HIPEP_ROOT%bin\llama-cli.exe" --list-devices
"@
Set-Content -Path (Join-Path $root "check-device.cmd") -Value $check -Encoding Ascii

Copy-Required (Join-Path $PSScriptRoot "README.md") $root

# Record what went in.
$hipEpRev = (git -C $PSScriptRoot rev-parse --short HEAD) 2>$null
$llamaSrc = (Get-Item $LlamaBin).Parent.Parent.FullName
$llamaRev = (git -C $llamaSrc describe --tags --always) 2>$null
@"
hip-ep commit:    $hipEpRev
llama.cpp:        $llamaRev
GPU architecture: $Arch
Built:            $(Get-Date -Format s)
"@ | Set-Content -Path (Join-Path $root "VERSION.txt") -Encoding Ascii

$size = (Get-ChildItem $root -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("package: {0} ({1:N0} MB)" -f $root, ($size / 1MB))

if ($Zip) {
    $zipPath = "$root.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path $root -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host ("zip:     {0} ({1:N0} MB)" -f $zipPath, ((Get-Item $zipPath).Length / 1MB))
}
