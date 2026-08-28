##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
<#
.SYNOPSIS
    Install oga-server and the hipep CLI on Windows, then report what the machine can do.

.EXAMPLE
    .\product\install.ps1
.EXAMPLE
    .\product\install.ps1 -Dev -Quant
.EXAMPLE
    .\product\install.ps1 -Venv D:\envs\hipep
#>
[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$Quant,
    [string]$Venv
)

$ErrorActionPreference = 'Stop'
$ProductDir = $PSScriptRoot
if (-not $Venv) { $Venv = Join-Path $ProductDir '.venv' }

$extras = @('site')
if ($Dev)   { $extras += 'dev' }
if ($Quant) { $extras += 'quant' }
$spec = "$ProductDir[$($extras -join ',')]"

function Find-Python {
    # The py launcher is the reliable way to reach a specific interpreter on Windows,
    # where "python" may be the Store alias that opens a web page instead of running.
    foreach ($candidate in @(
            @{ exe = 'py';     args = @('-3.12') },
            @{ exe = 'py';     args = @('-3') },
            @{ exe = 'python'; args = @() })) {
        $cmd = Get-Command $candidate.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $probe = @($candidate.args) + @('-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)')
        & $cmd.Source @probe 2>$null
        if ($LASTEXITCODE -eq 0) { return @{ Exe = $cmd.Source; Args = $candidate.args } }
    }
    throw 'Python 3.10 or newer was not found. Install it from python.org and re-run.'
}

$python = Find-Python

if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    Write-Host "==> creating virtualenv at $Venv" -ForegroundColor Cyan
    & $python.Exe @($python.Args) -m venv $Venv
}
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
$VenvScripts = Join-Path $Venv 'Scripts'

Write-Host "==> installing hipep-product[$($extras -join ',')]" -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip | Out-Null
& $VenvPython -m pip install -e $spec
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

Write-Host '==> building the documentation site' -ForegroundColor Cyan
& $VenvPython (Join-Path $ProductDir 'site\build.py')

if ($Dev) {
    Write-Host '==> running the test suite' -ForegroundColor Cyan
    Push-Location $ProductDir
    try { & $VenvPython -m pytest -q } finally { Pop-Location }
}

Write-Host ''
& (Join-Path $VenvScripts 'hipep.exe') doctor

Write-Host @"

Activate this environment with:
  $Venv\Scripts\Activate.ps1

Then:
  hipep serve                     start the server and console on :11434
  hipep models                    the certified model zoo
  hipep pull llama3.1:8b          download a model

The GPU stack (onnxruntime-genai) is installed separately because the wheel is built
for one ROCm version and one GPU architecture. Until it is present the server replays
published timings and says so.
"@
