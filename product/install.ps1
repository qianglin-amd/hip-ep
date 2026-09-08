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
    .\product\install.ps1 -GpuWheels D:\hipep-artifacts\hip-python-package\wheels
.EXAMPLE
    .\product\install.ps1 -Venv D:\envs\hipep
#>
[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$Quant,
    [string]$Venv,
    # Directory of AMD GPU wheels. These are not on PyPI: each is built for one ROCm
    # version, one GPU architecture and one Python version, so they ship as a bundle
    # rather than being resolved. Without them the server runs in replay mode.
    [string]$GpuWheels
)

$ErrorActionPreference = 'Stop'
$ProductDir = $PSScriptRoot
if (-not $Venv) { $Venv = Join-Path $ProductDir '.venv' }

$extras = @('site')
if ($Dev) { $extras += 'dev' }
# -Quant deliberately adds no extra here. The exporter caps Python below 3.14 and pulls
# in torch, so it goes into its own environment further down rather than this one.
$spec = "$ProductDir[$($extras -join ',')]"

function Get-WheelPython {
    # The GPU wheels are built per Python version and the version is in the file name,
    # so when a bundle is given it decides which interpreter the virtualenv needs. Any
    # other choice ends in "not a supported wheel on this platform", which is true and
    # explains nothing.
    param([string]$Dir)
    if (-not $Dir -or -not (Test-Path $Dir)) { return $null }
    foreach ($wheel in Get-ChildItem -Path $Dir -Filter *.whl) {
        if ($wheel.Name -match '-cp3(\d+)-') { return "3.$($Matches[1])" }
    }
    return $null
}

function Find-Python {
    # The py launcher is the reliable way to reach a specific interpreter on Windows,
    # where "python" may be the Store alias that opens a web page instead of running.
    param([string]$Wanted)

    $candidates = @()
    if ($Wanted) { $candidates += @{ exe = 'py'; args = @("-$Wanted") } }
    $candidates += @(
        @{ exe = 'py';     args = @('-3.12') },
        @{ exe = 'py';     args = @('-3') },
        @{ exe = 'python'; args = @() })

    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $probe = @($candidate.args) + @('-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)')
        & $cmd.Source @probe 2>$null
        if ($LASTEXITCODE -eq 0) { return @{ Exe = $cmd.Source; Args = $candidate.args } }
    }
    throw 'Python 3.10 or newer was not found. Install it from python.org and re-run.'
}

$wheelPython = Get-WheelPython -Dir $GpuWheels
if ($wheelPython) {
    Write-Host "==> the GPU wheels are built for Python $wheelPython" -ForegroundColor Cyan
}
$python = Find-Python -Wanted $wheelPython

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

if ($GpuWheels) {
    if (-not (Test-Path $GpuWheels)) { throw "no such directory: $GpuWheels" }
    $wheels = @(Get-ChildItem -Path $GpuWheels -Filter *.whl)
    if (-not $wheels) { throw "no .whl files in $GpuWheels" }

    # A wheel built for another Python fails with "not a supported wheel on this
    # platform", which is true and says nothing about why. The version is in the file
    # name, so the mismatch can be named before pip is asked.
    $venvTag = & $VenvPython -c "import sys; print('cp%d%d' % sys.version_info[:2])"
    $tagged = $wheels | Where-Object { $_.Name -match 'cp(\d+)-' }
    $wrong = $tagged | Where-Object { $_.Name -notmatch "$venvTag-" }
    if ($wrong -and -not ($tagged | Where-Object { $_.Name -match "$venvTag-" })) {
        $theirs = ($wrong[0].Name -replace '.*-(cp\d+)-.*', '$1')
        throw @"
These wheels are built for $theirs and this virtualenv is $venvTag.

  wheels : $GpuWheels
  venv   : $Venv

The GPU wheels are built per Python version, so the virtualenv has to match them.
Create one on the right interpreter and install into that:

  .\product\install.ps1 -Venv <path> -GpuWheels $GpuWheels

pointing PATH at a $theirs interpreter first, or use a wheel bundle built for $venvTag.
"@
    }

    Write-Host "==> installing the GPU stack ($($wheels.Count) wheels)" -ForegroundColor Cyan
    & $VenvPython -m pip install --force-reinstall @($wheels.FullName)
    if ($LASTEXITCODE -ne 0) { throw 'installing the GPU wheels failed' }
}

if ($Quant) {
    # Porting needs Python 3.11-3.13 and a multi-gigabyte torch, neither of which
    # belongs in the server's environment, so it gets its own.
    Write-Host '==> building the porting toolchain' -ForegroundColor Cyan
    & (Join-Path $ProductDir 'install-port-toolchain.ps1')
}

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

The GPU stack ships as a wheel bundle rather than on PyPI, because each wheel is built
for one ROCm version, one GPU architecture and one Python version. Point the installer
at it to run models on the GPU:

  .\product\install.ps1 -GpuWheels <path-to-wheels>

Until it is present the server replays published timings and says so. To port your own
models as well, re-run with -Quant.
"@
