<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# llama.cpp on HIP EP: Windows package

This package runs GGUF models with the stock llama.cpp tools (`llama-cli`,
`llama-server`, ...) on an AMD GPU through the HIP EP ggml backend
(`ggml-hipep.dll`). The backend compiles each ggml graph with the HIP EP
MLIR compiler and runs it with hipBLASLt and the HIP EP custom kernels.

Everything it needs is inside: the llama.cpp tools, the backend, the ROCm
runtime libraries and the MSVC runtime. No ROCm SDK, Python or Visual Studio
install is required.

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11, x64 |
| GPU | The architecture in `VERSION.txt` (`gfx1151`: Radeon 8060S / 8050S, Ryzen AI Max) |
| Driver | A recent AMD Software: Adrenalin Edition driver |
| CPU | x86-64 with AVX2 |
| Disk | About 0.5 GB for the package, plus the model and a compile cache |

The GPU kernels are compiled for one architecture. On a different GPU the
backend fails to load its kernels; rebuild the package for that architecture
(see [Rebuilding](#rebuilding-the-package)).

## Quick start

1. Unzip the package anywhere, for example `C:\hipep`. Paths with spaces work.
2. Open a Command Prompt or PowerShell in that folder and check that the GPU
   is found:

   ```bat
   check-device.cmd
   ```

   Expected output:

   ```text
   Available devices:
     HIPEP0: AMD Radeon(TM) 8060S Graphics (HIP EP) (... MiB, ... MiB free)
   ```

3. Get a GGUF model, for example from Hugging Face
   ([bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF),
   file `Llama-3.2-1B-Instruct-Q4_K_M.gguf`).
4. Chat with it:

   ```bat
   llama-cli.cmd -m C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf
   ```

The first run of a model compiles its graphs. That takes from a few seconds
to about a minute; the results are cached (see [Compile cache](#compile-cache)), so
later runs start quickly.

## The launchers

Each `*.cmd` launcher sets up the environment and runs the matching tool in
`bin\` with `-dev HIPEP0 -ngl 99` (all layers on the GPU) and flash attention
on. Any option you pass is appended and overrides these defaults, so all
normal llama.cpp options work.

| Launcher | Use |
|---|---|
| `llama-cli.cmd` | Interactive chat in the console |
| `llama-server.cmd` | OpenAI-compatible HTTP server |
| `llama-completion.cmd` | One-shot text completion |
| `llama-bench.cmd` | Prompt-processing and generation speed |
| `llama-perplexity.cmd` | Perplexity on a text file |
| `check-device.cmd` | Lists the devices llama.cpp can use |

Examples:

```bat
rem One question, then exit
llama-cli.cmd -m model.gguf -st -p "Why is the sky blue?" -n 128

rem Non-interactive completion, greedy
llama-completion.cmd -m model.gguf -no-cnv -p "The capital of France is" -n 32 --temp 0

rem Benchmark
llama-bench.cmd -m model.gguf -p 512 -n 128

rem Perplexity
llama-perplexity.cmd -m model.gguf -f wiki.test.raw -c 512
```

### llama-server

```bat
llama-server.cmd -m model.gguf -c 8192 --host 127.0.0.1 --port 8080
```

Then call the OpenAI-compatible API, for example from PowerShell:

```powershell
$body = @{ messages = @(@{ role = "user"; content = "Hello!" }); max_tokens = 128 } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -ContentType "application/json" -Body $body | ForEach-Object { $_.choices[0].message.content }
```

or with curl:

```bat
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":128}"
```

This build has no browser UI; use the HTTP API or any OpenAI-compatible
client. Use `--host 0.0.0.0` to accept connections from other machines, and
`--api-key <key>` to require a key.

### Running the tools directly

To call the executables yourself (for example from your own scripts), set up
the shell once with `hipep-env.cmd`, or set the two variables it sets:

```bat
call C:\hipep\hipep-env.cmd
llama-cli.exe -m model.gguf -dev HIPEP0 -ngl 99 -fa on
```

```powershell
$env:PATH = "C:\hipep\bin;$env:PATH"
$env:GGML_BACKEND_PATH = "C:\hipep\bin\ggml-hipep.dll"
llama-cli.exe -m model.gguf -dev HIPEP0 -ngl 99 -fa on
```

`-dev HIPEP0` is required with these tools: this llama.cpp build only loads
the backend named by `GGML_BACKEND_PATH` when a device is selected on the
command line. Without it, everything runs on the CPU.

## Supported models and options

- Llama-family dense models (Llama 3.x, and models with the same graph).
  Other architectures run as long as their ops are supported; unsupported
  ops fall back to the CPU automatically.
- All GGUF weight types, including mixed-bit files: F16, BF16, F32, Q8_0,
  Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q4_0, Q4_1, Q3_K, Q2_K, IQ4_XS,
  IQ4_NL, IQ3, IQ2 and MXFP4. 4-bit and 8-bit types run natively; the other
  types are converted to 8-bit when the model loads.
- Flash attention on, off or auto (`-fa on|off|auto`).
- f16 KV cache (the default). A quantized KV cache (`-ctk q8_0` / `-ctv q8_0`)
  runs attention on the CPU and is much slower.

## Compile cache

Compiled graphs and machine code are cached in
`%LOCALAPPDATA%\hip-ep\ggml-cache`. The cache is rebuilt automatically when
the package changes. To clear it, delete that folder.

| Variable | Effect |
|---|---|
| `GGML_HIPEP_CACHE_DIR=<dir>` | Use another cache directory |
| `GGML_HIPEP_NO_DISK_CACHE=1` | Do not reuse cached results |
| `GGML_HIPEP_VERBOSE=1` | Log compiles, module loads, graph segments and ops left to the CPU |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check-device.cmd` shows no `HIPEP0` | Update the AMD driver. Run from the package folder or through the launchers, so `bin\` is first on `PATH`. |
| `no custom kernel library loadable for arch=...` | The GPU architecture differs from the one in `VERSION.txt`; rebuild the package for it. |
| Model runs but slowly, GPU idle | `-dev HIPEP0` is missing (direct use of the `.exe`), or `-ngl` is too low. |
| Long pause on the first prompt | First-time compilation; later runs use the cache. |
| Some ops still on the CPU | Expected for ops the backend does not support yet; run with `GGML_HIPEP_VERBOSE=1` to list them. |

## Package layout

```text
<package>\
  README.md, VERSION.txt
  check-device.cmd, hipep-env.cmd, llama-*.cmd     launchers
  bin\
    llama-*.exe, test-backend-ops.exe, llama*.dll, ggml*.dll, mtmd.dll   llama.cpp
    ggml-hipep.dll                                 HIP EP ggml backend (compiler + JIT)
    custom_kernels_<arch>.dll                      HIP EP GPU kernels
    amdhip64_7.dll, amd_comgr*.dll, hipblaslt.dll  ROCm runtime
    hipblaslt\library\<arch>\                      hipBLASLt kernel data
    msvcp140*.dll, vcruntime140*.dll, vcomp140.dll MSVC runtime
```

## Rebuilding the package

On the build machine, after building hip-ep with `HIP_EP_BUILD_GGML_BACKEND=ON`
and a shared llama.cpp from the same llama.cpp source (see
`docs/design/ggml-frontend.md`):

```powershell
frontends\ggml\package\make_package.ps1 `
  -LlamaBin <llama.cpp>\build\bin `
  -HipEpBin <hip-ep build>\bin `
  -RocmBin  <TheRock dist>\bin `
  -Arch gfx1151 -Zip
```

Build llama.cpp with `-DGGML_NATIVE=OFF -DGGML_AVX2=ON` so the CPU backend
runs on other machines, and build hip-ep with `-DHIP_ARCHITECTURES=` the
target GPU architecture.
