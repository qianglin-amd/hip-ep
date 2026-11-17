<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GGML frontend: GGUF models on HIP EP through llama.cpp

This document describes how HIP EP runs GGUF models inside the llama.cpp
runtime. llama.cpp keeps its API, GGUF loader, tokenizer, sampling, KV cache
and scheduler. HIP EP plugs in underneath as a ggml backend and executes the
ggml graphs through the same compile-and-run path that the ONNX Runtime
interface uses.

```text
llama.cpp apps (llama-cli, llama-server, apps on libllama)
        |
llama.cpp runtime (GGUF, sampling, KV cache, ggml_backend_sched)
        |  ggml backend API: buffers, supports_op, graph_compute
        v
ggml-hipep  (frontends/ggml)                 ONNX Runtime plugin EP
  GGML importer: ggml_cgraph -> ONNX-dialect MLIR   ONNX subgraph -> MLIR
        \___________________________  ____________________/
                                    \/
          shared compile & run: CompilerDriver -> LLVM bitcode
          -> LlvmIrJit (in-memory) -> inference_compute(state, inputs)
                                    |
              ROCm HIP, hipBLASLt, custom kernels -> AMD GPU
```

## Components

| Piece | Path | Role |
|---|---|---|
| Backend | `frontends/ggml/src/ggml-hipep.cpp` | ggml backend registration, device, device-memory buffer type, `supports_op`, `graph_compute` |
| Importer | `frontends/ggml/src/GgmlImporter.{h,cpp}` | Converts a run of ggml nodes into an ONNX-dialect `main_graph` plus the tensor bindings |
| Engine | `frontends/ggml/src/HipEpEngine.{h,cpp}` | Compiles MLIR text with `CompilerDriver`, caches the bitcode in memory and on disk, JIT-loads it with `LlvmIrJit`, runs it with device pointers |
| Side-effect kernels | `lib/Runtime/Kernels/hip/ggml_kernels.hip` | `hip_ggml_set_rows` and `hip_ggml_cpy` for in-place writes into persistent tensors |

The backend is an out-of-tree ggml backend DLL. A stock llama.cpp build loads
it through the `GGML_BACKEND_PATH` environment variable, which
`ggml_backend_load_all()` honours, and its device shows up as `HIPEP0`.

## Execution model

### Buffers

The buffer type allocates device memory with `hipMalloc`. llama.cpp places the
GGUF weights, the KV cache and the per-graph compute buffer in it. Weights stay
in llama.cpp's buffers: the compiled code reads them through input pointers,
so nothing is copied or baked into the artifact.

### Graph segmentation

A ggml graph is imperative. Most nodes are functional, but some write in place
into persistent tensors (`GGML_OP_SET_ROWS` into the KV cache, `GGML_OP_CPY`
into a view of a cache). The compiled `inference_compute` ABI is functional:
inputs are read-only and outputs are allocated through the output-allocator
callback.

`graph_compute` therefore splits the node list into segments:

- A **functional segment** is a maximal run of functional and view nodes. It
  becomes one `main_graph`.
- A **side-effect node** is executed natively between segments with a
  `hip_ggml_*` kernel.

Nodes without `GGML_TENSOR_FLAG_COMPUTE` are skipped. `ggml_build_forward_select`
keeps the unselected input paths (llama.cpp's token, embedding and mixed
inputs) in the graph without that flag, and ggml-alloc may give their memory
to live tensors.

Segment inputs are every tensor a segment reads that it does not produce
(weights, KV cache, graph inputs, results of earlier segments). A produced
storage is a segment output when any of these holds:

- it is flagged `GGML_TENSOR_FLAG_OUTPUT`;
- a node outside the segment reads it;
- nothing in the graph reads it (a sink);
- its `use_counts` entry exceeds the references inside the graph. A scheduler
  split is a `ggml_graph_view` of the whole graph and keeps those counts, so
  this catches tensors read by another split, for example when one op of a
  layer falls back to the CPU backend.

The output-allocator callback returns the address ggml-alloc already assigned
to the tensor. When that address overlaps an input of the same segment, the
callback returns a scratch buffer and the result is copied back afterwards.

Because weights are inputs, the segments of all transformer layers are
structurally identical. They produce identical MLIR text and share one
compiled module. A Llama graph compiles four distinct modules: the first
layer's prologue, the V projection that llama.cpp orders after the K-cache
write, the repeated layer body, and the last layer with the LM head.

### Stream ordering

A compiled segment runs on its runtime state's stream and returns
synchronized. Scratch copy-backs and native kernels are queued on the backend's
own stream, in graph order, and that stream is drained before the next segment
runs. The backend stream is a blocking stream, so it also orders against
null-stream copies such as `cpy_tensor`. Device-to-device `hipMemcpy` does not
block the host, so copies must not be left on the null stream while native
kernels run on another stream.

### Views

ggml views carry byte strides and an offset into a root storage tensor
(`view_src`). The importer keeps each storage as a logically contiguous value
and lowers every view of it to `Reshape`, `Slice`, `Pad` and `Transpose`:

1. Flatten the storage to 1-D and slice from the view offset.
2. Sort the view dimensions by stride, reshape into the nested stride blocks,
   and slice each block to the view extent.
3. Transpose back to ggml dimension order.

Views whose strides do not nest (overlapping or broadcast views) are rejected
by `supports_op`.

### Op mapping

ggml dimensions `ne[0..3]` map to the ONNX shape `[ne3, ne2, ne1, ne0]`.

| ggml op | ONNX-dialect lowering |
|---|---|
| `MUL_MAT`, 2-D f16/bf16/f32 weight | `Cast` activations to the weight type, `Gemm` with `transB = 1`, `Cast` to f32 |
| `MUL_MAT`, 2-D `Q4_0` / `Q8_0` weight | `com.microsoft.MatMulNBits` on the repacked weight with f16 activations |
| `MUL_MAT`, batched (attention) | Grouped-query broadcast folded into M, then f32 `MatMul` with a transposed `B` |
| `GET_ROWS` | `Gather` on axis 0 (or `GatherBlockQuantized` for `Q4_0` / `Q8_0`), then `Cast` to f32 |
| `RMS_NORM` (+ `MUL` by weight) | `SimplifiedLayerNormalization` |
| `ROPE` (normal, NeoX) | `com.microsoft.RotaryEmbedding` on `[1, T, H*D]`; cos/sin tables built on the host from the ggml YaRN parameters and frequency factors |
| `SOFT_MAX` (mask, scale) | `Mul`, `Add` of the sliced mask, `Softmax` |
| `GLU` (SwiGLU), `UNARY` (SiLU) | `Sigmoid`, `Mul` |
| `ADD`, `SUB`, `MUL`, `DIV`, `SCALE` | Element-wise ops with NumPy broadcasting |
| `CPY`, `DUP`, `CONT` | `Cast` and `Reshape` |
| `VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE` | Strided-view lowering above |
| `SET_ROWS`, in-place `CPY` | Native `hip_ggml_set_rows` / `hip_ggml_cpy` between segments |

### Compile cache

The engine keys compiled modules by the hash of the MLIR text and the backend
build. A miss compiles with `CompilerDriver` into
`%LOCALAPPDATA%\hip-ep\ggml-cache\<hash>` (or `$XDG_CACHE_HOME`), which also
holds the externalized constants. The bitcode is then JIT-loaded with
`LlvmIrJit`. A plan cache keyed by the graph's structure skips importing when
llama.cpp submits a graph with the same topology and shapes. Plans and RoPE
tables refer to tensors by address, so freeing any backend buffer drops them.

## Building

The backend must be built against the ggml headers and the `ggml-base`
library of the llama.cpp build that loads it.

```powershell
cmake -B <build> -S . -DHIP_EP_BUILD_GGML_BACKEND=ON `
  -DHIP_EP_GGML_SOURCE_DIR=<llama.cpp> `
  -DHIP_EP_GGML_BASE_LIBRARY=<llama.cpp-build>/ggml/src/ggml-base.lib
cmake --build <build> --target ggml-hipep
```

`ggml-hipep.dll` is written next to `custom_kernels_<arch>.dll`, which it and
the JIT load at run time. The HIP runtime DLLs (`amdhip64`, `hipblaslt`) must
be on `PATH`, because llama.cpp loads the backend with a plain
`LoadLibraryW`.

## Usage

```powershell
$env:GGML_BACKEND_PATH = "<build>\bin\ggml-hipep.dll"
llama-cli -m Llama-3.2-1B-Instruct-f16.gguf --device HIPEP0 -ngl 99 -fa off
```

| Variable | Effect |
|---|---|
| `GGML_HIPEP_VERBOSE=1` | Log compiles, segments and unsupported ops; check for device faults after each module load |
| `GGML_HIPEP_DUMP=1` | Print each segment's MLIR |
| `GGML_HIPEP_DUMP_GRAPH=1` | Print every node of each new graph with its kind and flags |
| `GGML_HIPEP_CACHE_DIR` | Override the compile-cache directory |
| `GGML_HIPEP_NO_DISK_CACHE=1` | Recompile instead of reusing cached bitcode |
| `GGML_HIPEP_REJECT=<substring>` | Leave nodes whose name contains the substring to the CPU backend (bisection aid) |
| `GGML_HIPEP_TRACE_MEMORY=1` | Log segment input and output ranges and report inputs that change between reads |
| `GGML_HIPEP_DUMP_INPUT=<names>` | With tracing, write the named inputs to `<name>.bin` as segments read them |
| `GGML_HIPEP_FORCE_SCRATCH=1` | Route every segment output through the scratch buffer |

## Testing

| Tool | What it checks |
|---|---|
| `ggml-import-test <dir> [T] [n_kv] [layers] [f16\|q8_0\|q4_0]` | GPU-free: imports Llama-style graphs and writes each segment's MLIR; feed them to `hip-mlir-opt --hipdnn-pipeline` |
| `tests/ggml-graph-test.cpp` | Runs Llama-style graphs (random weights, KV cache, ggml-alloc memory reuse) on the CPU backend and on `HIPEP0` and compares logits and checkpoints |
| llama.cpp `test-backend-ops -b HIPEP0 -o <op>` | Per-op comparison against the CPU backend |
| `tests/compare_llama_gguf.py` | Greedy decoding of a GGUF model on CPU and `HIPEP0` |

On Llama 3.2 1B Instruct (4 chunks of 256 tokens, `llama-perplexity`):

| Weights | CPU PPL | HIPEP0 PPL | Greedy decode |
|---|---|---|---|
| F16 | 57.488 | 57.472 | Identical |
| Q8_0 | 57.517 | 57.655 | Identical |
| Q4_0 (with Q4_1 / Q6_K tensors on CPU) | 62.327 | 62.122 | Diverges after the first sentence |

The CPU backend quantizes activations to Q8 for `Q4_0` / `Q8_0` dot products,
while `MatMulNBits` uses f16 activations, so quantized models differ slightly.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | F16/F32/BF16 weights, Llama-family dense graphs without flash attention, static shapes with compile caching | Done |
| 2 | `Q8_0` and `Q4_0` weights repacked in place into the `MatMulNBits` layout (same byte size) when uploaded; quantized `GET_ROWS` through `GatherBlockQuantized` | Done |
| 3 | K-quants (`Q4_K`, `Q6_K`) and `Q4_1` through `get_alloc_size` repacking; `FLASH_ATTN_EXT` on the GQA flash kernels; symbolic `n_tokens` / `n_kv` dimensions so decode does not recompile as the cache grows | Planned |
| 4 | Performance: one shared JIT runtime instead of one per module, fewer segments by sinking KV writes, a stream shared with llama.cpp, KV writes inside the compiled graph | Planned |

## Limits

- Shapes are static per compiled module. llama.cpp pads the KV length to 256,
  so decode recompiles only when the padded length grows. Prefill compiles once
  per prompt length.
- Every compiled module JIT-loads its own copy of the runtime bitcode, which
  dominates first-use latency.
- Unsupported ops report `false` from `supports_op` and run on the CPU backend
  through `ggml_backend_sched`. Unsupported quantized weights stay in their
  ggml layout and are copied to the CPU for those ops.
- Weight matmuls cast activations to the weight type and round the result to
  it before casting back to f32.
