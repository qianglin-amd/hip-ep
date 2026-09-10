<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Dynamic shape support in HIP EP

中文版本：[dynamic-shape-support.zh-CN.md](dynamic-shape-support.zh-CN.md)

This report explains how HIP EP supports dynamic tensor shapes: how a dynamic
dimension is represented, what is decided at compile time versus run time, how memory
is managed when sizes are not known until the inference call, and how the KV cache is
managed for autoregressive LLM decoding.

It is a synthesis of the existing design documents and the current implementation. The
authoritative per-topic documents remain
[hip-shape-inference.md](hip-shape-inference.md),
[pool-allocs-memory-planning.md](pool-allocs-memory-planning.md),
[output-allocator-design.md](output-allocator-design.md) and
[compiler-runtime-contract.md](compiler-runtime-contract.md).

## 1. Summary

HIP EP compiles **one shape-polymorphic artifact per model** and reuses it for every
input shape. There is no per-shape recompilation, no shape bucketing and no padding to
a fixed profile.

The design splits the work in three:

| Concern | Where it is resolved |
|---|---|
| Rank, and any extent provably constant | Compile time, by shape inference passes |
| Extents derived from input dimensions | Run time, by shape arithmetic emitted into the graph |
| Extents that depend on tensor *values* | Run time, by a synchronized GPU-to-host readback |

Memory follows the same split. Buffer *offsets and grouping* are planned at compile
time; buffer *sizes* are computed per inference and satisfied from session-owned pools
that grow to a high-water mark and are then reused without further allocation.

## 2. Representing a dynamic dimension

HIP EP uses standard MLIR ranked types. A dynamic dimension is `?`
(`ShapedType::kDynamic`); the concrete extent travels separately as an SSA value of type
`index`.

| Form | Meaning |
|---|---|
| `tensor<2x4x8xf16>` | All extents fixed at compile time |
| `tensor<?x4x?xf16>` | Rank fixed, dimensions 0 and 2 known only at run time |
| `tensor<*xf16>` | Rank itself unknown; most converters reject this |
| `-1` in model metadata | Serialized encoding of `?` in the compiler/runtime contract |

There are **no named symbolic dimensions** (no `s0`, `s1` style symbols). An extent is
either a compile-time constant folded into the type, or an SSA value produced by
`tensor.dim` / `memref.dim` and combined with ordinary integer arithmetic. This is why
shape computation and data computation live in the same IR and are optimized by the same
passes.

Unranked tensors are a separate, narrower story; see
[unranked-tensor-handling.md](unranked-tensor-handling.md).

## 3. Compile time versus run time

### 3.1 What the compiler decides

The pass order that matters for shapes, from
`lib/Dialect/Transforms/Pipelines.cpp`:

```text
hip-infer-loop-body-shapes     establish rank on loop-carried values
convert-onnx-to-hip            stamp hipdnn.input_shapes / output_shapes metadata
hip-infer-shapes               narrow ? to static where reification proves a constant
hip-resolve-tensor-dims        fold tensor.dim through reify and reshape chains
one-shot-bufferize             tensors become memrefs
hip-use-output-allocator       graph outputs become hip.alloc_output
hip-materialize-host-scalars   tiny host-written buffers leave the GPU pool
hip-resolve-memref-dims        fold memref.dim after bufferization
hip-hoist-alloc-size-arith     hoist size computation above the allocations
hip-pool-allocs                assign pool domains, offsets and size expressions
generate-interface             embed metadata, emit inference_init / inference_compute
```

Shape refinement is built on MLIR's own interfaces rather than a bespoke solver:
`InferTypeOpInterface` for result types, `ReifyRankedShapedTypeOpInterface` for per
dimension `OpFoldResult`s, and the dialect's `HipDpsOpInterface`, whose default
reification lifts the mixed sizes of the destination-passing-style init operand.

`hip-infer-shapes` only moves a dimension from `?` to static when reification yields an
actual constant. It never invents a static extent from a runtime value, which is what
keeps the artifact valid across shapes.

### 3.2 What only the inference call knows

At run time ONNX Runtime hands the EP the real shapes, the generated code builds memref
descriptors from them, and the graph computes everything downstream:

- **Input-derived extents.** `%M = memref.dim %input, %c0`, then ordinary arithmetic.
- **Allocation sizes.** Each pooled buffer's byte size is emitted as
  `staticFactor × product(dynamic dimension SSA values)`.
- **Output shapes.** `hip.alloc_output(%ctx, %M, ...)` passes the freshly computed shape
  to the EP's allocation callback, which forwards it to ORT verbatim.

### 3.3 Value-dependent extents need a synchronized readback

Some extents depend on tensor contents rather than tensor shapes — the selected count of
`Compress`, the trip count of `Range`. A host read of a GPU-computed scalar must use a
synchronized readback operation. A bare `tensor.extract` or `memref.load` of device
memory lowers to an unsynchronized host load and reads stale bytes.

The supported path is `hip.readback_dim` / `hip.readback_scalar`, which lower to runtime
helpers that copy on the stream and then synchronize:

```172:207:lib/Runtime/real/memory.cpp
int32_t hipdnn_ep_readback_i32(RuntimeState *state, const void *device_scalar) {
  ...
  hipMemcpyAsync(&host_val, device_scalar, sizeof(int32_t), hipMemcpyDefault, stream);
  ...
  hipStreamSynchronize(stream);
  return host_val;
}
```

`Compress` is the canonical example: a device scan produces the count, the count is read
back to a host `index`, and only then does it size the allocation — so the shape reported
to ORT is correct at the moment the output buffer is requested. Extents that are only
knowable *after* the producing kernel has run, such as `NonZero` as a graph output,
remain a documented limitation.

### 3.4 One artifact, many shapes

The MorphiZen compilation cache key is an MD5 over the ONNX graph topology and its
protobuf shape fields — including the `-1` markers for dynamic dimensions. It is **not**
keyed on the concrete dimensions of any particular inference. Changing batch size or
sequence length on a dimension already declared dynamic therefore reuses the same
compiled artifact and triggers no recompilation.

## 4. Memory management

### 4.1 The pool

Transient intermediates are not allocated individually. `hip-pool-allocs` runs a
liveness analysis, partitions allocations into dominance *domains*, and rewrites each
`memref.alloc` into a `memref.view` over a session-owned pool obtained from
`hip.get_pool`.

- **Static allocations** get compile-time byte offsets from greedy best-fit packing;
  buffers with disjoint lifetimes share the same offset.
- **Dynamic allocations** get runtime offsets, computed as `slabBase + unitOffset × F`
  where `F` is the product of the dynamic dimension SSA values.
- Domains exist because a pool's size expression must be dominated by all the values it
  depends on. When a later allocation's size depends on a value defined after an earlier
  allocation, that allocation needs its own pool.

### 4.2 Sizing without compile-time sizes

The pool is **not** sized from a compile-time worst case. The compiler emits the size
*expression*; the runtime evaluates it per inference and grows the pool only when the
request exceeds current capacity:

```1102:1135:lib/Runtime/hipdnn_ep_runtime_state.cpp
  // Grow-on-demand, per domain: when dynamic shapes produce larger
  // intermediates than the current allocation for this domain, reallocate
  // its pool. Pools never shrink, and other domains are untouched ...
  if (needed_size > state->pool_size[domain_id]) {
    if (state->pool_base[domain_id]) {
      if (state->stream)
        HIP_CLEANUP(hipStreamSynchronize(state->stream));
      ...
      HIP_CLEANUP(hipFree(state->pool_base[domain_id]));
    }
    ...
  }
```

The stream synchronization before `hipFree` matters: the previous inference may still
have in-flight kernels holding pointers into that pool.

The consequences are worth stating plainly. Steady-state inference performs **no
transient allocator calls** — every intermediate is a view into memory that already
exists. Peak memory equals the largest shape seen so far in the session, because pools
grow but never shrink. A growth event costs a synchronize plus a reallocation, so the
first inference at a new larger shape is slower than its successors.

### 4.3 Graph outputs are deliberately not pooled

An output buffer must outlive the inference and is owned by ORT, so it must not come
from the pool. This is enforced structurally rather than by convention: `hip.alloc_output`
carries a generic `Write` effect but deliberately **omits** `MemoryEffects::Allocate`.

```88:102:lib/Dialect/IR/HipDialect.cpp
  //   (1) NOT MemoryEffects::Allocate. The returned buffer is EP/runtime-owned
  //       (a graph output): hip-pool-allocs must never pool it, and a custom
  //       pipeline that enables ownership-based buffer deallocation must never
  //       free it. Ownership is keyed on the Allocate effect, so we
  //       deliberately omit it (unlike AllocOp/GetPoolOp above).
  //   (2) A generic Write effect (no associated value) marks the side effect of
  //       calling into the EP output allocator ...
```

Because pooling only ever rewrites `memref.alloc`, `hip-use-output-allocator` **must run
before** `hip-pool-allocs`. If the order were reversed, the output would already be a
pool view and would be overwritten by the next inference.

The generated ABI is `inference_compute(state, inputs)` with no outputs span: outputs are
allocated inside the graph at the point their shape becomes known, through the EP
callback that calls ORT's `GetOutput`.

### 4.4 Host scalars, constants and op state

Three categories of memory sit outside the transient pool:

- **Tiny host-written shape buffers.** `hip-materialize-host-scalars` redirects small
  integer or index buffers with host store users into `hip.get_host_scratch`, backed by
  `hipHostMalloc(hipHostMallocMapped)` — host-writable and GPU-readable. Without this,
  a host store into a pool view is a store into device memory and faults.
- **Constants.** Uploaded once at session init into their own GPU blob, separate from the
  working pool.
- **Op state slots.** Per-operator-instance persistent state across inferences —
  attention descriptors, dequantization unpack buffers, autotuned algorithm choices —
  assigned a compile-time slot index and constructed once during `inference_init`.

### 4.5 The invariant

Every transient allocation must be pooled or rewritten as an output allocation. This is
not stylistic. A surviving `memref.alloc` lowers to `hip.alloc` and then to
`hip_device_malloc`, which has no runtime definition in the production pipeline. The
production pipeline also intentionally omits ownership-based buffer deallocation,
precisely because every transient is pooled and every output is runtime-owned.

## 5. KV cache management

### 5.1 Representation

Autoregressive LLM decoding goes through `com.microsoft.GroupQueryAttention`, converted
to `hip.gqa`. The generic multi-head attention runtime does **not** implement past and
present KV; Whisper's self-attention is rerouted into the GQA path instead.

| Tensor | Shape | Notes |
|---|---|---|
| `past_key` / `past_value` | `(B, G, P, d)` BNSD | `P` is buffer capacity, not valid length |
| `present_key` / `present_value` | `(B, G, T, d)` BNSD | `T` is the buffer stride |
| `seqlens_k` | `(B,)` int32 | `seqlens_k[b] = total_seq - 1` |

The cache dimension is **dimension 2** of a rank-4 tensor. The critical distinction is
between *capacity* and *valid length*: the tensor dimension is the allocated slot count,
while the number of tokens actually populated is carried out-of-band in `seqlens_k`. A
negative `seqlens_k` is ORT's prefill sentinel and means the past is empty.

### 5.2 The output-allocator invariant that makes aliasing work

Each `present.*` output is sized in-graph from `memref.dim` of its matching `past` input —
that is, from the past buffer's **capacity**, not from a growing total length.

This is what allows onnxruntime-genai's `past_present_share_buffer` mode to work. Under
that mode OGA binds the same `OrtValue` to the past input and the present output. Because
the graph asks for exactly the capacity shape, ORT's `GetOutput` returns the pre-bound
shared buffer, and the pointer identity `past == present` is preserved. The regression
test `test/lit/e2e/test_gqa_output_allocator_present_dim.mlir` pins this behavior.

### 5.3 In-place append versus concat

The runtime selects its update strategy by comparing pointers:

```285:305:lib/Runtime/real/gqa.cpp
  if (past_key && past_len > 0 && past_key != present_key) {
    // Separate-buffer concat: needs host-side past_len for stride computation.
    if (hip_gqa_kv_cache_concat(...
  } else {
    // In-place append: kernel can read past_len from device via seqlens_k_ptr.
    if (hip_gqa_kv_cache_append(...
  }
```

| Condition | Mechanism | Copies the past? |
|---|---|---|
| `past == present` (shared buffer) | `hip_gqa_kv_cache_append` scatters the new tokens at `[past_len, past_len + sq)` | No |
| Separate buffers with a non-empty past | `hip_gqa_kv_cache_concat` copies past and appends new | Yes |

The append path is the one that matters for decode throughput, and it has a second
benefit: the kernel reads the write offset from `seqlens_k` on the device, so no
device-to-host synchronization is needed on the hot path. The concat path *does* need
`past_len` on the host, which forces a copy and a stream synchronize per step.

### 5.4 Prefill versus decode

A single entry point branches on the query sequence length: `sq > 1` is prefill, `sq == 1`
is decode. Both perform the same cache update first, then dispatch to different flash
attention kernels tuned for their very different arithmetic intensity. Decode selects
between WMMA and scalar variants by geometry only — never by cache depth.

### 5.5 Ownership and limits

The KV cache is owned above the EP: onnxruntime-genai allocates the `OrtValue`s, ORT
binds them, and the generated graph merely writes through pointers handed back by the
output-allocator callback. The cache is never pooled.

Known limits:

- **No eviction and no sliding-window reclaim.** Capacity is fixed at bind time; the
  runtime range-checks `total_seq` against the buffer and errors rather than growing.
- **Caches are per layer.** There is no cross-layer buffer sharing.
- **Generic MHA with past/present is not implemented** on the HIP runtime path.
- Growing context increases memory monotonically within a session, consistent with the
  pool's high-water-mark behavior.

## 6. Compile time versus run time, at a glance

| Decision | Compile time | Run time |
|---|---|---|
| Rank | Yes | — |
| Provably constant extents | Yes | — |
| Input-derived extents | Expression only | Value |
| Value-dependent extents | Readback inserted | Value, after stream sync |
| Pool offsets, static | Yes | — |
| Pool offsets, dynamic | Expression only | Value |
| Pool capacity | — | Grow-on-demand high-water mark |
| Output buffers | Allocation site chosen | Allocated via EP callback |
| KV cache capacity | — | Bound by OGA before the run |
| Recompilation on shape change | Never | — |

## 7. Known limitations

- Unranked tensors are not promoted to ranked by the shape passes; converters that need
  a rank will refuse the graph.
- Loop-body rank inference has narrow ONNX coverage; it is not a whole-graph solver.
- Graph outputs whose extent is only known after the producing kernel ran (`NonZero`,
  `Range` in output position) are still deferred.
- Pools never shrink, so a single very large input shape raises the session's memory
  floor for its remaining lifetime.
- Nested-region allocations inside outlined loop bodies are not pooled by
  `hip-pool-allocs` and fall back to `hip-lower-allocs`.

## 8. Where the code lives

| Concern | Path |
|---|---|
| Pass ordering | `lib/Dialect/Transforms/Pipelines.cpp` |
| Shape refinement | `lib/Dialect/Transforms/InferShapesPass.cpp` |
| Dimension folding | `ResolveTensorDims.cpp`, `ResolveMemRefDims.cpp` |
| Host scalar scratch | `lib/Dialect/Transforms/MaterializeHostScalars.cpp` |
| Output rewrite | `lib/Dialect/Transforms/UseOutputAllocator.cpp` |
| Memory planning | `lib/Dialect/Transforms/PoolAllocs.cpp` |
| Pool and scratch runtime | `lib/Runtime/hipdnn_ep_runtime_state.cpp` |
| Readback runtime | `lib/Runtime/real/memory.cpp` |
| Output allocator effects | `lib/Dialect/IR/HipDialect.cpp` |
| EP output callback | `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` |
| KV cache update | `lib/Runtime/real/gqa.cpp` |
| KV cache kernels | `lib/Runtime/Kernels/hip/gqa_kernel.hip` |
| KV allocator invariant test | `test/lit/e2e/test_gqa_output_allocator_present_dim.mlir` |
