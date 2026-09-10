<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP EP 动态形状（Dynamic Shape）支持

English version: [dynamic-shape-support.md](dynamic-shape-support.md)

本报告说明 HIP EP 如何支持动态张量形状：动态维度如何表示、哪些在编译期确定、哪些
只能在运行期确定、当尺寸直到推理调用才已知时内存如何管理，以及自回归 LLM 解码中
KV cache 如何管理。

本文是对现有设计文档与当前实现的综合整理。各专题的权威文档仍为
[hip-shape-inference.md](hip-shape-inference.md)、
[pool-allocs-memory-planning.md](pool-allocs-memory-planning.md)、
[output-allocator-design.md](output-allocator-design.md) 与
[compiler-runtime-contract.md](compiler-runtime-contract.md)。

## 1. 结论概要

HIP EP 为每个模型编译**一个形状多态（shape-polymorphic）的产物**，并在所有输入形状
下复用它。不存在按形状重新编译、形状分桶（bucketing），也不会为固定档位做 padding。

整体设计把工作分成三层：

| 关注点 | 解决位置 |
|---|---|
| 秩（rank），以及可被证明为常量的维度 | 编译期，由形状推导 pass 完成 |
| 由输入维度派生的维度 | 运行期，由编译器写入图中的形状算术计算 |
| 依赖张量**数值**的维度 | 运行期，通过同步的 GPU→Host 回读获得 |

内存管理遵循同样的划分：缓冲区的**偏移与分组**在编译期规划；缓冲区的**大小**在每次
推理时计算，并由会话持有的内存池满足；内存池增长到历史最大值（high-water mark）后
即被反复复用，不再申请。

## 2. 动态维度的表示

HIP EP 使用标准 MLIR 的 ranked 类型。动态维度写作 `?`（`ShapedType::kDynamic`），
具体大小以 `index` 类型的 SSA 值单独传递。

| 形式 | 含义 |
|---|---|
| `tensor<2x4x8xf16>` | 所有维度在编译期已知 |
| `tensor<?x4x?xf16>` | 秩已知，第 0、2 维只有运行期才知道 |
| `tensor<*xf16>` | 连秩都未知；多数转换器会拒绝 |
| 元数据中的 `-1` | 编译器/运行时契约中 `?` 的序列化编码 |

系统中**没有具名符号维度**（不存在 `s0`、`s1` 这类符号）。一个维度要么是被折叠进类型
的编译期常量，要么是由 `tensor.dim` / `memref.dim` 产生、再用普通整数运算组合出来的
SSA 值。正因如此，形状计算与数据计算处于同一份 IR 中，可以被同一批 pass 优化。

无秩（unranked）张量是另一个更受限的话题，参见
[unranked-tensor-handling.md](unranked-tensor-handling.md)。

## 3. 编译期与运行期的分工

### 3.1 编译期确定的内容

与形状相关的 pass 顺序（见 `lib/Dialect/Transforms/Pipelines.cpp`）：

```text
hip-infer-loop-body-shapes     为循环携带值确立秩
convert-onnx-to-hip            写入 hipdnn.input_shapes / output_shapes 元数据
hip-infer-shapes               在 reification 能证明常量时把 ? 收窄为静态值
hip-resolve-tensor-dims        沿 reify 与 reshape 链折叠 tensor.dim
one-shot-bufferize             tensor 转为 memref
hip-use-output-allocator       图输出改写为 hip.alloc_output
hip-materialize-host-scalars   把 host 写入的小缓冲移出 GPU 池
hip-resolve-memref-dims        bufferize 之后继续折叠 memref.dim
hip-hoist-alloc-size-arith     把尺寸计算提升到分配点之前
hip-pool-allocs                分配池域、偏移与尺寸表达式
generate-interface             嵌入元数据，生成 inference_init / inference_compute
```

形状细化建立在 MLIR 自身的接口之上，而不是自研求解器：`InferTypeOpInterface` 负责
结果类型，`ReifyRankedShapedTypeOpInterface` 负责逐维度的 `OpFoldResult`，方言自身的
`HipDpsOpInterface` 提供默认 reification——直接取 DPS（destination-passing style）init
操作数的 mixed sizes。

`hip-infer-shapes` 只有在 reification 得到真正的常量时才会把某维从 `?` 变为静态值，
绝不会用运行期的值去"猜"一个静态维度。这正是同一份产物能在不同形状下保持正确的原因。

### 3.2 只有推理调用才知道的内容

运行期由 ONNX Runtime 把真实形状交给 EP，生成的代码据此构造 memref 描述符，其余全部
由图内计算得出：

- **由输入派生的维度**：`%M = memref.dim %input, %c0`，随后是普通算术。
- **分配尺寸**：每个池化缓冲区的字节数被生成为
  `staticFactor × product(动态维度 SSA 值)`。
- **输出形状**：`hip.alloc_output(%ctx, %M, ...)` 把刚算出的形状交给 EP 的分配回调，
  再原样转交给 ORT。

### 3.3 依赖数值的维度必须走同步回读

有些维度依赖张量内容而非张量形状，例如 `Compress` 选中的元素个数、`Range` 的迭代
次数。**在 host 侧读取 GPU 计算出的标量必须使用同步回读操作**：裸的 `tensor.extract`
或对设备内存的 `memref.load` 会被降级成未同步的 host load，读到过期数据。

受支持的路径是 `hip.readback_dim` / `hip.readback_scalar`，它们降级到"在流上拷贝并
同步"的运行时辅助函数：

```172:207:lib/Runtime/real/memory.cpp
int32_t hipdnn_ep_readback_i32(RuntimeState *state, const void *device_scalar) {
  ...
  hipMemcpyAsync(&host_val, device_scalar, sizeof(int32_t), hipMemcpyDefault, stream);
  ...
  hipStreamSynchronize(stream);
  return host_val;
}
```

`Compress` 是标准范例：设备端 scan 得到计数，回读为 host 的 `index`，**之后**才用它
确定分配尺寸——这样在向 ORT 申请输出缓冲的那一刻，上报的形状就是正确的。至于那些只有
在生产它的 kernel 跑完之后才可知的维度（例如作为图输出的 `NonZero`），目前仍是已知
限制。

### 3.4 一份产物，多种形状

MorphiZen 的编译缓存键是对 ONNX 图拓扑及其 protobuf 形状字段做的 MD5，其中**包含**
表示动态维度的 `-1` 标记；它**不**以某一次推理的具体维度为键。因此，在已经声明为动态
的维度上改变 batch size 或序列长度，会复用同一份编译产物，不触发任何重新编译。

## 4. 内存管理

### 4.1 内存池

中间张量不会被单独分配。`hip-pool-allocs` 先做活跃区间（liveness）分析，把分配按支配
关系划分为若干**域（domain）**，再把每个 `memref.alloc` 改写为对会话持有内存池的
`memref.view`，池本身由 `hip.get_pool` 获得。

- **静态分配**通过贪心 best-fit 装箱获得编译期字节偏移；生命周期不相交的缓冲区共享
  同一偏移。
- **动态分配**获得运行期偏移，形式为 `slabBase + unitOffset × F`，其中 `F` 是动态维度
  SSA 值的乘积。
- 之所以需要"域"，是因为池的尺寸表达式必须被它依赖的所有值所支配。当较晚的分配其
  尺寸依赖于定义在较早分配之后的值时，该分配就需要属于自己的池。

### 4.2 在没有编译期尺寸的情况下确定池大小

内存池**不是**按编译期最坏情况来开的。编译器只生成尺寸**表达式**，运行时逐次推理求值，
并且只有当请求超过当前容量时才增长：

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

`hipFree` 之前的流同步是必需的：上一次推理可能仍有在途 kernel 持有指向该池的指针。

由此带来三个值得明确的结论：稳态推理**不产生任何临时内存分配调用**，所有中间张量都
只是对已有内存的视图；会话峰值内存等于该会话见过的最大形状，因为池只增不减；一次
增长事件包含一次同步和一次重新分配，所以某个更大形状的**首次**推理会明显慢于其后续
推理。

### 4.3 图输出刻意不进池

输出缓冲区的生命周期必须超出本次推理，且归 ORT 所有，因此不能来自内存池。这一点不是
靠约定，而是靠结构强制：`hip.alloc_output` 带有一个通用 `Write` 效果，但**刻意省略**
了 `MemoryEffects::Allocate`。

```88:102:lib/Dialect/IR/HipDialect.cpp
  //   (1) NOT MemoryEffects::Allocate. The returned buffer is EP/runtime-owned
  //       (a graph output): hip-pool-allocs must never pool it, and a custom
  //       pipeline that enables ownership-based buffer deallocation must never
  //       free it. Ownership is keyed on the Allocate effect, so we
  //       deliberately omit it (unlike AllocOp/GetPoolOp above).
  //   (2) A generic Write effect (no associated value) marks the side effect of
  //       calling into the EP output allocator ...
```

由于池化只改写 `memref.alloc`，所以 `hip-use-output-allocator` **必须先于**
`hip-pool-allocs` 运行。若顺序颠倒，输出就已经是池视图，会被下一次推理覆盖。

生成的 ABI 是 `inference_compute(state, inputs)`，没有输出参数区：输出在图内、在其形状
刚好可知的位置分配，经由 EP 回调调用 ORT 的 `GetOutput`。

### 4.4 Host 标量、常量与算子状态

有三类内存位于临时池之外：

- **host 写入的微小形状缓冲**。`hip-materialize-host-scalars` 把带有 host store 用户的
  小整数/index 缓冲重定向到 `hip.get_host_scratch`，其后端是
  `hipHostMalloc(hipHostMallocMapped)`——host 可写、GPU 可读。若不这样做，对池视图的
  host store 就是对设备内存的写入，会直接崩溃。
- **常量**。会话初始化时一次性上传到独立的 GPU blob，与工作池分离。
- **算子状态槽（op state slots）**。跨推理持久保存的每算子实例状态——注意力描述符、
  反量化解包缓冲、autotune 选择的算法——在编译期分配槽位下标，在 `inference_init`
  期间构造一次。

### 4.5 核心不变量

**每一个临时分配都必须被池化，或被改写为输出分配。** 这不是风格问题：残留的
`memref.alloc` 会降级为 `hip.alloc`，进而降级到 `hip_device_malloc`，而后者在生产
流水线中没有运行时实现。生产流水线同样刻意不启用基于所有权的缓冲释放，正是因为
所有临时张量都已池化、所有输出都归运行时所有。

## 5. KV cache 管理

### 5.1 表示

自回归 LLM 解码走 `com.microsoft.GroupQueryAttention`，转换为 `hip.gqa`。通用的
multi-head attention 运行时**没有**实现 past/present KV；Whisper 的自注意力被改道到
GQA 路径。

| 张量 | 形状 | 说明 |
|---|---|---|
| `past_key` / `past_value` | `(B, G, P, d)` BNSD | `P` 是缓冲容量，不是有效长度 |
| `present_key` / `present_value` | `(B, G, T, d)` BNSD | `T` 是缓冲步长 |
| `seqlens_k` | `(B,)` int32 | `seqlens_k[b] = total_seq - 1` |

cache 维度是 rank-4 张量的**第 2 维**。这里最关键的区分是**容量**与**有效长度**：
张量维度是已分配的槽位数，而真正写入的 token 数由 `seqlens_k` 带外传递。`seqlens_k`
为负是 ORT 的 prefill 哨兵值，表示 past 为空。

### 5.2 使指针别名成立的输出分配器不变量

每个 `present.*` 输出在图内的尺寸取自其对应 `past` 输入的 `memref.dim`——也就是取自
past 缓冲的**容量**，而不是不断增长的总长度。

这正是 onnxruntime-genai 的 `past_present_share_buffer` 模式得以工作的前提。该模式下
OGA 把同一个 `OrtValue` 同时绑定到 past 输入与 present 输出。由于图申请的恰好是容量
形状，ORT 的 `GetOutput` 会返回那个预先绑定的共享缓冲，从而保持 `past == present` 的
指针同一性。回归测试 `test/lit/e2e/test_gqa_output_allocator_present_dim.mlir` 锁定了
这一行为。

### 5.3 原地追加 vs. 拼接

运行时通过比较指针来选择更新策略：

```285:305:lib/Runtime/real/gqa.cpp
  if (past_key && past_len > 0 && past_key != present_key) {
    // Separate-buffer concat: needs host-side past_len for stride computation.
    if (hip_gqa_kv_cache_concat(...
  } else {
    // In-place append: kernel can read past_len from device via seqlens_k_ptr.
    if (hip_gqa_kv_cache_append(...
  }
```

| 条件 | 机制 | 是否拷贝 past？ |
|---|---|---|
| `past == present`（共享缓冲） | `hip_gqa_kv_cache_append` 把新 token 散射写入 `[past_len, past_len + sq)` | 否 |
| 分离缓冲且 past 非空 | `hip_gqa_kv_cache_concat` 先拷贝 past 再追加新值 | 是 |

append 路径是决定解码吞吐的关键，并且还有第二个好处：kernel 直接从设备上的
`seqlens_k` 读取写入偏移，热路径上无需 device→host 同步。而 concat 路径**需要** host
侧的 `past_len`，因而每步都要一次拷贝加一次流同步。

### 5.4 Prefill 与 Decode

单一入口按 query 序列长度分支：`sq > 1` 为 prefill，`sq == 1` 为 decode。两者先执行
相同的 cache 更新，再分派到各自的 flash attention kernel——二者的算术强度差异极大，
因此需要不同实现。decode 仅按**几何形状**在 WMMA 与标量变体之间选择，从不依据 cache
深度。

### 5.5 归属与限制

KV cache 归 EP 之上的层所有：由 onnxruntime-genai 分配 `OrtValue`、由 ORT 绑定，生成
的图只是通过输出分配器回调拿回的指针进行写入。**KV cache 从不进入内存池。**

已知限制：

- **没有淘汰（eviction），也没有滑动窗口回收**。容量在绑定时固定；运行时会用
  `total_seq` 对缓冲做范围检查并直接报错，而不是自动扩容。
- **每层各自持有 cache**，不存在跨层缓冲共享。
- **通用 MHA 的 past/present 在 HIP 运行时路径上未实现**。
- 上下文增长会使会话内内存单调上升，这与内存池 high-water mark 的行为一致。

## 6. 编译期 / 运行期一览

| 决策项 | 编译期 | 运行期 |
|---|---|---|
| 秩 | 是 | — |
| 可证明为常量的维度 | 是 | — |
| 由输入派生的维度 | 仅表达式 | 具体值 |
| 依赖数值的维度 | 插入回读 | 同步后取值 |
| 静态池偏移 | 是 | — |
| 动态池偏移 | 仅表达式 | 具体值 |
| 池容量 | — | 按需增长的 high-water mark |
| 输出缓冲 | 仅确定分配点 | 经 EP 回调分配 |
| KV cache 容量 | — | 运行前由 OGA 绑定 |
| 形状变化触发重编译 | 从不 | — |

## 7. 已知限制

- 形状 pass 不会把无秩张量提升为 ranked；需要秩的转换器会拒绝该图。
- 循环体的秩推导对 ONNX 覆盖较窄，并非全图求解器。
- 维度只有在生产 kernel 执行后才可知的图输出（输出位置上的 `NonZero`、`Range`）仍被
  推迟支持。
- 内存池只增不减，因此一次超大输入形状会把该会话余下生命周期的内存下限抬高。
- 外联循环体内嵌套区域中的分配不由 `hip-pool-allocs` 池化，会回退到
  `hip-lower-allocs`。

## 8. 代码位置

| 关注点 | 路径 |
|---|---|
| Pass 顺序 | `lib/Dialect/Transforms/Pipelines.cpp` |
| 形状细化 | `lib/Dialect/Transforms/InferShapesPass.cpp` |
| 维度折叠 | `ResolveTensorDims.cpp`、`ResolveMemRefDims.cpp` |
| Host 标量 scratch | `lib/Dialect/Transforms/MaterializeHostScalars.cpp` |
| 输出改写 | `lib/Dialect/Transforms/UseOutputAllocator.cpp` |
| 内存规划 | `lib/Dialect/Transforms/PoolAllocs.cpp` |
| 池与 scratch 运行时 | `lib/Runtime/hipdnn_ep_runtime_state.cpp` |
| 回读运行时 | `lib/Runtime/real/memory.cpp` |
| 输出分配器副作用定义 | `lib/Dialect/IR/HipDialect.cpp` |
| EP 输出回调 | `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` |
| KV cache 更新 | `lib/Runtime/real/gqa.cpp` |
| KV cache kernel | `lib/Runtime/Kernels/hip/gqa_kernel.hip` |
| KV 分配器不变量测试 | `test/lit/e2e/test_gqa_output_allocator_present_dim.mlir` |
