/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ggml backend for HIP EP. llama.cpp loads it as an out-of-tree backend
// (GGML_BACKEND_PATH) and hands it ggml graphs, which run through the HIP EP
// compile & run path. See docs/design/ggml-frontend.md.

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"

#include "GgmlImporter.h"
#include "HipEpEngine.h"
#include "QuantRepack.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#define HIPEP_EXPORT __declspec(dllexport)
#else
#include <dlfcn.h>
#define HIPEP_EXPORT __attribute__((visibility("default")))
#endif

namespace fs = std::filesystem;
using namespace hipep_ggml;

namespace {

bool envFlag(const char *name) {
  const char *v = std::getenv(name);
  return v && *v && std::string(v) != "0";
}

bool verbose() {
  static const bool v = envFlag("GGML_HIPEP_VERBOSE");
  return v;
}

void logError(const char *fmt, ...) {
  std::fprintf(stderr, "[ggml-hipep] ");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

#define HIPEP_CHECK(expr)                                                      \
  do {                                                                         \
    hipError_t err_ = (expr);                                                  \
    if (err_ != hipSuccess)                                                    \
      logError("%s failed: %s", #expr, hipGetErrorString(err_));               \
  } while (0)

//===----------------------------------------------------------------------===//
// Side-effect kernels, resolved from the custom_kernels library that the JIT
// also loads. It sits next to this module.
//===----------------------------------------------------------------------===//

struct Kernels {
  decltype(&hip_ggml_set_rows) setRows = nullptr;
  decltype(&hip_ggml_cpy) cpy = nullptr;
};

fs::path thisModuleDirectory() {
#ifdef _WIN32
  HMODULE module = nullptr;
  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCWSTR>(&thisModuleDirectory), &module);
  wchar_t path[MAX_PATH];
  GetModuleFileNameW(module, path, MAX_PATH);
  return fs::path(path).parent_path();
#else
  Dl_info info;
  dladdr(reinterpret_cast<void *>(&thisModuleDirectory), &info);
  return fs::path(info.dli_fname).parent_path();
#endif
}

const Kernels &kernels(const std::string &arch) {
  static Kernels k;
  static std::once_flag once;
  std::call_once(once, [&] {
    std::vector<fs::path> candidates;
    std::error_code ec;
    for (const auto &entry :
         fs::directory_iterator(thisModuleDirectory(), ec)) {
      const std::string name = entry.path().filename().string();
      if (name.find("custom_kernels_") == std::string::npos)
        continue;
#ifdef _WIN32
      if (entry.path().extension() != ".dll")
        continue;
#else
      if (entry.path().extension() != ".so")
        continue;
#endif
      candidates.push_back(entry.path());
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&](const fs::path &a, const fs::path &b) {
          return (a.filename().string().find(arch) != std::string::npos) >
                 (b.filename().string().find(arch) != std::string::npos);
        });
    for (const auto &path : candidates) {
#ifdef _WIN32
      HMODULE lib = LoadLibraryExW(path.wstring().c_str(), nullptr,
                                   LOAD_WITH_ALTERED_SEARCH_PATH);
      if (!lib)
        continue;
      k.setRows = reinterpret_cast<decltype(k.setRows)>(
          GetProcAddress(lib, "hip_ggml_set_rows"));
      k.cpy = reinterpret_cast<decltype(k.cpy)>(
          GetProcAddress(lib, "hip_ggml_cpy"));
#else
      void *lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!lib)
        continue;
      k.setRows = reinterpret_cast<decltype(k.setRows)>(
          dlsym(lib, "hip_ggml_set_rows"));
      k.cpy = reinterpret_cast<decltype(k.cpy)>(dlsym(lib, "hip_ggml_cpy"));
#endif
      if (k.setRows && k.cpy)
        return;
    }
    logError("custom_kernels library with hip_ggml_* not found next to %s",
             thisModuleDirectory().string().c_str());
  });
  return k;
}

int hipDtype(ggml_type type) {
  switch (type) {
  case GGML_TYPE_F32:
    return HIP_DTYPE_FLOAT32;
  case GGML_TYPE_F16:
    return HIP_DTYPE_FLOAT16;
  case GGML_TYPE_BF16:
    return HIP_DTYPE_BFLOAT16;
  case GGML_TYPE_I32:
    return HIP_DTYPE_INT32;
  case GGML_TYPE_I64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

struct Layout {
  int64_t ne[4];
  int64_t nb[4];
  explicit Layout(const ggml_tensor *t) {
    for (int k = 0; k < 4; ++k) {
      ne[k] = t->ne[k];
      nb[k] = static_cast<int64_t>(t->nb[k]);
    }
  }
};

//===----------------------------------------------------------------------===//
// Device, buffer type and buffer
//===----------------------------------------------------------------------===//

struct DeviceContext {
  int index = 0;
  std::string name;
  std::string description;
  std::string arch;
  ggml_backend_buffer_type buft{};
  ggml_backend_device device{};
};

struct BufferContext {
  int device = 0;
  void *base = nullptr;
};

bool isHipepBuft(ggml_backend_buffer_type_t buft);

// Plans and RoPE tables refer to tensors by address; freeing a buffer lets
// new tensors reuse those addresses, so it invalidates both.
std::atomic<uint64_t> gBufferGeneration{0};

void bufferFree(ggml_backend_buffer_t buffer) {
  auto *ctx = static_cast<BufferContext *>(buffer->context);
  hipSetDevice(ctx->device);
  HIPEP_CHECK(hipFree(ctx->base));
  delete ctx;
  ++gBufferGeneration;
}

void *bufferBase(ggml_backend_buffer_t buffer) {
  return static_cast<BufferContext *>(buffer->context)->base;
}

// Block-quantized tensors are stored in the MatMulNBits layout (QuantRepack).
size_t repackedSize(const ggml_tensor *t) {
  if (t->view_src)
    return 0;
  return packedSize(t->type, ggml_nrows(t), t->ne[0]);
}

bool isRepacked(const ggml_tensor *t) { return repackedSize(t) != 0; }

// The repacked tensor holding t's bytes: t itself or the root of a view of it.
// Byte offsets into a view are ggml-layout offsets into that root.
const ggml_tensor *repackedRoot(const ggml_tensor *t) {
  const ggml_tensor *root = t->view_src ? t->view_src : t;
  return isRepacked(root) ? root : nullptr;
}

size_t rootOffset(const ggml_tensor *t, const ggml_tensor *root) {
  return static_cast<const char *>(t->data) -
         static_cast<const char *>(root->data);
}

// Whole repacked tensor, converted back to ggml blocks.
void bufferGetTensorRaw(const ggml_tensor *tensor, uint8_t *raw, int device) {
  hipSetDevice(device);
  std::vector<uint8_t> packed(repackedSize(tensor));
  HIPEP_CHECK(hipMemcpy(packed.data(), tensor->data, packed.size(),
                        hipMemcpyDeviceToHost));
  unpackWeights(tensor->type, packed.data(), raw, ggml_nrows(tensor),
                tensor->ne[0]);
}

// Applies `edit` to the ggml-layout bytes of a repacked root and repacks it.
// The old contents are only fetched when the edit does not cover the root.
template <typename Edit>
void editRepacked(const ggml_tensor *root, size_t offset, size_t size,
                  int device, Edit edit) {
  const size_t total = ggml_nbytes(root);
  std::vector<uint8_t> raw(total), packed(repackedSize(root));
  if (offset != 0 || size != total)
    bufferGetTensorRaw(root, raw.data(), device);
  edit(raw.data() + offset);
  packWeights(root->type, raw.data(), packed.data(), ggml_nrows(root),
              root->ne[0]);
  HIPEP_CHECK(hipMemcpy(root->data, packed.data(), packed.size(),
                        hipMemcpyHostToDevice));
}

void bufferMemsetTensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor,
                        uint8_t value, size_t offset, size_t size) {
  auto *ctx = static_cast<BufferContext *>(buffer->context);
  hipSetDevice(ctx->device);
  if (const ggml_tensor *root = repackedRoot(tensor)) {
    editRepacked(root, rootOffset(tensor, root) + offset, size, ctx->device,
                 [&](uint8_t *p) { std::memset(p, value, size); });
    return;
  }
  HIPEP_CHECK(
      hipMemset(static_cast<char *>(tensor->data) + offset, value, size));
}

void bufferSetTensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor,
                     const void *data, size_t offset, size_t size) {
  auto *ctx = static_cast<BufferContext *>(buffer->context);
  hipSetDevice(ctx->device);
  if (const ggml_tensor *root = repackedRoot(tensor)) {
    editRepacked(root, rootOffset(tensor, root) + offset, size, ctx->device,
                 [&](uint8_t *p) { std::memcpy(p, data, size); });
    return;
  }
  HIPEP_CHECK(hipMemcpy(static_cast<char *>(tensor->data) + offset, data, size,
                        hipMemcpyHostToDevice));
}

void bufferGetTensor(ggml_backend_buffer_t buffer, const ggml_tensor *tensor,
                     void *data, size_t offset, size_t size) {
  auto *ctx = static_cast<BufferContext *>(buffer->context);
  hipSetDevice(ctx->device);
  if (const ggml_tensor *root = repackedRoot(tensor)) {
    std::vector<uint8_t> raw(ggml_nbytes(root));
    bufferGetTensorRaw(root, raw.data(), ctx->device);
    std::memcpy(data, raw.data() + rootOffset(tensor, root) + offset, size);
    return;
  }
  HIPEP_CHECK(hipMemcpy(data, static_cast<const char *>(tensor->data) + offset,
                        size, hipMemcpyDeviceToHost));
}

bool bufferCpyTensor(ggml_backend_buffer_t buffer, const ggml_tensor *src,
                     ggml_tensor *dst) {
  if (!src->buffer || !isHipepBuft(src->buffer->buft))
    return false;
  // A raw copy only moves a repacked layout between two whole tensors of the
  // same type; ggml falls back to get + set otherwise.
  const ggml_tensor *srcRoot = repackedRoot(src);
  const ggml_tensor *dstRoot = repackedRoot(dst);
  if ((srcRoot || dstRoot) &&
      (srcRoot != src || dstRoot != dst || src->type != dst->type))
    return false;
  auto *ctx = static_cast<BufferContext *>(buffer->context);
  hipSetDevice(ctx->device);
  // Device-to-device hipMemcpy does not block the host; wait so later work on
  // any stream sees the copy.
  const size_t bytes = std::max(ggml_nbytes(src), repackedSize(src));
  HIPEP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyDeviceToDevice));
  HIPEP_CHECK(hipDeviceSynchronize());
  return true;
}

void bufferClear(ggml_backend_buffer_t buffer, uint8_t value) {
  auto *ctx = static_cast<BufferContext *>(buffer->context);
  hipSetDevice(ctx->device);
  HIPEP_CHECK(hipMemset(ctx->base, value, buffer->size));
}

const ggml_backend_buffer_i kBufferInterface = {
    /* .free_buffer   = */ bufferFree,
    /* .get_base      = */ bufferBase,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ bufferMemsetTensor,
    /* .set_tensor    = */ bufferSetTensor,
    /* .get_tensor    = */ bufferGetTensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ bufferCpyTensor,
    /* .clear         = */ bufferClear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

const char *buftName(ggml_backend_buffer_type_t buft) {
  return static_cast<DeviceContext *>(buft->context)->name.c_str();
}

ggml_backend_buffer_t buftAlloc(ggml_backend_buffer_type_t buft, size_t size) {
  auto *dev = static_cast<DeviceContext *>(buft->context);
  hipSetDevice(dev->index);
  void *ptr = nullptr;
  if (hipMalloc(&ptr, std::max<size_t>(size, 1)) != hipSuccess) {
    logError("hipMalloc of %zu bytes failed", size);
    return nullptr;
  }
  auto *ctx = new BufferContext{dev->index, ptr};
  return ggml_backend_buffer_init(buft, kBufferInterface, ctx, size);
}

size_t buftAlignment(ggml_backend_buffer_type_t) { return 256; }

// The packed layout of some quantized types is larger than their ggml blocks.
size_t buftAllocSize(ggml_backend_buffer_type_t, const ggml_tensor *tensor) {
  return std::max(ggml_nbytes(tensor), repackedSize(tensor));
}

bool buftIsHost(ggml_backend_buffer_type_t) { return false; }

const ggml_backend_buffer_type_i kBuftInterface = {
    /* .get_name         = */ buftName,
    /* .alloc_buffer     = */ buftAlloc,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ buftAlignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ buftAllocSize,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ buftIsHost,
};

bool isHipepBuft(ggml_backend_buffer_type_t buft) {
  return buft && buft->iface.get_name == buftName;
}

//===----------------------------------------------------------------------===//
// Execution plan
//===----------------------------------------------------------------------===//

// Locates a tensor in a graph so a cached plan rebinds to the current graph's
// tensors: slot -1 is the node, -2 its view_src, >= 0 its src[slot]. The
// resolved tensor is reduced to its storage root.
struct TensorRef {
  int node = -1;
  int slot = -1;
};

const ggml_tensor *resolve(ggml_cgraph *graph, const TensorRef &ref) {
  const ggml_tensor *node = ggml_graph_node(graph, ref.node);
  const ggml_tensor *t = ref.slot == -1   ? node
                         : ref.slot == -2 ? node->view_src
                                          : node->src[ref.slot];
  return storageRoot(t);
}

struct InputBinding {
  SegmentInput::Kind kind = SegmentInput::Kind::Tensor;
  TensorRef ref;
  int rope = -1; // backend rope-table index
};

struct Step {
  bool native = false;
  int node = -1; // native node
  int begin = 0, end = 0;
  CompiledModule *module = nullptr;
  std::vector<InputBinding> inputs;
  std::vector<TensorRef> outputs;
  std::vector<std::pair<TensorRef, size_t>> copies;
};

struct Plan {
  std::vector<Step> steps;
};

struct RopeTable {
  RopeTableSpec spec;
  void *cos = nullptr;
  void *sin = nullptr;
};

struct BackendContext {
  DeviceContext *device = nullptr;
  hipStream_t stream = nullptr;
  std::unordered_map<uint64_t, std::unique_ptr<Plan>> plans;
  std::vector<RopeTable> ropes;
  uint64_t generation = 0;
  // GGML_HIPEP_TRACE_MEMORY: input contents at their last read in this graph.
  std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> traceSeen;
  void *scratch = nullptr;
  size_t scratchSize = 0;
  std::string name;

  void dropAddressCaches() {
    plans.clear();
    for (auto &r : ropes) {
      (void)hipFree(r.cos);
      (void)hipFree(r.sin);
    }
    ropes.clear();
  }

  ~BackendContext() {
    (void)hipSetDevice(device->index);
    dropAddressCaches();
    if (scratch)
      (void)hipFree(scratch);
    if (stream)
      (void)hipStreamDestroy(stream);
  }
};

struct Hasher {
  uint64_t h = 1469598103934665603ull;
  void mix(const void *p, size_t n) {
    const auto *c = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= c[i];
      h *= 1099511628211ull;
    }
  }
  template <typename T> void add(const T &v) { mix(&v, sizeof(v)); }
};

void addTensorShape(Hasher &h, const ggml_tensor *t) {
  h.add(t->type);
  h.mix(t->ne, sizeof(t->ne));
  h.mix(t->nb, sizeof(t->nb));
  h.add(t->view_offs);
}

// Structure of the graph: ops, shapes, layouts, parameters and which tensors
// feed which nodes. Leaves are identified by pointer.
uint64_t fingerprint(ggml_cgraph *graph) {
  const int n = ggml_graph_n_nodes(graph);
  std::unordered_map<const ggml_tensor *, int> index;
  index.reserve(n);
  for (int i = 0; i < n; ++i)
    index[ggml_graph_node(graph, i)] = i;
  Hasher h;
  h.add(n);
  auto addRef = [&](const ggml_tensor *t) {
    if (!t) {
      h.add(int64_t{-1});
      return;
    }
    auto it = index.find(t);
    if (it != index.end()) {
      h.add(int64_t{it->second});
      return;
    }
    h.add(reinterpret_cast<uintptr_t>(t));
    addTensorShape(h, t);
  };
  for (int i = 0; i < n; ++i) {
    const ggml_tensor *node = ggml_graph_node(graph, i);
    h.add(node->op);
    addTensorShape(h, node);
    h.add(node->flags & GGML_TENSOR_FLAG_OUTPUT);
    h.mix(node->op_params, sizeof(node->op_params));
    addRef(node->view_src);
    for (const ggml_tensor *src : node->src)
      addRef(src);
  }
  return h.h;
}

void ropeYarn(float thetaExtrap, float freqScale, const float corr[2],
              int64_t i0, float extFactor, float mscale, float &c, float &s) {
  float thetaInterp = freqScale * thetaExtrap;
  float theta = thetaInterp;
  if (extFactor != 0.0f) {
    const float y = (i0 / 2 - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
    const float ramp = (1 - std::min(1.0f, std::max(0.0f, y))) * extFactor;
    theta = thetaInterp * (1 - ramp) + thetaExtrap * ramp;
    mscale *= 1.0f + 0.1f * std::log(1.0f / freqScale);
  }
  c = std::cos(theta) * mscale;
  s = std::sin(theta) * mscale;
}

// Mirrors ggml_rope_cache_init: row p holds cos/sin of p * base^(-2i/n_dims)
// with the YaRN correction and per-frequency factors.
int ensureRope(BackendContext *ctx, const RopeTableSpec &spec) {
  for (size_t i = 0; i < ctx->ropes.size(); ++i)
    if (ctx->ropes[i].spec == spec)
      return static_cast<int>(i);
  const int64_t half = spec.nDims / 2;
  std::vector<float> factors;
  if (spec.freqFactors) {
    factors.resize(half);
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(spec.freqFactors),
                            factors.data(), 0, half * sizeof(float));
  }
  float corr[2];
  ggml_rope_yarn_corr_dims(spec.nDims, spec.nCtxOrig, spec.freqBase,
                           spec.betaFast, spec.betaSlow, spec.truncate, corr);
  const float thetaScale = std::pow(spec.freqBase, -2.0f / spec.nDims);
  std::vector<float> cosT(spec.rows * half), sinT(spec.rows * half);
  for (int64_t p = 0; p < spec.rows; ++p) {
    float theta = static_cast<float>(p);
    for (int64_t i = 0; i < half; ++i) {
      const float ff = factors.empty() ? 1.0f : factors[i];
      ropeYarn(theta / ff, spec.freqScale, corr, 2 * i, spec.extFactor,
               spec.attnFactor, cosT[p * half + i], sinT[p * half + i]);
      theta *= thetaScale;
    }
  }
  RopeTable table;
  table.spec = spec;
  const size_t count = cosT.size();
  hipSetDevice(ctx->device->index);
  if (spec.f16) {
    std::vector<ggml_fp16_t> c16(count), s16(count);
    ggml_fp32_to_fp16_row(cosT.data(), c16.data(), count);
    ggml_fp32_to_fp16_row(sinT.data(), s16.data(), count);
    HIPEP_CHECK(hipMalloc(&table.cos, count * 2));
    HIPEP_CHECK(hipMalloc(&table.sin, count * 2));
    HIPEP_CHECK(
        hipMemcpy(table.cos, c16.data(), count * 2, hipMemcpyHostToDevice));
    HIPEP_CHECK(
        hipMemcpy(table.sin, s16.data(), count * 2, hipMemcpyHostToDevice));
  } else {
    HIPEP_CHECK(hipMalloc(&table.cos, count * 4));
    HIPEP_CHECK(hipMalloc(&table.sin, count * 4));
    HIPEP_CHECK(
        hipMemcpy(table.cos, cosT.data(), count * 4, hipMemcpyHostToDevice));
    HIPEP_CHECK(
        hipMemcpy(table.sin, sinT.data(), count * 4, hipMemcpyHostToDevice));
  }
  ctx->ropes.push_back(table);
  return static_cast<int>(ctx->ropes.size() - 1);
}

std::string describeRange(ggml_cgraph *graph, int begin, int end) {
  std::string ops;
  for (int i = begin; i < end; ++i) {
    const ggml_tensor *node = ggml_graph_node(graph, i);
    if (classifyNode(node) != NodeKind::Functional)
      continue;
    if (!ops.empty())
      ops += ' ';
    ops += ggml_op_desc(node);
  }
  return ops;
}

void dumpGraph(ggml_cgraph *graph) {
  auto desc = [](const ggml_tensor *t) {
    char buf[256];
    std::snprintf(
        buf, sizeof(buf), "%s:%s[%lld,%lld,%lld,%lld]nb[%zu,%zu,%zu,%zu]%s",
        t->name, ggml_type_name(t->type), (long long)t->ne[0],
        (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], t->nb[0],
        t->nb[1], t->nb[2], t->nb[3], t->view_src ? "(view)" : "");
    return std::string(buf);
  };
  const int n = ggml_graph_n_nodes(graph);
  for (int i = 0; i < n; ++i) {
    const ggml_tensor *node = ggml_graph_node(graph, i);
    static const char *kKinds[] = {"skip", "view", "func", "native"};
    std::fprintf(stderr, "[ggml-hipep] node %d %s %s flags=0x%x kind=%s\n", i,
                 ggml_op_desc(node), desc(node).c_str(), node->flags,
                 kKinds[static_cast<int>(classifyNode(node))]);
    for (const ggml_tensor *src : node->src)
      if (src)
        std::fprintf(stderr, "[ggml-hipep]     <- %s\n", desc(src).c_str());
  }
}

bool buildPlan(BackendContext *ctx, ggml_cgraph *graph, Plan &plan,
               std::string &error) {
  const int n = ggml_graph_n_nodes(graph);
  if (envFlag("GGML_HIPEP_DUMP_GRAPH"))
    dumpGraph(graph);
  std::unordered_map<const ggml_tensor *, TensorRef> refs;
  auto addRef = [&](const ggml_tensor *t, int node, int slot) {
    if (t)
      refs.emplace(storageRoot(t), TensorRef{node, slot});
  };
  for (int i = 0; i < n; ++i) {
    ggml_tensor *node = ggml_graph_node(graph, i);
    addRef(node, i, -1);
    addRef(node->view_src, i, -2);
    for (int s = 0; s < GGML_MAX_SRC; ++s)
      addRef(node->src[s], i, s);
  }

  for (const GraphStep &gs : partitionGraph(graph)) {
    if (gs.native) {
      Step step;
      step.native = true;
      step.node = gs.node;
      plan.steps.push_back(step);
      continue;
    }
    Segment seg;
    if (!importSegment(graph, gs.begin, gs.end, gs.outputs, seg, error))
      return false;
    if (seg.mlir.empty())
      continue;
    if (envFlag("GGML_HIPEP_DUMP"))
      std::fprintf(stderr, "[ggml-hipep] segment [%d, %d): %s\n%s\n", gs.begin,
                   gs.end, describeRange(graph, gs.begin, gs.end).c_str(),
                   seg.mlir.c_str());
    Step step;
    step.begin = gs.begin;
    step.end = gs.end;
    step.module = Engine::instance().get(seg.mlir, error);
    if (!step.module)
      return false;
    if (verbose()) {
      const hipError_t err = hipDeviceSynchronize();
      if (err != hipSuccess) {
        error = std::string("device fault after loading module ") +
                step.module->key() + ": " + hipGetErrorString(err);
        return false;
      }
    }
    for (const SegmentInput &in : seg.inputs) {
      InputBinding b;
      b.kind = in.kind;
      if (in.tensor) {
        auto it = refs.find(in.tensor);
        if (it == refs.end()) {
          error = "segment input is not reachable from the graph";
          return false;
        }
        b.ref = it->second;
      } else {
        b.rope = ensureRope(ctx, seg.ropeTables[in.rope]);
        if (verbose() && hipDeviceSynchronize() != hipSuccess) {
          error = "device fault after building a RoPE table";
          return false;
        }
      }
      step.inputs.push_back(b);
    }
    for (const ggml_tensor *out : seg.outputs)
      step.outputs.push_back(refs.at(out));
    for (const auto &[root, result] : seg.outputCopies)
      step.copies.push_back({refs.at(root), result});
    if (verbose())
      std::fprintf(stderr,
                   "[ggml-hipep] segment [%d, %d) -> module %s (%zu in, %zu "
                   "out)\n",
                   gs.begin, gs.end, step.module->key().c_str(),
                   step.inputs.size(), step.outputs.size());
    plan.steps.push_back(std::move(step));
  }
  return true;
}

bool runNative(BackendContext *ctx, ggml_tensor *node, std::string &error) {
  const Kernels &k = kernels(ctx->device->arch);
  if (!k.setRows || !k.cpy) {
    error = "hip_ggml_* kernels are unavailable";
    return false;
  }
  int rc = 0;
  if (node->op == GGML_OP_SET_ROWS) {
    // The node is a view of the destination (src[2]) with its layout.
    const ggml_tensor *dst = node;
    const ggml_tensor *src = node->src[0];
    const ggml_tensor *idx = node->src[1];
    Layout d(dst), s(src), x(idx);
    rc = k.setRows(ctx->stream, dst->data, hipDtype(dst->type), d.ne, d.nb,
                   src->data, hipDtype(src->type), s.ne, s.nb, idx->data,
                   hipDtype(idx->type), x.ne, x.nb);
  } else if (node->op == GGML_OP_CPY || node->op == GGML_OP_DUP) {
    const ggml_tensor *src = node->src[0];
    Layout d(node), s(src);
    if (src->data == node->data && src->type == node->type &&
        std::equal(d.ne, d.ne + 4, s.ne) && std::equal(d.nb, d.nb + 4, s.nb))
      return true;
    rc = k.cpy(ctx->stream, node->data, hipDtype(node->type), d.ne, d.nb,
               src->data, hipDtype(src->type), s.ne, s.nb);
  } else {
    error = std::string("no native implementation for ") + ggml_op_desc(node);
    return false;
  }
  if (rc != 0) {
    error = std::string("native ") + ggml_op_desc(node) + " failed (" +
            std::to_string(rc) + ")";
    return false;
  }
  return true;
}

bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const char *pa = static_cast<const char *>(a);
  const char *pb = static_cast<const char *>(b);
  return pa < pb + bn && pb < pa + an;
}

bool runSegment(BackendContext *ctx, ggml_cgraph *graph, const Step &step,
                std::string &error) {
  std::vector<DeviceTensor> inputs;
  std::vector<std::pair<const void *, size_t>> inputRanges;
  for (const InputBinding &b : step.inputs) {
    DeviceTensor t;
    if (b.kind == SegmentInput::Kind::Tensor) {
      const ggml_tensor *root = resolve(graph, b.ref);
      t.data = root->data;
      t.shape = {root->ne[3], root->ne[2], root->ne[1], root->ne[0]};
      t.elementSize = static_cast<int64_t>(ggml_type_size(root->type));
      inputRanges.push_back({root->data, ggml_nbytes(root)});
    } else if (b.kind == SegmentInput::Kind::QuantData ||
               b.kind == SegmentInput::Kind::QuantScales ||
               b.kind == SegmentInput::Kind::QuantZeroPoints) {
      const ggml_tensor *w = resolve(graph, b.ref);
      QuantLayout layout;
      quantLayout(w->type, layout);
      const int64_t rows = w->ne[1], blocks = w->ne[0] / layout.blockSize;
      char *base = static_cast<char *>(w->data);
      if (b.kind == SegmentInput::Kind::QuantData) {
        t.data = base;
        t.shape = {rows, blocks, layout.blob};
        t.elementSize = 1;
      } else {
        const int64_t scalesAt = rows * blocks * layout.blob;
        t.data = b.kind == SegmentInput::Kind::QuantScales
                     ? base + scalesAt
                     : base + scalesAt + rows * blocks * 2;
        t.shape = {rows, blocks};
        t.elementSize = 2;
      }
      inputRanges.push_back({w->data, repackedSize(w)});
    } else {
      const RopeTable &r = ctx->ropes[b.rope];
      t.data = b.kind == SegmentInput::Kind::RopeCos ? r.cos : r.sin;
      t.shape = {r.spec.rows, r.spec.nDims / 2};
      t.elementSize = r.spec.f16 ? 2 : 4;
    }
    inputs.push_back(std::move(t));
  }

  std::vector<const ggml_tensor *> outputs;
  size_t scratchNeeded = 0;
  std::vector<size_t> scratchOffset;
  for (const TensorRef &ref : step.outputs) {
    const ggml_tensor *root = resolve(graph, ref);
    outputs.push_back(root);
    static const bool forceScratch = envFlag("GGML_HIPEP_FORCE_SCRATCH");
    bool clash = forceScratch;
    for (const auto &[p, n] : inputRanges)
      clash |= overlaps(root->data, ggml_nbytes(root), p, n);
    if (clash) {
      scratchOffset.push_back(scratchNeeded);
      scratchNeeded += (ggml_nbytes(root) + 255) & ~size_t{255};
    } else {
      scratchOffset.push_back(SIZE_MAX);
    }
  }
  if (scratchNeeded > ctx->scratchSize) {
    if (ctx->scratch)
      hipFree(ctx->scratch);
    if (hipMalloc(&ctx->scratch, scratchNeeded) != hipSuccess) {
      ctx->scratch = nullptr;
      ctx->scratchSize = 0;
      error = "scratch allocation failed";
      return false;
    }
    ctx->scratchSize = scratchNeeded;
  }

  bool sizeMismatch = false;
  auto allocate = [&](int64_t idx, const int64_t *shape, int64_t rank,
                      int64_t elemSize) -> void * {
    const ggml_tensor *root = outputs[idx];
    int64_t bytes = elemSize;
    for (int64_t i = 0; i < rank; ++i)
      bytes *= shape[i];
    if (bytes != static_cast<int64_t>(ggml_nbytes(root)))
      sizeMismatch = true;
    if (scratchOffset[idx] != SIZE_MAX)
      return static_cast<char *>(ctx->scratch) + scratchOffset[idx];
    return root->data;
  };
  if (envFlag("GGML_HIPEP_TRACE_MEMORY")) {
    std::fprintf(stderr, "[ggml-hipep] run [%d, %d)\n", step.begin, step.end);
    for (const InputBinding &b : step.inputs)
      if (b.kind == SegmentInput::Kind::Tensor) {
        const ggml_tensor *t = resolve(graph, b.ref);
        std::fprintf(stderr, "[ggml-hipep]   in  %-28s %p +%zu\n", t->name,
                     t->data, ggml_nbytes(t));
      }
    for (size_t i = 0; i < outputs.size(); ++i)
      std::fprintf(stderr, "[ggml-hipep]   out %-28s %p +%zu%s\n",
                   outputs[i]->name, outputs[i]->data, ggml_nbytes(outputs[i]),
                   scratchOffset[i] != SIZE_MAX ? " (scratch)" : "");
  }
  // Debug aid: report inputs the compiled segment modified.
  std::vector<std::pair<const ggml_tensor *, std::vector<uint8_t>>> before;
  if (envFlag("GGML_HIPEP_TRACE_MEMORY")) {
    for (const InputBinding &b : step.inputs) {
      if (b.kind != SegmentInput::Kind::Tensor)
        continue;
      const ggml_tensor *t = resolve(graph, b.ref);
      if (ggml_nbytes(t) > (4u << 20))
        continue;
      std::vector<uint8_t> host(ggml_nbytes(t));
      HIPEP_CHECK(
          hipMemcpy(host.data(), t->data, host.size(), hipMemcpyDeviceToHost));
      before.push_back({t, std::move(host)});
    }
  }
  if (!step.module->run(inputs, allocate, error))
    return false;
  for (const auto &[t, host] : before) {
    auto seen = ctx->traceSeen.find(t);
    if (seen != ctx->traceSeen.end() && seen->second != host)
      std::fprintf(stderr,
                   "[ggml-hipep]   INPUT CHANGED since last read, before "
                   "[%d, %d): %s\n",
                   step.begin, step.end, t->name);
    ctx->traceSeen[t] = host;
    if (const char *dump = std::getenv("GGML_HIPEP_DUMP_INPUT")) {
      const std::string list = std::string(",") + dump + ",";
      if (list.find(std::string(",") + t->name + ",") != std::string::npos) {
        const std::string path = std::string(t->name) + ".bin";
        if (FILE *f = std::fopen(path.c_str(), "wb")) {
          std::fwrite(host.data(), 1, host.size(), f);
          std::fclose(f);
        }
      }
    }
    std::vector<uint8_t> after(host.size());
    HIPEP_CHECK(
        hipMemcpy(after.data(), t->data, after.size(), hipMemcpyDeviceToHost));
    if (after != host)
      std::fprintf(stderr, "[ggml-hipep]   INPUT MODIFIED by [%d, %d): %s\n",
                   step.begin, step.end, t->name);
  }
  if (sizeMismatch) {
    error = "compiled output size does not match the ggml tensor";
    return false;
  }
  // Queued on the backend stream so the native kernels that follow see them;
  // the caller synchronizes before the next segment runs.
  for (size_t i = 0; i < outputs.size(); ++i)
    if (scratchOffset[i] != SIZE_MAX)
      HIPEP_CHECK(hipMemcpyAsync(
          outputs[i]->data,
          static_cast<char *>(ctx->scratch) + scratchOffset[i],
          ggml_nbytes(outputs[i]), hipMemcpyDeviceToDevice, ctx->stream));
  for (const auto &[ref, result] : step.copies) {
    const ggml_tensor *root = resolve(graph, ref);
    HIPEP_CHECK(hipMemcpyAsync(root->data, outputs[result]->data,
                               ggml_nbytes(root), hipMemcpyDeviceToDevice,
                               ctx->stream));
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Backend (stream)
//===----------------------------------------------------------------------===//

const char *backendName(ggml_backend_t backend) {
  return static_cast<BackendContext *>(backend->context)->name.c_str();
}

void backendFree(ggml_backend_t backend) {
  delete static_cast<BackendContext *>(backend->context);
  delete backend;
}

void backendSynchronize(ggml_backend_t backend) {
  auto *ctx = static_cast<BackendContext *>(backend->context);
  hipSetDevice(ctx->device->index);
  HIPEP_CHECK(hipStreamSynchronize(ctx->stream));
}

ggml_status backendGraphCompute(ggml_backend_t backend, ggml_cgraph *graph) {
  auto *ctx = static_cast<BackendContext *>(backend->context);
  hipSetDevice(ctx->device->index);
  if (ctx->generation != gBufferGeneration) {
    ctx->dropAddressCaches();
    ctx->generation = gBufferGeneration;
  }
  const uint64_t key = fingerprint(graph);
  auto it = ctx->plans.find(key);
  if (it == ctx->plans.end()) {
    auto plan = std::make_unique<Plan>();
    std::string error;
    if (!buildPlan(ctx, graph, *plan, error)) {
      logError("graph import failed: %s", error.c_str());
      return GGML_STATUS_FAILED;
    }
    it = ctx->plans.emplace(key, std::move(plan)).first;
  }
  ctx->traceSeen.clear();
  // Copy-backs and native kernels are queued on ctx->stream; compiled
  // segments run on their own stream and return synchronized, so the backend
  // stream is drained before each segment.
  bool pending = false;
  for (const Step &step : it->second->steps) {
    std::string error;
    if (step.native) {
      if (!runNative(ctx, ggml_graph_node(graph, step.node), error)) {
        logError("%s", error.c_str());
        return GGML_STATUS_FAILED;
      }
      pending = true;
      continue;
    }
    if (pending) {
      HIPEP_CHECK(hipStreamSynchronize(ctx->stream));
      pending = false;
    }
    if (!runSegment(ctx, graph, step, error)) {
      logError("segment [%d, %d) failed: %s", step.begin, step.end,
               error.c_str());
      return GGML_STATUS_FAILED;
    }
    pending = true;
  }
  HIPEP_CHECK(hipStreamSynchronize(ctx->stream));
  return GGML_STATUS_SUCCESS;
}

const ggml_backend_i kBackendInterface = {
    /* .get_name            = */ backendName,
    /* .free                = */ backendFree,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ backendSynchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backendGraphCompute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

ggml_guid_t backendGuid() {
  static ggml_guid guid = {0x68, 0x69, 0x70, 0x65, 0x70, 0x2d, 0x67, 0x67,
                           0x6d, 0x6c, 0x2d, 0x62, 0x61, 0x63, 0x6b, 0x01};
  return &guid;
}

//===----------------------------------------------------------------------===//
// Device interface
//===----------------------------------------------------------------------===//

DeviceContext *deviceOf(ggml_backend_dev_t dev) {
  return static_cast<DeviceContext *>(dev->context);
}

const char *devName(ggml_backend_dev_t dev) {
  return deviceOf(dev)->name.c_str();
}

const char *devDescription(ggml_backend_dev_t dev) {
  return deviceOf(dev)->description.c_str();
}

void devMemory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
  hipSetDevice(deviceOf(dev)->index);
  if (hipMemGetInfo(free, total) != hipSuccess)
    *free = *total = 0;
}

enum ggml_backend_dev_type devType(ggml_backend_dev_t) {
  return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void devProps(ggml_backend_dev_t dev, ggml_backend_dev_props *props) {
  props->name = devName(dev);
  props->description = devDescription(dev);
  props->type = devType(dev);
  props->device_id = nullptr;
  devMemory(dev, &props->memory_free, &props->memory_total);
  props->caps = {/* async */ false, /* host_buffer */ false,
                 /* buffer_from_host_ptr */ false, /* events */ false,
                 /* mmap_support */ false};
}

ggml_backend_t devInitBackend(ggml_backend_dev_t dev, const char *) {
  DeviceContext *device = deviceOf(dev);
  hipSetDevice(device->index);
  auto *ctx = new BackendContext();
  ctx->device = device;
  ctx->name = device->name;
  // A blocking stream also orders against null-stream buffer copies.
  if (hipStreamCreate(&ctx->stream) != hipSuccess) {
    delete ctx;
    return nullptr;
  }
  return new ggml_backend{backendGuid(), kBackendInterface, dev, ctx};
}

ggml_backend_buffer_type_t devBufferType(ggml_backend_dev_t dev) {
  return &deviceOf(dev)->buft;
}

bool devSupportsOp(ggml_backend_dev_t, const ggml_tensor *op) {
  // Debug aid: GGML_HIPEP_REJECT=<substring> leaves matching nodes to the CPU.
  static const char *reject = std::getenv("GGML_HIPEP_REJECT");
  if (reject && *reject && std::strstr(op->name, reject))
    return false;
  std::string reason;
  if (canExecuteNode(op, &reason))
    return true;
  if (verbose()) {
    static std::mutex mutex;
    static std::set<std::string> reported;
    std::lock_guard<std::mutex> lock(mutex);
    std::string what = std::string(ggml_op_desc(op)) + ": " + reason;
    if (reported.insert(what).second)
      std::fprintf(stderr, "[ggml-hipep] unsupported %s\n", what.c_str());
  }
  return false;
}

bool devSupportsBuft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
  return buft == &deviceOf(dev)->buft;
}

const ggml_backend_device_i kDeviceInterface = {
    /* .get_name             = */ devName,
    /* .get_description      = */ devDescription,
    /* .get_memory           = */ devMemory,
    /* .get_type             = */ devType,
    /* .get_props            = */ devProps,
    /* .init_backend         = */ devInitBackend,
    /* .get_buffer_type      = */ devBufferType,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ devSupportsOp,
    /* .supports_buft        = */ devSupportsBuft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

struct RegContext {
  std::vector<std::unique_ptr<DeviceContext>> devices;
};

const char *regName(ggml_backend_reg_t) { return "HIPEP"; }

size_t regDeviceCount(ggml_backend_reg_t reg) {
  return static_cast<RegContext *>(reg->context)->devices.size();
}

ggml_backend_dev_t regDevice(ggml_backend_reg_t reg, size_t index) {
  return &static_cast<RegContext *>(reg->context)->devices[index]->device;
}

void *regProcAddress(ggml_backend_reg_t, const char *) { return nullptr; }

const ggml_backend_reg_i kRegInterface = {
    /* .get_name         = */ regName,
    /* .get_device_count = */ regDeviceCount,
    /* .get_device       = */ regDevice,
    /* .get_proc_address = */ regProcAddress,
};

ggml_backend_reg_t hipepReg() {
  static ggml_backend_reg reg = [] {
    auto *ctx = new RegContext();
    ggml_backend_reg r{GGML_BACKEND_API_VERSION, kRegInterface, ctx};
    return r;
  }();
  static std::once_flag once;
  std::call_once(once, [] {
    auto *ctx = static_cast<RegContext *>(reg.context);
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess)
      count = 0;
    for (int i = 0; i < count; ++i) {
      auto dev = std::make_unique<DeviceContext>();
      dev->index = i;
      dev->name = "HIPEP" + std::to_string(i);
      hipDeviceProp_t prop{};
      if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
        dev->description = std::string(prop.name) + " (HIP EP)";
        dev->arch = prop.gcnArchName;
        dev->arch = dev->arch.substr(0, dev->arch.find(':'));
      } else {
        dev->description = "HIP EP device";
      }
      dev->buft = {kBuftInterface, &dev->device, dev.get()};
      dev->device = {kDeviceInterface, &reg, dev.get()};
      ctx->devices.push_back(std::move(dev));
    }
  });
  return &reg;
}

} // namespace

extern "C" HIPEP_EXPORT ggml_backend_reg_t ggml_backend_hipep_reg(void) {
  return hipepReg();
}

extern "C" HIPEP_EXPORT ggml_backend_reg_t ggml_backend_init(void) {
  return hipepReg();
}
