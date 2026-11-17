/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Shared compile & run for the GGML frontend: ONNX-dialect MLIR text ->
// CompilerDriver -> LLVM bitcode -> LlvmIrJit -> inference_compute.

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipep_ggml {

struct DeviceTensor {
  void *data = nullptr;
  std::vector<int64_t> shape;
  int64_t elementSize = 0;
};

// Returns device memory for result `index` of the shape reported by the
// compiled graph.
using OutputAllocator = std::function<void *(
    int64_t index, const int64_t *shape, int64_t rank, int64_t elementSize)>;

class CompiledModule {
public:
  ~CompiledModule();
  CompiledModule(const CompiledModule &) = delete;
  CompiledModule &operator=(const CompiledModule &) = delete;

  // Runs synchronously: results are complete when this returns.
  bool run(const std::vector<DeviceTensor> &inputs,
           const OutputAllocator &allocate, std::string &error);

  const std::string &key() const { return key_; }

private:
  friend class Engine;
  CompiledModule() = default;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string key_;
};

class Engine {
public:
  static Engine &instance();

  // Compiles (or reuses) the module for `mlir`. Returns nullptr with `error`
  // set on failure.
  CompiledModule *get(const std::string &mlir, std::string &error);

  size_t moduleCount() const;

private:
  Engine();

  std::string cacheRoot_;
  bool diskCache_ = true;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<CompiledModule>> modules_;
};

} // namespace hipep_ggml
