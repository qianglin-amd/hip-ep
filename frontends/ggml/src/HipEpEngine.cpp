/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipEpEngine.h"

#include "LlvmIrJit.h"
#include "compilation_options_generated.h"
#include "hip/Compiler/CompilerDriver.h"
#include "hip/Support/DiskFileSystem.h"
#include "hip/artifact_abi.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace hipep_ggml {
namespace {

// Wire types of the generated inference_compute ABI (see tools/hip-test).
enum { kTensorMemoryGpu = 1 };

struct TensorT {
  void *data;
  int64_t *shape;
  size_t rank;
  size_t element_size;
  int memory_type;
};

struct SpanT {
  TensorT *data;
  size_t count;
};

struct OutputAllocatorT {
  void *self;
  void *(*allocate)(void *self, int64_t out_idx, const int64_t *shape,
                    int64_t rank, int64_t elem_size);
};

using InitFn = int (*)(void **, void *, const void *);
using ComputeFn = int (*)(void *, SpanT *);
using SetAllocFn = void (*)(void *, const OutputAllocatorT *);
using CleanupFn = int (*)(void *);
using BeginFn = void (*)(void *);

// The compiler and the embedded runtime live in this module, so its size and
// timestamp identify the toolchain that produced a cached artifact.
const std::string &toolchainId() {
  static const std::string id = [] {
    fs::path self;
#ifdef _WIN32
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&toolchainId), &module);
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(module, path, MAX_PATH);
    self = path;
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&toolchainId), &info))
      self = info.dli_fname;
#endif
    std::error_code ec;
    const auto size = fs::file_size(self, ec);
    const auto time = fs::last_write_time(self, ec).time_since_epoch().count();
    return std::to_string(size) + ":" + std::to_string(time);
  }();
  return id;
}

std::string hashKey(const std::string &text) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const char *p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      h ^= static_cast<unsigned char>(p[i]);
      h *= 1099511628211ull;
    }
  };
  mix(toolchainId().data(), toolchainId().size());
  mix(text.data(), text.size());
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return buf;
}

bool envFlag(const char *name) {
  const char *v = std::getenv(name);
  return v && *v && std::string(v) != "0";
}

std::string defaultCacheRoot() {
  if (const char *dir = std::getenv("GGML_HIPEP_CACHE_DIR"))
    return dir;
#ifdef _WIN32
  if (const char *local = std::getenv("LOCALAPPDATA"))
    return (fs::path(local) / "hip-ep" / "ggml-cache").string();
#else
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"))
    return (fs::path(xdg) / "hip-ep" / "ggml-cache").string();
  if (const char *home = std::getenv("HOME"))
    return (fs::path(home) / ".cache" / "hip-ep" / "ggml-cache").string();
#endif
  return (fs::temp_directory_path() / "hip-ep-ggml-cache").string();
}

bool readFile(const fs::path &path, std::vector<uint8_t> &bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  bytes.assign(std::istreambuf_iterator<char>(in),
               std::istreambuf_iterator<char>());
  return true;
}

} // namespace

struct CompiledModule::Impl {
  std::unique_ptr<mlir::hip::DiskFileSystem> fs;
  std::unique_ptr<mlir_compilation::customop::LlvmIrJit> jit;
  void *state = nullptr;
  ComputeFn compute = nullptr;
  SetAllocFn setAlloc = nullptr;
  CleanupFn cleanup = nullptr;
  BeginFn begin = nullptr;
};

CompiledModule::~CompiledModule() {
  if (impl_ && impl_->state && impl_->cleanup)
    impl_->cleanup(impl_->state);
}

bool CompiledModule::run(const std::vector<DeviceTensor> &inputs,
                         const OutputAllocator &allocate, std::string &error) {
  std::vector<TensorT> tensors(inputs.size());
  std::vector<std::vector<int64_t>> shapes(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    shapes[i] = inputs[i].shape;
    tensors[i] = {inputs[i].data, shapes[i].data(), shapes[i].size(),
                  static_cast<size_t>(inputs[i].elementSize), kTensorMemoryGpu};
  }
  SpanT span{tensors.data(), tensors.size()};

  struct Ctx {
    const OutputAllocator *fn;
  } ctx{&allocate};
  OutputAllocatorT alloc{&ctx,
                         [](void *self, int64_t idx, const int64_t *shape,
                            int64_t rank, int64_t elemSize) -> void * {
                           return (*static_cast<Ctx *>(self)->fn)(
                               idx, shape, rank, elemSize);
                         }};

  impl_->setAlloc(impl_->state, &alloc);
  if (impl_->begin)
    impl_->begin(impl_->state);
  const int rc = impl_->compute(impl_->state, &span);
  impl_->setAlloc(impl_->state, nullptr);
  if (rc != 0) {
    error = "inference_compute failed with code " + std::to_string(rc) +
            " (module " + key_ + ")";
    return false;
  }
  return true;
}

Engine &Engine::instance() {
  // Never destroyed: module cleanup at static-destruction time would run after
  // the HIP runtime may already have been torn down.
  static Engine *engine = new Engine();
  return *engine;
}

Engine::Engine()
    : cacheRoot_(defaultCacheRoot()),
      diskCache_(!envFlag("GGML_HIPEP_NO_DISK_CACHE")) {}

size_t Engine::moduleCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return modules_.size();
}

CompiledModule *Engine::get(const std::string &mlir, std::string &error) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key = hashKey(mlir);
  auto found = modules_.find(key);
  if (found != modules_.end())
    return found->second.get();

  const fs::path dir = fs::path(cacheRoot_) / key;
  const fs::path bitcode = dir / "model.bc";
  const fs::path done = dir / "complete";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    error =
        "cannot create cache directory " + dir.string() + ": " + ec.message();
    return nullptr;
  }

  auto impl = std::make_unique<CompiledModule::Impl>();
  impl->fs = std::make_unique<mlir::hip::DiskFileSystem>(dir.string().c_str());

  if (!diskCache_ || !fs::exists(done)) {
    fs::remove(done, ec);
    { std::ofstream(dir / "model.mlir", std::ios::binary) << mlir; }
    const auto t0 = std::chrono::steady_clock::now();
    hip::compiler::CompilerDriver driver;
    driver.setFileSystem(impl->fs.get());
    mlir::hip::CompilationOptionsT options;
    options.output_mode = mlir::hip::OutputMode::LLVM_IR;
    std::string compileError;
    if (!driver.compile(mlir, bitcode.string(), options, compileError)) {
      error = "HIP EP compile failed for " + (dir / "model.mlir").string() +
              ": " + compileError;
      return nullptr;
    }
    std::ofstream(done) << "ok\n";
    if (envFlag("GGML_HIPEP_VERBOSE")) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      std::fprintf(stderr, "[ggml-hipep] compiled %s in %.0f ms\n", key.c_str(),
                   ms);
    }
  }

  std::vector<uint8_t> bytes;
  if (!readFile(bitcode, bytes) || bytes.empty()) {
    error = "cannot read " + bitcode.string();
    return nullptr;
  }
  impl->jit =
      mlir_compilation::customop::LlvmIrJit::create(bytes, "ggml_" + key);
  if (!impl->jit) {
    error = "LlvmIrJit failed to load " + bitcode.string();
    return nullptr;
  }
  auto init = impl->jit->get_method<int, void **, void *, const void *>(
      hipdnn::abi::kInferenceInit);
  impl->compute = reinterpret_cast<ComputeFn>(
      impl->jit->lookup_raw(hipdnn::abi::kInferenceCompute));
  impl->setAlloc = reinterpret_cast<SetAllocFn>(
      impl->jit->lookup_raw(hipdnn::abi::kSetOutputAllocator));
  impl->cleanup = reinterpret_cast<CleanupFn>(
      impl->jit->lookup_raw(hipdnn::abi::kInferenceCleanup));
  impl->begin = reinterpret_cast<BeginFn>(
      impl->jit->lookup_raw(hipdnn::abi::kRuntimeBeginCompute));
  if (!init || !impl->compute || !impl->setAlloc || !impl->cleanup) {
    error = "compiled module " + key + " is missing an inference entry point";
    return nullptr;
  }
  morphizen::FileSystem *fsBase = impl->fs.get();
  const int rc = init(&impl->state, fsBase, nullptr);
  if (rc != 0) {
    error = "inference_init failed with code " + std::to_string(rc) +
            " (module " + key + ")";
    return nullptr;
  }

  std::unique_ptr<CompiledModule> module(new CompiledModule());
  module->impl_ = std::move(impl);
  module->key_ = key;
  CompiledModule *raw = module.get();
  modules_.emplace(key, std::move(module));
  return raw;
}

} // namespace hipep_ggml
