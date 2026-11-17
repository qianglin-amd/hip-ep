/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// In-place writes for the GGML frontend (frontends/ggml). Shapes and strides
// follow ggml: ne[0] is the innermost dimension and nb[] are byte strides.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

struct Dims4 {
  int64_t ne[4];
  int64_t nb[4];
};

__device__ __forceinline__ float loadAsFloat(const char *p, int dtype) {
  switch (dtype) {
  case HIP_DTYPE_FLOAT16:
    return __half2float(*reinterpret_cast<const __half *>(p));
  case HIP_DTYPE_BFLOAT16:
    return __bfloat162float(*reinterpret_cast<const __hip_bfloat16 *>(p));
  default:
    return *reinterpret_cast<const float *>(p);
  }
}

__device__ __forceinline__ void storeFromFloat(char *p, int dtype, float v) {
  switch (dtype) {
  case HIP_DTYPE_FLOAT16:
    *reinterpret_cast<__half *>(p) = __float2half(v);
    break;
  case HIP_DTYPE_BFLOAT16:
    *reinterpret_cast<__hip_bfloat16 *>(p) = __float2bfloat16(v);
    break;
  default:
    *reinterpret_cast<float *>(p) = v;
    break;
  }
}

__device__ __forceinline__ void copyElement(char *dst, int dstType,
                                            const char *src, int srcType) {
  if (dstType == srcType) {
    if (dstType == HIP_DTYPE_FLOAT32)
      *reinterpret_cast<float *>(dst) = *reinterpret_cast<const float *>(src);
    else
      *reinterpret_cast<uint16_t *>(dst) =
          *reinterpret_cast<const uint16_t *>(src);
    return;
  }
  storeFromFloat(dst, dstType, loadAsFloat(src, srcType));
}

// Element i of src (row-major over src.ne) goes to element i of dst
// (row-major over dst.ne). ggml_cpy requires equal element counts only.
__global__ void ggmlCpyKernel(char *dst, Dims4 d, int dstType,
                              const char *src, Dims4 s, int srcType,
                              int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  int64_t r = i;
  int64_t s0 = r % s.ne[0]; r /= s.ne[0];
  int64_t s1 = r % s.ne[1]; r /= s.ne[1];
  int64_t s2 = r % s.ne[2]; int64_t s3 = r / s.ne[2];
  r = i;
  int64_t d0 = r % d.ne[0]; r /= d.ne[0];
  int64_t d1 = r % d.ne[1]; r /= d.ne[1];
  int64_t d2 = r % d.ne[2]; int64_t d3 = r / d.ne[2];
  const char *sp =
      src + s0 * s.nb[0] + s1 * s.nb[1] + s2 * s.nb[2] + s3 * s.nb[3];
  char *dp = dst + d0 * d.nb[0] + d1 * d.nb[1] + d2 * d.nb[2] + d3 * d.nb[3];
  copyElement(dp, dstType, sp, srcType);
}

// ggml_set_rows: dst[:, idx[i1, i2 % ne11, i3 % ne12], i2, i3] =
//                src[:, i1, i2, i3]
__global__ void ggmlSetRowsKernel(char *dst, Dims4 d, int dstType,
                                  const char *src, Dims4 s, int srcType,
                                  const char *idx, Dims4 x, int idxIs64,
                                  int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  int64_t r = i;
  int64_t i0 = r % s.ne[0]; r /= s.ne[0];
  int64_t i1 = r % s.ne[1]; r /= s.ne[1];
  int64_t i2 = r % s.ne[2]; int64_t i3 = r / s.ne[2];
  const char *ip = idx + i1 * x.nb[0] + (i2 % x.ne[1]) * x.nb[1] +
                   (i3 % x.ne[2]) * x.nb[2];
  int64_t row = idxIs64 ? *reinterpret_cast<const int64_t *>(ip)
                        : *reinterpret_cast<const int32_t *>(ip);
  if (row < 0 || row >= d.ne[1])
    return;
  const char *sp =
      src + i0 * s.nb[0] + i1 * s.nb[1] + i2 * s.nb[2] + i3 * s.nb[3];
  char *dp = dst + i0 * d.nb[0] + row * d.nb[1] + i2 * d.nb[2] + i3 * d.nb[3];
  copyElement(dp, dstType, sp, srcType);
}

bool isSupportedFloat(int dtype) {
  return dtype == HIP_DTYPE_FLOAT32 || dtype == HIP_DTYPE_FLOAT16 ||
         dtype == HIP_DTYPE_BFLOAT16;
}

Dims4 makeDims(const int64_t *ne, const int64_t *nb) {
  Dims4 d;
  for (int k = 0; k < 4; ++k) {
    d.ne[k] = ne[k];
    d.nb[k] = nb[k];
  }
  return d;
}

constexpr int kBlockSize = 256;

} // namespace

extern "C" int hip_ggml_cpy(void *stream, void *dst, int dst_dtype,
                            const int64_t *dst_ne, const int64_t *dst_nb,
                            const void *src, int src_dtype,
                            const int64_t *src_ne, const int64_t *src_nb) {
  if (!isSupportedFloat(dst_dtype) || !isSupportedFloat(src_dtype)) {
    fprintf(stderr, "[custom_kernels] hip_ggml_cpy: unsupported dtype %d->%d\n",
            src_dtype, dst_dtype);
    return -1;
  }
  int64_t total = src_ne[0] * src_ne[1] * src_ne[2] * src_ne[3];
  if (total != dst_ne[0] * dst_ne[1] * dst_ne[2] * dst_ne[3]) {
    fprintf(stderr, "[custom_kernels] hip_ggml_cpy: element count mismatch\n");
    return -1;
  }
  if (total == 0)
    return 0;
  int64_t grid = (total + kBlockSize - 1) / kBlockSize;
  (void)hipGetLastError();
  hipLaunchKernelGGL(ggmlCpyKernel, dim3(static_cast<unsigned>(grid)),
                     dim3(kBlockSize), 0, static_cast<hipStream_t>(stream),
                     static_cast<char *>(dst), makeDims(dst_ne, dst_nb),
                     dst_dtype, static_cast<const char *>(src),
                     makeDims(src_ne, src_nb), src_dtype, total);
  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_ggml_cpy: %d->%d n=%lld\n",
                           src_dtype, dst_dtype, (long long)total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hip_ggml_set_rows(void *stream, void *dst, int dst_dtype,
                                 const int64_t *dst_ne, const int64_t *dst_nb,
                                 const void *src, int src_dtype,
                                 const int64_t *src_ne, const int64_t *src_nb,
                                 const void *idx, int idx_dtype,
                                 const int64_t *idx_ne,
                                 const int64_t *idx_nb) {
  if (!isSupportedFloat(dst_dtype) || !isSupportedFloat(src_dtype) ||
      (idx_dtype != HIP_DTYPE_INT64 && idx_dtype != HIP_DTYPE_INT32)) {
    fprintf(stderr,
            "[custom_kernels] hip_ggml_set_rows: unsupported dtypes "
            "src=%d dst=%d idx=%d\n",
            src_dtype, dst_dtype, idx_dtype);
    return -1;
  }
  int64_t total = src_ne[0] * src_ne[1] * src_ne[2] * src_ne[3];
  if (total == 0)
    return 0;
  int64_t grid = (total + kBlockSize - 1) / kBlockSize;
  (void)hipGetLastError();
  hipLaunchKernelGGL(ggmlSetRowsKernel, dim3(static_cast<unsigned>(grid)),
                     dim3(kBlockSize), 0, static_cast<hipStream_t>(stream),
                     static_cast<char *>(dst), makeDims(dst_ne, dst_nb),
                     dst_dtype, static_cast<const char *>(src),
                     makeDims(src_ne, src_nb), src_dtype,
                     static_cast<const char *>(idx), makeDims(idx_ne, idx_nb),
                     idx_dtype == HIP_DTYPE_INT64 ? 1 : 0, total);
  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_ggml_set_rows: n=%lld\n",
                           (long long)total);
  return static_cast<int>(hipGetLastError());
}
