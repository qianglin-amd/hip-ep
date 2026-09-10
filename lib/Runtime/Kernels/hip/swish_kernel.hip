/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Swish activation: y = x * sigmoid(alpha * x), element-wise (ONNX opset 24).
 * At alpha == 1 this is SiLU.
 *
 * The sigmoid is branch-split on the sign of the argument so neither exp()
 * call can overflow: exp(-z) blows up for large negative z, exp(z) for large
 * positive z, and each branch only ever evaluates the bounded one.
 *
 * FP16 promotes to float for the exp (same pattern as leaky_relu_kernel.hip).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ float sigmoid_f32(float z) {
  if (z >= 0.f)
    return 1.f / (1.f + expf(-z));
  const float e = expf(z);
  return e / (1.f + e);
}

__device__ __forceinline__ double sigmoid_f64(double z) {
  if (z >= 0.0)
    return 1.0 / (1.0 + exp(-z));
  const double e = exp(z);
  return e / (1.0 + e);
}

__global__ void swish_f32_kernel(const float *__restrict__ input,
                                 float *__restrict__ output,
                                 int64_t num_elements, float alpha) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = input[idx];
    output[idx] = x * sigmoid_f32(alpha * x);
  }
}

__global__ void swish_f16_kernel(const __half *__restrict__ input,
                                 __half *__restrict__ output,
                                 int64_t num_elements, float alpha) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(x * sigmoid_f32(alpha * x));
  }
}

__global__ void swish_f64_kernel(const double *__restrict__ input,
                                 double *__restrict__ output,
                                 int64_t num_elements, double alpha) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    double x = input[idx];
    output[idx] = x * sigmoid_f64(alpha * x);
  }
}

extern "C" int hip_swish(void *stream, const void *input, void *output,
                         int64_t num_elements, int hip_dtype, double alpha) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_swish: f32, alpha=%f, num=%lld\n", alpha,
        (long long)num_elements);
    hipLaunchKernelGGL(swish_f32_kernel, dim3(grid_size), dim3(kBlockSize), 0,
                       hip_stream, static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements,
                       static_cast<float>(alpha));
    break;
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_swish: f16, alpha=%f, num=%lld\n", alpha,
        (long long)num_elements);
    hipLaunchKernelGGL(swish_f16_kernel, dim3(grid_size), dim3(kBlockSize), 0,
                       hip_stream, static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements,
                       static_cast<float>(alpha));
    break;
  case HIP_DTYPE_FLOAT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_swish: f64, alpha=%f, num=%lld\n", alpha,
        (long long)num_elements);
    hipLaunchKernelGGL(swish_f64_kernel, dim3(grid_size), dim3(kBlockSize), 0,
                       hip_stream, static_cast<const double *>(input),
                       static_cast<double *>(output), num_elements, alpha);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_swish: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_swish launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
