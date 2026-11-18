/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Converts block-quantized ggml weights to and from the MatMulNBits layout
// described by QuantLayout (GgmlImporter.h). Rows are ggml rows (ne[0] long).

#include <cstddef>
#include <cstdint>

namespace hipep_ggml {

// Bytes the packed layout needs for `rows` rows of `cols` elements, or 0 when
// `type` is not handled.
size_t packedSize(int type, int64_t rows, int64_t cols);

// `src` holds rows of ggml blocks; `dst` receives packedSize() bytes.
void packWeights(int type, const void *src, void *dst, int64_t rows,
                 int64_t cols);

// Inverse of packWeights. Exact for Q4_0 and Q8_0; other types are rebuilt by
// dequantizing the packed values and quantizing them back to `type`.
void unpackWeights(int type, const void *src, void *dst, int64_t rows,
                   int64_t cols);

} // namespace hipep_ggml
