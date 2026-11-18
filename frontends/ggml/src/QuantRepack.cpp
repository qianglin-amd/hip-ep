/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "QuantRepack.h"
#include "GgmlImporter.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace hipep_ggml {
namespace {

constexpr int64_t kBlock = 32;

// ggml block layouts (ggml-common.h, not an installed header).
struct BlockQ4_0 {
  uint16_t d;
  uint8_t qs[16]; // element j in the low nibble of qs[j], j + 16 in the high
};
struct BlockQ4_1 {
  uint16_t d, m;
  uint8_t qs[16];
};
struct BlockQ5_0 {
  uint16_t d;
  uint8_t qh[4];
  uint8_t qs[16];
};
struct BlockQ8_0 {
  uint16_t d;
  int8_t qs[32];
};
struct BlockQ4_K {
  uint16_t d, dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 layout");
static_assert(sizeof(BlockQ4_1) == 20, "block_q4_1 layout");
static_assert(sizeof(BlockQ5_0) == 22, "block_q5_0 layout");
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 layout");
static_assert(sizeof(BlockQ4_K) == 144, "block_q4_K layout");

float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }
uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }

// Destination views of one packed tensor.
struct Packed {
  uint8_t *q;
  uint16_t *scale;
  uint16_t *zp; // null for symmetric layouts
  int64_t blob;
};

Packed viewPacked(void *dst, const QuantLayout &layout, int64_t blocks) {
  auto *base = static_cast<uint8_t *>(dst);
  Packed p;
  p.blob = layout.blob;
  p.q = base;
  p.scale = reinterpret_cast<uint16_t *>(base + blocks * layout.blob);
  p.zp = layout.zeroPoints ? p.scale + blocks : nullptr;
  return p;
}

void put4(Packed &p, int64_t b, const uint8_t q[32]) {
  for (int i = 0; i < 16; ++i)
    p.q[b * 16 + i] = static_cast<uint8_t>(q[2 * i] | (q[2 * i + 1] << 4));
}

// value = q * scale - mn, q in [0, 15], as (q - zp) * scale. The 4-bit WMMA
// dequant forms 1024 + floor(zp) in fp16, which is exact only for |zp| <= 1024.
// A block whose value does not depend on q (scale 0) or whose zero point is
// larger becomes q = 0, zp = -1, scale = -mn.
void put4Affine(Packed &p, int64_t b, const uint8_t q[32], float scale,
                float mn) {
  const uint16_t sh = f2h(scale);
  const float s = h2f(sh);
  const float zp = s != 0.0f ? mn / s : 0.0f;
  if (s == 0.0f || !std::isfinite(zp) || std::fabs(zp) > 1000.0f) {
    const uint8_t zero[32] = {};
    put4(p, b, zero);
    p.scale[b] = f2h(-mn);
    p.zp[b] = f2h(-1.0f);
    return;
  }
  put4(p, b, q);
  p.scale[b] = sh;
  p.zp[b] = f2h(zp);
}

// One 8-bit block of p.blob values.
void put8(Packed &p, int64_t b, const int *q, float scale) {
  for (int64_t i = 0; i < p.blob; ++i)
    p.q[b * p.blob + i] = static_cast<uint8_t>(q[i] + 128);
  p.scale[b] = f2h(scale);
}

// Symmetric 8-bit quantization of one block of p.blob floats.
void requant8(Packed &p, int64_t b, const float *v) {
  float amax = 0.0f;
  for (int64_t i = 0; i < p.blob; ++i)
    amax = std::max(amax, std::fabs(v[i]));
  const float s = h2f(f2h(amax / 127.0f));
  int q[kBlock];
  for (int64_t i = 0; i < p.blob; ++i)
    q[i] = s > 0.0f
               ? std::clamp(static_cast<int>(std::lround(v[i] / s)), -127, 127)
               : 0;
  put8(p, b, q, s);
}

void packRows(ggml_type type, const uint8_t *src, Packed p, int64_t row0,
              int64_t row1, int64_t cols, int64_t blockSize) {
  // Exact repacks below use 32-element blocks; the requantized default case
  // uses the layout's block size.
  const int64_t kb = cols / blockSize;
  const size_t rowBytes = ggml_row_size(type, cols);
  std::vector<float> f;
  for (int64_t r = row0; r < row1; ++r) {
    const uint8_t *row = src + r * rowBytes;
    const int64_t b0 = r * kb;
    switch (type) {
    case GGML_TYPE_Q4_0:
      for (int64_t j = 0; j < kb; ++j) {
        const auto &blk = reinterpret_cast<const BlockQ4_0 *>(row)[j];
        uint8_t q[32];
        for (int i = 0; i < 16; ++i) {
          q[i] = blk.qs[i] & 0x0F;
          q[i + 16] = blk.qs[i] >> 4;
        }
        put4(p, b0 + j, q);
        p.scale[b0 + j] = blk.d;
      }
      break;
    case GGML_TYPE_Q8_0:
      for (int64_t j = 0; j < kb; ++j) {
        const auto &blk = reinterpret_cast<const BlockQ8_0 *>(row)[j];
        for (int i = 0; i < 32; ++i)
          p.q[(b0 + j) * 32 + i] = static_cast<uint8_t>(blk.qs[i]) ^ 0x80;
        p.scale[b0 + j] = blk.d;
      }
      break;
    case GGML_TYPE_Q4_1:
      for (int64_t j = 0; j < kb; ++j) {
        const auto &blk = reinterpret_cast<const BlockQ4_1 *>(row)[j];
        uint8_t q[32];
        for (int i = 0; i < 16; ++i) {
          q[i] = blk.qs[i] & 0x0F;
          q[i + 16] = blk.qs[i] >> 4;
        }
        put4Affine(p, b0 + j, q, h2f(blk.d), -h2f(blk.m));
      }
      break;
    case GGML_TYPE_Q4_K:
      for (int64_t s = 0; s < kb / 8; ++s) {
        const auto &blk = reinterpret_cast<const BlockQ4_K *>(row)[s];
        const float d = h2f(blk.d), dmin = h2f(blk.dmin);
        for (int j = 0; j < 8; ++j) {
          uint8_t sc, m;
          if (j < 4) {
            sc = blk.scales[j] & 63;
            m = blk.scales[j + 4] & 63;
          } else {
            sc = (blk.scales[j + 4] & 0x0F) | ((blk.scales[j - 4] >> 6) << 4);
            m = (blk.scales[j + 4] >> 4) | ((blk.scales[j] >> 6) << 4);
          }
          const uint8_t *qs = blk.qs + 32 * (j / 2);
          uint8_t q[32];
          for (int i = 0; i < 32; ++i)
            q[i] = (j % 2) ? (qs[i] >> 4) : (qs[i] & 0x0F);
          put4Affine(p, b0 + s * 8 + j, q, d * sc, dmin * m);
        }
      }
      break;
    case GGML_TYPE_Q5_0:
      for (int64_t j = 0; j < kb; ++j) {
        const auto &blk = reinterpret_cast<const BlockQ5_0 *>(row)[j];
        uint32_t qh;
        std::memcpy(&qh, blk.qh, 4);
        int q[32];
        for (int i = 0; i < 16; ++i) {
          const int h0 = ((qh >> i) << 4) & 0x10;
          const int h1 = (qh >> (i + 12)) & 0x10;
          q[i] = ((blk.qs[i] & 0x0F) | h0) - 16;
          q[i + 16] = ((blk.qs[i] >> 4) | h1) - 16;
        }
        put8(p, b0 + j, q, h2f(blk.d));
      }
      break;
    default:
      f.resize(cols);
      ggml_get_type_traits(type)->to_float(row, f.data(), cols);
      for (int64_t j = 0; j < kb; ++j)
        requant8(p, b0 + j, f.data() + j * blockSize);
      break;
    }
  }
}

void dequantRows(const QuantLayout &layout, const void *src, float *dst,
                 int64_t rows, int64_t cols) {
  const int64_t bs = layout.blockSize;
  const int64_t kb = cols / bs;
  Packed p = viewPacked(const_cast<void *>(src), layout, rows * kb);
  const float zp0 = layout.bits == 4 ? 8.0f : 128.0f;
  for (int64_t b = 0; b < rows * kb; ++b) {
    const float s = h2f(p.scale[b]);
    const float zp = p.zp ? h2f(p.zp[b]) : zp0;
    for (int64_t i = 0; i < bs; ++i) {
      const float q =
          layout.bits == 4
              ? static_cast<float>((p.q[b * p.blob + i / 2] >> (4 * (i % 2))) &
                                   0x0F)
              : static_cast<float>(p.q[b * p.blob + i]);
      dst[b * bs + i] = (q - zp) * s;
    }
  }
}

template <typename Fn> void parallelRows(int64_t rows, const Fn &fn) {
  const int64_t workers = std::clamp<int64_t>(
      std::thread::hardware_concurrency(), 1, std::max<int64_t>(rows / 64, 1));
  if (workers <= 1) {
    fn(0, rows);
    return;
  }
  std::vector<std::thread> pool;
  const int64_t chunk = (rows + workers - 1) / workers;
  for (int64_t w = 0; w < workers; ++w) {
    const int64_t r0 = w * chunk, r1 = std::min(rows, r0 + chunk);
    if (r0 < r1)
      pool.emplace_back([&, r0, r1] { fn(r0, r1); });
  }
  for (auto &t : pool)
    t.join();
}

} // namespace

size_t packedSize(int type, int64_t rows, int64_t cols) {
  QuantLayout layout;
  if (!quantLayout(type, layout) || cols % kBlock != 0)
    return 0;
  const int64_t blocks = rows * (cols / layout.blockSize);
  return static_cast<size_t>(blocks *
                             (layout.blob + 2 + (layout.zeroPoints ? 2 : 0)));
}

void packWeights(int type, const void *src, void *dst, int64_t rows,
                 int64_t cols) {
  QuantLayout layout;
  quantLayout(type, layout);
  Packed p = viewPacked(dst, layout, rows * (cols / layout.blockSize));
  parallelRows(rows, [&](int64_t r0, int64_t r1) {
    packRows(static_cast<ggml_type>(type), static_cast<const uint8_t *>(src), p,
             r0, r1, cols, layout.blockSize);
  });
}

void unpackWeights(int type, const void *src, void *dst, int64_t rows,
                   int64_t cols) {
  QuantLayout layout;
  quantLayout(type, layout);
  const auto t = static_cast<ggml_type>(type);
  const int64_t kb = cols / layout.blockSize;
  Packed p = viewPacked(const_cast<void *>(src), layout, rows * kb);
  if (t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0) {
    for (int64_t b = 0; b < rows * kb; ++b) {
      if (t == GGML_TYPE_Q4_0) {
        auto &blk = static_cast<BlockQ4_0 *>(dst)[b];
        blk.d = p.scale[b];
        uint8_t q[32];
        for (int i = 0; i < 16; ++i) {
          q[2 * i] = p.q[b * 16 + i] & 0x0F;
          q[2 * i + 1] = p.q[b * 16 + i] >> 4;
        }
        for (int j = 0; j < 16; ++j)
          blk.qs[j] = static_cast<uint8_t>(q[j] | (q[j + 16] << 4));
      } else {
        auto &blk = static_cast<BlockQ8_0 *>(dst)[b];
        blk.d = p.scale[b];
        for (int i = 0; i < 32; ++i)
          blk.qs[i] = static_cast<int8_t>(p.q[b * 32 + i] ^ 0x80);
      }
    }
    return;
  }
  std::vector<float> f(static_cast<size_t>(rows * cols));
  dequantRows(layout, src, f.data(), rows, cols);
  if (ggml_quantize_requires_imatrix(t)) {
    std::memset(dst, 0, ggml_row_size(t, cols) * rows);
    return;
  }
  ggml_quantize_chunk(t, f.data(), dst, 0, rows, cols, nullptr);
}

} // namespace hipep_ggml
