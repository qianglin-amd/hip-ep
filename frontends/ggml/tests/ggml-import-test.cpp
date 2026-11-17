/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free check of the GGML importer: builds Llama-style decoder graphs with
// the ggml API (no tensor data), partitions and imports them, and writes each
// segment's ONNX-dialect MLIR so it can be run through hip-mlir-opt.
//
//   ggml-import-test <out-dir> [n_tokens] [n_kv] [n_layer] [f16|q8_0|q4_0]

#include "GgmlImporter.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace hipep_ggml;

namespace {

struct Config {
  int64_t nEmbd = 2048, nHead = 32, nHeadKv = 8, nFf = 8192;
  int64_t nVocab = 128256, kvSize = 4096;
  int64_t nTokens = 7, nKv = 256;
  int nLayer = 2;
  float eps = 1e-5f;
  ggml_type wtype = GGML_TYPE_F16; // matmul weights and token embedding
};

ggml_tensor *weight(ggml_context *ctx, ggml_type type, int64_t ne0, int64_t ne1,
                    const char *name, int il = -1) {
  ggml_tensor *t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
  if (il >= 0)
    ggml_format_name(t, "blk.%d.%s", il, name);
  else
    ggml_set_name(t, name);
  return t;
}

// Mirrors llama.cpp's Llama graph without flash attention (transposed V
// cache, F32 mask, set_rows KV writes).
ggml_cgraph *buildLlama(ggml_context *ctx, const Config &c) {
  const int64_t d = c.nEmbd / c.nHead;
  const int64_t gqa = d * c.nHeadKv;
  const int64_t T = c.nTokens;

  ggml_tensor *tokEmbd =
      weight(ctx, c.wtype, c.nEmbd, c.nVocab, "token_embd.weight");
  ggml_tensor *outNorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.nEmbd);
  ggml_tensor *ropeFreqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d / 2);
  ggml_tensor *tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  ggml_tensor *kIdx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
  ggml_tensor *vIdx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T * gqa);
  ggml_tensor *mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.nKv, T, 1, 1);
  ggml_tensor *outIds = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8192, false);
  ggml_tensor *inpL = ggml_get_rows(ctx, tokEmbd, tokens);
  const float kqScale = 1.0f / std::sqrt(static_cast<float>(d));

  for (int il = 0; il < c.nLayer; ++il) {
    ggml_tensor *attnNorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.nEmbd);
    ggml_tensor *ffnNorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.nEmbd);
    ggml_tensor *wq = weight(ctx, c.wtype, c.nEmbd, c.nEmbd, "wq", il);
    ggml_tensor *wk = weight(ctx, c.wtype, c.nEmbd, gqa, "wk", il);
    ggml_tensor *wv = weight(ctx, c.wtype, c.nEmbd, gqa, "wv", il);
    ggml_tensor *wo = weight(ctx, c.wtype, c.nEmbd, c.nEmbd, "wo", il);
    ggml_tensor *wg = weight(ctx, c.wtype, c.nEmbd, c.nFf, "wg", il);
    ggml_tensor *wu = weight(ctx, c.wtype, c.nEmbd, c.nFf, "wu", il);
    ggml_tensor *wd = weight(ctx, c.wtype, c.nFf, c.nEmbd, "wd", il);
    ggml_tensor *kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, gqa, c.kvSize);
    ggml_tensor *vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, gqa, c.kvSize);

    ggml_tensor *inpSA = inpL;
    ggml_tensor *cur = ggml_rms_norm(ctx, inpL, c.eps);
    cur = ggml_mul(ctx, cur, attnNorm);

    ggml_tensor *q =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wq, cur), d, c.nHead, T);
    ggml_tensor *k =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wk, cur), d, c.nHeadKv, T);
    ggml_tensor *v =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wv, cur), d, c.nHeadKv, T);
    q = ggml_rope_ext(ctx, q, pos, ropeFreqs, d, 0, 131072, 500000.0f, 1.0f,
                      0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, pos, ropeFreqs, d, 0, 131072, 500000.0f, 1.0f,
                      0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor *k2 = ggml_view_2d(ctx, k, gqa, T, k->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kc, k2, kIdx));
    ggml_tensor *vView = ggml_reshape_2d(ctx, vc, 1, ggml_nelements(vc));
    ggml_tensor *v2 =
        ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v, gqa, T), 1, gqa * T);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vView, v2, vIdx));

    ggml_tensor *kv =
        ggml_view_4d(ctx, kc, d, c.nHeadKv, c.nKv, 1,
                     ggml_row_size(kc->type, d), ggml_row_size(kc->type, gqa),
                     ggml_row_size(kc->type, gqa * c.kvSize), 0);
    ggml_tensor *vv = ggml_view_4d(ctx, vc, c.nKv, c.nHeadKv, d, 1,
                                   ggml_row_size(vc->type, c.kvSize * d),
                                   ggml_row_size(vc->type, c.kvSize),
                                   ggml_row_size(vc->type, c.kvSize * gqa), 0);
    ggml_tensor *q4 = ggml_view_4d(ctx, q, d, c.nHead, T, 1, q->nb[1], q->nb[2],
                                   q->nb[2] * T, 0);
    q4 = ggml_permute(ctx, q4, 0, 2, 1, 3);
    kv = ggml_permute(ctx, kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx, vv, 0, 2, 1, 3);
    ggml_tensor *kq = ggml_mul_mat(ctx, kv, q4);
    kq = ggml_soft_max_ext(ctx, kq, mask, kqScale, 0.0f);
    ggml_tensor *kqv = ggml_mul_mat(ctx, vv, kq);
    cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx, cur, c.nEmbd, T);
    cur = ggml_mul_mat(ctx, wo, cur);

    if (il == c.nLayer - 1) {
      cur = ggml_get_rows(ctx, cur, outIds);
      inpSA = ggml_get_rows(ctx, inpSA, outIds);
    }
    ggml_tensor *ffnInp = ggml_add(ctx, cur, inpSA);
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, ffnInp, c.eps), ffnNorm);
    ggml_tensor *up = ggml_mul_mat(ctx, wu, cur);
    ggml_tensor *gate = ggml_mul_mat(ctx, wg, cur);
    cur = ggml_swiglu_split(ctx, gate, up);
    cur = ggml_mul_mat(ctx, wd, cur);
    inpL = ggml_add(ctx, cur, ffnInp);
  }
  ggml_tensor *cur = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, c.eps), outNorm);
  ggml_tensor *logits = ggml_mul_mat(ctx, tokEmbd, cur);
  ggml_set_output(logits);
  ggml_build_forward_expand(gf, logits);
  return gf;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <out-dir> [n_tokens] [n_kv] [n_layer] "
                 "[f16|q8_0|q4_0]\n",
                 argv[0]);
    return 2;
  }
  Config c;
  if (argc > 2)
    c.nTokens = std::atoll(argv[2]);
  if (argc > 3)
    c.nKv = std::atoll(argv[3]);
  if (argc > 4)
    c.nLayer = std::atoi(argv[4]);
  if (argc > 5) {
    const std::string w = argv[5];
    c.wtype = w == "q8_0"   ? GGML_TYPE_Q8_0
              : w == "q4_0" ? GGML_TYPE_Q4_0
                            : GGML_TYPE_F16;
  }
  std::filesystem::path out(argv[1]);
  std::filesystem::create_directories(out);

  ggml_init_params params{256 * 1024 * 1024, nullptr, /*no_alloc=*/true};
  ggml_context *ctx = ggml_init(params);
  ggml_cgraph *gf = buildLlama(ctx, c);

  int failures = 0;
  for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
    std::string reason;
    ggml_tensor *node = ggml_graph_node(gf, i);
    if (!canExecuteNode(node, &reason)) {
      std::printf("unsupported node %d %s: %s\n", i, ggml_op_desc(node),
                  reason.c_str());
      ++failures;
    }
  }

  int index = 0;
  for (const GraphStep &step : partitionGraph(gf)) {
    if (step.native) {
      std::printf("native  node %d %s\n", step.node,
                  ggml_op_desc(ggml_graph_node(gf, step.node)));
      continue;
    }
    Segment seg;
    std::string error;
    if (!importSegment(gf, step.begin, step.end, step.outputs, seg, error)) {
      std::printf("segment [%d, %d) FAILED: %s\n", step.begin, step.end,
                  error.c_str());
      ++failures;
      continue;
    }
    if (seg.mlir.empty())
      continue;
    const auto file = out / ("segment_" + std::to_string(index++) + ".mlir");
    std::ofstream(file) << seg.mlir;
    std::printf("segment [%d, %d): %zu inputs, %zu outputs, %zu rope tables "
                "-> %s\n",
                step.begin, step.end, seg.inputs.size(), seg.outputs.size(),
                seg.ropeTables.size(), file.string().c_str());
  }
  ggml_free(ctx);
  return failures ? 1 : 0;
}
