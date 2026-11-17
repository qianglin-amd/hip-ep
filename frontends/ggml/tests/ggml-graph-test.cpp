/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runs a Llama-style decoder graph (random weights, partially filled KV cache,
// causal mask) on the CPU backend and on ggml-hipep and compares the logits
// plus every intermediate tensor named "chk.*".
//
//   ggml-graph-test <ggml-hipep.dll> [n_tokens] [n_past] [f16|q8_0|q4_0]

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Config {
  int64_t nEmbd = 256, nHead = 8, nHeadKv = 2, nFf = 512, nVocab = 1000;
  int64_t kvSize = 512, nKv = 256, nTokens = 7, nPast = 100;
  int nLayer = 2;
  float eps = 1e-5f;
  ggml_type wtype = GGML_TYPE_F16;
  bool checkpoints = true; // flag intermediates as outputs
  bool allLogits = false;  // logits for every token (no output-row gather)
  bool embdInput = false;  // embeddings arrive as a graph input (llama.cpp)
};

bool gCheckpoints = true;

struct Model {
  ggml_tensor *tokEmbd, *outNorm, *ropeFreqs;
  struct Layer {
    ggml_tensor *attnNorm, *ffnNorm, *wq, *wk, *wv, *wo, *wg, *wu, *wd, *kc,
        *vc;
  };
  std::vector<Layer> layers;
  ggml_tensor *tokens, *pos, *kIdx, *vIdx, *mask, *outIds, *embd;
};

void check(ggml_tensor *t, const char *name, int il) {
  ggml_format_name(t, "chk.%s.%d", name, il);
  if (gCheckpoints)
    ggml_set_output(t);
}

ggml_cgraph *build(ggml_context *w, ggml_context *ctx, const Config &c,
                   Model &m) {
  const int64_t d = c.nEmbd / c.nHead, gqa = d * c.nHeadKv, T = c.nTokens;
  m.tokEmbd = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, c.nVocab);
  m.outNorm = ggml_new_tensor_1d(w, GGML_TYPE_F32, c.nEmbd);
  m.ropeFreqs = ggml_new_tensor_1d(w, GGML_TYPE_F32, d / 2);
  for (int il = 0; il < c.nLayer; ++il) {
    Model::Layer l;
    l.attnNorm = ggml_new_tensor_1d(w, GGML_TYPE_F32, c.nEmbd);
    l.ffnNorm = ggml_new_tensor_1d(w, GGML_TYPE_F32, c.nEmbd);
    l.wq = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, c.nEmbd);
    l.wk = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, gqa);
    l.wv = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, gqa);
    l.wo = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, c.nEmbd);
    l.wg = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, c.nFf);
    l.wu = ggml_new_tensor_2d(w, c.wtype, c.nEmbd, c.nFf);
    l.wd = ggml_new_tensor_2d(w, c.wtype, c.nFf, c.nEmbd);
    l.kc = ggml_new_tensor_2d(w, GGML_TYPE_F16, gqa, c.kvSize);
    l.vc = ggml_new_tensor_2d(w, GGML_TYPE_F16, gqa, c.kvSize);
    ggml_format_name(l.kc, "cache_k_l%d", il);
    ggml_format_name(l.vc, "cache_v_l%d", il);
    m.layers.push_back(l);
  }
  m.tokens = ggml_new_tensor_1d(w, GGML_TYPE_I32, T);
  m.pos = ggml_new_tensor_1d(w, GGML_TYPE_I32, T);
  m.kIdx = ggml_new_tensor_1d(w, GGML_TYPE_I64, T);
  m.vIdx = ggml_new_tensor_1d(w, GGML_TYPE_I64, T * gqa);
  m.mask = ggml_new_tensor_4d(w, GGML_TYPE_F32, c.nKv, T, 1, 1);
  m.outIds = ggml_new_tensor_1d(w, GGML_TYPE_I32, 1);
  m.embd = ggml_new_tensor_2d(w, GGML_TYPE_F32, c.nEmbd, T);

  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 4096, false);
  ggml_tensor *inpL =
      c.embdInput ? m.embd : ggml_get_rows(ctx, m.tokEmbd, m.tokens);
  if (!c.embdInput)
    check(inpL, "embd", 0);
  const float kqScale = 1.0f / std::sqrt(static_cast<float>(d));
  for (int il = 0; il < c.nLayer; ++il) {
    const auto &l = m.layers[il];
    ggml_tensor *inpSA = inpL;
    ggml_tensor *cur =
        ggml_mul(ctx, ggml_rms_norm(ctx, inpL, c.eps), l.attnNorm);
    ggml_tensor *q =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wq, cur), d, c.nHead, T);
    ggml_tensor *k =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wk, cur), d, c.nHeadKv, T);
    ggml_tensor *v =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wv, cur), d, c.nHeadKv, T);
    q = ggml_rope_ext(ctx, q, m.pos, m.ropeFreqs, d, 0, 131072, 500000.0f, 1.0f,
                      0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, m.pos, m.ropeFreqs, d, 0, 131072, 500000.0f, 1.0f,
                      0.0f, 1.0f, 32.0f, 1.0f);
    check(q, "q", il);
    check(k, "k", il);
    // llama.cpp expands q, v, k together before the KV writes.
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, k);
    ggml_tensor *k2 = ggml_view_2d(ctx, k, gqa, T, k->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, l.kc, k2, m.kIdx));
    ggml_tensor *vView = ggml_reshape_2d(ctx, l.vc, 1, ggml_nelements(l.vc));
    ggml_tensor *v2 =
        ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v, gqa, T), 1, gqa * T);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vView, v2, m.vIdx));

    ggml_tensor *kv = ggml_view_4d(
        ctx, l.kc, d, c.nHeadKv, c.nKv, 1, ggml_row_size(l.kc->type, d),
        ggml_row_size(l.kc->type, gqa),
        ggml_row_size(l.kc->type, gqa * c.kvSize), 0);
    ggml_tensor *vv =
        ggml_view_4d(ctx, l.vc, c.nKv, c.nHeadKv, d, 1,
                     ggml_row_size(l.vc->type, c.kvSize * d),
                     ggml_row_size(l.vc->type, c.kvSize),
                     ggml_row_size(l.vc->type, c.kvSize * gqa), 0);
    ggml_tensor *q4 = ggml_permute(ctx, q, 0, 2, 1, 3);
    kv = ggml_permute(ctx, kv, 0, 2, 1, 3);
    vv = ggml_permute(ctx, vv, 0, 2, 1, 3);
    ggml_tensor *kq = ggml_mul_mat(ctx, kv, q4);
    kq = ggml_soft_max_ext(ctx, kq, m.mask, kqScale, 0.0f);
    check(kq, "kq", il);
    ggml_tensor *kqv = ggml_mul_mat(ctx, vv, kq);
    cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), c.nEmbd, T);
    check(cur, "attn", il);
    cur = ggml_mul_mat(ctx, l.wo, cur);
    if (il == c.nLayer - 1 && !c.allLogits) {
      cur = ggml_get_rows(ctx, cur, m.outIds);
      inpSA = ggml_get_rows(ctx, inpSA, m.outIds);
    }
    ggml_tensor *ffnInp = ggml_add(ctx, cur, inpSA);
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, ffnInp, c.eps), l.ffnNorm);
    cur = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, l.wg, cur),
                            ggml_mul_mat(ctx, l.wu, cur));
    cur = ggml_mul_mat(ctx, l.wd, cur);
    inpL = ggml_add(ctx, cur, ffnInp);
    check(inpL, "out", il);
  }
  ggml_tensor *cur = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, c.eps), m.outNorm);
  ggml_tensor *logits = ggml_mul_mat(ctx, m.tokEmbd, cur);
  ggml_set_name(logits, "logits");
  ggml_set_output(logits);
  ggml_build_forward_expand(gf, logits);
  return gf;
}

void fillFloat(ggml_tensor *t, std::mt19937 &rng, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  const int64_t n = ggml_nelements(t);
  std::vector<float> f(n);
  for (auto &x : f)
    x = dist(rng);
  if (t->type == GGML_TYPE_F32) {
    ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    return;
  }
  std::vector<uint8_t> q(ggml_nbytes(t));
  ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / t->ne[0], t->ne[0],
                      nullptr);
  ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

void fillInputs(const Config &c, Model &m, uint32_t seed) {
  std::mt19937 rng(seed);
  const int64_t d = c.nEmbd / c.nHead, gqa = d * c.nHeadKv, T = c.nTokens;
  fillFloat(m.tokEmbd, rng, -0.5f, 0.5f);
  fillFloat(m.outNorm, rng, 0.5f, 1.5f);
  fillFloat(m.ropeFreqs, rng, 1.0f, 4.0f);
  fillFloat(m.embd, rng, -0.05f, 0.05f);
  for (auto &l : m.layers) {
    fillFloat(l.attnNorm, rng, 0.5f, 1.5f);
    fillFloat(l.ffnNorm, rng, 0.5f, 1.5f);
    for (ggml_tensor *w : {l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd})
      fillFloat(w, rng, -0.1f, 0.1f);
    fillFloat(l.kc, rng, -1.0f, 1.0f);
    fillFloat(l.vc, rng, -1.0f, 1.0f);
  }
  std::vector<int32_t> tokens(T), pos(T);
  std::vector<int64_t> kIdx(T), vIdx(T * gqa);
  for (int64_t t = 0; t < T; ++t) {
    tokens[t] = static_cast<int32_t>((t * 37 + 11) % c.nVocab);
    pos[t] = static_cast<int32_t>(c.nPast + t);
    kIdx[t] = c.nPast + t;
    for (int64_t e = 0; e < gqa; ++e)
      vIdx[t * gqa + e] = e * c.kvSize + c.nPast + t;
  }
  std::vector<float> mask(c.nKv * T);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < c.nKv; ++j)
      mask[t * c.nKv + j] = j <= c.nPast + t ? 0.0f : -INFINITY;
  int32_t outId = static_cast<int32_t>(T - 1);
  ggml_backend_tensor_set(m.tokens, tokens.data(), 0, T * 4);
  ggml_backend_tensor_set(m.pos, pos.data(), 0, T * 4);
  ggml_backend_tensor_set(m.kIdx, kIdx.data(), 0, T * 8);
  ggml_backend_tensor_set(m.vIdx, vIdx.data(), 0, vIdx.size() * 8);
  ggml_backend_tensor_set(m.mask, mask.data(), 0, mask.size() * 4);
  ggml_backend_tensor_set(m.outIds, &outId, 0, 4);
}

std::vector<float> readF32(ggml_tensor *t) {
  std::vector<float> v(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
  return v;
}

struct RunResult {
  std::vector<std::pair<std::string, std::vector<float>>> tensors;
};

RunResult runOn(ggml_backend_t backend, const Config &c) {
  ggml_init_params params{64 * 1024 * 1024, nullptr, true};
  ggml_context *wctx = ggml_init(params);
  ggml_context *ctx = ggml_init(params);
  Model m;
  ggml_cgraph *gf = build(wctx, ctx, c, m);
  // Weights, caches and inputs live in `buf`; graph nodes go through
  // ggml-alloc, which reuses memory across node lifetimes as llama.cpp does.
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, backend);
  ggml_gallocr_t galloc =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_alloc_graph(galloc, gf);
  fillInputs(c, m, 1234);
  if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "graph compute failed on %s\n",
                 ggml_backend_name(backend));
    std::exit(1);
  }
  RunResult r;
  if (ggml_backend_is_cpu(backend))
    for (const auto &l : m.layers)
      for (ggml_tensor *cache : {l.kc, l.vc}) {
        std::vector<uint8_t> bytes(ggml_nbytes(cache));
        ggml_backend_tensor_get(cache, bytes.data(), 0, bytes.size());
        if (FILE *f = std::fopen(
                (std::string(cache->name) + ".cpu.bin").c_str(), "wb")) {
          std::fwrite(bytes.data(), 1, bytes.size(), f);
          std::fclose(f);
        }
      }
  for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
    ggml_tensor *n = ggml_graph_node(gf, i);
    if ((n->flags & GGML_TENSOR_FLAG_OUTPUT) && n->type == GGML_TYPE_F32)
      r.tensors.push_back({n->name, readF32(n)});
  }
  ggml_gallocr_free(galloc);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_free(wctx);
  return r;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <ggml-hipep library> [n_tokens] [n_past] "
                 "[f16|q8_0|q4_0]\n",
                 argv[0]);
    return 2;
  }
  Config c;
  if (argc > 2)
    c.nTokens = std::atoll(argv[2]);
  if (argc > 3)
    c.nPast = std::atoll(argv[3]);
  if (argc > 4) {
    const std::string w = argv[4];
    c.wtype = w == "q8_0"   ? GGML_TYPE_Q8_0
              : w == "q4_0" ? GGML_TYPE_Q4_0
                            : GGML_TYPE_F16;
  }
  // Remaining flags: "llama1b" (Llama 3.2 1B dimensions), "nochk" (no
  // intermediate outputs), "all" (logits for every token).
  for (int i = 5; i < argc; ++i) {
    const std::string f = argv[i];
    if (f == "llama1b") {
      c.nEmbd = 2048, c.nHead = 32, c.nHeadKv = 8, c.nFf = 8192;
      c.nVocab = 32768, c.nLayer = 2;
    } else if (f == "fullvocab") {
      c.nVocab = 128256;
    } else if (f == "nochk") {
      c.checkpoints = false;
    } else if (f == "embd") {
      c.embdInput = true;
    } else if (f == "all") {
      c.allLogits = true;
    }
  }
  c.kvSize = std::max<int64_t>(c.kvSize, c.nPast + c.nTokens);
  c.nKv = std::min<int64_t>(
      c.kvSize,
      std::max<int64_t>(256, (c.nPast + c.nTokens + 255) / 256 * 256));
  gCheckpoints = c.checkpoints;
  ggml_backend_reg_t reg = ggml_backend_load(argv[1]);
  if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
    std::fprintf(stderr, "cannot load %s\n", argv[1]);
    return 1;
  }
  ggml_backend_t hipep =
      ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
  ggml_backend_t cpu = ggml_backend_cpu_init();

  // The CPU reference always keeps the intermediates (they do not change its
  // result) and saves them as <name>.cpu.bin for comparison with tensors the
  // backend dumps (GGML_HIPEP_DUMP_INPUT).
  gCheckpoints = true;
  RunResult ref = runOn(cpu, c);
  for (const auto &[name, data] : ref.tensors)
    if (FILE *f = std::fopen((name + ".cpu.bin").c_str(), "wb")) {
      std::fwrite(data.data(), sizeof(float), data.size(), f);
      std::fclose(f);
    }
  gCheckpoints = c.checkpoints;
  RunResult got = runOn(hipep, c);
  if (!c.checkpoints) {
    // Compare only tensors both runs produced.
    std::vector<std::pair<std::string, std::vector<float>>> keep;
    for (auto &t : ref.tensors)
      for (auto &g : got.tensors)
        if (g.first == t.first)
          keep.push_back(t);
    ref.tensors = keep;
  }
  int bad = 0;
  for (size_t i = 0; i < ref.tensors.size(); ++i) {
    const auto &[name, a] = ref.tensors[i];
    const auto &b = got.tensors[i].second;
    double err = 0, norm = 0, maxAbs = 0;
    for (size_t j = 0; j < a.size(); ++j) {
      const double diff = static_cast<double>(a[j]) - b[j];
      err += diff * diff;
      norm += static_cast<double>(a[j]) * a[j];
      maxAbs = std::max(maxAbs, std::fabs(diff));
    }
    const double nmse = norm > 0 ? err / norm : err;
    // The CPU backend quantizes activations to Q8 for quantized weights while
    // MatMulNBits keeps them in f16, so quantized runs get a looser bound.
    const double bound = ggml_is_quantized(c.wtype) ? 2e-3 : 1e-4;
    const bool ok = nmse < bound && std::isfinite(nmse);
    bad += ok ? 0 : 1;
    std::printf("%-14s n=%-7zu nmse=%.3e max_abs=%.3e %s\n", name.c_str(),
                a.size(), nmse, maxAbs, ok ? "OK" : "FAIL");
  }
  ggml_backend_free(hipep);
  ggml_backend_free(cpu);
  return bad ? 1 : 0;
}
