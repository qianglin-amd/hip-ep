/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "GgmlImporter.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace hipep_ggml {
namespace {

enum class Elt { F32, F16, BF16, I32, I64, I16, I8, U8 };

bool eltOf(ggml_type type, Elt &elt) {
  switch (type) {
  case GGML_TYPE_F32:
    elt = Elt::F32;
    return true;
  case GGML_TYPE_F16:
    elt = Elt::F16;
    return true;
  case GGML_TYPE_BF16:
    elt = Elt::BF16;
    return true;
  case GGML_TYPE_I32:
    elt = Elt::I32;
    return true;
  case GGML_TYPE_I64:
    elt = Elt::I64;
    return true;
  case GGML_TYPE_I16:
    elt = Elt::I16;
    return true;
  case GGML_TYPE_I8:
    elt = Elt::I8;
    return true;
  default:
    return false;
  }
}

bool isFloat(ggml_type type) {
  return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
         type == GGML_TYPE_BF16;
}

const char *eltName(Elt elt) {
  switch (elt) {
  case Elt::F32:
    return "f32";
  case Elt::F16:
    return "f16";
  case Elt::BF16:
    return "bf16";
  case Elt::I32:
    return "i32";
  case Elt::I64:
    return "i64";
  case Elt::I16:
    return "i16";
  case Elt::I8:
    return "i8";
  case Elt::U8:
    return "ui8";
  }
  return "?";
}

bool isQuant(ggml_type type) {
  QuantLayout layout;
  return quantLayout(type, layout);
}

// A block-quantized weight consumed directly by MUL_MAT / GET_ROWS.
bool quantWeightUsable(const ggml_tensor *t, std::string *reason) {
  QuantLayout layout;
  if (!quantLayout(t->type, layout))
    return false;
  if (t->view_src || !ggml_is_contiguous(t)) {
    if (reason)
      *reason = "view of a block-quantized tensor";
    return false;
  }
  if (t->ne[2] != 1 || t->ne[3] != 1 || t->ne[0] % layout.blockSize != 0) {
    if (reason)
      *reason = "block-quantized weight shape";
    return false;
  }
  return true;
}

using Shape = std::vector<int64_t>;

// ggml ne[0] is innermost; ONNX shapes are outermost-first.
Shape shape4(const ggml_tensor *t) {
  return {t->ne[3], t->ne[2], t->ne[1], t->ne[0]};
}

int64_t product(const Shape &s) {
  return std::accumulate(s.begin(), s.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

float opParamF32(const ggml_tensor *t, int i) {
  float v;
  std::memcpy(&v, &t->op_params[i], sizeof(float));
  return v;
}

std::string fmtFloat(double v) {
  if (std::isinf(v))
    return v > 0 ? "0x7F800000" : "0xFF800000";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9e", v);
  return buf;
}

std::string joinI64(const Shape &v) {
  std::string r;
  for (size_t i = 0; i < v.size(); ++i)
    r += (i ? ", " : "") + std::to_string(v[i]);
  return r;
}

// A view expressed over its storage root: the root is flattened, sliced from
// the view offset, reshaped into nested stride blocks, sliced per block and
// transposed back into ggml dimension order.
struct ViewPlan {
  bool contiguous = false;
  int64_t offset = 0; // elements
  struct Dim {
    int k;
    int64_t n, stride;
  };
  std::vector<Dim> dims; // sorted by stride, outermost first
};

bool planView(const ggml_tensor *t, ViewPlan &plan, std::string *reason) {
  const ggml_tensor *root = storageRoot(t);
  if (ggml_blck_size(root->type) != 1) {
    if (reason)
      *reason = "view of a block-quantized tensor";
    return false;
  }
  const int64_t es = static_cast<int64_t>(ggml_type_size(root->type));
  if (static_cast<int64_t>(t->view_offs) % es != 0) {
    if (reason)
      *reason = "view offset is not element aligned";
    return false;
  }
  plan.offset = static_cast<int64_t>(t->view_offs) / es;
  if (plan.offset + 1 > ggml_nelements(root) && ggml_nelements(t) > 0) {
    if (reason)
      *reason = "view offset outside its storage";
    return false;
  }
  if (ggml_is_contiguous(t)) {
    plan.contiguous = true;
    return true;
  }
  plan.dims.clear();
  for (int k = 0; k < 4; ++k) {
    if (t->ne[k] <= 1)
      continue;
    if (static_cast<int64_t>(t->nb[k]) % es != 0 || t->nb[k] == 0) {
      if (reason)
        *reason = "broadcast or unaligned view stride";
      return false;
    }
    plan.dims.push_back({k, t->ne[k], static_cast<int64_t>(t->nb[k]) / es});
  }
  std::stable_sort(plan.dims.begin(), plan.dims.end(),
                   [](const ViewPlan::Dim &a, const ViewPlan::Dim &b) {
                     return a.stride > b.stride;
                   });
  for (size_t i = 0; i + 1 < plan.dims.size(); ++i) {
    const auto &outer = plan.dims[i];
    const auto &inner = plan.dims[i + 1];
    if (outer.stride % inner.stride != 0 ||
        inner.n * inner.stride > outer.stride) {
      if (reason)
        *reason = "view strides do not nest";
      return false;
    }
  }
  return true;
}

struct Val {
  std::string id;
  Shape shape;
  Elt elt = Elt::F32;
  bool none = false;
};

// Emits ONNX-dialect ops in MLIR generic form.
class Builder {
public:
  static std::string type(const Val &v) {
    if (v.none)
      return "none";
    return type(v.shape, v.elt);
  }
  static std::string type(const Shape &s, Elt e) {
    std::string r = "tensor<";
    for (int64_t d : s)
      r += std::to_string(d) + "x";
    return r + eltName(e) + ">";
  }

  Val emit(const std::string &op, const std::vector<Val> &ins,
           const std::string &attrs, Shape shape, Elt elt) {
    Val out{fresh(), std::move(shape), elt};
    body_ << "    " << out.id << " = \"onnx." << op << "\"(";
    for (size_t i = 0; i < ins.size(); ++i)
      body_ << (i ? ", " : "") << ins[i].id;
    body_ << ")";
    if (!attrs.empty())
      body_ << " {" << attrs << "}";
    body_ << " : (";
    for (size_t i = 0; i < ins.size(); ++i)
      body_ << (i ? ", " : "") << type(ins[i]);
    body_ << ") -> " << type(out) << "\n";
    return out;
  }

  Val none() {
    Val v{fresh(), {}, Elt::F32, true};
    body_ << "    " << v.id << " = \"onnx.NoValue\"() {value} : () -> none\n";
    return v;
  }

  Val constI64(const Shape &values) {
    Shape shape{static_cast<int64_t>(values.size())};
    std::string key = joinI64(values);
    auto it = i64Consts_.find(key);
    if (it != i64Consts_.end())
      return it->second;
    Val v{fresh(), shape, Elt::I64};
    body_ << "    " << v.id << " = \"onnx.Constant\"() {value = dense<[" << key
          << "]> : " << type(v) << "} : () -> " << type(v) << "\n";
    i64Consts_.emplace(key, v);
    return v;
  }

  Val splat(double value, Elt elt, const Shape &shape) {
    Val v{fresh(), shape, elt};
    std::string lit = (elt == Elt::I32 || elt == Elt::I64)
                          ? std::to_string(static_cast<int64_t>(value))
                          : fmtFloat(value);
    body_ << "    " << v.id << " = \"onnx.Constant\"() {value = dense<" << lit
          << "> : " << type(v) << "} : () -> " << type(v) << "\n";
    return v;
  }

  Val reshape(const Val &v, const Shape &shape) {
    if (v.shape == shape)
      return v;
    return emit("Reshape", {v, constI64(shape)}, "", shape, v.elt);
  }

  Val transpose(const Val &v, const std::vector<int64_t> &perm) {
    bool identity = true;
    for (size_t i = 0; i < perm.size(); ++i)
      identity &= perm[i] == static_cast<int64_t>(i);
    if (identity)
      return v;
    Shape out;
    for (int64_t p : perm)
      out.push_back(v.shape[p]);
    return emit("Transpose", {v}, "perm = [" + joinI64(perm) + "]", out, v.elt);
  }

  Val slice(const Val &v, const Shape &starts, const Shape &ends,
            const Shape &axes) {
    Shape out = v.shape;
    bool trivial = true;
    for (size_t i = 0; i < axes.size(); ++i) {
      out[axes[i]] = ends[i] - starts[i];
      trivial &= starts[i] == 0 && ends[i] == v.shape[axes[i]];
    }
    if (trivial)
      return v;
    return emit("Slice", {v, constI64(starts), constI64(ends), constI64(axes)},
                "", out, v.elt);
  }

  Val pad1d(const Val &v, int64_t after) {
    if (after <= 0)
      return v;
    Val n = none();
    return emit("Pad", {v, constI64({0, after}), n, n}, "mode = \"constant\"",
                {v.shape[0] + after}, v.elt);
  }

  Val cast(const Val &v, Elt to) {
    if (v.elt == to)
      return v;
    return emit("Cast", {v},
                std::string("saturate = 1 : si64, to = ") + eltName(to),
                v.shape, to);
  }

  std::string body() const { return body_.str(); }

private:
  std::string fresh() { return "%v" + std::to_string(next_++); }

  std::ostringstream body_;
  int next_ = 0;
  std::unordered_map<std::string, Val> i64Consts_;
};

bool wholeCoverage(const ggml_tensor *node, const ggml_tensor *root) {
  return node->view_offs == 0 && node->type == root->type &&
         ggml_nelements(node) == ggml_nelements(root) &&
         ggml_is_contiguous(node);
}

bool srcLowerable(const ggml_tensor *t, std::string *reason) {
  if (!t)
    return true;
  if (!t->view_src) {
    if (!ggml_is_contiguous(t)) {
      if (reason)
        *reason = "non-contiguous storage tensor";
      return false;
    }
    return true;
  }
  ViewPlan plan;
  return planView(t, plan, reason);
}

bool sameOrOne(int64_t b, int64_t a) { return b == a || b == 1; }

class Importer {
public:
  Importer(ggml_cgraph *graph, int begin, int end, Segment &seg)
      : graph_(graph), begin_(begin), end_(end), seg_(seg) {}

  bool run(const std::vector<const ggml_tensor *> &outputs,
           std::string &error) {
    const int n = ggml_graph_n_nodes(graph_);
    for (int i = 0; i < n; ++i) {
      ggml_tensor *node = ggml_graph_node(graph_, i);
      index_[node] = i;
      for (ggml_tensor *src : node->src)
        if (src)
          ++uses_[src];
    }
    std::unordered_set<const ggml_tensor *> wanted(outputs.begin(),
                                                   outputs.end());
    planNormFusion(wanted);

    for (int i = begin_; i < end_ && error_.empty(); ++i) {
      ggml_tensor *node = ggml_graph_node(graph_, i);
      if (classifyNode(node) != NodeKind::Functional)
        continue;
      if (fusedNorms_.count(node))
        continue;
      lowerNode(node);
    }
    if (!error_.empty()) {
      error = error_;
      return false;
    }

    std::vector<Val> results;
    std::unordered_map<std::string, size_t> resultById;
    for (const ggml_tensor *root : outputs) {
      auto it = storage_.find(root);
      if (it == storage_.end() || !(it->second.produced || it->second.dirty))
        continue;
      Elt elt;
      if (!eltOf(root->type, elt)) {
        error = "output has an unsupported type";
        return false;
      }
      Val v = b_.cast(b_.reshape(it->second.val, shape4(root)), elt);
      auto dup = resultById.find(v.id);
      if (dup != resultById.end()) {
        seg_.outputCopies.push_back({root, dup->second});
        continue;
      }
      resultById.emplace(v.id, results.size());
      results.push_back(v);
      seg_.outputs.push_back(root);
    }
    if (results.empty()) {
      seg_.mlir.clear();
      return true;
    }

    std::ostringstream m;
    m << "module {\n  func.func @main_graph(";
    for (size_t i = 0; i < args_.size(); ++i)
      m << (i ? ", " : "") << args_[i].id << ": " << Builder::type(args_[i])
        << " {onnx.name = \"in" << i << "\"}";
    m << ") -> (";
    for (size_t i = 0; i < results.size(); ++i)
      m << (i ? ", " : "") << Builder::type(results[i]) << " {onnx.name = \"out"
        << i << "\"}";
    m << ") {\n" << b_.body() << "    \"onnx.Return\"(";
    for (size_t i = 0; i < results.size(); ++i)
      m << (i ? ", " : "") << results[i].id;
    m << ") : (";
    for (size_t i = 0; i < results.size(); ++i)
      m << (i ? ", " : "") << Builder::type(results[i]);
    m << ") -> ()\n  }\n  \"onnx.EntryPoint\"() {func = @main_graph} : () "
         "-> ()\n}\n";
    seg_.mlir = m.str();
    return true;
  }

private:
  struct Storage {
    Val val;
    bool produced = false;
    bool dirty = false;
    int version = 0;
  };

  bool fail(const std::string &msg, const ggml_tensor *node = nullptr) {
    if (error_.empty()) {
      error_ = msg;
      if (node)
        error_ += std::string(" (node '") + node->name + "', op " +
                  ggml_op_desc(node) + ")";
    }
    return false;
  }

  bool inSegment(const ggml_tensor *t) const {
    auto it = index_.find(t);
    return it != index_.end() && it->second >= begin_ && it->second < end_;
  }

  // RMS_NORM followed by MUL with a 1-D weight becomes one
  // SimplifiedLayerNormalization when the norm has no other reader.
  void planNormFusion(const std::unordered_set<const ggml_tensor *> &wanted) {
    for (int i = begin_; i < end_; ++i) {
      ggml_tensor *node = ggml_graph_node(graph_, i);
      if (node->op != GGML_OP_MUL || node->view_src)
        continue;
      ggml_tensor *norm = node->src[0];
      ggml_tensor *w = node->src[1];
      if (!norm || norm->op != GGML_OP_RMS_NORM || norm->view_src ||
          !inSegment(norm) || uses_[norm] != 1 || wanted.count(norm) ||
          (norm->flags & GGML_TENSOR_FLAG_OUTPUT))
        continue;
      if (w->type != GGML_TYPE_F32 || w->ne[0] != norm->ne[0] ||
          w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1)
        continue;
      fusedNorms_.insert(norm);
      fusedMuls_.insert(node);
    }
  }

  Val addInput(const Shape &shape, Elt elt, const SegmentInput &binding) {
    Val v{"%arg" + std::to_string(args_.size()), shape, elt};
    args_.push_back(v);
    seg_.inputs.push_back(binding);
    return v;
  }

  Storage *storageFor(const ggml_tensor *root) {
    auto it = storage_.find(root);
    if (it != storage_.end())
      return &it->second;
    Elt elt;
    if (!eltOf(root->type, elt)) {
      fail(std::string("unsupported storage type ") +
               ggml_type_name(root->type),
           root);
      return nullptr;
    }
    if (!ggml_is_contiguous(root)) {
      fail("non-contiguous storage tensor", root);
      return nullptr;
    }
    SegmentInput in;
    in.tensor = root;
    Storage st;
    st.val = addInput(shape4(root), elt, in);
    return &storage_.emplace(root, st).first->second;
  }

  // Packed values [rows, blocks, blob] and fp16 scales [rows, blocks] of a
  // repacked block-quantized weight.
  std::pair<Val, Val> quantOperands(const ggml_tensor *w) {
    auto it = quant_.find(w);
    if (it != quant_.end())
      return it->second;
    QuantLayout layout;
    quantLayout(w->type, layout);
    const int64_t rows = w->ne[1], blocks = w->ne[0] / layout.blockSize;
    SegmentInput data, scales;
    data.kind = SegmentInput::Kind::QuantData;
    data.tensor = w;
    scales.kind = SegmentInput::Kind::QuantScales;
    scales.tensor = w;
    Val d = addInput({rows, blocks, layout.blob}, Elt::U8, data);
    Val s = addInput({rows, blocks}, Elt::F16, scales);
    return quant_.emplace(w, std::make_pair(d, s)).first->second;
  }

  Val valueOf(const ggml_tensor *t) {
    const ggml_tensor *root = storageRoot(t);
    if (isQuant(root->type)) {
      fail("block-quantized tensor used as a dense value", t);
      return {};
    }
    Storage *st = storageFor(root);
    if (!st)
      return {};
    if (t == root)
      return st->val;
    auto key = std::make_pair(t, st->version);
    auto it = viewMemo_.find(key);
    if (it != viewMemo_.end())
      return it->second;
    Val v = lowerView(st->val, root, t);
    viewMemo_.emplace(key, v);
    return v;
  }

  Val lowerView(const Val &storage, const ggml_tensor *root,
                const ggml_tensor *t) {
    ViewPlan plan;
    std::string reason;
    if (!planView(t, plan, &reason)) {
      fail(reason, t);
      return {};
    }
    const int64_t rootN = ggml_nelements(root);
    const int64_t nel = ggml_nelements(t);
    Val flat = b_.reshape(storage, {rootN});
    if (plan.contiguous || plan.dims.empty()) {
      Val v = b_.slice(flat, {plan.offset}, {plan.offset + nel}, {0});
      return b_.reshape(v, shape4(t));
    }
    const auto &dims = plan.dims;
    const size_t m = dims.size();
    const int64_t inner = dims.back().stride;
    Shape blocks{dims[0].n};
    for (size_t i = 1; i < m; ++i)
      blocks.push_back(dims[i - 1].stride / dims[i].stride);
    if (inner > 1)
      blocks.push_back(inner);
    const int64_t span = dims[0].n * dims[0].stride;
    Val v = b_.pad1d(flat, plan.offset + span - rootN);
    v = b_.slice(v, {plan.offset}, {plan.offset + span}, {0});
    v = b_.reshape(v, blocks);
    Shape starts, ends, axes;
    for (size_t i = 1; i < m; ++i) {
      if (blocks[i] != dims[i].n) {
        starts.push_back(0);
        ends.push_back(dims[i].n);
        axes.push_back(static_cast<int64_t>(i));
      }
    }
    if (inner > 1) {
      starts.push_back(0);
      ends.push_back(1);
      axes.push_back(static_cast<int64_t>(m));
    }
    if (!axes.empty())
      v = b_.slice(v, starts, ends, axes);
    Shape sized;
    for (const auto &d : dims)
      sized.push_back(d.n);
    v = b_.reshape(v, sized);
    std::vector<int64_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t c) {
      return dims[a].k > dims[c].k;
    });
    v = b_.transpose(v, perm);
    return b_.reshape(v, shape4(t));
  }

  void lowerNode(const ggml_tensor *node) {
    Val r;
    switch (node->op) {
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
      r = lowerBinary(node);
      break;
    case GGML_OP_SCALE:
      r = lowerScale(node);
      break;
    case GGML_OP_RMS_NORM:
      r = lowerRmsNorm(node, nullptr);
      break;
    case GGML_OP_MUL_MAT:
      r = lowerMulMat(node);
      break;
    case GGML_OP_GET_ROWS:
      r = lowerGetRows(node);
      break;
    case GGML_OP_ROPE:
      r = lowerRope(node);
      break;
    case GGML_OP_SOFT_MAX:
      r = lowerSoftMax(node);
      break;
    case GGML_OP_GLU:
      r = lowerGlu(node);
      break;
    case GGML_OP_UNARY:
      r = lowerUnary(node);
      break;
    case GGML_OP_CPY:
    case GGML_OP_DUP:
    case GGML_OP_CONT:
      r = lowerCopy(node);
      break;
    default:
      fail("unsupported op", node);
      return;
    }
    if (!error_.empty())
      return;

    const ggml_tensor *root = storageRoot(node);
    if (root == node) {
      Storage st;
      st.val = r;
      st.produced = true;
      storage_[node] = st;
      return;
    }
    // In-place op covering its whole storage: the storage takes the new value.
    Storage *st = storageFor(root);
    if (!st)
      return;
    Elt elt;
    eltOf(root->type, elt);
    st->val = b_.cast(b_.reshape(r, shape4(root)), elt);
    st->dirty = !st->produced;
    ++st->version;
  }

  Elt resultElt(const ggml_tensor *node) {
    Elt e = Elt::F32;
    if (!eltOf(node->type, e))
      fail("unsupported result type", node);
    return e;
  }

  Val lowerBinary(const ggml_tensor *node) {
    if (fusedMuls_.count(node))
      return lowerRmsNorm(node->src[0], node->src[1]);
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *c = node->src[1];
    for (int k = 0; k < 4; ++k)
      if (a->ne[k] != node->ne[k] || !sameOrOne(c->ne[k], node->ne[k])) {
        fail("binary op broadcast is not NumPy compatible", node);
        return {};
      }
    const Elt elt = resultElt(node);
    // Mixed operand types compute in f32 and round once, as ggml does.
    const Elt ct = a->type == c->type ? elt : Elt::F32;
    Val va = b_.cast(valueOf(a), ct);
    Val vc = b_.cast(valueOf(c), ct);
    const char *op = node->op == GGML_OP_ADD   ? "Add"
                     : node->op == GGML_OP_SUB ? "Sub"
                     : node->op == GGML_OP_MUL ? "Mul"
                                               : "Div";
    return b_.cast(b_.emit(op, {va, vc}, "", shape4(node), ct), elt);
  }

  Val lowerScale(const ggml_tensor *node) {
    const float s = opParamF32(node, 0);
    const float bias = opParamF32(node, 1);
    Elt elt = resultElt(node);
    Val x = b_.cast(valueOf(node->src[0]), elt);
    Val r = x;
    if (s != 1.0f)
      r = b_.emit("Mul", {r, b_.splat(s, elt, {1})}, "", shape4(node), elt);
    if (bias != 0.0f)
      r = b_.emit("Add", {r, b_.splat(bias, elt, {1})}, "", shape4(node), elt);
    return b_.reshape(r, shape4(node));
  }

  // RMS_NORM (optionally fused with the following MUL by `weight`).
  Val lowerRmsNorm(const ggml_tensor *norm, const ggml_tensor *weight) {
    const float eps = opParamF32(norm, 0);
    Val x = b_.cast(valueOf(norm->src[0]), Elt::F32);
    const int64_t d = norm->ne[0];
    Val w = weight ? b_.reshape(b_.cast(valueOf(weight), Elt::F32), {d})
                   : b_.splat(1.0, Elt::F32, {d});
    return b_.emit("Custom", {x, w},
                   "function_name = \"SimplifiedLayerNormalization\", "
                   "axis = -1 : si64, domain_name = \"\", epsilon = " +
                       fmtFloat(eps) + " : f32, stash_type = 1 : si64",
                   shape4(norm), Elt::F32);
  }

  Val lowerQuantMulMat(const ggml_tensor *node) {
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *c = node->src[1];
    QuantLayout layout;
    quantLayout(a->type, layout);
    const int64_t K = a->ne[0], N = a->ne[1];
    const int64_t rows = c->ne[1] * c->ne[2] * c->ne[3];
    auto [packed, scales] = quantOperands(a);
    Val x = b_.reshape(b_.cast(valueOf(c), Elt::F16), {1, rows, K});
    Val y =
        b_.emit("Custom", {x, packed, scales},
                "function_name = \"MatMulNBits\", K = " + std::to_string(K) +
                    " : si64, N = " + std::to_string(N) +
                    " : si64, accuracy_level = 0 : si64, bits = " +
                    std::to_string(layout.bits) + " : si64, block_size = " +
                    std::to_string(layout.blockSize) +
                    " : si64, domain_name = \"com.microsoft\"",
                {1, rows, N}, Elt::F16);
    return b_.cast(b_.reshape(y, shape4(node)), resultElt(node));
  }

  Val lowerMulMat(const ggml_tensor *node) {
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *c = node->src[1];
    if (isQuant(a->type))
      return lowerQuantMulMat(node);
    Elt ta;
    if (!eltOf(a->type, ta) || !isFloat(a->type) || !isFloat(c->type)) {
      fail("unsupported MUL_MAT operand types", node);
      return {};
    }
    // Weight-typed compute: f16/bf16 weights multiply f16/bf16 activations.
    const Elt ct = (ta == Elt::F16 || ta == Elt::BF16) ? ta : Elt::F32;
    const int64_t K = a->ne[0], N = a->ne[1], M = c->ne[1];
    const int64_t B2a = a->ne[2], B3a = a->ne[3];
    const int64_t B2b = c->ne[2], B3b = c->ne[3];
    Val R;
    if (B2a == 1 && B3a == 1) {
      Val A = b_.cast(valueOf(a), ct);
      Val B = b_.cast(valueOf(c), ct);
      const int64_t rows = M * B2b * B3b;
      Val A2 = b_.reshape(A, {N, K});
      Val B2 = b_.reshape(B, {rows, K});
      Val G = b_.emit("Gemm", {B2, A2},
                      "alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, "
                      "transA = 0 : si64, transB = 1 : si64",
                      {rows, N}, ct);
      R = b_.reshape(G, {B3b, B2b, M, N});
    } else {
      if (B3a != B3b || B2b % B2a != 0) {
        fail("MUL_MAT broadcast pattern is not supported", node);
        return {};
      }
      // Activation-by-activation products (attention scores and the
      // probability-weighted values) run in f32: ggml produces them in f32,
      // and f16 results overflow or round attention logits.
      Val A = b_.cast(valueOf(a), Elt::F32);
      Val B = b_.cast(valueOf(c), Elt::F32);
      // Query heads sharing one KV head fold into the M dimension.
      const int64_t r2 = B2b / B2a;
      Val Bm = b_.reshape(B, {B3b, B2a, r2 * M, K});
      Val At = b_.transpose(A, {0, 1, 3, 2});
      Val P = b_.emit("MatMul", {Bm, At}, "", {B3b, B2a, r2 * M, N}, Elt::F32);
      R = b_.reshape(P, {B3b, B2b, M, N});
    }
    return b_.cast(R, resultElt(node));
  }

  Val lowerGetRows(const ggml_tensor *node) {
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *idx = node->src[1];
    if (a->ne[2] != 1 || a->ne[3] != 1) {
      fail("GET_ROWS on batched data is not supported", node);
      return {};
    }
    const int64_t count = idx->ne[0] * idx->ne[1] * idx->ne[2];
    Val ids = b_.reshape(b_.cast(valueOf(idx), Elt::I64), {count});
    QuantLayout layout;
    if (quantLayout(a->type, layout)) {
      auto [packed, scales] = quantOperands(a);
      const int64_t rowBytes = packed.shape[1] * packed.shape[2];
      Val data = b_.reshape(packed, {a->ne[1], rowBytes});
      Val g = b_.emit(
          "Custom", {data, ids, scales},
          "function_name = \"GatherBlockQuantized\", bits = " +
              std::to_string(layout.bits) +
              " : si64, block_size = " + std::to_string(layout.blockSize) +
              " : si64, domain_name = \"com.microsoft\", gather_axis = 0 : "
              "si64, quantize_axis = 1 : si64",
          {count, a->ne[0]}, Elt::F16);
      return b_.cast(b_.reshape(g, shape4(node)), resultElt(node));
    }
    Val data = b_.reshape(valueOf(a), {a->ne[1], a->ne[0]});
    Val g = b_.emit("Gather", {data, ids}, "axis = 0 : si64", {count, a->ne[0]},
                    data.elt);
    return b_.cast(b_.reshape(g, shape4(node)), resultElt(node));
  }

  std::pair<Val, Val> ropeTables(const RopeTableSpec &spec, Elt elt) {
    for (size_t i = 0; i < seg_.ropeTables.size(); ++i)
      if (seg_.ropeTables[i] == spec)
        return ropeArgs_[i];
    const int index = static_cast<int>(seg_.ropeTables.size());
    seg_.ropeTables.push_back(spec);
    const Shape shape{spec.rows, spec.nDims / 2};
    SegmentInput cosIn, sinIn;
    cosIn.kind = SegmentInput::Kind::RopeCos;
    cosIn.rope = index;
    sinIn.kind = SegmentInput::Kind::RopeSin;
    sinIn.rope = index;
    Val c = addInput(shape, elt, cosIn);
    Val s = addInput(shape, elt, sinIn);
    ropeArgs_.push_back({c, s});
    return ropeArgs_.back();
  }

  Val lowerRope(const ggml_tensor *node) {
    const ggml_tensor *x = node->src[0];
    const ggml_tensor *pos = node->src[1];
    const int mode = node->op_params[2];
    const int64_t d = x->ne[0], H = x->ne[1], T = x->ne[2];
    RopeTableSpec spec;
    spec.nDims = node->op_params[1];
    spec.nCtxOrig = node->op_params[4];
    spec.freqBase = opParamF32(node, 5);
    spec.freqScale = opParamF32(node, 6);
    spec.extFactor = opParamF32(node, 7);
    spec.attnFactor = opParamF32(node, 8);
    spec.betaFast = opParamF32(node, 9);
    spec.betaSlow = opParamF32(node, 10);
    spec.truncate = ggml_rope_get_truncate(node);
    spec.freqFactors = node->src[2];
    spec.f16 = x->type == GGML_TYPE_F16;
    spec.rows = std::max<int64_t>(spec.nCtxOrig, 32768);
    const Elt elt = spec.f16 ? Elt::F16 : Elt::F32;

    Val x3 = b_.reshape(b_.cast(valueOf(x), elt), {1, T, H * d});
    Val p = b_.reshape(b_.cast(valueOf(pos), Elt::I64), {1, T});
    auto [cosV, sinV] = ropeTables(spec, elt);
    Val r = b_.emit(
        "Custom", {x3, p, cosV, sinV},
        "function_name = \"RotaryEmbedding\", domain_name = \"com.microsoft\", "
        "interleaved = " +
            std::to_string(mode == GGML_ROPE_TYPE_NEOX ? 0 : 1) +
            " : si64, num_heads = " + std::to_string(H) +
            " : si64, rotary_embedding_dim = " + std::to_string(spec.nDims) +
            " : si64",
        {1, T, H * d}, elt);
    return b_.cast(b_.reshape(r, shape4(node)), resultElt(node));
  }

  Val lowerSoftMax(const ggml_tensor *node) {
    const float scale = opParamF32(node, 0);
    const ggml_tensor *mask = node->src[1];
    Val s = b_.cast(valueOf(node->src[0]), Elt::F32);
    if (scale != 1.0f)
      s = b_.emit("Mul", {s, b_.splat(scale, Elt::F32, {1})}, "", s.shape,
                  Elt::F32);
    if (mask) {
      Val m = valueOf(mask);
      if (mask->ne[1] > node->ne[1])
        m = b_.slice(m, {0}, {node->ne[1]}, {2});
      m = b_.cast(m, Elt::F32);
      s = b_.emit("Add", {s, m}, "", shape4(node), Elt::F32);
    }
    Val r = b_.emit("Softmax", {s}, "axis = -1 : si64", shape4(node), Elt::F32);
    return b_.cast(r, resultElt(node));
  }

  Val silu(const Val &x) {
    Val sig = b_.emit("Sigmoid", {x}, "", x.shape, x.elt);
    return b_.emit("Mul", {x, sig}, "", x.shape, x.elt);
  }

  Val lowerGlu(const ggml_tensor *node) {
    const bool swapped = node->op_params[1] != 0;
    Elt elt = resultElt(node);
    Val gate, up;
    if (node->src[1]) {
      gate = b_.cast(valueOf(node->src[0]), elt);
      up = b_.cast(valueOf(node->src[1]), elt);
    } else {
      Val x = b_.cast(valueOf(node->src[0]), elt);
      const int64_t nc = node->ne[0];
      Val lo = b_.slice(x, {0}, {nc}, {3});
      Val hi = b_.slice(x, {nc}, {2 * nc}, {3});
      gate = swapped ? hi : lo;
      up = swapped ? lo : hi;
    }
    return b_.emit("Mul", {silu(gate), up}, "", shape4(node), elt);
  }

  Val lowerUnary(const ggml_tensor *node) {
    Elt elt = resultElt(node);
    Val x = b_.cast(valueOf(node->src[0]), elt);
    switch (ggml_get_unary_op(node)) {
    case GGML_UNARY_OP_SILU:
      return silu(x);
    case GGML_UNARY_OP_NEG:
      return b_.emit("Neg", {x}, "", x.shape, elt);
    default:
      fail("unsupported unary op", node);
      return {};
    }
  }

  Val lowerCopy(const ggml_tensor *node) {
    Val x = b_.cast(valueOf(node->src[0]), resultElt(node));
    return b_.reshape(x, shape4(node));
  }

  ggml_cgraph *graph_;
  int begin_, end_;
  Segment &seg_;
  Builder b_;
  std::vector<Val> args_;
  std::vector<std::pair<Val, Val>> ropeArgs_;
  std::unordered_map<const ggml_tensor *, std::pair<Val, Val>> quant_;
  std::unordered_map<const ggml_tensor *, Storage> storage_;
  std::map<std::pair<const ggml_tensor *, int>, Val> viewMemo_;
  std::unordered_map<const ggml_tensor *, int> index_;
  std::unordered_map<const ggml_tensor *, int> uses_;
  std::unordered_set<const ggml_tensor *> fusedNorms_, fusedMuls_;
  std::string error_;
};

bool checkUnaryShape(const ggml_tensor *node, std::string *reason) {
  return srcLowerable(node->src[0], reason);
}

} // namespace

bool quantLayout(int ggmlType, QuantLayout &layout) {
  switch (static_cast<ggml_type>(ggmlType)) {
  case GGML_TYPE_Q4_0:
    layout = {4, 32, 16};
    return true;
  case GGML_TYPE_Q8_0:
    layout = {8, 32, 32};
    return true;
  default:
    return false;
  }
}

bool RopeTableSpec::operator==(const RopeTableSpec &o) const {
  return nDims == o.nDims && nCtxOrig == o.nCtxOrig && freqBase == o.freqBase &&
         freqScale == o.freqScale && extFactor == o.extFactor &&
         attnFactor == o.attnFactor && betaFast == o.betaFast &&
         betaSlow == o.betaSlow && truncate == o.truncate && f16 == o.f16 &&
         freqFactors == o.freqFactors && rows == o.rows;
}

const ggml_tensor *storageRoot(const ggml_tensor *t) {
  return t->view_src ? t->view_src : t;
}

// Kind by op and layout alone, ignoring whether the node is scheduled.
static NodeKind classifyOp(const ggml_tensor *node) {
  switch (node->op) {
  case GGML_OP_NONE:
  case GGML_OP_VIEW:
  case GGML_OP_RESHAPE:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
    return NodeKind::View;
  default:
    break;
  }
  if (!node->view_src)
    return NodeKind::Functional;
  if (node->op == GGML_OP_SET_ROWS)
    return NodeKind::SideEffect;
  return wholeCoverage(node, node->view_src) ? NodeKind::Functional
                                             : NodeKind::SideEffect;
}

NodeKind classifyNode(const ggml_tensor *node) {
  // ggml_build_forward_select keeps unselected branches in the graph without
  // the compute flag; ggml-alloc may hand their memory to live tensors.
  if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE))
    return NodeKind::Skip;
  return classifyOp(node);
}

bool canExecuteNode(const ggml_tensor *node, std::string *reason) {
  auto no = [&](const char *why) {
    if (reason)
      *reason = why;
    return false;
  };
  // supports_op is also asked before a graph is built, when no node carries
  // the compute flag yet, so capability is judged by op alone.
  const NodeKind kind = classifyOp(node);
  if (kind == NodeKind::View)
    return srcLowerable(node, reason);

  if (kind == NodeKind::SideEffect) {
    if (node->op == GGML_OP_SET_ROWS) {
      // src[0] = values, src[1] = row indices, src[2] = destination; the node
      // itself is a view of the destination.
      const ggml_tensor *idx = node->src[1];
      if (!isFloat(node->type) || !isFloat(node->src[0]->type))
        return no("SET_ROWS type");
      if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64)
        return no("SET_ROWS index type");
      return true;
    }
    if (node->op == GGML_OP_CPY || node->op == GGML_OP_DUP) {
      if (!isFloat(node->type) || !isFloat(node->src[0]->type))
        return no("in-place copy type");
      return true;
    }
    return no("partial in-place write");
  }

  // MUL_MAT and GET_ROWS read a block-quantized src[0] directly.
  const bool quantSrc0 =
      (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_GET_ROWS) &&
      isQuant(node->src[0]->type);
  if (quantSrc0 && !quantWeightUsable(node->src[0], reason))
    return false;
  for (int s = quantSrc0 ? 1 : 0; s < GGML_MAX_SRC; ++s) {
    const ggml_tensor *src = node->src[s];
    if (!src)
      continue;
    if (!srcLowerable(src, reason))
      return false;
    Elt e;
    if (!eltOf(storageRoot(src)->type, e))
      return no("source storage type");
  }
  Elt e;
  if (!eltOf(node->type, e))
    return no("result type");

  switch (node->op) {
  case GGML_OP_ADD:
  case GGML_OP_SUB:
  case GGML_OP_MUL:
  case GGML_OP_DIV: {
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *c = node->src[1];
    if (!isFloat(node->type) || !isFloat(a->type) || !isFloat(c->type))
      return no("binary op type");
    for (int k = 0; k < 4; ++k)
      if (a->ne[k] != node->ne[k] || !sameOrOne(c->ne[k], node->ne[k]))
        return no("binary op broadcast");
    return true;
  }
  case GGML_OP_SCALE:
  case GGML_OP_CONT:
  case GGML_OP_CPY:
  case GGML_OP_DUP:
    if (!isFloat(node->type) || !isFloat(node->src[0]->type))
      return no("copy type");
    return checkUnaryShape(node, reason);
  case GGML_OP_RMS_NORM:
    return node->src[0]->type == GGML_TYPE_F32 ? true : no("RMS_NORM type");
  case GGML_OP_MUL_MAT: {
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *c = node->src[1];
    if (quantSrc0)
      return isFloat(c->type) ? true : no("MUL_MAT activation type");
    if (!isFloat(a->type) || !isFloat(c->type))
      return no("MUL_MAT type");
    if (a->ne[2] == 1 && a->ne[3] == 1)
      return true;
    if (a->ne[3] != c->ne[3] || c->ne[2] % a->ne[2] != 0)
      return no("MUL_MAT broadcast");
    return true;
  }
  case GGML_OP_GET_ROWS: {
    const ggml_tensor *a = node->src[0];
    const ggml_tensor *idx = node->src[1];
    if ((!quantSrc0 && !isFloat(a->type)) || idx->type != GGML_TYPE_I32)
      return no("GET_ROWS type");
    if (a->ne[2] != 1 || a->ne[3] != 1)
      return no("GET_ROWS batched data");
    return true;
  }
  case GGML_OP_ROPE: {
    const int mode = node->op_params[2];
    const ggml_tensor *x = node->src[0];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX)
      return no("ROPE mode");
    if (node->op_params[15] != 0)
      return no("ROPE offset");
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16)
      return no("ROPE type");
    if (x->ne[3] != 1 || node->src[1]->ne[0] != x->ne[2])
      return no("ROPE shape");
    if (node->src[2] && node->src[2]->type != GGML_TYPE_F32)
      return no("ROPE frequency factor type");
    return true;
  }
  case GGML_OP_SOFT_MAX: {
    if (opParamF32(node, 1) != 0.0f)
      return no("SOFT_MAX ALiBi");
    if (node->src[2])
      return no("SOFT_MAX sinks");
    const ggml_tensor *mask = node->src[1];
    if (mask) {
      if (mask->ne[0] != node->ne[0] || mask->ne[1] < node->ne[1] ||
          !sameOrOne(mask->ne[2], node->ne[2]) ||
          !sameOrOne(mask->ne[3], node->ne[3]))
        return no("SOFT_MAX mask shape");
      if (!isFloat(mask->type))
        return no("SOFT_MAX mask type");
    }
    return true;
  }
  case GGML_OP_GLU:
    if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU)
      return no("GLU op");
    if (!isFloat(node->type))
      return no("GLU type");
    return true;
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(node)) {
    case GGML_UNARY_OP_SILU:
    case GGML_UNARY_OP_NEG:
      return isFloat(node->type) ? true : no("unary type");
    default:
      return no("unary op");
    }
  default:
    return no("op");
  }
}

std::vector<GraphStep> partitionGraph(ggml_cgraph *graph) {
  const int n = ggml_graph_n_nodes(graph);
  std::vector<NodeKind> kinds(n);
  std::unordered_map<const ggml_tensor *, int> index;
  std::unordered_map<const ggml_tensor *, std::vector<int>> readers;
  std::unordered_map<const ggml_tensor *, int32_t> refs;
  for (int i = 0; i < n; ++i) {
    const ggml_tensor *node = ggml_graph_node(graph, i);
    kinds[i] = classifyNode(node);
    index[node] = i;
    for (const ggml_tensor *src : node->src)
      if (src)
        ++refs[src];
    if (kinds[i] == NodeKind::View || kinds[i] == NodeKind::Skip)
      continue;
    for (const ggml_tensor *src : node->src)
      if (src)
        readers[storageRoot(src)].push_back(i);
  }

  // A scheduler split is a view of the whole graph and keeps its use counts:
  // a node used more often than this split references it is read by another
  // split, so its storage must be written back.
  std::unordered_set<const ggml_tensor *> usedElsewhere;
  for (int i = 0; i < n; ++i) {
    const ggml_tensor *node = ggml_graph_node(graph, i);
    const size_t pos = ggml_hash_find(&graph->visited_hash_set, node);
    if (pos == GGML_HASHSET_FULL ||
        !ggml_bitset_get(graph->visited_hash_set.used, pos))
      continue;
    auto it = refs.find(node);
    const int32_t local = it == refs.end() ? 0 : it->second;
    if (graph->use_counts[pos] > local)
      usedElsewhere.insert(storageRoot(node));
  }

  std::vector<GraphStep> steps;
  int i = 0;
  while (i < n) {
    if (kinds[i] == NodeKind::SideEffect) {
      GraphStep step;
      step.native = true;
      step.node = i;
      steps.push_back(step);
      ++i;
      continue;
    }
    int j = i;
    while (j < n && kinds[j] != NodeKind::SideEffect)
      ++j;
    GraphStep step;
    step.begin = i;
    step.end = j;
    std::unordered_set<const ggml_tensor *> seen;
    for (int k = i; k < j; ++k) {
      if (kinds[k] != NodeKind::Functional)
        continue;
      const ggml_tensor *root = storageRoot(ggml_graph_node(graph, k));
      if (!seen.insert(root).second)
        continue;
      auto at = index.find(root);
      const bool producedHere =
          at != index.end() && at->second >= i && at->second < j;
      bool needed = !producedHere || (root->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                    usedElsewhere.count(root);
      auto rd = readers.find(root);
      if (rd == readers.end() || rd->second.empty())
        needed = true;
      else
        for (int r : rd->second)
          needed |= r < i || r >= j;
      if (needed)
        step.outputs.push_back(root);
    }
    steps.push_back(std::move(step));
    i = j;
  }
  return steps;
}

bool importSegment(ggml_cgraph *graph, int begin, int end,
                   const std::vector<const ggml_tensor *> &outputs,
                   Segment &segment, std::string &error) {
  segment = Segment{};
  Importer importer(graph, begin, end, segment);
  return importer.run(outputs, error);
}

} // namespace hipep_ggml
