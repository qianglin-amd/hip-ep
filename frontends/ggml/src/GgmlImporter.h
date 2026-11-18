/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Converts a run of ggml graph nodes into an ONNX-dialect `main_graph` module
// that the HIP EP compiler consumes (the same input form the ONNX Runtime
// interface produces). See docs/design/ggml-frontend.md.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;

namespace hipep_ggml {

enum class NodeKind {
  Skip,       // not selected for computation (no GGML_TENSOR_FLAG_COMPUTE)
  View,       // VIEW / RESHAPE / PERMUTE / TRANSPOSE / NONE: no computation
  Functional, // lowered into a compiled segment
  SideEffect, // partial in-place write into a persistent tensor (native)
};

NodeKind classifyNode(const ggml_tensor *node);

// The tensor that owns `t`'s memory (`t` itself unless it is a view).
const ggml_tensor *storageRoot(const ggml_tensor *t);

// Whether the backend can execute `node`. Fills `reason` on rejection.
bool canExecuteNode(const ggml_tensor *node, std::string *reason);

// Host-built cos/sin table consumed by com.microsoft.RotaryEmbedding. Rows are
// positions, columns are the n_dims/2 rotation frequencies.
struct RopeTableSpec {
  int nDims = 0;
  int nCtxOrig = 0;
  float freqBase = 0, freqScale = 0, extFactor = 0, attnFactor = 0;
  float betaFast = 0, betaSlow = 0;
  bool truncate = false;
  bool f16 = false;
  const ggml_tensor *freqFactors = nullptr;
  int64_t rows = 0;

  bool operator==(const RopeTableSpec &o) const;
};

// Block-quantized weights live in the backend buffer repacked into a
// MatMulNBits layout with 32-element blocks, or 16 for requantized types whose
// ggml block is not a multiple of 32 (QuantRepack.cpp):
//   [rows * blocks * blob] packed values, [rows * blocks] fp16 scales, and
//   for zeroPoints layouts [rows * blocks] fp16 zero points;
//   value = (q - zero_point) * scale.
// 4-bit symmetric (Q4_0): low nibble first, implicit zero point 8.
// 4-bit with zero points (Q4_1, Q4_K): fp16 zero point per block.
// 8-bit symmetric (Q8_0, Q5_0 exactly; any other ggml quantization through
// dequantize and requantize): stored as q + 128, implicit zero point 128.
struct QuantLayout {
  int bits = 0;
  int64_t blockSize = 32;
  int64_t blob = 0; // packed bytes per block
  bool zeroPoints = false;
};
bool quantLayout(int ggmlType, QuantLayout &layout);

struct SegmentInput {
  enum class Kind {
    Tensor,
    RopeCos,
    RopeSin,
    QuantData,
    QuantScales,
    QuantZeroPoints
  };
  Kind kind = Kind::Tensor;
  const ggml_tensor *tensor = nullptr; // storage root (Tensor, Quant*)
  int rope = -1;                       // index into Segment::ropeTables
};

struct Segment {
  std::string mlir;
  std::vector<SegmentInput> inputs;         // main_graph arguments, in order
  std::vector<const ggml_tensor *> outputs; // storage roots, result order
  // Roots whose value equals an earlier result: copied from that result's
  // root after the run, since one compiled result cannot fill two buffers.
  std::vector<std::pair<const ggml_tensor *, size_t>> outputCopies;
  std::vector<RopeTableSpec> ropeTables;
};

// One execution step of a graph: a functional segment or a native node.
struct GraphStep {
  bool native = false;
  int node = -1;          // native node index
  int begin = 0, end = 0; // segment node range [begin, end)
  // Storage roots the segment must write: produced roots read outside the
  // segment, flagged as graph outputs, or never read (graph sinks), plus
  // persistent tensors the segment overwrites in place.
  std::vector<const ggml_tensor *> outputs;
};

// Splits a graph at its side-effect nodes.
std::vector<GraphStep> partitionGraph(ggml_cgraph *graph);

// Lowers nodes [begin, end) of `graph`. `outputs` lists the storage roots the
// caller needs written back; roots the segment does not produce are dropped
// from Segment::outputs. Returns false with `error` set when a node cannot be
// lowered.
bool importSegment(ggml_cgraph *graph, int begin, int end,
                   const std::vector<const ggml_tensor *> &outputs,
                   Segment &segment, std::string &error);

} // namespace hipep_ggml
