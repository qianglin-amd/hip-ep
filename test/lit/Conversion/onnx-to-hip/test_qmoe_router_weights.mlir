// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a QMoE custom op carrying the optional ONNX v1.25 `router_weights`
// input (index 14) reaches hip.qmoe with that operand attached.
//
// The interesting part is the operand layout: `router_weights` is the last
// optional input, and everything between fc1_bias and fc3_zero_points is
// absent, so the conversion has to skip seven `none` operands without
// shifting router_weights into another optional slot.
//
// This is the shape the ONNX Runtime GenAI builder emits for MoE models that
// route and weight with different tensors: router_probs is the raw gate
// logits and router_weights is the softmax of those same logits.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(
      %input: tensor<1x128x2880xf16>,
      %router_probs: tensor<128x32xf16>,
      %router_weights: tensor<128x32xf16>) -> tensor<1x128x2880xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s,
                       %none, %none, %none, %none, %none, %none, %none, %router_weights) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_type = "swiglu",
      block_size = 128 : si64, expert_weight_bits = 4 : si64, k = 8 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      onnx_node_name = "QMoE_router_weights"
    } : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, none,
         tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>,
         none, none, none, none, none, none, none, tensor<128x32xf16>) -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:[^:]*]]: tensor<1x128x2880xf16>, %[[PROBS:[^:]*]]: tensor<128x32xf16>, %[[WEIGHTS:[^:]*]]: tensor<128x32xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x2880xf16>

  // The absent optionals print nothing, so router_weights following ins(...)
  // directly is what proves it did not land in an earlier optional slot.
  // CHECK: hip.qmoe(%[[CTX]]) ins(%[[INPUT]], %[[PROBS]],
  // CHECK-SAME: ) router_weights(%[[WEIGHTS]] : tensor<128x32xf16>) outs(%[[INIT]] : tensor<1x128x2880xf16>)
  // CHECK-SAME: k = 8
  // CHECK-NOT: onnx.Custom
}
