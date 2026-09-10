// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @swish_default(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %y = "onnx.Swish"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @swish_default
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>)
  // CHECK-NOT: onnx.Swish
  // CHECK: hip.swish(%[[CTX]]) ins(%[[X]] : tensor<3x4xf32>)
  // CHECK-NOT: alpha
  // CHECK-SAME: : tensor<3x4xf32>

  func.func @swish_alpha(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
    %y = "onnx.Swish"(%x) {alpha = 0.2 : f32} : (tensor<2x3xf16>) -> tensor<2x3xf16>
    return %y : tensor<2x3xf16>
  }

  // CHECK-LABEL: func.func @swish_alpha
  // CHECK-NOT: onnx.Swish
  // CHECK: hip.swish
  // CHECK-SAME: {alpha = 0.200000002980232{{[0-9]*}} : f64}

  func.func @swish_dynamic(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %y = "onnx.Swish"(%x) {alpha = 0.05 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @swish_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[X3:.*]]: tensor<?x?xf32>)
  // CHECK-NOT: onnx.Swish
  // CHECK: tensor.dim
  // CHECK: hip.swish(%[[CTX3]])
  // CHECK-SAME: {alpha = 0.0500000007450580{{[0-9]*}} : f64}
}
