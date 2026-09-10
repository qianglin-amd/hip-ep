/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SwishConversion.cpp - onnx.Swish -> hip.swish --------------------===//
//
// Converts onnx.Swish (opset 24) to a single hip.swish op backed by a
// dedicated HIP kernel.
//
//   Before:
//     %y = "onnx.Swish"(%x) {alpha = 1.0 : f32}
//          : (tensor<...xT>) -> tensor<...xT>
//
//   After:
//     %init = tensor.empty(...) : tensor<...xT>
//     %y    = hip.swish(%ctx) ins(%x : tensor<...xT>)
//                             outs(%init : tensor<...xT>)
//                             {alpha = 1.0 : f64}
//
// Type T is one of f16/f32/f64 per the ONNX spec.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct SwishToHip : public mlir::RewritePattern {
  SwishToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Swish", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Swish expects 1 operand and 1 result");

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "ranked tensor required");
    auto ft = mlir::dyn_cast<mlir::FloatType>(xType.getElementType());
    if (!ft)
      return rewriter.notifyMatchFailure(op, "float element type required");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    double alphaVal = 1.0;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alphaVal = attr.getValueAsDouble();

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

    auto alphaAttr = rewriter.getF64FloatAttr(alphaVal);
    auto hipOp = mlir::hip::SwishOp::create(rewriter, loc, resultType, context,
                                            x, init, alphaAttr);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateSwishConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<SwishToHip>(ctx);
}

} // namespace hip
} // namespace mlir
