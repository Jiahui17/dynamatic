//===- CfShrinkBitWidth.h --------------------------------- -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --cf-shrink-bit-width pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/CfShrinkBitWidth.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {
class ShrinkBitwidthPass
    : public dynamatic::experimental::impl::CfShrinkBitWidthBase<
          ShrinkBitwidthPass> {
  unsigned targetBitwidth;

public:
  ShrinkBitwidthPass(const unsigned &targetBitwidth)
      : targetBitwidth(targetBitwidth) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShrinkBitwidthPass)

  StringRef getArgument() const final { return "shrink-bitwidth"; }
  StringRef getDescription() const final {
    return "Shrink bitwidth of ops and arguments";
  }

  void runDynamaticPass() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());

    module.walk([&](Operation *op) {
      // Update result types
      for (auto result : op->getResults()) {
        if (auto intType = result.getType().dyn_cast<IntegerType>()) {
          if (intType.getWidth() > targetBitwidth) {
            result.setType(builder.getIntegerType(targetBitwidth));
          }
        }
      }

      // Clip integer constants
      int64_t maxValue = (1 << targetBitwidth) - 1;
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
          int64_t val = intAttr.getInt();
          int64_t clippedVal = val < maxValue ? val : maxValue;
          auto newAttr = builder.getIntegerAttr(
              builder.getIntegerType(targetBitwidth), clippedVal);
          constOp.setValueAttr(newAttr);
        }
      }
    });

    // Update function arguments
    module.walk([&](func::FuncOp func) {
      auto funcType = func.getFunctionType();
      SmallVector<Type, 4> newInputs;
      for (Type ty : funcType.getInputs()) {
        if (auto intType = ty.dyn_cast<IntegerType>()) {
          if (intType.getWidth() > targetBitwidth)
            newInputs.push_back(builder.getIntegerType(targetBitwidth));
          else
            newInputs.push_back(ty);
        } else {
          newInputs.push_back(ty);
        }
      }
      // if (funcType.getInputs() != newInputs) {
      func.setType(builder.getFunctionType(newInputs, funcType.getResults()));
      // }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createShrinkBitWidth(const unsigned &targetBitwidth) {
  return std::make_unique<ShrinkBitwidthPass>(targetBitwidth);
}