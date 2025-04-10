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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {
class ShrinkBitWidthPass
    : public dynamatic::experimental::impl::CfShrinkBitWidthBase<
          ShrinkBitWidthPass> {

  int64_t clipValue(int64_t val) {
    int64_t maxValue = (1 << targetBitWidth) - 1;
    if (val > maxValue)
      return maxValue;
    return val;
  }

  // Helper function
  Type shrinkType(Type ty, MLIRContext *ctx) {
    Builder builder(ctx);

    if (auto intType = ty.dyn_cast<IntegerType>()) {
      if (intType.getWidth() > targetBitWidth)
        return builder.getIntegerType(targetBitWidth);
    }

    if (auto memrefType = ty.dyn_cast<MemRefType>()) {
      auto newElemType = shrinkType(memrefType.getElementType(), ctx);
      if (newElemType != memrefType.getElementType()) {

        SmallVector<int64_t, 4> newShape;
        assert(memrefType.getShape().size() == 1 &&
               "We assume that Dynamatic has flattened the array shape");
        for (int64_t dim : memrefType.getShape()) {
          newShape.push_back(clipValue(dim));
        }

        ArrayRef<int64_t> newShapeRef = newShape;

        return MemRefType::get(newShapeRef, newElemType, memrefType.getLayout(),
                               memrefType.getMemorySpace());
      }
    }
    return ty;
  }

public:
  ShrinkBitWidthPass(unsigned targetBitWidth) {
    this->targetBitWidth = targetBitWidth;
  }

  void runDynamaticPass() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    MLIRContext *ctx = &getContext();

    module.walk([&](Operation *op) {
      // Update result types
      for (auto result : op->getResults()) {
        result.setType(shrinkType(result.getType(), ctx));
      }

      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto memref = load.getMemRef();
        auto newType = shrinkType(memref.getType(), ctx);
        if (newType != memref.getType()) {
          memref.setType(newType);
        }
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        auto memref = store.getMemRef();
        auto newType = shrinkType(memref.getType(), ctx);
        if (newType != memref.getType()) {
          memref.setType(newType);
        }
      }

      // Clip integer constants
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
          if (constOp.getValue().getType() == builder.getIndexType()) {
            int64_t val = intAttr.getInt();
            auto newAttr = builder.getIndexAttr(clipValue(val));
            constOp.setValueAttr(newAttr);
          } else {
            int64_t val = intAttr.getInt();
            auto newAttr = builder.getIntegerAttr(
                builder.getIntegerType(targetBitWidth), clipValue(val));
            constOp.setValueAttr(newAttr);
          }
        }
      }
    });

    // Update function arguments
    module.walk([&](func::FuncOp func) {
      // Rewrite func signature
      auto funcType = func.getFunctionType();

      SmallVector<Type> newInputs;
      for (Type ty : funcType.getInputs())
        newInputs.push_back(shrinkType(ty, ctx));

      SmallVector<Type> newResults;
      for (Type ty : funcType.getResults())
        newResults.push_back(shrinkType(ty, ctx));

      auto newFuncType = builder.getFunctionType(newInputs, newResults);
      func.setType(newFuncType);

      // Don't forget to update block argument types:
      for (auto [arg, newTy] : llvm::zip(func.getArguments(), newInputs))
        arg.setType(newTy);

      for (Block &block : func.getBlocks()) {
        for (BlockArgument arg : block.getArguments()) {
          if (arg.getType().isIndex())
            continue;

          if (auto intType = arg.getType().dyn_cast<IntegerType>()) {
            if (intType.getWidth() > targetBitWidth) {
              arg.setType(builder.getIntegerType(targetBitWidth));
            }
          }
        }
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createShrinkBitWidth(unsigned targetBitWidth) {
  return std::make_unique<ShrinkBitWidthPass>(targetBitWidth);
}