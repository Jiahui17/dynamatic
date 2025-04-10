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

#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#ifndef EXPERIMENTAL_TRANSFORMS_CF_SHRINK_BIT_WIDTH_H
#define EXPERIMENTAL_TRANSFORMS_CF_SHRINK_BIT_WIDTH_H
namespace dynamatic {
namespace experimental {

std::unique_ptr<DynamaticPass>
createShrinkBitWidth(const unsigned &targetBitwidth = 8);

#define GEN_PASS_DECL_CFSHRINKBITWIDTH
#define GEN_PASS_DEF_CFSHRINKBITWIDTH
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_CF_SHRINK_BIT_WIDTH_H
