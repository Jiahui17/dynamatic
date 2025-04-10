//===- export-rtl.cpp - Export RTL from HW-level IR -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exports RTL from HW-level IR. Files corresponding to internal and external
// modules are written inside a provided output directory (which is created if
// necessary).
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Conversion/HandshakeToHW.h"
#include "dynamatic/Dialect/HW/HWDialect.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "dynamatic/Support/System.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

void printInstantiation(hw::HWModuleOp modOp) {

  llvm::outs() << "" << modOp.getSymName() << " " << modOp.getSymName()
               << " (\n";
  for (auto [arg, portAttr] : llvm::zip_equal(
           modOp.getBodyBlock()->getArguments(), modOp.getInputNamesStr())) {
    if (isa<handshake::ChannelType>(arg.getType())) {
      llvm::outs() << "." << portAttr.str() << "(" << portAttr.str() << "),\n";
      llvm::outs() << "." << portAttr.str() << "_valid(go & !" << portAttr.str()
                   << "_taken),\n";
      llvm::outs() << "." << portAttr.str() << "_ready(" << portAttr.str()
                   << "_ready),\n";
    } else if (isa<handshake::ControlType>(arg.getType())) {
      llvm::outs() << "." << portAttr.str() << "_valid(go & !" << portAttr.str()
                   << "_taken),\n";
      llvm::outs() << "." << portAttr.str() << "_ready(" << portAttr.str()
                   << "_ready),\n";
    } else if (isa<IntegerType>(arg.getType())) {
      llvm::outs() << "." << portAttr.str() << "(" << portAttr.str() << "),\n";
    }
  }
  bool first;
  first = true;
  for (auto [resType, portAttr] :
       llvm::zip_equal(modOp.getOutputTypes(), modOp.getOutputNamesStr())) {
    if (!first) {
      llvm::outs() << ",\n";
    }
    first = false;
    if (isa<handshake::ChannelType>(resType)) {
      llvm::outs() << "." << portAttr.str() << "(" << portAttr.str() << "),\n";
      llvm::outs() << "." << portAttr.str() << "_valid(" << portAttr.str()
                   << "_valid"
                   << "),\n";
      llvm::outs() << "." << portAttr.str() << "_ready(1)";
    } else if (isa<handshake::ControlType>(resType)) {
      llvm::outs() << "." << portAttr.str() << "_valid(" << portAttr.str()
                   << "_valid"
                   << "),\n";
      llvm::outs() << "." << portAttr.str() << "_ready(1)";
    } else if (isa<IntegerType>(resType)) {
      llvm::outs() << "." << portAttr.str() << "(" << portAttr.str() << ")";
    }
  }
  llvm::outs() << "\n);\n";
}

std::string formatSignalWidth(Type type) {

  if (auto channelType = dyn_cast<ChannelType>(type)) {
    llvm::errs() << "!!Channel type: " << channelType << "\n";
    int width = channelType.getDataBitWidth();
    llvm::errs() << "!!!Channel type: " << channelType << "\n";
    if (width == 1) {
      return "";
    }
    return "[" + std::to_string(width - 1) + ":0]";
  }

  if (auto intType = dyn_cast<IntegerType>(type)) {
    int width = intType.getWidth();
    if (width == 1) {
      return "";
    }
    return "[" + std::to_string(width - 1) + ":0]";
  }
  llvm::errs() << "Unsupported type: " << type << "\n";
  return "";
}

void writeWrapper(hw::HWModuleOp modOp) {

  llvm::outs() << "module " << modOp.getSymName() << "_rigid(\n";

  // Iterate over input ports.
  for (auto [arg, portAttr] : llvm::zip_equal(
           modOp.getBodyBlock()->getArguments(), modOp.getInputNamesStr())) {
    if (isa<handshake::ChannelType>(arg.getType())) {
      llvm::outs() << "input " << formatSignalWidth(arg.getType())
                   << portAttr.str() << ",\n";
      llvm::errs() << "Port type: " << arg.getType() << "\n";
      llvm::errs() << "Port name: " << portAttr << "\n";
    } else if (isa<handshake::ControlType>(arg.getType())) {
      llvm::errs() << "Port type: " << formatSignalWidth(arg.getType())
                   << arg.getType() << "\n";
      llvm::errs() << "Port name: " << portAttr << "\n";
    } else if (isa<IntegerType>(arg.getType())) {
      llvm::outs() << "input " << formatSignalWidth(arg.getType())
                   << portAttr.str() << ",\n";
    }
  }
  llvm::outs() << "go,\n";
  llvm::outs() << "clk,\n";
  llvm::outs() << "rst,\n";

  for (auto [resType, portAttr] :
       llvm::zip_equal(modOp.getOutputTypes(), modOp.getOutputNamesStr())) {
    if (isa<handshake::ChannelType>(resType)) {
      llvm::outs() << "output " << formatSignalWidth(resType) << portAttr.str()
                   << ",\n";
      llvm::errs() << "Port type: " << resType << "\n";
      llvm::errs() << "Port name: " << portAttr << "\n";
    } else if (isa<handshake::ControlType>(resType)) {
      llvm::errs() << "Port type: " << formatSignalWidth(resType) << resType
                   << "\n";
      llvm::errs() << "Port name: " << portAttr << "\n";
    } else if (isa<IntegerType>(resType)) {
      llvm::outs() << "output " << portAttr.str() << ",\n";
    }
  }
  llvm::outs() << "done\n);\n";

  llvm::outs() << "// Internal signals\n";

  // Input channel has taken the token
  for (auto [arg, portAttr] : llvm::zip_equal(
           modOp.getBodyBlock()->getArguments(), modOp.getInputNamesStr())) {
    if (isa<handshake::ChannelType>(arg.getType()) or
        isa<handshake::ControlType>(arg.getType())) {
      llvm::outs() << "wire " << portAttr.str() << "_ready;\n";
      llvm::outs() << "reg " << portAttr.str() << "_taken = 0;\n";
      llvm::outs() << "always @(posedge clk) begin\n";
      llvm::outs() << "  if (rst) begin\n";
      llvm::outs() << "    " << portAttr.str() << "_taken <= 0;\n";
      llvm::outs() << "  end else begin\n";
      llvm::outs() << "    if (go & " << portAttr.str() << "_ready) begin\n";
      llvm::outs() << "      " << portAttr.str() << "_taken <= 1;\n";
      llvm::outs() << "    end\n";
      llvm::outs() << "  end\n";
      llvm::outs() << "end\n\n";
    }
  }

  // Output channel has received a token
  SmallVector<std::string> outputFullFlags;
  for (auto [resType, portAttr] :
       llvm::zip_equal(modOp.getOutputTypes(), modOp.getOutputNamesStr())) {
    if (isa<handshake::ChannelType>(resType) or
        isa<handshake::ControlType>(resType)) {
      llvm::outs() << "wire " << portAttr.str() << "_valid;\n";
      llvm::outs() << "reg " << portAttr.str() << "_full = 0;\n";
      llvm::outs() << "always @(posedge clk) begin\n";
      llvm::outs() << "  if (rst) begin\n";
      llvm::outs() << "    " << portAttr.str() << "_full <= 0;\n";
      llvm::outs() << "  end else begin\n";
      llvm::outs() << "    if (" << portAttr.str() << "_valid & go) begin\n";
      llvm::outs() << "      " << portAttr.str() << "_full <= 1;\n";
      llvm::outs() << "    end\n";
      llvm::outs() << "  end\n";
      llvm::outs() << "end\n\n";
      outputFullFlags.push_back(portAttr.str() + "_full");
    }
  }

  // Output is valid

  llvm::outs() << "assign done = " << llvm::join(outputFullFlags, " & ")
               << ";\n";

  // // Iterate over output ports.
  // for (auto &output : ports.outputs) {
  //   llvm::StringRef name = output.getName(); // Port name.
  //   mlir::Type type = output.type;           // Port type.
  //   // Do something with output port.
  // }
  printInstantiation(modOp);
  llvm::outs() << "endmodule\n";
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv, "Create a rigid wrapper for a given HW module.\n");

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, hw::HWDialect>();

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Write each module's RTL implementation to a separate file
  for (hw::HWModuleOp hwModOp : modOp->getOps<hw::HWModuleOp>()) {
    writeWrapper(hwModOp);
  }
}