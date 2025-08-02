#include "polly/DependenceInfo.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include <stdexcept>
#include <stdlib.h>
#include <utility>

#include "llvm/Analysis/ValueTracking.h"

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/MemoryDependency.h"
#include "polly/Support/ISLTools.h"

using namespace llvm;
using namespace polly;

namespace {

struct CFGPath {
  std::vector<BasicBlock *> blocks;
  std::map<BasicBlock *, std::set<llvm::Value *>> vals;
};

bool inLoopLatches(const BasicBlock *bb, const std::set<Loop *> &loopSet) {

  return std::any_of(loopSet.begin(), loopSet.end(),
                     [&bb](Loop *loop) { return loop->getLoopLatch() == bb; });
}

/// \brief: Recursive routine that checks if there is always a path from
/// starting from srcInst.
bool instructionAlwaysDepends(const CFGPath &currentPath, Instruction *srcInst,
                              const std::set<Loop *> &loopSet) {
  BasicBlock *curBB = currentPath.blocks.back();
  std::set<llvm::Value *> activeVals = currentPath.vals.at(curBB);
  std::map<BasicBlock *, std::set<llvm::Value *>> phiDepends;

  /// Determine active values of the current basic block
  /// For each instruction in the current basic block, determine whether it
  /// has been marked as an active dependence in a previous call to
  /// tokenDepends (or if path P was initialized with an active dependence).
  /// when an active dependence is found, add all of its own arguments are
  /// themselves added as active dependences.
  for (auto rit = curBB->rbegin(); rit != curBB->rend(); ++rit) {
    auto *inst = &*rit;
    if (isa<BranchInst>(inst) || isa<DbgInfoIntrinsic>(inst))
      continue;

    // If this instruction is an not active dependence, ignore it
    if (activeVals.find(inst) == activeVals.end())
      continue;
    // Else, its operands are active dependences too
    if (auto *phiNode = dyn_cast<PHINode>(inst)) {
      // For Phi nodes, the active dependent values may be different in the
      // different predecessors BB, so we store them in this map for now. We add
      // it to the Path.Val set before recursive calls
      for (auto &predBB : phiNode->blocks()) {
        auto *value = phiNode->getIncomingValueForBlock(predBB);
        if (!(isa<Argument>(value) || isa<Constant>(value)))
          phiDepends[predBB].insert(value);
      }
    } else {
      for (auto *op : inst->operand_values())
        if (!(isa<Argument>(op) || isa<Constant>(op)))
          activeVals.emplace(op);
    }
  }

  bool depends = true;
  if (srcInst->getParent() == curBB) {
    // If through successive tokenDepends calls we have reached the basic
    // block containing I_A, check whether I_A has been added to the list of
    // active dependences. If so, dependency is met.
    depends = activeVals.find(srcInst) != activeVals.end();
  } else {
    SmallVector<BasicBlock *, 2> validPredBBs;
    for (auto *predBB : predecessors(curBB)) {
      // This depends on having a canonical loop structure. Loops will have a
      // single latch with a single successor: the loop header.  Continuing
      // across an edge from a latch to header for any loop in LS is not
      // allowed.
      if (!(inLoopLatches(predBB, loopSet) ||
            (currentPath.vals.count(predBB) &&
             currentPath.vals.at(predBB) == activeVals))) {
        validPredBBs.push_back(predBB);
      }
    }

    // If we have reached a point where active values has been determined,
    // but there are no valid predecessors to produce these values, then
    // dependency is not met.
    depends = !validPredBBs.empty();

    // Else, for each predecessor block, propagate the active values from
    // this basic block (plus any potential active values from phi-nodes
    // with incoming values for the given predecessor block) into a
    // successive call to tokenDepends.
    for (auto *predBB : validPredBBs) {
      if (!depends)
        break;

      CFGPath predBBPath = currentPath;
      predBBPath.blocks.emplace_back(predBB);
      predBBPath.vals[predBB] = activeVals;
      auto it = phiDepends.find(predBB);
      if (it != phiDepends.end())
        for (const auto &val : it->second)
          predBBPath.vals[predBB].insert(val);

      // NOTE: we use "&" because our aim is that the dependency is always
      // there, regardless of the control flow.
      depends &= instructionAlwaysDepends(predBBPath, srcInst, loopSet);
    }
  }

  return depends;
}

} // namespace

using InstPairType = std::pair<Instruction *, Instruction *>;

struct IndexAnalysis {

  ~IndexAnalysis() = default;

  std::set<InstPairType> instRAWlist;
  std::set<InstPairType> instWAWlist;
};

void getAllRegions(llvm::Region &r, std::deque<llvm::Region *> &rq) {
  rq.push_back(&r);
  for (const auto &e : r)
    getAllRegions(*e, rq);
}

// Returns the base address produced by the alloca instruction or the global
// constant declaration.
Value *findBaseInternal(Value *addr) {
  if (auto *arg = dyn_cast<Argument>(addr)) {
    if (!arg->getType()->isPointerTy())
      llvm_unreachable("Only pointer arguments are considered addresses");
    return addr;
  }

  // Example: returns a global constant or variable
  if (isa<Constant>(addr))
    return addr;

  if (auto *inst = dyn_cast_or_null<Instruction>(addr)) {
    if (isa<AllocaInst>(inst))
      return addr;
    if (auto *gepi = dyn_cast<GetElementPtrInst>(inst))
      return findBaseInternal(gepi->getPointerOperand());
    if (auto *si = dyn_cast<SelectInst>(inst)) {
      auto *trueBase = findBaseInternal(si->getTrueValue());
      auto *falseBase = findBaseInternal(si->getFalseValue());

      // Select must choose pointers to same array. Otherwise cannot
      // choose relevant arrayRAM in elastic circuit
      assert(trueBase == falseBase);
      return trueBase;
    }
  }

  // We try to find a few known cases of pointer expression. For others,
  // implement when you come across them
  llvm_unreachable("Cannot  determine base array, aborting...");
}

Value *findBase(Instruction *inst) {
  Value *addr;
  if (auto *loadInst = dyn_cast<LoadInst>(inst)) {
    addr = loadInst->getPointerOperand();
  } else if (auto *storeInst = dyn_cast<StoreInst>(inst)) {
    addr = storeInst->getPointerOperand();
  } else {
    llvm_unreachable("Instruction is not a memory access");
  }

  return findBaseInternal(addr);
}

bool equalBase(Instruction *a, Instruction *b) {
  return findBase(a) == findBase(b);
}

namespace {

/// Metadata for loops
struct LoopMetaData {

  Loop *loop;
  LoopMetaData() = default;
  ~LoopMetaData() = default;

  std::set<LoadInst *> readInstructions;
  std::set<StoreInst *> writeInstructions;
};

/// \brief: an LLVM pass that combines polyhedral and alias analysis to compute
/// a set of dependency edges from the LLVM IR. It further uses dataflow
/// analysis to eliminate dependency edges enforced by the dataflow.
struct MemDepAnalysisPass : PassInfoMixin<MemDepAnalysisPass> {

  IndexAnalysis indexAnalysis;
  AAManager::Result *aliasAnalysis;
  unsigned memCount = 0;

  /// \brief: Loops through the scop regions in the IR and applies index and
  /// dataflow analysis to compute the minimum set of dependency edges.
  void processScop(Scop &s, unsigned scopId);

  /// \brief: Loops through the loops in the IR and collect the loads and
  /// stores.
  void processLoop(Loop *l, std::vector<struct LoopMetaData> &loopMetaInfos);
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);

  /// \brief: returns a list of (srcInst, dstInst) pairs that might have a WAR
  /// or WAW conflict.
  std::vector<InstPairType> getDependencyPairs(struct LoopMetaData &loopInfo);
  std::map<Instruction *, std::string> nameAllLoadStores(Function &f);

  // NOTE: An instruction should not be present in multiple Scops (?), so a
  // single set is a good container for it.
  std::map<Instruction *, int> instToScopMap;

  bool sameScop(Instruction *i, Instruction *j) const {
    if (!instToScopMap.count(i))
      return false;

    if (!instToScopMap.count(j))
      return false;

    return instToScopMap.count(i) == instToScopMap.count(j);
  }
};

std::map<Instruction *, std::string>
MemDepAnalysisPass::nameAllLoadStores(Function &f) {
  llvm::LLVMContext &context = f.getContext();

  std::map<Instruction *, std::string> nameMapping;

  for (llvm::BasicBlock &bb : f) {
    for (llvm::Instruction &instr : bb) {
      if (llvm::LoadInst *loadInstr = llvm::dyn_cast<llvm::LoadInst>(&instr)) {

        std::string name = "load" + std::to_string(memCount);

        // Create a metadata string
        llvm::MDString *mdStr = llvm::MDString::get(context, name);

        // Create an MDNode containing the MDString
        // MDNode::get takes a context and an arrayref of llvm::Value*
        llvm::MDNode *md = llvm::MDNode::get(context, mdStr);

        loadInstr->setMetadata(dynamatic::NameAnalysis::ATTR_NAME, md);
        nameMapping[&instr] = name;
        memCount++;
      } else if (llvm::StoreInst *storeInstr =
                     llvm::dyn_cast<llvm::StoreInst>(&instr)) {

        std::string name = "store" + std::to_string(memCount);

        // Create a metadata string
        llvm::MDString *mdStr = llvm::MDString::get(context, name);

        // Create an MDNode containing the MDString
        llvm::MDNode *md = llvm::MDNode::get(context, mdStr);

        storeInstr->setMetadata(dynamatic::NameAnalysis::ATTR_NAME, md);
        nameMapping[&instr] = name;
        memCount++;
      }
    }
  }
  return nameMapping;
}

bool hasGlobalInOrderInstrDependency(Instruction *dstInst, Instruction *srcInst,
                                     LoopInfo &loopInfo) {

  CFGPath cfgPath;
  auto *bb = dstInst->getParent();
  cfgPath.blocks.push_back(bb);
  cfgPath.vals[bb] = {dstInst};

  auto loopSet = std::set<Loop *>();
  for (Loop *loop = loopInfo.getLoopFor(dstInst->getParent()); loop != nullptr;
       loop = loop->getParentLoop())
    loopSet.insert(loop);

  return instructionAlwaysDepends(cfgPath, srcInst, loopSet);
}

void MemDepAnalysisPass::processScop(Scop &scop, unsigned scopId) {

  // The range of indices accessed by each instruction in the scop
  std::map<Instruction *, isl::set> instToAccessRangeMap;

  std::map<Instruction *, Value *> instToBaseMap;

  std::vector<std::pair<Instruction *, Instruction *>> intersections;

  // Collect the index access range of each memory access statement in the scop
  for (auto &stmt : scop) {
    for (auto *inst : stmt.getInstructions()) {
      if (inst->mayReadOrWriteMemory()) {

        instToScopMap[inst] = scopId;

        auto *memoryAccess = stmt.getArrayAccessOrNULLFor(inst);

        isl::map currentMap = memoryAccess->getLatestAccessRelation();

        dumpPw(currentMap);

        // The domain of the iterators, e.g., 0 <= i < N (the upper and lower
        // bounds).
        isl::set domain = stmt.getDomain();

        // Access index space of this *inst
        isl::set accessRange = currentMap.intersect_domain(domain).range();
        instToAccessRangeMap[inst] = accessRange;

        instToBaseMap[inst] =
            memoryAccess->getOriginalBaseAddr(); // Base address of the array
      }
    }
  }

  for (auto &[storeInst, storeAccessRange] : instToAccessRangeMap) {
    if (!isa<StoreInst>(storeInst))
      continue;

    for (auto &[proceedingInst, proceedingAccessRange] : instToAccessRangeMap) {
      // Skip checking with self
      if (proceedingInst == storeInst)
        continue;
      // Skip checking if instructions are in different arrays
      if (instToBaseMap[storeInst] != instToBaseMap[proceedingInst])
        continue;

      bool hasDependency = hasGlobalInOrderInstrDependency(
          storeInst, proceedingInst, *scop.getLI());

      if (auto *loadInst = dyn_cast_or_null<LoadInst>(proceedingInst);
          loadInst && hasDependency) {
        // If there is a dependency and the proceeding instruction is a load,
        // then we have a WAR enforced by data dependency. This dependency does
        // not need to be stored.
        llvm::errs() << "Skipping WAR dependency between " << *storeInst
                     << " and " << *proceedingInst
                     << " as it is enforced by data dependency.\n";
        continue;
      }

      // Otherwise, we check if the two instructions may access the same index
      if (storeAccessRange.intersect(proceedingAccessRange)
              .is_empty()
              .is_false()) {
        // If the two instructions might access the same index:
        auto pair = InstPairType(storeInst, proceedingInst);

        // If the intersection is not empty, then we have a dependency
        intersections.emplace_back(storeInst, proceedingInst);
        llvm::errs() << "Found dependency between " << *storeInst << " and "
                     << *proceedingInst << "\n";
      } else {
        llvm::errs() << "No dependency between " << *storeInst << " and "
                     << *proceedingInst << "\n";
      }
    }
  }

  for (auto pair : intersections) {
    // The convention used in ScopMeta class is that the first element in an
    // instPair is a store instruction. Thus, checking the type of the second
    // instruction tells us whther it is a RAW/WAW dependency
    if (pair.second->mayWriteToMemory())
      indexAnalysis.instWAWlist.insert(pair);
    else
      indexAnalysis.instRAWlist.insert(pair);
  }
}

void MemDepAnalysisPass::processLoop(Loop *l,
                                     std::vector<LoopMetaData> &loopMetaInfos) {

  loopMetaInfos.emplace_back();
  auto &loopMetaData = loopMetaInfos.back();

  loopMetaData.loop = l;

  for (auto *bb : l->getBlocks()) {
    for (auto &inst : *bb) {
      if (!inst.mayReadOrWriteMemory())
        continue;
      if (isa<CallInst>(&inst))
        continue;

      // NOTE: In legacy dynamatic here uses mayReadFromMemory and
      // mayWriteToMemory, which I think is quite redundant for our need
      if (auto *loadInst = dyn_cast<llvm::LoadInst>(&inst))
        loopMetaData.readInstructions.emplace(loadInst);
      if (auto *storeInst = dyn_cast<llvm::StoreInst>(&inst))
        loopMetaData.writeInstructions.emplace(storeInst);
    }
  }
}

std::vector<InstPairType>
MemDepAnalysisPass::getDependencyPairs(LoopMetaData &loopInfo) {
  std::vector<InstPairType> depPairList;

  for (auto *storeInst : loopInfo.writeInstructions) {
    // Find RAW dependencies
    for (auto *loadInst : loopInfo.readInstructions) {
      InstPairType rawPair = std::make_pair(storeInst, loadInst);

      // NOTE: In dynamatic we assume that memory with different base addresses
      // are store in separate RAMs. Two instructions targetting differing base
      // arrays can never conflict.
      if (!equalBase(storeInst, loadInst))
        continue;

      // Instructions are in the same scop: use the result from IndexAnalysis
      if (sameScop(loadInst, storeInst)) {
        if (indexAnalysis.instRAWlist.count(rawPair) > 0) {
          llvm::errs() << "Register RAW dependency between " << *storeInst
                       << " and " << *loadInst << "\n";
          depPairList.push_back(rawPair);
        }
        continue;
      }

      // Instruction are in different Scops: use the result from alias analysis
      AliasResult aliasResult = aliasAnalysis->alias(
          MemoryLocation::get(loadInst), MemoryLocation::get(storeInst));

      // If they always or sometimes alias:
      if (aliasResult != AliasResult::NoAlias) {
        depPairList.push_back(rawPair);
      }
    }
    // Find WAW dependencies
    for (auto *secondStoreInst : loopInfo.writeInstructions) {
      if (secondStoreInst == storeInst)
        continue;

      // NOTE: In dynamatic we assume that memory with different base addresses
      // are store in separate RAMs. Two instructions targetting differing base
      // arrays can never conflict.
      if (!equalBase(storeInst, secondStoreInst))
        continue;

      auto pair = InstPairType(secondStoreInst, storeInst);
      auto pairRev = InstPairType(storeInst, secondStoreInst);

      // Instructions are in the same scop: use the result from IndexAnalysis
      if (sameScop(storeInst, secondStoreInst)) {
        if (indexAnalysis.instWAWlist.count(pair) > 0)
          depPairList.push_back(pair);
        else if (indexAnalysis.instWAWlist.count(pairRev) > 0)
          depPairList.push_back(pairRev);
        continue;
      }

      // Otherwise, use results from alias analysis:
      AliasResult aliasResult = aliasAnalysis->alias(
          MemoryLocation::get(storeInst), MemoryLocation::get(secondStoreInst));
      // If they always or sometimes alias:
      if (aliasResult != AliasResult::NoAlias) {
        depPairList.push_back(pair);
      }
    }
  }

  return depPairList;
}

PreservedAnalyses MemDepAnalysisPass::run(Function &f,
                                          FunctionAnalysisManager &fam) {

  llvm::LLVMContext &ctx = f.getContext();

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(f);

  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(f);

  auto &loopAnalysis = fam.getResult<LoopAnalysis>(f);

  std::vector<LoopMetaData> loopMetaInfos;

  aliasAnalysis = &fam.getResult<AAManager>(f);

  std::deque<Region *> rq;
  getAllRegions(*regionInfoAnalysis.getTopLevelRegion(), rq);

  Scop *s;
  unsigned scopId = 0;
  for (Region *r : rq) {
    if ((s = scopInfoAnalysis.getScop(r))) {
      processScop(*s, scopId);
      scopId += 1;
    }
  }

  // Process loops according to AA
  for (Loop *loop : loopAnalysis) {
    // Currently, we shall analyze only top-level loops. TODO: Properly handle
    // multi-level loops.
    //
    // @Jiahui17: I don't see why processLoop doesn't work for depth > 1.
    if (loop->getLoopDepth() > 1)
      continue;

    processLoop(loop, loopMetaInfos);
  }

  auto nameMapping = nameAllLoadStores(f);

  std::map<Instruction *, LLVMMemDependency> deps;
  for (auto &meta : loopMetaInfos) {
    for (auto &[src, dst] : getDependencyPairs(meta)) {
      assert(nameMapping.count(src) > 0 && "Unnamed load/store op!");
      if (deps.count(src) == 0) {
        LLVMMemDependency newDep;
        newDep.name = nameMapping[src];
        newDep.destAndDepth.emplace_back(nameMapping[dst],
                                         meta.loop->getLoopDepth());
        deps[src] = newDep;
      } else {
        deps[src].destAndDepth.emplace_back(nameMapping[dst],
                                            meta.loop->getLoopDepth());
      }
    }
  }

  for (auto [src, dests] : deps) {
    llvm::errs() << "Adding dependency between " << *src << " and .."
                 << "\n";
    dests.serializeToLLVMMetaDataNode(ctx, src);
  }

  return PreservedAnalyses::all();
}

} // end anonymous namespace

// Register the pass for opt-style loading
// Important note: you need to enable shared libarary in LLVM to load pass
// plugin:
// https://stackoverflow.com/questions/51474188/using-shared-object-so-by-command-opt-in-llvm
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MemDepAnalysis", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "mem-dep-analysis") {
                    fpm.addPass(MemDepAnalysisPass());
                    return true;
                  }
                  return false;
                });
          }};
}
