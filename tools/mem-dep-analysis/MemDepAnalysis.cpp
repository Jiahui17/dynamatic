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
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include <stdexcept>
#include <stdlib.h>

using namespace llvm;
using namespace polly;

// We need to link against the MLIR library if we import this from NameAnalysis,
// which is an overkill...
const std::string HANDSHAKE_NAME = "handshake.name";
const std::string DEST_NAMES = "dest.ops";

namespace {

class InstructionDependenceInfo {
public:
  InstructionDependenceInfo(const LoopInfo &li) : loopInfo(li) {}

  /// Query whether I_B is dependant on I_A: I_A -D-> I_B
  /// Returns true if every token coming to I_A has passed through I_B
  /// without traversing any BB-edge that would increment common induction
  /// variables
  bool hasDependency(const Instruction *iB, const Instruction *iA);

  /// Query whether I_B is reversely dependant on I_A: I_A -RD-> I_B
  /// Returns true if every token coming to I_A will pass through I_B
  /// without traversing any BB-edge that would increment common induction
  /// variables
  bool hasReverseDependency(const Instruction *iA, const Instruction *iB);

private:
  const LoopInfo &loopInfo;
};

using Path = struct Path {
  std::vector<const BasicBlock *> blocks;
  std::map<const BasicBlock *, std::set<const Value *>> vals;
};

bool inLoopLatches(const BasicBlock *bb, const std::set<Loop *> &loopSet) {

  return std::any_of(loopSet.begin(), loopSet.end(),
                     [&bb](Loop *loop) { return loop->getLoopLatch() == bb; });
}

bool tokenDepends(const Path &p, const Instruction *instA,
                  const std::set<Loop *> &loopSet) {
  int len = p.blocks.size();
  const BasicBlock *curBB = p.blocks.back();
  std::set<const Value *> activeVals = p.vals.at(curBB);
  std::map<const BasicBlock *, std::set<const Value *>> phiDepends;

  llvm::errs().indent(len * 4) << curBB->getName() << "\n";

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

    /* If this instruction is an not active dependence, ignore it */
    if (activeVals.find(inst) == activeVals.end())
      continue;
    /* Else, its operands are active dependences too */
    if (const auto *phiNode = dyn_cast<PHINode>(inst)) {
      /* For Phi nodes, the active dependent values may be different
          in the different predecessors BB, so we store them in this map
          for now. We add it to the Path.Val set before recursive calls */
      for (const auto &predBB : phiNode->blocks()) {
        auto *value = phiNode->getIncomingValueForBlock(predBB);
        if (!(isa<Argument>(value) || isa<Constant>(value)))
          phiDepends[predBB].insert(value);
      }
    } else {
      for (const auto *op : inst->operand_values())
        if (!(isa<Argument>(op) || isa<Constant>(op)))
          activeVals.insert(op);
    }
  }

  bool depends = true;
  if (instA->getParent() == curBB) {
    // If through successive tokenDepends calls we have reached the basic
    // block containing I_A, check whether I_A has been added to the list of
    // active dependences. If so, dependency is met.
    depends = activeVals.find(instA) != activeVals.end();
  } else {
    std::vector<const BasicBlock *> validPreds;
    for (const auto *predBB : predecessors(curBB)) {
      /* This depends on having a canonical loop structure. Loops will
       * have a single latch with a single successor: the loop header.
       * Continuing across an edge from a latch to header for any loop in
       * LS is not allowed. */
      if (!(inLoopLatches(predBB, loopSet) ||
            (p.vals.count(predBB) && p.vals.at(predBB) == activeVals))) {
        validPreds.push_back(predBB);
      }
    }

    // if we have reached a point where active values has been determined,
    // but there are no valid predecessors to produce these values, then
    // dependency is not met.
    depends = !validPreds.empty();

    // Else, for each predecessor block, propagate the active values from
    // this basic block (plus any potential active values from phi-nodes
    // with incoming values for the given predecessor block) into a
    // successive call to tokenDepends.
    for (const auto *predBB : validPreds) {
      if (!depends)
        break;

      Path predBBPath = p;
      predBBPath.blocks.push_back(predBB);
      predBBPath.vals[predBB] = activeVals;
      auto it = phiDepends.find(predBB);
      if (it != phiDepends.end())
        for (const auto &val : it->second)
          predBBPath.vals[predBB].insert(val);

      depends &= tokenDepends(predBBPath, instA, loopSet);
    }
  }

  llvm::errs().indent(len * 4) << depends << "\n";
  return depends;
}

static bool tokenRevDepends(Path path, const Instruction *instA,
                            const std::set<Loop *> &loopSet) {
  const int len = path.blocks.size();
  const BasicBlock *curBB = path.blocks.back();
  const BasicBlock *predBB = (len > 1) ? path.blocks[len - 2] : nullptr;
  auto activeVals = path.vals[curBB];

  llvm::errs().indent(len * 4) << curBB->getName() << "\n";
  /// Determine active values of the current basic block
  /// For each instruction in the current basic block, its operands are
  /// checked to see whether they are present in the current set of active
  /// dependences.
  /// If so, the instruction which has the operand is itself reverse
  /// dependent.
  for (const auto &inst : *curBB) {
    if (isa<BranchInst>(&inst) || isa<DbgInfoIntrinsic>(&inst))
      continue;

    std::vector<const Value *> operands;
    if (const auto *phiNode = dyn_cast<PHINode>(&inst)) {
      /* For a PHI node, the only relevant operand is decided by the
       * prev BB */
      if (predBB != nullptr)
        operands.push_back(phiNode->getIncomingValueForBlock(predBB));
    } else {
      for (const auto *op : inst.operand_values())
        operands.push_back(op);
    }
    /* If any of the operands has revdep on LI, this value does too */
    for (const auto *op : operands)
      if (activeVals.find(op) != activeVals.end())
        activeVals.insert(&inst);
  }

  bool depends = true;
  if (instA->getParent() == curBB) {
    depends = activeVals.find(instA) != activeVals.end();
  } else if (inLoopLatches(curBB, loopSet)) {
    /* This depends on having a canonical loop structure. Loops will
     * have a single latch with a single successor: the loop header.
     * Continuing across an edge from a latch to header for any loop in
     * LS is not allowed. */
  } else {
    const unsigned numSucc = curBB->getTerminator()->getNumSuccessors();
    depends = (numSucc > 0);

    for (const auto *succBB : successors(curBB)) {
      if (!depends)
        break;

      /* Skip successor BB if no active values have been added in this
       * call to TokenRevDepends */
      if (std::find(path.blocks.begin(), path.blocks.end(), succBB) !=
          path.blocks.end()) {
        if (path.vals[succBB] == activeVals)
          continue;
      }

      /* If next BB has not been sufficiently explored, explore again */
      Path succBBPath = path;
      succBBPath.blocks.push_back(succBB);
      succBBPath.vals[succBB] = activeVals;

      depends &= tokenRevDepends(succBBPath, instA, loopSet);
    }
  }
  llvm::errs().indent(len * 4) << depends << "\n";
  return depends;
}

} // namespace

bool InstructionDependenceInfo::hasDependency(const Instruction *iB,
                                              const Instruction *iA) {

  Path p;
  const auto *bb = iB->getParent();
  p.blocks.push_back(bb);
  p.vals[bb] = {iB};

  auto loopSet = std::set<Loop *>();
  for (Loop *loop = loopInfo.getLoopFor(iB->getParent()); loop != nullptr;
       loop = loop->getParentLoop())
    loopSet.insert(loop);

  llvm::errs() << "I_B = " << *iB << " depends " << *iA << " ? \n";
  return tokenDepends(p, iA, loopSet);
}

bool InstructionDependenceInfo::hasReverseDependency(const Instruction *iA,
                                                     const Instruction *iB) {
  Path p;
  const auto *bb = iA->getParent();
  p.blocks.push_back(bb);
  p.vals[bb] = {iA};

  auto ls = std::set<Loop *>();
  for (Loop *l = loopInfo.getLoopFor(iB->getParent()); l != nullptr;
       l = l->getParentLoop())
    ls.insert(l);

  return tokenRevDepends(p, iB, ls);
}

using InstrPairType = std::pair<Instruction *, Instruction *>;

class ScopMetaInfo {
  LoopInfo *loopInfo;
  InstructionDependenceInfo instrDependenceInfo;

  int scopMinDepth;
  std::vector<Instruction *> memInsts;
  std::map<Instruction *, isl::map> instToCurrentMap;
  std::map<Instruction *, int> instToLoopDepth;
  std::set<InstrPairType> intersections;
  std::set<InstrPairType> nonIntersections;
  std::map<Instruction *, Value *> instToBase;
  /* Each Minimized Scop has a separate context. This ensures that
   * trying to intersect maps for instructions from separate Scops
   * will raise an error */
  isl::ctx ctx;
  /* Used by the dependsInternal() function */
  std::map<InstrPairType, bool> dependsCache;
  std::set<InstrPairType> outstandingDependsQueries;

  int getMaxCommonDepth(Instruction *i0, Instruction *i1) {
    // DEBUG(dbgs() << *I0 << " and " << *I1 << " \n");
    const auto *bb0 = i0->getParent();
    const auto *bb1 = i1->getParent();
    int depth0 = loopInfo->getLoopDepth(bb0);
    int depth1 = loopInfo->getLoopDepth(bb1);
    Loop *l0 = loopInfo->getLoopFor(bb0);
    Loop *l1 = loopInfo->getLoopFor(bb1);

    while (depth0 > depth1) {
      l0 = l0->getParentLoop();
      depth0--;
    }
    while (depth1 > depth0) {
      l1 = l1->getParentLoop();
      depth1--;
    }

    /* Keep reducing loop depths until they match,
     * or we reach outside all loops */
    while (l1 != l0 && depth0-- > 0) {
      l0 = l0->getParentLoop();
      l1 = l1->getParentLoop();
    }

    return depth0;
  }

  isl::map getMap(Instruction *inst, const unsigned int depthToKeep,
                  const bool getFuture) {

    const auto currentMap = instToCurrentMap[inst];

    auto inDimsToBeChecked = currentMap.dim(isl::dim::in);

    assert(!inDimsToBeChecked.is_error());

    unsigned inDimValue = static_cast<unsigned>(inDimsToBeChecked);

    assert(inDimValue >= depthToKeep);

    isl::map retMap = currentMap.project_out(isl::dim::in, depthToKeep,
                                             inDimValue - depthToKeep);
    if (getFuture && depthToKeep > 0) {
      retMap = makeFutureMap(retMap);
    }
    return removeMapMeta(retMap);
  }

  /* Functions for modifying isl::map to future forms */

  isl::map makeFutureMap(const isl::map &map) {
    isl::map fMap, tmpMap;

    auto nInsToBeChecked = map.dim(isl::dim::in);

    assert(!nInsToBeChecked.is_error());

    unsigned nIns = static_cast<unsigned>(nInsToBeChecked);

    /* Add input vars */
    tmpMap = map.add_dims(isl::dim::in, nIns);
    /* Add future constraints on new input variables */
    for (unsigned int i = 1; i <= nIns; i++) {
      isl::map constrMapN = addFutureCondition(tmpMap, i);
      if (i == 1)
        fMap = constrMapN;
      else
        fMap = fMap.unite(constrMapN);
    }
    /* Project out old input vars */
    fMap = fMap.project_out(isl::dim::in, 0, nIns);
    assert(fMap.get() != nullptr);
    return fMap;
  }

  /* Add constraints on the 'n' most significant dimensions */
  isl::map addFutureCondition(const isl::map &map, int n) {

    auto nInsToBeChecked = map.dim(isl::dim::in);

    assert(!nInsToBeChecked.is_error());

    int nIns = static_cast<unsigned>(nInsToBeChecked) / 2;
    isl::map constrMap = map;

    isl_local_space *lsp =
        isl_local_space_from_space(map.get_space().release());
    isl::local_space ls = isl::manage(lsp);

    /* Add equality constraints on the first 'n - 1' dims,
        Inequality on the last dim
    */
    for (int i = 0; i < n; i++) {
      isl::constraint c;
      if (i == n - 1) {
        c = isl::constraint::alloc_inequality(ls);
        c = c.set_constant_si(-1);
      } else
        c = isl::constraint::alloc_equality(ls);
      c = c.set_coefficient_si(isl::dim::in, i, 1);
      c = c.set_coefficient_si(isl::dim::in, i + nIns, -1);
      constrMap = constrMap.add_constraint(c);
    }

    return isl::map(constrMap);
  }

  isl::map copyMapMeta(isl::map map, const isl::map &templateMap) {

    isl::id inTupleID = templateMap.get_tuple_id(isl::dim::in);
    isl::id outTupleID = templateMap.get_tuple_id(isl::dim::out);

    map = map.set_tuple_id(isl::dim::in, inTupleID);
    map = map.set_tuple_id(isl::dim::out, outTupleID);

    return map;
  }

  isl::map removeMapMeta(isl::map map) {
    auto emptyID = isl::id::alloc(ctx, "", nullptr);

    map = map.set_tuple_id(isl::dim::in, emptyID);
    map = map.set_tuple_id(isl::dim::out, emptyID);

    return map;
  }

public:
  ScopMetaInfo(Scop &scop)
      : instrDependenceInfo(*scop.getLI()), ctx(isl::ctx(isl_ctx_alloc())) {
    // ctx = isl::ctx(isl_ctx_alloc());
    loopInfo = scop.getLI();

    /* Calculate scopMinDepth based on first scopStmt */
    auto *bb = scop.begin()->getBasicBlock();
    auto *l = loopInfo->getLoopFor(bb);
    scopMinDepth = loopInfo->getLoopDepth(bb) - scop.getRelativeLoopDepth(l);
    assert(scopMinDepth > 0);
  }

  ~ScopMetaInfo() = default;
  /* Use addScopStmt() to add all ScopStmt's in a Scop. Then,
   * computeIntersections() and finally getIntersectionList() */

  void addScopStmt(ScopStmt &stmt) {
    int depth = loopInfo->getLoopDepth(stmt.getBasicBlock());

    for (auto *inst : stmt.getInstructions())
      if (inst->mayReadOrWriteMemory()) {
        auto &ma = stmt.getArrayAccessFor(inst);

        isl::map currentMap = ma.getLatestAccessRelation();

        isl::map domain = isl::map::from_domain(stmt.getDomain());

        auto outDim = currentMap.dim(isl::dim::out);

        assert(!outDim.is_error());

        domain = domain.add_dims(isl::dim::out, static_cast<unsigned>(outDim));

        domain = copyMapMeta(domain, currentMap);

        instToCurrentMap.emplace(inst, currentMap.intersect(domain));

        instToLoopDepth[inst] = depth;
        instToBase[inst] = ma.getOriginalBaseAddr();
        memInsts.push_back(inst);
      }
  }

  void computeIntersections() {
    /* Checking for RAW and WAW conflicts */
    for (auto *wrInst : memInsts) {
      if (!wrInst->mayWriteToMemory())
        continue;

      // clang-format off
        /* The following describes nodes corresponding to LoadInst and StoreInst 
        * in an elastic circuit.  
        *
        * For ... -> SI -> LI -> ... , SI may affect LI in this and future iterations
        *      ↱---------------↵
        * intersect store-set with current and future load-set 
        * 
        * For ... -> LI -> SI -> ... , SI may affect LI only in future iterations
        *      ↱---------------↵
        * intersect store-set with future load-set
        * 
        * For ... -> SI -> ......     , SI and LI iterations are independent 
        *      ↱  -> LI ->     |
        *      |---------------↵
        * intersect entire store-set with entire load-set 
        * 
        * Foreach write access, compare with relevant sets of read accesses 
        * 
        * Similarly, two stores are checked for possible WAW conflicts
        * */
      // clang-format on

      for (auto *inst : memInsts) {
        /* Skip checking with self */
        if (inst == wrInst)
          continue;
        /* No need to check between different arrays */
        if (instToBase[inst] != instToBase[wrInst]) {
          // DEBUG(dbgs() << "Skipping:Diff bases " << *Inst << " and " <<
          // *WrInst
          //              << "\n");
          continue;
        }

        const int commonDepth = getMaxCommonDepth(inst, wrInst);

        auto pair = InstrPairType(wrInst, inst);
        auto *rdInst = dyn_cast_or_null<LoadInst>(inst);

        isl::map instMap, wrInstMap;

        bool depends = instrDependenceInfo.hasDependency(wrInst, inst) ||
                       instrDependenceInfo.hasReverseDependency(inst, wrInst);

        /* Only WrInst may only depend on Inst if Inst is a load */
        if (rdInst != nullptr && depends) {
          /* Consecutive top-level loops will finish the load before any
           * store, since there is an operand dependency */
          if (commonDepth == 0 && scopMinDepth == 1)
            continue;

          const int depthToKeep = commonDepth - scopMinDepth + 1;
          if (depthToKeep < 0) {
            llvm_unreachable("Cannot keep negative depth!");
          }
          instMap = getMap(inst, static_cast<unsigned int>(depthToKeep), true);
          wrInstMap =
              getMap(wrInst, static_cast<unsigned int>(depthToKeep), false);
        } else {
          /* Generic case: we cannot put any restrictions on the indices
           * being processed by the instructions, if there are no token
           * flow that can be established between them. Therefore, we
           * intersect the sets of all possible indices ever accessed */
          wrInstMap = getMap(wrInst, 0, false);
          instMap = getMap(inst, 0, false);
        }

        isl::map intersect = instMap.intersect(wrInstMap);
        if (intersect.is_empty().is_false()) {
          // DEBUG(dbgs() << *WrInst << "\t intersects \t" << *Inst << "\n");
          // DEBUG(dbgs() << "Intersection is: " << intersect.to_str() << "\n");
          intersections.insert(pair);
        }
      }
    }
  }

  std::set<InstrPairType> &getIntersectionList() { return intersections; }

  std::map<Instruction *, Value *> &getInstsToBase() { return instToBase; }

  using iterator = std::vector<Instruction *>::iterator;
  iterator begin() { return memInsts.begin(); }
  iterator end() { return memInsts.end(); }
};

struct IndexAnalysis {

  IndexAnalysis() : otherInsts() {}
  ~IndexAnalysis() = default;

  /// Returns all memory instructions in SCoPs which do not require an LSQ
  /// connection
  std::vector<const Instruction *> &getOtherInsts() { return otherInsts; }

  /// Query whether any SCoP contains BB
  bool isInScop(const BasicBlock *bb) {
    return bbList.find(bb) != bbList.end();
  }

  /// Returns an integer uniquely identifying the SCoP which contains BB
  int getScopID(const BasicBlock *bb) {
    return (isInScop(bb)) ? bbToScopMap[bb] : -1;
  }

  /// Returns a set of instruction pairs which exist in some SCoP and have RAW
  /// dependencies between them
  const std::set<InstrPairType> &getRAWlist() { return instRAWlist; }

  /// Returns a set of instruction pairs which exist in some SCoP and have
  /// WAW dependencies between them.
  const std::set<InstrPairType> &getWAWlist() { return instWAWlist; }

  // std::vector<std::set<Instruction *>> instSets;
  std::vector<const Instruction *> otherInsts;
  std::set<InstrPairType> instRAWlist;
  std::set<InstrPairType> instWAWlist;
  std::set<const BasicBlock *> bbList;
  std::map<const BasicBlock *, int> bbToScopMap;
  std::map<const Instruction *, const Value *> instToBase;
};

void getAllRegions(Region &r, std::deque<Region *> &rq) {
  rq.push_back(&r);
  for (const auto &e : r)
    getAllRegions(*e, rq);
}

bool hasMemoryReadOrWrite(ScopStmt &stmt) {
  bool hasRdWr = false;
  for (auto *inst : stmt.getInstructions()) {
    hasRdWr |= inst->mayReadOrWriteMemory();
  }
  return hasRdWr;
}

Value *findBaseInternal(Value *addr) {
  if (auto *arg = dyn_cast<Argument>(addr)) {
    if (!arg->getType()->isPointerTy())
      llvm_unreachable("Only pointer arguments are considered addresses");
    return addr;
  }

  if (isa<Constant>(addr))
    llvm_unreachable("Cannot determine base address of Constant");

  if (auto *inst = dyn_cast_or_null<Instruction>(addr)) {
    if (isa<AllocaInst>(inst))
      return addr;
    if (auto *gepi = dyn_cast<GetElementPtrInst>(inst))
      return findBaseInternal(gepi->getPointerOperand());
    if (auto *si = dyn_cast<SelectInst>(inst)) {
      auto *trueBase = findBaseInternal(si->getTrueValue());
      auto *falseBase = findBaseInternal(si->getFalseValue());

      /* Select must choose pointers to same array. Otherwise cannot
       * choose relevant arrayRAM in elastic circuit */
      assert(trueBase == falseBase);
      return trueBase;
    }
  }

  /* We try to find a few known cases of pointer expression. For others,
   * implement when you come across them */
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

/* An LSQset is a set instructions that have *possible* memory
 * dependences in a function and hence, need runtime address
 * aliasing via an LSQ */
using LSQset = struct LSQset {
  LSQset(const Value *v, Instruction *i0, Instruction *i1) {
    base = v;
    insts.insert(i0);
    insts.insert(i1);
  }
  LSQset(const Value *v) { base = v; }

  const Value *base;
  std::set<Instruction *> insts;

  using iterator = std::set<Instruction *>::iterator;
  iterator begin() { return insts.begin(); }
  iterator end() { return insts.end(); }
};

namespace {
struct MemDepAnalysisPass : PassInfoMixin<MemDepAnalysisPass> {

  /// Memory metadata for top-level loops
  struct TopLevelLoopMetaInfo {
    TopLevelLoopMetaInfo() = default;
    ~TopLevelLoopMetaInfo() = default;

    std::set<Instruction *> readInstructions;
    std::set<Instruction *> writeInstructions;
    std::map<Instruction *, int> instToScop;
  };

  IndexAnalysis indexAnalysis;

  void processScop(Scop &s, std::vector<ScopMetaInfo> &scopMeta);
  void processLoop(Loop *l,
                   std::vector<struct TopLevelLoopMetaInfo> &loopMetaInfos);
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);

  std::vector<InstrPairType>
  getDependencyPairs(struct TopLevelLoopMetaInfo &loopInfo);

  AAManager::Result *aliasAnalysis;
};

std::map<Instruction *, std::string> nameAllLoadStores(Function &f) {
  unsigned memCount = 0;
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

        loadInstr->setMetadata(HANDSHAKE_NAME, md);
        nameMapping[&instr] = name;
        memCount++;
      } else if (llvm::StoreInst *storeInstr =
                     llvm::dyn_cast<llvm::StoreInst>(&instr)) {

        std::string name = "store" + std::to_string(memCount);

        // Create a metadata string
        llvm::MDString *mdStr = llvm::MDString::get(context, name);

        // Create an MDNode containing the MDString
        llvm::MDNode *md = llvm::MDNode::get(context, mdStr);

        storeInstr->setMetadata(HANDSHAKE_NAME, md);
        nameMapping[&instr] = name;
        memCount++;
      }
    }
  }
  return nameMapping;
}

PreservedAnalyses MemDepAnalysisPass::run(Function &f,
                                          FunctionAnalysisManager &fam) {

  auto &regionInfoAnalysis = fam.getResult<RegionInfoAnalysis>(f);

  auto &scopInfoAnalysis = fam.getResult<ScopInfoAnalysis>(f);

  auto &loopAnalysis = fam.getResult<LoopAnalysis>(f);

  std::vector<TopLevelLoopMetaInfo> loopMetaInfos;

  std::vector<ScopMetaInfo> scopMetaInfos;

  aliasAnalysis = &fam.getResult<AAManager>(f);

  std::deque<Region *> rq;
  getAllRegions(*regionInfoAnalysis.getTopLevelRegion(), rq);

  Scop *s;
  for (Region *r : rq) {
    if ((s = scopInfoAnalysis.getScop(r)))
      processScop(*s, scopMetaInfos);
  }

  /* Process loops according to AA */
  for (Loop *loop : loopAnalysis) {
    /* Currently, we shall analyze only top-level loops */
    // TODO: Properly handle multi-level loops
    if (loop->getLoopDepth() > 1)
      continue;

    processLoop(loop, loopMetaInfos);
  }

  auto nameMapping = nameAllLoadStores(f);
  llvm::LLVMContext &ctx = f.getContext();

  std::map<Instruction *, std::vector<std::string /*names*/>>
      instrToListOfDependentDestinations;

  for (auto &meta : loopMetaInfos) {
    for (auto &[src, dst] : getDependencyPairs(meta)) {
      assert(nameMapping.count(src) > 0 && "Unnamed load/store op!");
      // Get the name meta data
      if (instrToListOfDependentDestinations.count(src) == 0) {
        instrToListOfDependentDestinations[src] = {nameMapping[dst]};
      } else {
        instrToListOfDependentDestinations[src].emplace_back(nameMapping[dst]);
      }
    }
  }

  for (auto [src, dests] : instrToListOfDependentDestinations) {

    SmallVector<llvm::Metadata *, 10> mdVals;
    for (const auto &name : dests) {
      mdVals.push_back(MDString::get(ctx, name));
    }
    llvm::MDNode *destNamesNode = llvm::MDNode::get(ctx, ArrayRef(mdVals));
    destNamesNode->dump();

    src->setMetadata(DEST_NAMES, destNamesNode);
  }

  return PreservedAnalyses::all();
}

void MemDepAnalysisPass::processScop(Scop &scop,
                                     std::vector<ScopMetaInfo> &scopMeta) {

  auto meta = ScopMetaInfo(scop);

  for (auto &stmt : scop) {
    auto *bb = stmt.getBasicBlock();
    indexAnalysis.bbList.insert(bb);
    indexAnalysis.bbToScopMap[bb] = scopMeta.size();

    if (!hasMemoryReadOrWrite(stmt))
      continue;

    meta.addScopStmt(stmt);
  }

  meta.computeIntersections();
  auto intersectList = meta.getIntersectionList();

  for (auto it : meta.getInstsToBase()) {
    const auto *i = it.first;
    const auto *v = it.second;
    indexAnalysis.instToBase[i] = v;
  }

  /* The convention used in ScopMeta class is that the first element
   * in an instPair is a store instruction. Thus, checking the type
   * of the second instruction tells us whther it is a RAW/WAW dependency */
  for (auto pair : intersectList) {
    if (pair.second->mayWriteToMemory())
      indexAnalysis.instWAWlist.insert(pair);
    else
      indexAnalysis.instRAWlist.insert(pair);
  }

  scopMeta.push_back(meta);
}

void MemDepAnalysisPass::processLoop(
    Loop *l, std::vector<struct TopLevelLoopMetaInfo> &loopMetaInfos) {

  loopMetaInfos.emplace_back();
  auto &loopMetaData = loopMetaInfos.back();

  for (auto *bb : l->getBlocks()) {
    int scopId = indexAnalysis.getScopID(bb);
    bool isInScop = indexAnalysis.isInScop(bb);

    for (auto &inst : *bb) {
      if (!inst.mayReadOrWriteMemory())
        continue;
      if (isa<CallInst>(&inst))
        continue;

      if (inst.mayReadFromMemory())
        loopMetaData.readInstructions.insert(&inst);
      if (inst.mayWriteToMemory())
        loopMetaData.writeInstructions.insert(&inst);

      if (isInScop)
        loopMetaData.instToScop[&inst] = scopId;
    }
  }
}

std::vector<InstrPairType>
MemDepAnalysisPass::getDependencyPairs(struct TopLevelLoopMetaInfo &loopInfo) {
  std::vector<InstrPairType> intersectList;
  auto rdInstrSet = loopInfo.readInstructions;
  auto wrInstrSet = loopInfo.writeInstructions;

  for (auto *wrInst : wrInstrSet) {
    /* Find RAW dependencies */
    for (auto *rdInst : rdInstrSet) {
      auto pair = InstrPairType(wrInst, rdInst);

      /* Each base array is emitted as a separate RAM in the design. Two
       * instructions targetting differing base arrays can never depend */
      if (!equalBase(wrInst, rdInst))
        continue;

      /*  If both instructions are in the same scop,
          use the result from IndexAnalysis */
      auto rdIt = loopInfo.instToScop.find(rdInst);
      auto wrIt = loopInfo.instToScop.find(wrInst);
      if (rdIt != loopInfo.instToScop.end() &&
          wrIt != loopInfo.instToScop.end() && rdIt->second == wrIt->second) {
        if (indexAnalysis.getRAWlist().find(pair) !=
            indexAnalysis.getRAWlist().end())
          intersectList.push_back(pair);
        continue;
      }

      /* Otherwise, use results from AA */
      auto *loadInst = dyn_cast<LoadInst>(rdInst);
      auto *storeInst = dyn_cast<StoreInst>(wrInst);

      if (loadInst == nullptr || storeInst == nullptr) {
        llvm_unreachable("Expecting only Read-Write pairs of "
                         "instructions when locating RAW dependencies");
      }

      if (aliasAnalysis->alias(MemoryLocation::get(loadInst),
                               MemoryLocation::get(storeInst)) !=
          AliasResult::NoAlias)
        intersectList.push_back(pair);
    }
    /* Find WAW dependencies */
    for (auto *secondStoreInst : wrInstrSet) {
      if (secondStoreInst == wrInst)
        continue;

      /* Each base array is emitted as a separate RAM in the design. Two
       * instructions targetting differing base arrays can never depend */
      if (!equalBase(wrInst, secondStoreInst))
        continue;

      auto pair = InstrPairType(secondStoreInst, wrInst);
      auto pairRev = InstrPairType(wrInst, secondStoreInst);
      /*  If both instructions are in the same scop,
          use the result rom IndexAnalysis */
      auto wr1It = loopInfo.instToScop.find(secondStoreInst);
      auto wrIt = loopInfo.instToScop.find(wrInst);
      if (wr1It != loopInfo.instToScop.end() &&
          wrIt != loopInfo.instToScop.end() && wr1It->second == wrIt->second) {
        if (indexAnalysis.getWAWlist().find(pair) !=
            indexAnalysis.getWAWlist().end())
          intersectList.push_back(pair);
        else if (indexAnalysis.getWAWlist().find(pairRev) !=
                 indexAnalysis.getWAWlist().end())
          intersectList.push_back(pairRev);
        continue;
      }

      // Otherwise, use results from AA
      auto *storeInst0 = dyn_cast<StoreInst>(wrInst);
      auto *storeInst1 = dyn_cast<StoreInst>(secondStoreInst);

      if (storeInst0 == nullptr || storeInst1 == nullptr) {
        llvm_unreachable("Expecting only Write-Write pairs of "
                         "instructions when locating WAW dependencies");
      }

      // If they always or sometimes alias:
      if (aliasAnalysis->alias(MemoryLocation::get(storeInst0),
                               MemoryLocation::get(storeInst1)) !=
          AliasResult::NoAlias)
        intersectList.push_back(pair);
    }
  }

  return intersectList;
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
