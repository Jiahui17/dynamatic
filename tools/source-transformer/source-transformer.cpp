// Declares clang::SyntaxOnlyAction.
#include "clang/AST/Decl.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace llvm;
using namespace clang::tooling;

using namespace clang::ast_matchers;

// clang-format off
// Matches "for (i=0; ....)""
auto zeroInit = hasLoopInit(declStmt(
    hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0)))))));

// Matches "for (...; var < const_int; ...)"
auto constCompare = hasCondition(
    binaryOperator(
                    hasOperatorName("<"),
                    hasLHS(ignoringParenImpCasts(declRefExpr( to(varDecl(hasType(isInteger())).bind("condVarName"))))),
                    hasRHS(integerLiteral().bind("constbound"))
                  )
                );
auto incrementConstraint = hasIncrement(unaryOperator(
  hasOperatorName("++"),
  hasUnaryOperand(declRefExpr(to(
    varDecl(hasType(isInteger())).bind("incrementVariable"))))));
// clang-format on

StatementMatcher loopMatcher =
    forStmt(zeroInit, constCompare, incrementConstraint).bind("forLoop");

class LoopPrinter : public MatchFinder::MatchCallback {

  Rewriter &rewrite;

public:
  void run(const MatchFinder::MatchResult &result) override {
    if (const ForStmt *fs = result.Nodes.getNodeAs<clang::ForStmt>("forLoop")) {
      fs->dump();
      const auto *decl = result.Nodes.getNodeAs<VarDecl>("varDecl");
      if (decl && decl->getType()->isUnsignedIntegerType()) {
        SourceLocation loc = decl->getDefinition()->getBeginLoc();
        rewrite.ReplaceText(loc, 3,
                            "long"); // Replace 'int' (3 chars) with 'long'
      }
    }
  }
  LoopPrinter(Rewriter &r) : rewrite(r) {}
};

class MyFrontendAction : public ASTFrontendAction {
public:
  void EndSourceFileAction() override {
    SourceManager &sourceMgr = rewriter.getSourceMgr();
    llvm::outs() << "--- Rewritten File ---\n";
    rewriter.getEditBuffer(sourceMgr.getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &ci, StringRef inputFileName) override {
    rewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());

    finder.addMatcher(loopMatcher, &callback);

    return finder.newASTConsumer();
  }

private:
  Rewriter rewriter;
  LoopPrinter callback{rewriter};
  MatchFinder finder;
};

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory myToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp commonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp moreHelp("\nMore help text...\n");

int main(int argc, const char **argv) {
  auto expectedParser = CommonOptionsParser::create(argc, argv, myToolCategory);
  if (!expectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << expectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &optionsParser = expectedParser.get();
  ClangTool tool(optionsParser.getCompilations(),
                 optionsParser.getSourcePathList());

  if (int retCode =
          tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>().get());
      retCode != 0) {

    llvm::errs() << "Error - syntax checking failed!\n";
    return retCode;
  }

  return tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}