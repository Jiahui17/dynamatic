// Declares clang::SyntaxOnlyAction.
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Format/Format.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace clang;
using namespace llvm;
using namespace clang::tooling;

using namespace clang::ast_matchers;
using namespace clang::format;

std::string formatCode(llvm::StringRef code) {

  FormatStyle style = getLLVMStyle(); // or getGoogleStyle(), etc.

  tooling::Replacements replaces =
      reformat(style, code, tooling::Range(0, code.size()));

  auto changed = tooling::applyAllReplacements(code, replaces);
  if (!changed)
    return code.str(); // fallback

  return *changed;
}

/// \brief: This matches loops of the form
/// - for (int i = 0; i < N; ++i), where N is a constant.
/// And performs unrolling by 3

const std::string BOUND_VALUE = "constant-bound-value";
const std::string FOR_LOOP = "simple-for-loop";

class SimpleLoopUnroller : public MatchFinder::MatchCallback {
  Rewriter &rewrite;

public:
  SimpleLoopUnroller(Rewriter &r) : rewrite(r) {}
  void registerSimpleLoopUnrollingRewrite(MatchFinder &finder) {

    // clang-format off
    // Matches "for (i=0; ....)""

    // auto isUnsignedDecl = varDecl(hasType(isUnsignedInteger()), hasInitializer(integerLiteral(equals(0))));
    auto isUnsignedDecl = varDecl(hasType(isUnsignedInteger()));

    auto zeroInit = hasLoopInit(declStmt(hasSingleDecl(isUnsignedDecl)));

    // Matches "for (...; var < const_int; ...)"
    auto constCompare = hasCondition(
        binaryOperator(
                        hasOperatorName("<"),
                        hasLHS(ignoringParenImpCasts(declRefExpr(to(varDecl(hasType(isUnsignedInteger())))))),
                        hasRHS(ignoringParenImpCasts(integerLiteral().bind(BOUND_VALUE)))
                      )
                    );
    auto incrementConstraint = hasIncrement(unaryOperator(
      hasOperatorName("++"),
      hasUnaryOperand(declRefExpr(to(varDecl(hasType(isUnsignedInteger())).bind("incrementVariable"))))));
    // clang-format on

    StatementMatcher loopMatcher = forStmt(constCompare).bind(FOR_LOOP);

    finder.addMatcher(loopMatcher, this);
  }

  SourceRange getLoopBodyRange(const ForStmt *fs, SourceManager &smgr,
                               ASTContext *context) {
    const auto *loopBody = fs->getBody();
    if (auto const *compound = llvm::dyn_cast<CompoundStmt>(loopBody)) {
      // return SourceRange( compound->body_front()->getBeginLoc(),
      //     Lexer::getLocForEndOfToken(compound->body_back()->getEndLoc(), 0,
      //                                smgr, options));
      SourceLocation begin = compound->body_front()->getBeginLoc();
      SourceLocation end = compound->body_back()->getEndLoc();
      compound->dumpPretty(*context);
      auto range = SourceRange(begin, end);

      auto actualEnd = Lexer::findLocationAfterToken(
          end, tok::TokenKind::semi, smgr, context->getLangOpts(), true);
      return SourceRange(begin, actualEnd);
    }
    return loopBody->getSourceRange();
  }

  void run(const MatchFinder::MatchResult &result) override {
    if (const ForStmt *fs = result.Nodes.getNodeAs<clang::ForStmt>(FOR_LOOP)) {
      // fs->dump();
      llvm::errs() << "Matched!\n";
      const auto *loopBody = fs->getBody();
      for (const Stmt *s : loopBody->children()) {
        if (isa<ForStmt>(s)) {
          return;
        }
      }
      SourceManager &sm = *result.SourceManager;
      LangOptions options = result.Context->getLangOpts();
      // SourceRange range(
      //     loopBody->getBeginLoc(),
      //     Lexer::getLocForEndOfToken(loopBody->getEndLoc(), 0, sm,
      //                                result.Context->getLangOpts()));

      auto range = getLoopBodyRange(fs, sm, result.Context);

      auto bodyText = Lexer::getSourceText(
                          CharSourceRange::getTokenRange(range), sm, options)
                          .str();

      llvm::errs() << "New loop body!\n";
      std::string newBodyText = bodyText + "\n" + bodyText + "\n" + bodyText;
      StringRef replacedRef(newBodyText);
      llvm::errs() << newBodyText << "\n";

      // rewrite.ReplaceText(CharSourceRange::getTokenRange(range), bodyText);
      rewrite.ReplaceText(range, replacedRef);
    }
  }
};

class MyFrontendAction : public ASTFrontendAction {
public:
  void EndSourceFileAction() override {
    SourceManager &sourceMgr = rewriter.getSourceMgr();
    llvm::outs() << "// --- Rewritten File ---\n";
    rewriter.getEditBuffer(sourceMgr.getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &ci, StringRef inputFileName) override {
    rewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());

    callback.registerSimpleLoopUnrollingRewrite(finder);

    return finder.newASTConsumer();
  }

private:
  Rewriter rewriter;
  SimpleLoopUnroller callback{rewriter};
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