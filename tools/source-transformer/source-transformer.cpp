// Declares clang::SyntaxOnlyAction.
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <regex>
#include <sstream>
#include <string>

#include "PragmaHandler.h"

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
const std::string INCREMENT_VAR = "increment-var";

class SimpleLoopUnroller : public MatchFinder::MatchCallback {
  Rewriter &rewrite;
  LabelToPragmaMap &pragmas;

public:
  SimpleLoopUnroller(Rewriter &r, LabelToPragmaMap &pragmas)
      : rewrite(r), pragmas(pragmas) {}
  void registerSimpleLoopUnrollingRewrite(MatchFinder &finder) {
    // clang-format off
    // Matches "for (i=0; ....)""
    // auto isUnsignedDecl = varDecl(hasType(isUnsignedInteger()), hasInitializer(integerLiteral(equals(0))));
    auto isUnsignedDecl = varDecl(hasType(isUnsignedInteger())).bind(INCREMENT_VAR);
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

    StatementMatcher loopMatcher =
        forStmt(zeroInit, constCompare, incrementConstraint).bind(FOR_LOOP);

    finder.addMatcher(loopMatcher, this);
  }

  SourceRange getLoopBodyRange(const ForStmt *fs, SourceManager &smgr,
                               ASTContext *context) {
    const auto *loopBody = fs->getBody();
    if (auto const *compound = llvm::dyn_cast<CompoundStmt>(loopBody)) {
      SourceLocation begin = compound->body_front()->getBeginLoc();
      SourceLocation end = compound->body_back()->getEndLoc();
      auto range = SourceRange(begin, end);
      // NOTE: this is a quirk in Clang LibTooling. compound->body_back() skips
      // the trailing ";" token
      auto actualEnd =
          Lexer::findLocationAfterToken(end, tok::TokenKind::semi /* ";" */,
                                        smgr, context->getLangOpts(), true);
      if (actualEnd.isInvalid())
        return SourceRange(begin, end);
      return SourceRange(begin, actualEnd);
    }
    return loopBody->getSourceRange();
  }

  std::string getUnrolledLoopBody(const std::string &originalLoopBody,
                                  unsigned factor,
                                  const std::string &incrementVariable,
                                  int incrementValue) {
    SmallVector<std::string> bodyBlocks;
    std::string patternLeft = R"DELIM((^|[^a-zA-Z0-9_]))DELIM";
    std::string patternRight = R"DELIM((^|[^a-zA-Z0-9_]))DELIM";
    std::regex replacePattern(patternLeft + incrementVariable + patternRight);
    for (unsigned i = 0; i < factor; ++i) {

      std::string newBody =
          std::regex_replace(originalLoopBody, replacePattern,
                             "$1" + incrementVariable + " + " +
                                 std::to_string(i * incrementValue) + "$2");

      bodyBlocks.push_back(newBody);
    }
    bodyBlocks.push_back(incrementVariable +
                         "+=" + std::to_string(factor * incrementValue) + ";");
    return llvm::join(bodyBlocks, "\n");
  }

  std::optional<std::string> getLoopLabel(const ForStmt *fs,
                                          ASTContext *context) {

    auto parents = context->getParents(*fs);
    if (parents.empty()) {
      // no parent available
      return std::nullopt;
    }

    const Stmt *parentStmt = parents[0].get<Stmt>();
    if (!parentStmt)
      return std::nullopt;

    if (const LabelStmt *ls = llvm::dyn_cast<LabelStmt>(parentStmt)) {

      return ls->getName();
    }
    return std::nullopt;
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

      auto range = getLoopBodyRange(fs, sm, result.Context);

      auto bodyText = Lexer::getSourceText(
                          CharSourceRange::getTokenRange(range), sm, options)
                          .str();
      // Get the increment variable:
      const auto *decl = result.Nodes.getNodeAs<clang::VarDecl>(INCREMENT_VAR);
      if (!decl) {
        return;
      }

      auto declName = decl->getDeclName().getAsString();

      auto label = getLoopLabel(fs, result.Context);

      if (!label)
        return;

      if (!pragmas.count(label.value()))
        return;

      llvm::errs() << "Name of the stuff: " << declName << "\n";

      std::string newBodyText = getUnrolledLoopBody(
          bodyText, pragmas[label.value()].factor, declName, 1);
      StringRef replacedRef(newBodyText);
      llvm::errs() << newBodyText << "\n";

      // rewrite.ReplaceText(CharSourceRange::getTokenRange(range), bodyText);
      rewrite.ReplaceText(range, replacedRef);

      auto rangeIncrement = fs->getInc()->getSourceRange();

      rewrite.ReplaceText(rangeIncrement, "");
    }
  }
};

class MyFrontendAction : public ASTFrontendAction {
  Rewriter rewriter;
  LabelToPragmaMap pragmas;
  SimpleLoopUnroller callback{rewriter, pragmas};
  MatchFinder finder;

public:
  void EndSourceFileAction() override {
    SourceManager &sourceMgr = rewriter.getSourceMgr();
    llvm::outs() << "// --- Rewritten File ---\n";
    rewriter.getEditBuffer(sourceMgr.getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &ci, StringRef inputFileName) override {
    // Install the pragma handler
    ci.getPreprocessor().AddPragmaHandler(
        new UnrollPragmaHandler(pragmas, rewriter));
    rewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());

    callback.registerSimpleLoopUnrollingRewrite(finder);

    return finder.newASTConsumer();
  }
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
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]); // <--- this is key

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