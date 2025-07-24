// Declares clang::SyntaxOnlyAction.
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
// Declares llvm::cl::extrahelp.
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <charconv>
#include <optional>
#include <string>

using namespace clang;
using namespace llvm;
using namespace clang::tooling;

inline std::optional<int> strToInt(const std::string &str) {
  int value;
  auto result = std::from_chars(str.data(), str.data() + str.size(), value);
  if (result.ec == std::errc()) {
    return value;
  }
  return std::nullopt;
}

struct UnrollPragmaInfo {
  unsigned factor;
};

using LabelToPragmaMap = std::map<std::string, UnrollPragmaInfo>;

class UnrollPragmaHandler : public PragmaHandler {
  LabelToPragmaMap &pragmaInfo;

  Rewriter &rewriter;

public:
  UnrollPragmaHandler(LabelToPragmaMap &pragmaInfo, Rewriter &rewriter)
      : PragmaHandler("HLS_UNROLL"), pragmaInfo(pragmaInfo),
        rewriter(rewriter) {}

  void HandlePragma(Preprocessor &pp, PragmaIntroducer introducer,
                    Token &firstToken) override {
    Token tok;

    pp.Lex(tok);
    if (tok.isNot(tok::identifier)) {
      pp.Diag(tok.getLocation(), diag::err_expected) << "expected 'identifier'";
      return;
    }
    SourceLocation beginLoc = tok.getLocation();

    std::string identifier = pp.getSpelling(tok);

    // Step 1: Expect identifier "factor"
    pp.Lex(tok);
    if (tok.isNot(tok::identifier) || pp.getSpelling(tok) != "factor") {
      pp.Diag(tok.getLocation(), diag::err_expected_either)
          << "expected 'factor'";
      return;
    }

    // Step 2: Expect '='
    pp.Lex(tok);
    if (tok.isNot(tok::equal)) {
      pp.Diag(tok.getLocation(), diag::err_expected) << "'='";
      return;
    }

    // Step 3: Expect an integer literal (e.g., 3)
    pp.Lex(tok);
    if (tok.isNot(tok::numeric_constant)) {
      pp.Diag(tok.getLocation(), diag::err_expected) << "integer literal";
      return;
    }

    // Extract the integer value
    std::string valStr;
    valStr = pp.getSpelling(tok);
    auto factor = strToInt(valStr);

    if (!factor) {
      pp.Diag(tok.getLocation(), diag::err_invalid_numeric_udl);
      return;
    }
    SourceLocation endLoc = tok.getEndLoc();

    llvm::errs() << "Pragma " << factor << "\n";
    UnrollPragmaInfo info{(unsigned)factor.value()};
    pragmaInfo[identifier] = info;

    rewriter.RemoveText(SourceRange(firstToken.getLocation(), endLoc));
  }
};
