set(LLVM_LINK_COMPONENTS support)

add_clang_executable(source-transformer
  source-transformer.cpp
)
target_link_libraries(source-transformer
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  LLVMFrontendOpenMP
  clangRewrite
)