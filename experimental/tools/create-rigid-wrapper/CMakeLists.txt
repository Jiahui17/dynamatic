set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(create-rigid-wrapper create-rigid-wrapper.cpp)
llvm_update_compile_flags(create-rigid-wrapper)
target_link_libraries(create-rigid-wrapper PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticHW
)
