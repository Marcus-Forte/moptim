option(FORMAT_CODE "Code formatter" OFF)

if(FORMAT_CODE)
  find_program(FORMATTER clang-format)
  if(FORMATTER)
    file(GLOB_RECURSE SRC_FILES
      ${CMAKE_SOURCE_DIR}/**/*.cc
      ${CMAKE_SOURCE_DIR}/**/*.hh)
    add_custom_target(code-format ALL
    clang-format -i ${SRC_FILES})
  endif()
endif()