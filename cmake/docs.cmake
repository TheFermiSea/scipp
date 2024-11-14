# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# ~~~
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter)
function(add_docs_target name)
  set(oneValueArgs BUILDER)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(
    PARSE_ARGV 0 ADD_DOCS_TARGET "" "${oneValueArgs}" "${multiValueArgs}"
  )
  add_custom_target(
    ${name}
    COMMAND
      ${Python_EXECUTABLE} -m sphinx -j2 -v -b ${ADD_DOCS_TARGET_BUILDER} -d
      ${CMAKE_BINARY_DIR}/.doctrees ${CMAKE_SOURCE_DIR}/docs
      ${CMAKE_BINARY_DIR}/html
    DEPENDS ${ADD_DOCS_TARGET_DEPENDS}
  )
endfunction()
# Documentation build configuration
function(add_docs_target target)
  cmake_parse_arguments(PARSE_ARGV 1 DOCS "" "BUILDER" "DEPENDS")
  if(NOT DOCS_BUILDER)
    message(FATAL_ERROR "BUILDER argument required")
  endif()
  
  add_custom_target(
    ${target}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}
            sphinx-build -M ${DOCS_BUILDER}
            "${CMAKE_SOURCE_DIR}/docs"
            "${CMAKE_BINARY_DIR}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Building documentation with Sphinx"
  )
  
  if(DOCS_DEPENDS)
    add_dependencies(${target} ${DOCS_DEPENDS})
  endif()
endfunction()
