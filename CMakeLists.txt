cmake_minimum_required(VERSION 3.18...3.30)
project(cmf LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING "Configure this project with pip/scikit-build-core, not raw CMake.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _cmf_native
  STABLE_ABI
  csrc/src/bindings.cpp
)

target_include_directories(_cmf_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  target_compile_options(_cmf_native PRIVATE -O3 -Wall -Wextra -Wpedantic -fno-math-errno)
endif()

install(TARGETS _cmf_native LIBRARY DESTINATION cmf)
