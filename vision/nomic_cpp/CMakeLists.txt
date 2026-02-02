# ============================================================================
# MODUL: nomic_cpp CMakeLists.txt
# ZWECK: Build-Konfiguration fuer Nomic Vision Encoder Library
# OUTPUT: libnomic.a / nomic.lib
# ABHAENGIGKEITEN: ggml, ggml-alloc, ggml-backend
# ============================================================================

cmake_minimum_required(VERSION 3.14)
project(nomic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output-Pfad auf build/ setzen
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ============================================================================
# Source-Dateien
# ============================================================================

set(NOMIC_SOURCES
    nomic_core.cpp
    nomic_inference.cpp
    nomic_preprocess.cpp
)

set(NOMIC_HEADERS
    nomic.h
    nomic_internal.h
)

# ============================================================================
# Library Target
# ============================================================================

# Library Name: "nomic" damit Go CGO mit -lnomic funktioniert
add_library(nomic STATIC ${NOMIC_SOURCES} ${NOMIC_HEADERS})

target_include_directories(nomic
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/ml/backend/ggml/ggml/include
        ${CMAKE_SOURCE_DIR}/ml/backend/ggml/ggml/src
        ${CMAKE_SOURCE_DIR}/ml/backend/ggml/ggml/src/include
)

# GGML Libraries - ggml-base ist das Haupt-Target im Ollama Build
if(TARGET ggml-base)
    target_link_libraries(nomic PRIVATE ggml-base)
elseif(TARGET ggml)
    target_link_libraries(nomic PRIVATE ggml)
endif()

# ============================================================================
# Compiler-Optionen
# ============================================================================

if(MSVC)
    target_compile_options(nomic PRIVATE /W4)
else()
    target_compile_options(nomic PRIVATE -Wall -Wextra -O3)
endif()

# AVX2 fuer x86 (beschleunigt GGML)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(MSVC)
        target_compile_options(nomic PRIVATE /arch:AVX2)
    else()
        target_compile_options(nomic PRIVATE -mavx2 -mfma)
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS nomic
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES nomic.h DESTINATION include)

# ============================================================================
# Test-Executable (optional)
# ============================================================================

option(NOMIC_BUILD_TESTS "Build Nomic test executable" OFF)

if(NOMIC_BUILD_TESTS)
    add_executable(nomic_test test/test_nomic.cpp)
    target_link_libraries(nomic_test PRIVATE nomic)
endif()
