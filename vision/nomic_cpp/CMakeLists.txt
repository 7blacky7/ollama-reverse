# ============================================================================
# MODUL: nomic_cpp CMakeLists.txt
# ZWECK: Build-Konfiguration fuer Nomic Vision Encoder Library
# OUTPUT: libnomic_vision.a / nomic_vision.lib
# ABHAENGIGKEITEN: ggml, ggml-alloc, ggml-backend
# ============================================================================

cmake_minimum_required(VERSION 3.14)
project(nomic_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

add_library(nomic_vision STATIC ${NOMIC_SOURCES} ${NOMIC_HEADERS})

target_include_directories(nomic_vision
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/ml/backend/ggml/ggml/include
        ${CMAKE_SOURCE_DIR}/llama.cpp.upstream/ggml/include
)

# GGML Libraries (falls vorhanden)
if(TARGET ggml)
    target_link_libraries(nomic_vision PRIVATE ggml)
endif()

# ============================================================================
# Compiler-Optionen
# ============================================================================

if(MSVC)
    target_compile_options(nomic_vision PRIVATE /W4)
else()
    target_compile_options(nomic_vision PRIVATE -Wall -Wextra -O3)
endif()

# AVX2 fuer x86 (beschleunigt GGML)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(MSVC)
        target_compile_options(nomic_vision PRIVATE /arch:AVX2)
    else()
        target_compile_options(nomic_vision PRIVATE -mavx2 -mfma)
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS nomic_vision
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
    target_link_libraries(nomic_test PRIVATE nomic_vision)
endif()
