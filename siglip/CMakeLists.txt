# ============================================================================
# SigLIP Vision Encoder - CMake Build-Konfiguration
# ============================================================================
#
# SigLIP (Sigmoid Loss for Language Image Pre-Training) Library
# Unterstützt CPU, CUDA und Metal Backends
#
# Hauptfunktionen:
#   - siglip_core.cpp      - Kern-Funktionalität und Kontext-Management
#   - siglip_image.cpp     - Bild-Preprocessing
#   - siglip_inference.cpp - Modell-Inferenz
#   - siglip_transformer.cpp - ViT Transformer Implementierung
#   - siglip_gguf.cpp      - GGUF Modell-Loading
#   - siglip_serialize.cpp - Serialisierung
#   - siglip_system.cpp    - System-Utilities
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(siglip VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Build-Optionen
# ============================================================================
option(SIGLIP_CUDA     "CUDA Backend aktivieren"  OFF)
option(SIGLIP_METAL    "Metal Backend aktivieren" OFF)
option(SIGLIP_VULKAN   "Vulkan Backend aktivieren (experimentell)" OFF)
option(SIGLIP_BUILD_CLI "siglip-cli Tool bauen" ON)
option(SIGLIP_SHARED   "Shared Library bauen" OFF)
option(SIGLIP_STATIC   "Static Library bauen" ON)

# ============================================================================
# Plattform-Erkennung
# ============================================================================
if(WIN32)
    set(SIGLIP_PLATFORM_WINDOWS TRUE)
    set(SIGLIP_PLATFORM "Windows")
elseif(APPLE)
    set(SIGLIP_PLATFORM_APPLE TRUE)
    set(SIGLIP_PLATFORM "macOS")
else()
    set(SIGLIP_PLATFORM_LINUX TRUE)
    set(SIGLIP_PLATFORM "Linux")
endif()

message(STATUS "SigLIP: Plattform = ${SIGLIP_PLATFORM}")

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Source Files
# ============================================================================
set(SIGLIP_SOURCES
    siglip_core.cpp
    siglip_image.cpp
    siglip_inference.cpp
    siglip_transformer.cpp
    siglip_gguf.cpp
    siglip_serialize.cpp
    siglip_system.cpp
)

set(SIGLIP_HEADERS
    siglip.h
    siglip_internal.h
)

# ============================================================================
# GGML Integration
# ============================================================================
# Pfad zu GGML relativ zum Projekt-Root
set(GGML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ml/backend/ggml/ggml")

if(EXISTS "${GGML_ROOT}/src")
    set(GGML_INCLUDE_DIRS
        "${GGML_ROOT}/src"
        "${GGML_ROOT}/src/include"
        "${GGML_ROOT}/src/ggml-cpu"
    )
    message(STATUS "SigLIP: GGML gefunden in ${GGML_ROOT}")
else()
    message(WARNING "SigLIP: GGML nicht gefunden - manuelles Linken erforderlich")
endif()

# ============================================================================
# STB Image (Header-Only)
# ============================================================================
set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp.upstream/vendor")
if(NOT EXISTS "${STB_INCLUDE_DIR}/stb_image.h")
    set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor")
endif()

if(EXISTS "${STB_INCLUDE_DIR}")
    message(STATUS "SigLIP: stb_image.h gefunden in ${STB_INCLUDE_DIR}")
else()
    message(WARNING "SigLIP: stb_image.h nicht gefunden")
endif()

# ============================================================================
# Backend-Erkennung: CUDA
# ============================================================================
if(SIGLIP_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "SigLIP: CUDA Toolkit gefunden (${CUDAToolkit_VERSION})")
        set(SIGLIP_HAS_CUDA TRUE)
    else()
        message(WARNING "SigLIP: CUDA angefordert aber nicht gefunden")
        set(SIGLIP_HAS_CUDA FALSE)
    endif()
endif()

# ============================================================================
# Backend-Erkennung: Metal (nur macOS)
# ============================================================================
if(SIGLIP_METAL AND SIGLIP_PLATFORM_APPLE)
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)

    if(FOUNDATION_LIBRARY AND METAL_FRAMEWORK AND METALKIT_FRAMEWORK)
        message(STATUS "SigLIP: Metal Framework gefunden")
        set(SIGLIP_HAS_METAL TRUE)
    else()
        message(WARNING "SigLIP: Metal angefordert aber nicht gefunden")
        set(SIGLIP_HAS_METAL FALSE)
    endif()
elseif(SIGLIP_METAL AND NOT SIGLIP_PLATFORM_APPLE)
    message(WARNING "SigLIP: Metal nur auf macOS verfügbar")
    set(SIGLIP_HAS_METAL FALSE)
endif()

# ============================================================================
# Library Target
# ============================================================================
if(SIGLIP_SHARED)
    add_library(siglip SHARED ${SIGLIP_SOURCES} ${SIGLIP_HEADERS})
    set_target_properties(siglip PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
else()
    add_library(siglip STATIC ${SIGLIP_SOURCES} ${SIGLIP_HEADERS})
endif()

# ============================================================================
# Include Directories
# ============================================================================
target_include_directories(siglip PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(siglip PRIVATE
    ${GGML_INCLUDE_DIRS}
    ${STB_INCLUDE_DIR}
)

# ============================================================================
# GGML Linking
# ============================================================================
if(TARGET ggml)
    target_link_libraries(siglip PUBLIC ggml)
elseif(TARGET ggml-base)
    target_link_libraries(siglip PUBLIC ggml-base)
endif()

# ============================================================================
# CUDA Backend Linking
# ============================================================================
if(SIGLIP_HAS_CUDA)
    target_compile_definitions(siglip PUBLIC GGML_USE_CUDA)
    target_link_libraries(siglip PRIVATE CUDA::cudart)
    if(TARGET ggml-cuda)
        target_link_libraries(siglip PUBLIC ggml-cuda)
    endif()
endif()

# ============================================================================
# Metal Backend Linking
# ============================================================================
if(SIGLIP_HAS_METAL)
    target_compile_definitions(siglip PUBLIC GGML_USE_METAL)
    target_link_libraries(siglip PRIVATE
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
    if(TARGET ggml-metal)
        target_link_libraries(siglip PUBLIC ggml-metal)
    endif()
endif()

# ============================================================================
# Thread Support
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(siglip PRIVATE Threads::Threads)

# ============================================================================
# Math Library (Linux/Unix)
# ============================================================================
if(NOT SIGLIP_PLATFORM_WINDOWS)
    target_link_libraries(siglip PRIVATE m)
endif()

# ============================================================================
# Compile Definitions
# ============================================================================
target_compile_definitions(siglip PRIVATE
    SIGLIP_VERSION="${PROJECT_VERSION}"
    $<$<BOOL:${SIGLIP_PLATFORM_WINDOWS}>:_CRT_SECURE_NO_WARNINGS>
    $<$<BOOL:${SIGLIP_PLATFORM_WINDOWS}>:NOMINMAX>
)

# ============================================================================
# Compiler-Warnungen
# ============================================================================
if(MSVC)
    target_compile_options(siglip PRIVATE /W4 /utf-8)
else()
    target_compile_options(siglip PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
endif()

# ============================================================================
# Subdirectory: siglip-cli
# ============================================================================
if(SIGLIP_BUILD_CLI)
    add_subdirectory(tools/siglip-cli)
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS siglip
    EXPORT siglip-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${SIGLIP_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/siglip
)

# ============================================================================
# Build-Information ausgeben
# ============================================================================
message(STATUS "")
message(STATUS "=== SigLIP Build-Konfiguration ===")
message(STATUS "  Version:     ${PROJECT_VERSION}")
message(STATUS "  Plattform:   ${SIGLIP_PLATFORM}")
message(STATUS "  CUDA:        ${SIGLIP_HAS_CUDA}")
message(STATUS "  Metal:       ${SIGLIP_HAS_METAL}")
message(STATUS "  CLI:         ${SIGLIP_BUILD_CLI}")
message(STATUS "  Library:     $<IF:$<BOOL:${SIGLIP_SHARED}>,SHARED,STATIC>")
message(STATUS "=================================")
message(STATUS "")
