# ============================================================================
# siglip-cli - Command-Line Interface für SigLIP Vision Encoder
# ============================================================================
#
# Standalone CLI Tool für SigLIP Image Embeddings
# Modular aufgeteilt: main.cpp (Workflow), args.cpp (Parsing), output.cpp (Formate)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Prüfe ob siglip Target existiert
if(NOT TARGET siglip)
    message(FATAL_ERROR
        "siglip-cli muss als Teil des Haupt-Projekts gebaut werden.\n"
        "Bitte baue vom Root-Verzeichnis: cmake -B build && cmake --build build"
    )
endif()

# ============================================================================
# Source Files (modular)
# ============================================================================
set(SIGLIP_CLI_SOURCES
    main.cpp
    args.cpp
    output.cpp
)

set(SIGLIP_CLI_HEADERS
    args.h
    output.h
)

add_executable(siglip-cli ${SIGLIP_CLI_SOURCES} ${SIGLIP_CLI_HEADERS})

# C++17 und Link gegen siglip
target_compile_features(siglip-cli PRIVATE cxx_std_17)
target_link_libraries(siglip-cli PRIVATE siglip)

# Include für siglip.h
target_include_directories(siglip-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../siglip
)

# Backend Definitions durchreichen
target_compile_definitions(siglip-cli PRIVATE
    $<$<BOOL:${SIGLIP_CUDA}>:GGML_USE_CUDA>
    $<$<BOOL:${SIGLIP_METAL}>:GGML_USE_METAL>
)

# Thread Support
find_package(Threads REQUIRED)
target_link_libraries(siglip-cli PRIVATE Threads::Threads)

# Plattform-spezifisch
if(WIN32)
    target_compile_definitions(siglip-cli PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_link_libraries(siglip-cli PRIVATE m)
endif()

# Output Directory
set_target_properties(siglip-cli PROPERTIES
    OUTPUT_NAME "siglip-cli"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)
install(TARGETS siglip-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT cli)

# ============================================================================
# Compiler-Warnungen
# ============================================================================
if(MSVC)
    target_compile_options(siglip-cli PRIVATE /W4 /WX- /wd4100 /wd4244 /utf-8)
else()
    target_compile_options(siglip-cli PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

message(STATUS "siglip-cli konfiguriert: ${CMAKE_BINARY_DIR}/bin/siglip-cli")
