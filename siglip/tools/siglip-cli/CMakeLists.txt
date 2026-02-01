# ============================================================================
# siglip-cli - Command-Line Interface für SigLIP Vision Encoder
# ============================================================================

# Prüfe ob als Subdirectory oder standalone gebaut wird
if(NOT TARGET siglip)
    message(FATAL_ERROR
        "siglip-cli muss als Teil des Haupt-Projekts gebaut werden.\n"
        "Bitte baue vom Root-Verzeichnis: cmake -B build && cmake --build build"
    )
endif()

# ============================================================================
# CLI Executable
# ============================================================================
set(SIGLIP_CLI_SOURCES
    main.cpp
)

add_executable(siglip-cli ${SIGLIP_CLI_SOURCES})

# C++17 Standard
target_compile_features(siglip-cli PRIVATE cxx_std_17)

# SigLIP Library linken
target_link_libraries(siglip-cli PRIVATE siglip)

# STB Image Include (für direktes Image-Loading im CLI)
if(STB_INCLUDE_DIR)
    target_include_directories(siglip-cli PRIVATE ${STB_INCLUDE_DIR})
endif()

# Compile Definitions
target_compile_definitions(siglip-cli PRIVATE
    $<$<BOOL:${SIGLIP_CUDA}>:GGML_USE_CUDA>
    $<$<BOOL:${SIGLIP_METAL}>:GGML_USE_METAL>
)

# Plattform-spezifische Einstellungen
if(SIGLIP_PLATFORM_WINDOWS)
    target_compile_definitions(siglip-cli PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
    # Windows Console Application
    set_target_properties(siglip-cli PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
endif()

# Thread Support
find_package(Threads REQUIRED)
target_link_libraries(siglip-cli PRIVATE Threads::Threads)

# Math Library (Linux/Unix)
if(NOT SIGLIP_PLATFORM_WINDOWS)
    target_link_libraries(siglip-cli PRIVATE m)
endif()

# ============================================================================
# RPATH Einstellungen (Unix/macOS)
# ============================================================================
if(NOT SIGLIP_PLATFORM_WINDOWS)
    set_target_properties(siglip-cli PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ============================================================================
# Output Name und Properties
# ============================================================================
set_target_properties(siglip-cli PROPERTIES
    OUTPUT_NAME "siglip-cli"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Debug/Release unterschiedliche Ausgabeverzeichnisse (optional)
set_target_properties(siglip-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
)

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS siglip-cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT cli
)

# Optionale Installation von Shell-Completion Scripts
if(UNIX)
    # Bash Completion (falls vorhanden)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/completions/siglip-cli.bash")
        install(FILES completions/siglip-cli.bash
            DESTINATION ${CMAKE_INSTALL_DATADIR}/bash-completion/completions
            RENAME siglip-cli
            COMPONENT cli
        )
    endif()

    # Zsh Completion (falls vorhanden)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/completions/_siglip-cli")
        install(FILES completions/_siglip-cli
            DESTINATION ${CMAKE_INSTALL_DATADIR}/zsh/site-functions
            COMPONENT cli
        )
    endif()
endif()

# ============================================================================
# Compile-Warnungen aktivieren
# ============================================================================
if(MSVC)
    target_compile_options(siglip-cli PRIVATE
        /W4         # Hohe Warnstufe
        /WX-        # Warnungen nicht als Fehler (für Entwicklung)
        /wd4100     # Unreferenced formal parameter
        /wd4244     # Conversion warnings
        /utf-8      # UTF-8 Source Files
    )
else()
    target_compile_options(siglip-cli PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-function
    )

    # Debug-spezifische Flags
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(siglip-cli PRIVATE
            -g3
            -O0
            -fno-omit-frame-pointer
        )
    endif()
endif()

# ============================================================================
# Build-Information
# ============================================================================
message(STATUS "siglip-cli konfiguriert")
message(STATUS "  Output: ${CMAKE_BINARY_DIR}/bin/siglip-cli")
