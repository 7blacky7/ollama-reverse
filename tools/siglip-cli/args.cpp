/**
 * args.cpp - CLI Argument-Parsing Implementation
 *
 * Implementiert:
 * - parse_args(): Parst alle CLI-Argumente
 * - validate_args(): Prueft auf konsistente Eingaben
 * - print_usage(): Hilfe-Text
 * - print_version(): Versions-Info
 */

#include "args.h"
#include "../../siglip/siglip.h"

#include <cstdio>
#include <cstdlib>

// ============================================================================
// Version
// ============================================================================

#define SIGLIP_CLI_VERSION "1.0.0"

// ============================================================================
// Hilfe und Version
// ============================================================================

void print_usage(const char * program) {
    printf("SigLIP CLI - Standalone Image Embedding Tool\n\n");

    printf("VERWENDUNG:\n");
    printf("  %s --model <path> [MODE] [OPTIONS]\n\n", program);

    printf("MODI:\n");
    printf("  --encode <image>           Einzelbild zu Embedding\n");
    printf("  --batch <dir>              Alle Bilder im Verzeichnis\n");
    printf("  --similarity <img1> <img2> Zwei Bilder vergleichen\n\n");

    printf("OPTIONEN:\n");
    printf("  -m, --model <path>   Pfad zur GGUF-Modelldatei (Pflicht)\n");
    printf("  -o, --output <path>  Output-Datei (Default: stdout)\n");
    printf("  --format <type>      json, binary, numpy (Default: json)\n");
    printf("  --no-normalize       Embeddings nicht L2-normalisieren\n");
    printf("  -t, --threads <n>    CPU Threads (Default: 4)\n");
    printf("  --gpu-layers <n>     Layers auf GPU (-1 = alle)\n");
    printf("  -v, --verbose        Ausfuehrliche Ausgabe\n");
    printf("  -h, --help           Diese Hilfe anzeigen\n");
    printf("  --version            Version anzeigen\n\n");

    printf("BEISPIELE:\n");
    printf("  # Embedding fuer ein Bild\n");
    printf("  %s -m model.gguf --encode image.jpg\n\n", program);

    printf("  # Batch-Verarbeitung mit Binary-Output\n");
    printf("  %s -m model.gguf --batch ./images -o out.bin --format binary\n\n", program);

    printf("  # Aehnlichkeit zweier Bilder\n");
    printf("  %s -m model.gguf --similarity cat.jpg dog.jpg\n", program);
}

void print_version() {
    printf("siglip-cli %s\n", SIGLIP_CLI_VERSION);
    printf("siglip library %s\n", siglip_version());
    printf("Build: %s\n", siglip_build_info());
}

// ============================================================================
// Argument-Parsing
// ============================================================================

bool parse_args(int argc, char ** argv, cli_args & args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Hilfe und Version
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return true;
        }
        if (arg == "--version") {
            args.show_version = true;
            return true;
        }

        // Modell-Pfad
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --model benoetigt einen Pfad\n");
                return false;
            }
            args.model_path = argv[i];
            continue;
        }

        // Encode-Modus
        if (arg == "--encode") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --encode benoetigt einen Bild-Pfad\n");
                return false;
            }
            args.image_path = argv[i];
            continue;
        }

        // Batch-Modus
        if (arg == "--batch") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --batch benoetigt ein Verzeichnis\n");
                return false;
            }
            args.batch_dir = argv[i];
            continue;
        }

        // Similarity-Modus
        if (arg == "--similarity") {
            if (i + 2 >= argc) {
                fprintf(stderr, "Fehler: --similarity benoetigt zwei Bild-Pfade\n");
                return false;
            }
            args.sim_images.push_back(argv[++i]);
            args.sim_images.push_back(argv[++i]);
            continue;
        }

        // Output
        if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --output benoetigt einen Pfad\n");
                return false;
            }
            args.output_path = argv[i];
            continue;
        }

        // Format
        if (arg == "--format") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --format benoetigt json, binary oder numpy\n");
                return false;
            }
            args.format = argv[i];
            if (args.format != "json" && args.format != "binary" && args.format != "numpy") {
                fprintf(stderr, "Fehler: Ungueltiges Format '%s'\n", args.format.c_str());
                return false;
            }
            continue;
        }

        // Normalize-Flag
        if (arg == "--no-normalize") {
            args.normalize = false;
            continue;
        }

        // Threads
        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --threads benoetigt eine Zahl\n");
                return false;
            }
            args.n_threads = std::atoi(argv[i]);
            continue;
        }

        // GPU Layers
        if (arg == "--gpu-layers") {
            if (++i >= argc) {
                fprintf(stderr, "Fehler: --gpu-layers benoetigt eine Zahl\n");
                return false;
            }
            args.n_gpu_layers = std::atoi(argv[i]);
            continue;
        }

        // Verbose
        if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
            continue;
        }

        // Unbekanntes Argument
        fprintf(stderr, "Fehler: Unbekanntes Argument '%s'\n", arg.c_str());
        return false;
    }

    return true;
}

// ============================================================================
// Validierung
// ============================================================================

bool validate_args(const cli_args & args) {
    // Hilfe/Version braucht keine weitere Validierung
    if (args.show_help || args.show_version) {
        return true;
    }

    // Modell ist Pflicht
    if (args.model_path.empty()) {
        fprintf(stderr, "Fehler: Modell-Pfad erforderlich (-m, --model)\n");
        return false;
    }

    // Mindestens ein Modus muss gesetzt sein
    int modes = 0;
    if (!args.image_path.empty()) modes++;
    if (!args.batch_dir.empty()) modes++;
    if (!args.sim_images.empty()) modes++;

    if (modes == 0) {
        fprintf(stderr, "Fehler: Kein Modus angegeben (--encode, --batch, --similarity)\n");
        return false;
    }

    if (modes > 1) {
        fprintf(stderr, "Fehler: Nur ein Modus erlaubt\n");
        return false;
    }

    // Binary/NumPy braucht Output-Datei
    if ((args.format == "binary" || args.format == "numpy") && args.output_path.empty()) {
        if (!args.sim_images.empty()) {
            // Similarity gibt Text aus, kein Binary
        } else {
            fprintf(stderr, "Fehler: %s Format benoetigt Output-Datei (-o)\n", args.format.c_str());
            return false;
        }
    }

    return true;
}
