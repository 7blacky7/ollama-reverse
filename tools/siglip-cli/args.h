/**
 * args.h - CLI Argument-Parsing fuer SigLIP CLI
 *
 * Dieses Modul definiert:
 * - cli_args Struktur fuer alle CLI-Optionen
 * - Argument-Parsing Funktionen
 * - Validierung der Eingaben
 */

#ifndef SIGLIP_CLI_ARGS_H
#define SIGLIP_CLI_ARGS_H

#include <string>
#include <vector>

// ============================================================================
// CLI Optionen Struktur
// ============================================================================

/**
 * Alle CLI-Argumente in einer Struktur
 */
struct cli_args {
    // Modell-Pfad (Pflicht)
    std::string model_path;

    // Input-Modi (einer davon muss gesetzt sein)
    std::string image_path;           // --encode: Einzelbild
    std::string batch_dir;            // --batch: Verzeichnis
    std::vector<std::string> sim_images; // --similarity: 2 Bilder

    // Output
    std::string output_path;          // -o, --output
    std::string format = "json";      // --format: json, binary, numpy

    // Flags
    bool show_help    = false;
    bool show_version = false;
    bool verbose      = false;
    bool normalize    = true;

    // Hardware
    int n_threads    = 4;
    int n_gpu_layers = -1;            // -1 = alle auf GPU
};

// ============================================================================
// Funktionen
// ============================================================================

/**
 * Zeigt Hilfe-Text an
 *
 * @param program  Name des Programms (argv[0])
 */
void print_usage(const char * program);

/**
 * Zeigt Version an
 */
void print_version();

/**
 * Parst CLI-Argumente
 *
 * @param argc  Argument-Anzahl
 * @param argv  Argument-Vektor
 * @param args  Output: Geparste Argumente
 * @return      true bei Erfolg, false bei Fehler
 */
bool parse_args(int argc, char ** argv, cli_args & args);

/**
 * Validiert geparste Argumente
 *
 * @param args  Zu validierende Argumente
 * @return      true wenn gueltig, false bei Fehler
 */
bool validate_args(const cli_args & args);

#endif // SIGLIP_CLI_ARGS_H
