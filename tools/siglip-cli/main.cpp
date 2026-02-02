/**
 * main.cpp - SigLIP CLI Haupt-Programm
 *
 * Standalone CLI Tool fuer SigLIP Image-Embeddings.
 *
 * Unterstuetzte Modi:
 * - --encode: Einzelbild zu Embedding
 * - --batch: Mehrere Bilder aus Verzeichnis
 * - --similarity: Zwei Bilder vergleichen
 *
 * Beispiele:
 *   siglip-cli --model model.gguf --encode image.jpg --format json
 *   siglip-cli --model model.gguf --batch images/ -o embeddings.bin
 *   siglip-cli --model model.gguf --similarity img1.jpg img2.jpg
 */

#include "args.h"
#include "output.h"
#include "../../siglip/siglip.h"

#include <chrono>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ============================================================================
// Hilfsfunktionen
// ============================================================================

/**
 * Sammelt alle Bilddateien aus einem Verzeichnis
 */
static std::vector<std::string> get_image_files(const std::string & dir) {
    std::vector<std::string> files;
    const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};

    for (const auto & entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================================
// Modus: Einzelbild (--encode)
// ============================================================================

static int run_encode(siglip_ctx * ctx, const cli_args & args) {
    if (args.verbose) {
        printf("Lade Bild: %s\n", args.image_path.c_str());
    }

    siglip_image * img = siglip_image_load(args.image_path.c_str());
    if (!img) {
        fprintf(stderr, "Fehler: Kann Bild nicht laden: %s\n", siglip_get_last_error());
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    siglip_embedding * emb = siglip_encode(ctx, img);
    auto end = std::chrono::high_resolution_clock::now();

    siglip_image_free(img);

    if (!emb) {
        fprintf(stderr, "Fehler: Encoding fehlgeschlagen: %s\n", siglip_get_last_error());
        return 1;
    }

    if (args.normalize) {
        siglip_normalize(emb);
    }

    if (args.verbose) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Encoding: %lld ms\n", static_cast<long long>(ms.count()));
        printf("Dimension: %d\n", emb->size);
    }

    // Output schreiben
    bool ok = false;
    if (args.format == "json") {
        ok = write_json(emb, args.output_path);
    } else if (args.format == "binary") {
        ok = write_binary(emb, args.output_path);
    } else if (args.format == "numpy") {
        ok = write_numpy(emb, args.output_path);
    }

    siglip_embedding_free(emb);
    return ok ? 0 : 1;
}

// ============================================================================
// Modus: Batch (--batch)
// ============================================================================

static int run_batch(siglip_ctx * ctx, const cli_args & args) {
    auto files = get_image_files(args.batch_dir);

    if (files.empty()) {
        fprintf(stderr, "Fehler: Keine Bilder in '%s'\n", args.batch_dir.c_str());
        return 1;
    }

    if (args.verbose) {
        printf("Gefunden: %zu Bilder\n", files.size());
    }

    std::vector<siglip_embedding *> embeddings;
    std::vector<std::string> filenames;
    int errors = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < files.size(); i++) {
        if (args.verbose) {
            printf("[%zu/%zu] %s\n", i + 1, files.size(), files[i].c_str());
        }

        siglip_image * img = siglip_image_load(files[i].c_str());
        if (!img) {
            fprintf(stderr, "Warnung: Kann '%s' nicht laden\n", files[i].c_str());
            errors++;
            continue;
        }

        siglip_embedding * emb = siglip_encode(ctx, img);
        siglip_image_free(img);

        if (!emb) {
            fprintf(stderr, "Warnung: Encoding fehlgeschlagen fuer '%s'\n", files[i].c_str());
            errors++;
            continue;
        }

        if (args.normalize) {
            siglip_normalize(emb);
        }

        embeddings.push_back(emb);
        filenames.push_back(fs::path(files[i]).filename().string());
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (args.verbose) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Verarbeitet: %zu, Fehler: %d\n", embeddings.size(), errors);
        printf("Zeit: %lld ms (%.1f ms/Bild)\n",
               static_cast<long long>(ms.count()),
               embeddings.empty() ? 0.0 : ms.count() / (double)embeddings.size());
    }

    // Output schreiben
    bool ok = false;
    if (args.format == "json") {
        ok = write_json_batch(embeddings, filenames, args.output_path);
    } else if (args.format == "binary") {
        ok = write_binary_batch(embeddings, args.output_path);
    } else if (args.format == "numpy") {
        ok = write_numpy_batch(embeddings, args.output_path);
    }

    // Cleanup
    for (auto * emb : embeddings) {
        siglip_embedding_free(emb);
    }

    return ok ? 0 : 1;
}

// ============================================================================
// Modus: Similarity (--similarity)
// ============================================================================

// Hilfsfunktion: Bild laden und encodieren
static siglip_embedding * load_and_encode(siglip_ctx * ctx, const char * path, bool norm) {
    siglip_image * img = siglip_image_load(path);
    if (!img) {
        fprintf(stderr, "Fehler: Kann '%s' nicht laden\n", path);
        return nullptr;
    }
    siglip_embedding * emb = siglip_encode(ctx, img);
    siglip_image_free(img);
    if (!emb) {
        fprintf(stderr, "Fehler: Encoding fehlgeschlagen fuer '%s'\n", path);
        return nullptr;
    }
    if (norm) siglip_normalize(emb);
    return emb;
}

static int run_similarity(siglip_ctx * ctx, const cli_args & args) {
    siglip_embedding * emb1 = load_and_encode(ctx, args.sim_images[0].c_str(), args.normalize);
    if (!emb1) return 1;

    siglip_embedding * emb2 = load_and_encode(ctx, args.sim_images[1].c_str(), args.normalize);
    if (!emb2) { siglip_embedding_free(emb1); return 1; }

    float sim = siglip_cosine_similarity(emb1, emb2);

    printf("Bild 1: %s\n", fs::path(args.sim_images[0]).filename().string().c_str());
    printf("Bild 2: %s\n", fs::path(args.sim_images[1]).filename().string().c_str());
    printf("Cosine Similarity: %.6f\n", sim);
    printf("Bewertung: %s\n", sim > 0.9f ? "Sehr aehnlich" : sim > 0.7f ? "Aehnlich" :
           sim > 0.5f ? "Teilweise aehnlich" : sim > 0.3f ? "Unterschiedlich" : "Sehr unterschiedlich");

    siglip_embedding_free(emb1);
    siglip_embedding_free(emb2);
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    cli_args args;

    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    if (args.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.show_version) {
        print_version();
        return 0;
    }

    if (!validate_args(args)) {
        fprintf(stderr, "\nVerwende --help fuer Hilfe\n");
        return 1;
    }

    // SigLIP Parameter konfigurieren
    siglip_params params = siglip_params_default();
    params.n_threads = args.n_threads;
    params.n_gpu_layers = args.n_gpu_layers;
    params.log_level = args.verbose ? SIGLIP_LOG_INFO : SIGLIP_LOG_WARN;

    // Modell laden
    if (args.verbose) {
        printf("Lade Modell: %s\n", args.model_path.c_str());
    }

    auto load_start = std::chrono::high_resolution_clock::now();
    siglip_ctx * ctx = siglip_load_model(args.model_path.c_str(), params);
    auto load_end = std::chrono::high_resolution_clock::now();

    if (!ctx) {
        fprintf(stderr, "Fehler: Kann Modell nicht laden: %s\n", siglip_get_last_error());
        return 1;
    }

    if (args.verbose) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
        printf("Modell geladen: %lld ms\n", static_cast<long long>(ms.count()));
        printf("Modell: %s\n", siglip_get_model_name(ctx));
        printf("Embedding Dim: %d\n\n", siglip_get_embedding_dim(ctx));
    }

    // Modus ausfuehren
    int result = 0;

    if (!args.image_path.empty()) {
        result = run_encode(ctx, args);
    } else if (!args.batch_dir.empty()) {
        result = run_batch(ctx, args);
    } else if (!args.sim_images.empty()) {
        result = run_similarity(ctx, args);
    }

    siglip_free(ctx);

    return result;
}
