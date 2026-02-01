/**
 * siglip-cli - Standalone CLI Tool für SigLIP Vision Encoder
 *
 * Funktionen:
 *   - Einzelbild-Embedding-Generierung
 *   - Batch-Processing für Verzeichnisse
 *   - Similarity-Vergleich zwischen Bildern
 *   - Benchmark-Modus für Performance-Tests
 *   - Multiple Output-Formate (JSON, Binary, NumPy)
 *
 * Beispiele:
 *   siglip-cli -m model.gguf -i image.jpg -o embedding.json
 *   siglip-cli -m model.gguf --dir ./images --format binary
 *   siglip-cli -m model.gguf --compare img1.jpg img2.jpg
 *   siglip-cli -m model.gguf --benchmark -n 100
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "siglip.h"

namespace fs = std::filesystem;

// ============================================================================
// Version und Build-Info
// ============================================================================

#define SIGLIP_CLI_VERSION "1.0.0"

// ============================================================================
// Argument Parser
// ============================================================================

struct cli_args {
    // Model
    std::string model_path;

    // Input
    std::string image_path;
    std::string dir_path;
    std::vector<std::string> compare_images;

    // Output
    std::string output_path;
    std::string format = "json";  // json, binary, numpy

    // Modes
    bool benchmark_mode = false;
    bool similarity_mode = false;
    bool batch_mode = false;
    bool verbose = false;
    bool normalize = true;
    bool show_help = false;
    bool show_version = false;

    // Similarity options
    int top_k = 5;

    // Benchmark options
    int benchmark_iterations = 100;
    int warmup_iterations = 10;

    // Hardware
    int n_threads = 4;
    int n_gpu_layers = -1;
    std::string backend = "cpu";
};

static void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("\n");
    printf("SigLIP CLI - Standalone Image Embedding Generator\n");
    printf("\n");
    printf("Required:\n");
    printf("  -m, --model <path>      Path to SigLIP GGUF model file\n");
    printf("\n");
    printf("Input (one of):\n");
    printf("  -i, --image <path>      Single image file to encode\n");
    printf("  --dir <path>            Directory with images for batch processing\n");
    printf("  --compare <img1> <img2> Compare two images (similarity mode)\n");
    printf("\n");
    printf("Output:\n");
    printf("  -o, --output <path>     Output file path (default: stdout)\n");
    printf("  --format <type>         Output format: json, binary, numpy (default: json)\n");
    printf("\n");
    printf("Similarity Mode:\n");
    printf("  --top-k <n>             Show top-k similar images (default: 5)\n");
    printf("\n");
    printf("Benchmark Mode:\n");
    printf("  --benchmark             Run benchmark mode\n");
    printf("  -n, --iterations <n>    Number of benchmark iterations (default: 100)\n");
    printf("  --warmup <n>            Warmup iterations (default: 10)\n");
    printf("\n");
    printf("Hardware:\n");
    printf("  -t, --threads <n>       Number of CPU threads (default: 4)\n");
    printf("  --gpu-layers <n>        Number of layers to offload to GPU (-1 = all)\n");
    printf("  --backend <type>        Backend: cpu, cuda, metal, vulkan (default: cpu)\n");
    printf("\n");
    printf("Options:\n");
    printf("  --no-normalize          Don't L2-normalize embeddings\n");
    printf("  -v, --verbose           Verbose output\n");
    printf("  -h, --help              Show this help message\n");
    printf("  --version               Show version information\n");
    printf("\n");
    printf("Examples:\n");
    printf("  # Generate embedding for a single image\n");
    printf("  %s -m siglip.gguf -i photo.jpg -o embedding.json\n", program);
    printf("\n");
    printf("  # Batch process a directory\n");
    printf("  %s -m siglip.gguf --dir ./images --format binary -o embeddings.bin\n", program);
    printf("\n");
    printf("  # Compare two images\n");
    printf("  %s -m siglip.gguf --compare dog.jpg cat.jpg\n", program);
    printf("\n");
    printf("  # Run benchmark\n");
    printf("  %s -m siglip.gguf --benchmark -n 1000 --gpu-layers -1\n", program);
    printf("\n");
}

static void print_version() {
    printf("siglip-cli version %s\n", SIGLIP_CLI_VERSION);
    printf("siglip library version %s\n", siglip_version());
    printf("Build info: %s\n", siglip_build_info());
    printf("System: %s\n", siglip_system_info());
}

static bool parse_args(int argc, char** argv, cli_args& args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Help & Version
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return true;
        }
        if (arg == "--version") {
            args.show_version = true;
            return true;
        }

        // Model
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --model requires a path argument\n");
                return false;
            }
            args.model_path = argv[i];
            continue;
        }

        // Image Input
        if (arg == "-i" || arg == "--image") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --image requires a path argument\n");
                return false;
            }
            args.image_path = argv[i];
            continue;
        }

        // Directory Input
        if (arg == "--dir") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --dir requires a path argument\n");
                return false;
            }
            args.dir_path = argv[i];
            args.batch_mode = true;
            continue;
        }

        // Compare Mode
        if (arg == "--compare") {
            if (i + 2 >= argc) {
                fprintf(stderr, "Error: --compare requires two image paths\n");
                return false;
            }
            args.compare_images.push_back(argv[++i]);
            args.compare_images.push_back(argv[++i]);
            args.similarity_mode = true;
            continue;
        }

        // Output
        if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --output requires a path argument\n");
                return false;
            }
            args.output_path = argv[i];
            continue;
        }

        // Format
        if (arg == "--format") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --format requires an argument (json, binary, numpy)\n");
                return false;
            }
            args.format = argv[i];
            if (args.format != "json" && args.format != "binary" && args.format != "numpy") {
                fprintf(stderr, "Error: Invalid format '%s'. Use json, binary, or numpy\n", args.format.c_str());
                return false;
            }
            continue;
        }

        // Top-K
        if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --top-k requires a number\n");
                return false;
            }
            args.top_k = std::atoi(argv[i]);
            continue;
        }

        // Benchmark Mode
        if (arg == "--benchmark") {
            args.benchmark_mode = true;
            continue;
        }

        // Iterations
        if (arg == "-n" || arg == "--iterations") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --iterations requires a number\n");
                return false;
            }
            args.benchmark_iterations = std::atoi(argv[i]);
            continue;
        }

        // Warmup
        if (arg == "--warmup") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --warmup requires a number\n");
                return false;
            }
            args.warmup_iterations = std::atoi(argv[i]);
            continue;
        }

        // Threads
        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --threads requires a number\n");
                return false;
            }
            args.n_threads = std::atoi(argv[i]);
            continue;
        }

        // GPU Layers
        if (arg == "--gpu-layers") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --gpu-layers requires a number\n");
                return false;
            }
            args.n_gpu_layers = std::atoi(argv[i]);
            continue;
        }

        // Backend
        if (arg == "--backend") {
            if (++i >= argc) {
                fprintf(stderr, "Error: --backend requires an argument\n");
                return false;
            }
            args.backend = argv[i];
            continue;
        }

        // No Normalize
        if (arg == "--no-normalize") {
            args.normalize = false;
            continue;
        }

        // Verbose
        if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
            continue;
        }

        // Unknown argument
        fprintf(stderr, "Error: Unknown argument '%s'\n", arg.c_str());
        fprintf(stderr, "Use --help for usage information\n");
        return false;
    }

    return true;
}

static bool validate_args(const cli_args& args) {
    if (args.show_help || args.show_version) {
        return true;
    }

    if (args.model_path.empty()) {
        fprintf(stderr, "Error: Model path is required (-m, --model)\n");
        return false;
    }

    // Check input modes
    int input_modes = 0;
    if (!args.image_path.empty()) input_modes++;
    if (!args.dir_path.empty()) input_modes++;
    if (args.similarity_mode) input_modes++;
    if (args.benchmark_mode) input_modes++;

    if (input_modes == 0) {
        fprintf(stderr, "Error: No input specified. Use -i, --dir, --compare, or --benchmark\n");
        return false;
    }

    if (input_modes > 1 && !args.benchmark_mode) {
        fprintf(stderr, "Error: Multiple input modes specified. Choose one.\n");
        return false;
    }

    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

static siglip_backend parse_backend(const std::string& backend) {
    if (backend == "cuda") return SIGLIP_BACKEND_CUDA;
    if (backend == "metal") return SIGLIP_BACKEND_METAL;
    if (backend == "vulkan") return SIGLIP_BACKEND_VULKAN;
    return SIGLIP_BACKEND_CPU;
}

static std::vector<std::string> get_image_files(const std::string& dir_path) {
    std::vector<std::string> files;

    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"
    };

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

static bool write_output(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", path.c_str());
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), size);
    return true;
}

static bool write_output_string(const std::string& path, const std::string& data) {
    if (path.empty()) {
        printf("%s\n", data.c_str());
        return true;
    }

    std::ofstream file(path);
    if (!file) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", path.c_str());
        return false;
    }
    file << data;
    return true;
}

// ============================================================================
// Main Processing Functions
// ============================================================================

static int run_single_image(siglip_ctx* ctx, const cli_args& args) {
    if (args.verbose) {
        printf("Loading image: %s\n", args.image_path.c_str());
    }

    siglip_image* img = siglip_image_load(args.image_path.c_str());
    if (!img) {
        fprintf(stderr, "Error: Cannot load image '%s': %s\n",
                args.image_path.c_str(), siglip_get_last_error());
        return 1;
    }

    if (args.verbose) {
        printf("Image size: %dx%d (%d channels)\n", img->width, img->height, img->channels);
        printf("Encoding...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();
    siglip_embedding* emb = siglip_encode(ctx, img);
    auto end = std::chrono::high_resolution_clock::now();

    siglip_image_free(img);

    if (!emb) {
        fprintf(stderr, "Error: Encoding failed: %s\n", siglip_get_last_error());
        return 1;
    }

    if (args.normalize) {
        siglip_normalize(emb);
    }

    if (args.verbose) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Encoding time: %lld ms\n", static_cast<long long>(duration.count()));
        printf("Embedding dimension: %d\n", emb->size);
    }

    // Write output
    bool success = false;

    if (args.format == "json") {
        char* json = siglip_embedding_to_json(emb);
        if (json) {
            success = write_output_string(args.output_path, json);
            free(json);
        }
    } else if (args.format == "binary") {
        size_t size;
        uint8_t* data = siglip_embedding_to_binary(emb, &size);
        if (data) {
            if (args.output_path.empty()) {
                fprintf(stderr, "Error: Binary format requires output file (-o)\n");
            } else {
                success = write_output(args.output_path, data, size);
            }
            free(data);
        }
    } else if (args.format == "numpy") {
        size_t size;
        uint8_t* data = siglip_embedding_to_numpy(emb, &size);
        if (data) {
            if (args.output_path.empty()) {
                fprintf(stderr, "Error: NumPy format requires output file (-o)\n");
            } else {
                success = write_output(args.output_path, data, size);
            }
            free(data);
        }
    }

    siglip_embedding_free(emb);

    if (success && args.verbose) {
        if (!args.output_path.empty()) {
            printf("Output written to: %s\n", args.output_path.c_str());
        }
    }

    return success ? 0 : 1;
}

static int run_batch(siglip_ctx* ctx, const cli_args& args) {
    std::vector<std::string> files = get_image_files(args.dir_path);

    if (files.empty()) {
        fprintf(stderr, "Error: No image files found in '%s'\n", args.dir_path.c_str());
        return 1;
    }

    if (args.verbose) {
        printf("Found %zu image files\n", files.size());
    }

    // Für JSON: Array von Embeddings mit Dateinamen
    std::stringstream json_output;
    json_output << "[\n";

    // Für Binary: Header + Embeddings
    std::vector<uint8_t> binary_output;

    int processed = 0;
    int errors = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < files.size(); i++) {
        const std::string& file = files[i];

        if (args.verbose) {
            printf("[%zu/%zu] Processing: %s\n", i + 1, files.size(), file.c_str());
        }

        siglip_image* img = siglip_image_load(file.c_str());
        if (!img) {
            fprintf(stderr, "Warning: Cannot load '%s': %s\n", file.c_str(), siglip_get_last_error());
            errors++;
            continue;
        }

        siglip_embedding* emb = siglip_encode(ctx, img);
        siglip_image_free(img);

        if (!emb) {
            fprintf(stderr, "Warning: Cannot encode '%s': %s\n", file.c_str(), siglip_get_last_error());
            errors++;
            continue;
        }

        if (args.normalize) {
            siglip_normalize(emb);
        }

        if (args.format == "json") {
            if (processed > 0) {
                json_output << ",\n";
            }
            json_output << "  {\n";
            json_output << "    \"file\": \"" << file << "\",\n";
            json_output << "    \"embedding\": [";
            for (int j = 0; j < emb->size; j++) {
                if (j > 0) json_output << ", ";
                json_output << std::setprecision(8) << emb->data[j];
            }
            json_output << "]\n  }";
        } else if (args.format == "binary" || args.format == "numpy") {
            // Append embedding data
            const uint8_t* data = reinterpret_cast<const uint8_t*>(emb->data);
            binary_output.insert(binary_output.end(), data, data + emb->size * sizeof(float));
        }

        siglip_embedding_free(emb);
        processed++;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    if (args.verbose) {
        printf("\nBatch processing complete:\n");
        printf("  Processed: %d images\n", processed);
        printf("  Errors: %d\n", errors);
        printf("  Total time: %lld ms\n", static_cast<long long>(total_duration.count()));
        if (processed > 0) {
            printf("  Avg time: %.2f ms/image\n",
                   static_cast<double>(total_duration.count()) / processed);
        }
    }

    // Write output
    bool success = false;

    if (args.format == "json") {
        json_output << "\n]";
        success = write_output_string(args.output_path, json_output.str());
    } else {
        if (args.output_path.empty()) {
            fprintf(stderr, "Error: Binary/NumPy format requires output file (-o)\n");
            return 1;
        }

        if (args.format == "numpy") {
            // NumPy .npy header
            std::vector<uint8_t> npy_output;

            // Magic number
            npy_output.push_back(0x93);
            npy_output.push_back('N');
            npy_output.push_back('U');
            npy_output.push_back('M');
            npy_output.push_back('P');
            npy_output.push_back('Y');

            // Version 1.0
            npy_output.push_back(0x01);
            npy_output.push_back(0x00);

            int embed_dim = siglip_get_embedding_dim(ctx);
            std::stringstream header;
            header << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                   << processed << ", " << embed_dim << "), }";

            std::string header_str = header.str();
            // Pad to multiple of 64 bytes
            while ((header_str.size() + 10) % 64 != 0) {
                header_str += ' ';
            }
            header_str += '\n';

            uint16_t header_len = static_cast<uint16_t>(header_str.size());
            npy_output.push_back(header_len & 0xFF);
            npy_output.push_back((header_len >> 8) & 0xFF);

            npy_output.insert(npy_output.end(), header_str.begin(), header_str.end());
            npy_output.insert(npy_output.end(), binary_output.begin(), binary_output.end());

            success = write_output(args.output_path, npy_output.data(), npy_output.size());
        } else {
            success = write_output(args.output_path, binary_output.data(), binary_output.size());
        }
    }

    return success ? 0 : 1;
}

static int run_similarity(siglip_ctx* ctx, const cli_args& args) {
    if (args.compare_images.size() < 2) {
        fprintf(stderr, "Error: Need at least 2 images for comparison\n");
        return 1;
    }

    std::vector<siglip_embedding*> embeddings;
    std::vector<std::string> names;

    for (const auto& path : args.compare_images) {
        if (args.verbose) {
            printf("Loading: %s\n", path.c_str());
        }

        siglip_image* img = siglip_image_load(path.c_str());
        if (!img) {
            fprintf(stderr, "Error: Cannot load '%s': %s\n", path.c_str(), siglip_get_last_error());
            // Cleanup
            for (auto emb : embeddings) siglip_embedding_free(emb);
            return 1;
        }

        siglip_embedding* emb = siglip_encode(ctx, img);
        siglip_image_free(img);

        if (!emb) {
            fprintf(stderr, "Error: Cannot encode '%s': %s\n", path.c_str(), siglip_get_last_error());
            for (auto e : embeddings) siglip_embedding_free(e);
            return 1;
        }

        if (args.normalize) {
            siglip_normalize(emb);
        }

        embeddings.push_back(emb);
        names.push_back(fs::path(path).filename().string());
    }

    // Compute pairwise similarities
    printf("\nSimilarity Matrix:\n");
    printf("%-20s", "");
    for (const auto& name : names) {
        printf("%-15s", name.substr(0, 14).c_str());
    }
    printf("\n");

    for (size_t i = 0; i < embeddings.size(); i++) {
        printf("%-20s", names[i].substr(0, 19).c_str());
        for (size_t j = 0; j < embeddings.size(); j++) {
            float sim = siglip_cosine_similarity(embeddings[i], embeddings[j]);
            printf("%-15.4f", sim);
        }
        printf("\n");
    }

    // If only 2 images, show a simple result
    if (embeddings.size() == 2) {
        float similarity = siglip_cosine_similarity(embeddings[0], embeddings[1]);
        printf("\nCosine Similarity: %.6f\n", similarity);

        // Interpretation
        printf("Interpretation: ");
        if (similarity > 0.9) {
            printf("Very similar (likely same object/scene)\n");
        } else if (similarity > 0.7) {
            printf("Similar (related content)\n");
        } else if (similarity > 0.5) {
            printf("Somewhat similar\n");
        } else if (similarity > 0.3) {
            printf("Different but some relation\n");
        } else {
            printf("Very different\n");
        }
    }

    // Cleanup
    for (auto emb : embeddings) {
        siglip_embedding_free(emb);
    }

    return 0;
}

static int run_benchmark(siglip_ctx* ctx, const cli_args& args) {
    printf("SigLIP Benchmark\n");
    printf("================\n\n");

    // Model info
    const siglip_hparams* hparams = siglip_get_hparams(ctx);
    printf("Model: %s\n", siglip_get_model_name(ctx));
    printf("Image size: %d x %d\n", hparams->image_size, hparams->image_size);
    printf("Embedding dim: %d\n", hparams->hidden_size);
    printf("Layers: %d\n", hparams->num_hidden_layers);
    printf("Threads: %d\n", args.n_threads);
    printf("\n");

    // Create synthetic test image
    int img_size = hparams->image_size;
    std::vector<uint8_t> test_data(img_size * img_size * 3);

    // Fill with random-ish pattern (deterministic)
    for (int i = 0; i < img_size * img_size * 3; i++) {
        test_data[i] = static_cast<uint8_t>((i * 17 + 43) % 256);
    }

    siglip_image* img = siglip_image_from_raw(test_data.data(), img_size, img_size, 3);
    if (!img) {
        fprintf(stderr, "Error: Cannot create test image\n");
        return 1;
    }

    // Warmup
    printf("Warmup (%d iterations)...\n", args.warmup_iterations);
    for (int i = 0; i < args.warmup_iterations; i++) {
        siglip_embedding* emb = siglip_encode(ctx, img);
        if (emb) siglip_embedding_free(emb);
    }

    // Benchmark
    printf("Benchmarking (%d iterations)...\n\n", args.benchmark_iterations);

    std::vector<double> times;
    times.reserve(args.benchmark_iterations);

    for (int i = 0; i < args.benchmark_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        siglip_embedding* emb = siglip_encode(ctx, img);
        auto end = std::chrono::high_resolution_clock::now();

        if (!emb) {
            fprintf(stderr, "Error at iteration %d: %s\n", i, siglip_get_last_error());
            siglip_image_free(img);
            return 1;
        }
        siglip_embedding_free(emb);

        auto duration = std::chrono::duration<double, std::milli>(end - start);
        times.push_back(duration.count());

        if (args.verbose && (i + 1) % 10 == 0) {
            printf("  Progress: %d/%d\n", i + 1, args.benchmark_iterations);
        }
    }

    siglip_image_free(img);

    // Statistics
    std::sort(times.begin(), times.end());

    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / times.size();

    double variance = 0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    variance /= times.size();
    double stddev = std::sqrt(variance);

    double min = times.front();
    double max = times.back();
    double median = times[times.size() / 2];
    double p95 = times[static_cast<size_t>(times.size() * 0.95)];
    double p99 = times[static_cast<size_t>(times.size() * 0.99)];

    printf("Results:\n");
    printf("  Mean:     %.3f ms\n", mean);
    printf("  Median:   %.3f ms\n", median);
    printf("  Std Dev:  %.3f ms\n", stddev);
    printf("  Min:      %.3f ms\n", min);
    printf("  Max:      %.3f ms\n", max);
    printf("  P95:      %.3f ms\n", p95);
    printf("  P99:      %.3f ms\n", p99);
    printf("\n");
    printf("Throughput: %.2f images/sec\n", 1000.0 / mean);

    // JSON output if requested
    if (!args.output_path.empty() && args.format == "json") {
        std::stringstream json;
        json << "{\n";
        json << "  \"model\": \"" << siglip_get_model_name(ctx) << "\",\n";
        json << "  \"image_size\": " << img_size << ",\n";
        json << "  \"embedding_dim\": " << hparams->hidden_size << ",\n";
        json << "  \"iterations\": " << args.benchmark_iterations << ",\n";
        json << "  \"threads\": " << args.n_threads << ",\n";
        json << "  \"results\": {\n";
        json << "    \"mean_ms\": " << std::setprecision(4) << mean << ",\n";
        json << "    \"median_ms\": " << median << ",\n";
        json << "    \"stddev_ms\": " << stddev << ",\n";
        json << "    \"min_ms\": " << min << ",\n";
        json << "    \"max_ms\": " << max << ",\n";
        json << "    \"p95_ms\": " << p95 << ",\n";
        json << "    \"p99_ms\": " << p99 << ",\n";
        json << "    \"throughput_ips\": " << (1000.0 / mean) << "\n";
        json << "  }\n";
        json << "}\n";

        write_output_string(args.output_path, json.str());
        printf("\nResults saved to: %s\n", args.output_path.c_str());
    }

    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
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
        fprintf(stderr, "\nUse --help for usage information\n");
        return 1;
    }

    // Setup siglip params
    siglip_params params = siglip_params_default();
    params.n_threads = args.n_threads;
    params.n_gpu_layers = args.n_gpu_layers;
    params.backend = parse_backend(args.backend);
    params.log_level = args.verbose ? SIGLIP_LOG_INFO : SIGLIP_LOG_WARN;
    params.embed_format = args.normalize ? SIGLIP_EMBED_NORMALIZED : SIGLIP_EMBED_F32;

    // Check backend availability
    if (!siglip_backend_available(params.backend)) {
        fprintf(stderr, "Warning: Backend '%s' not available, falling back to CPU\n", args.backend.c_str());
        params.backend = SIGLIP_BACKEND_CPU;
    }

    // Load model
    if (args.verbose) {
        printf("Loading model: %s\n", args.model_path.c_str());
    }

    auto load_start = std::chrono::high_resolution_clock::now();
    siglip_ctx* ctx = siglip_load_model(args.model_path.c_str(), params);
    auto load_end = std::chrono::high_resolution_clock::now();

    if (!ctx) {
        fprintf(stderr, "Error: Cannot load model '%s': %s\n",
                args.model_path.c_str(), siglip_get_last_error());
        return 1;
    }

    if (args.verbose) {
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
        printf("Model loaded in %lld ms\n", static_cast<long long>(load_duration.count()));
        printf("Model: %s\n", siglip_get_model_name(ctx));
        printf("Embedding dimension: %d\n", siglip_get_embedding_dim(ctx));
        printf("\n");
    }

    // Run appropriate mode
    int result = 0;

    if (args.benchmark_mode) {
        result = run_benchmark(ctx, args);
    } else if (args.similarity_mode) {
        result = run_similarity(ctx, args);
    } else if (args.batch_mode) {
        result = run_batch(ctx, args);
    } else {
        result = run_single_image(ctx, args);
    }

    siglip_free(ctx);

    return result;
}
