/**
 * siglip_image.cpp - SigLIP Bild-Verarbeitung und Preprocessing
 *
 * Dieser Teil der SigLIP-Implementierung kuemmert sich um:
 * - Bild-Laden (via stb_image)
 * - Bild-Speicherverwaltung
 * - Preprocessing (Resize, Normalisierung, HWC->CHW Konvertierung)
 */

#include "siglip.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// STB fuer Bild-Loading und Resize
// WICHTIG: STB_IMAGE_IMPLEMENTATION nur einmal definieren!
// Falls bereits in siglip_core.cpp definiert, hier weglassen
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif
#include "stb_image.h"
#include "stb_image_resize2.h"

// ============================================================================
// Interne Hilfsfunktionen
// ============================================================================

// Fehler-String setzen (extern aus siglip_core.cpp)
extern void set_error(const char * fmt, ...);

// ============================================================================
// Oeffentliche API - Bild-Laden und Speicherverwaltung
// ============================================================================

/**
 * Laedt ein Bild von der Festplatte
 * Unterstuetzte Formate: JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PIC
 *
 * @param path Pfad zur Bild-Datei
 * @return Neues siglip_image oder nullptr bei Fehler
 */
siglip_image * siglip_image_load(const char * path) {
    int width, height, channels;

    // Bild laden und auf 3 Kanaele (RGB) konvertieren
    uint8_t * data = stbi_load(path, &width, &height, &channels, 3);

    if (!data) {
        set_error("Konnte Bild nicht laden: %s", path);
        return nullptr;
    }

    // Neue Bild-Struktur erstellen
    siglip_image * img = new siglip_image();
    img->data = data;
    img->width = width;
    img->height = height;
    img->channels = 3;  // Immer RGB nach Konvertierung

    return img;
}

/**
 * Erstellt ein Bild aus Rohdaten (kopiert die Daten)
 *
 * @param data Zeiger auf Pixel-Daten (HWC Format, uint8)
 * @param width Bild-Breite in Pixeln
 * @param height Bild-Hoehe in Pixeln
 * @param channels Anzahl Kanaele (1=Graustufen, 3=RGB, 4=RGBA)
 * @return Neues siglip_image oder nullptr bei Fehler
 */
siglip_image * siglip_image_from_raw(const uint8_t * data, int width, int height, int channels) {
    // Parameter validieren
    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        set_error("Ungueltige Bild-Parameter");
        return nullptr;
    }

    // Neue Bild-Struktur erstellen
    siglip_image * img = new siglip_image();
    size_t size = width * height * channels;

    // Daten kopieren (nicht nur Zeiger uebernehmen!)
    img->data = new uint8_t[size];
    memcpy(img->data, data, size);

    img->width = width;
    img->height = height;
    img->channels = channels;

    return img;
}

/**
 * Gibt den Speicher eines Bildes frei
 *
 * @param img Das freizugebende Bild (kann nullptr sein)
 */
void siglip_image_free(siglip_image * img) {
    if (img) {
        if (img->data) {
            // stbi_image_free ist aequivalent zu free()
            // Funktioniert auch fuer manuell allozierte Daten
            stbi_image_free(img->data);
        }
        delete img;
    }
}

/**
 * Erstellt eine tiefe Kopie eines Bildes
 *
 * @param img Das zu kopierende Bild
 * @return Neue Kopie oder nullptr bei Fehler
 */
siglip_image * siglip_image_clone(const siglip_image * img) {
    if (!img) return nullptr;
    return siglip_image_from_raw(img->data, img->width, img->height, img->channels);
}

// ============================================================================
// Oeffentliche API - Preprocessing
// ============================================================================

/**
 * Fuehrt Preprocessing mit Kontext-Parametern durch
 * Verwendet die im Modell gespeicherten Preprocessing-Parameter
 *
 * @param ctx Der SigLIP-Kontext
 * @param img Das zu verarbeitende Bild
 * @return Float-Array im CHW Format oder nullptr bei Fehler
 *         Muss mit siglip_preprocess_free() freigegeben werden!
 */
float * siglip_preprocess(const siglip_ctx * ctx, const siglip_image * img) {
    if (!ctx || !img) return nullptr;

    // Preprocessing-Parameter aus dem Kontext holen
    // Hinweis: ctx->hparams.preprocess enthaelt die Modell-spezifischen Werte
    extern const siglip_hparams * siglip_get_hparams(const siglip_ctx * ctx);
    const siglip_hparams * hparams = siglip_get_hparams(ctx);
    if (!hparams) return nullptr;

    return siglip_preprocess_with_params(img, &hparams->preprocess);
}

/**
 * Fuehrt Preprocessing mit benutzerdefinierten Parametern durch
 *
 * Ablauf:
 * 1. Resize auf target_size x target_size (bilinear/bicubic)
 * 2. Normalisierung: (pixel/255 - mean) / std
 * 3. Konvertierung von HWC nach CHW Format
 *
 * @param img Das zu verarbeitende Bild
 * @param params Die Preprocessing-Parameter
 * @return Float-Array im CHW Format [3, H, W] oder nullptr bei Fehler
 */
float * siglip_preprocess_with_params(const siglip_image * img, const siglip_preprocess_params * params) {
    if (!img || !params) return nullptr;

    int target_size = params->target_size;

    // ====================================
    // Schritt 1: Resize
    // ====================================
    std::vector<uint8_t> resized(target_size * target_size * 3);

    // stbir_resize_uint8_linear fuer bilineare Interpolation
    // Fuer bicubic waere stbir_resize_uint8_srgb mit STBIR_FILTER_CUBIC noetig
    stbir_resize_uint8_linear(
        img->data,                              // Input
        img->width, img->height,                // Input Dimensionen
        img->width * img->channels,             // Input Stride
        resized.data(),                         // Output
        target_size, target_size,               // Output Dimensionen
        target_size * 3,                        // Output Stride
        static_cast<stbir_pixel_layout>(STBIR_RGB)  // Pixel Layout
    );

    // ====================================
    // Schritt 2+3: Normalisierung und HWC -> CHW
    // ====================================
    // Ausgabe-Format: [Channels, Height, Width]
    float * output = new float[3 * target_size * target_size];

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                // Source Index im HWC Format
                int src_idx = (y * target_size + x) * 3 + c;

                // Destination Index im CHW Format
                int dst_idx = c * target_size * target_size + y * target_size + x;

                // Pixel normalisieren: [0, 255] -> [0, 1] -> normalisiert
                float pixel = resized[src_idx] / 255.0f;
                output[dst_idx] = (pixel - params->mean[c]) / params->std[c];
            }
        }
    }

    return output;
}

/**
 * Gibt den Speicher eines Preprocessing-Ergebnisses frei
 *
 * @param preprocessed Das freizugebende Float-Array (kann nullptr sein)
 */
void siglip_preprocess_free(float * preprocessed) {
    delete[] preprocessed;
}

