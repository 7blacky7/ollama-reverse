/**
 * MODUL: nomic_preprocess.cpp
 * ZWECK: Bild-Preprocessing fuer Nomic Vision Encoder
 * INPUT: RGB Bilddaten (uint8, HWC), Bildgroesse
 * OUTPUT: Float-Array [3, 384, 384] normalisiert (CHW)
 * NEBENEFFEKTE: Speicherallokation fuer Output
 * ABHAENGIGKEITEN: nomic_internal.h
 * HINWEISE: Bilineare Interpolation, ImageNet Normalisierung
 *
 * Preprocessing-Pipeline:
 * 1. Resize auf 384x384 (bilinear)
 * 2. HWC -> CHW Konvertierung
 * 3. [0,255] -> [0,1] Skalierung
 * 4. ImageNet Normalisierung: (x - mean) / std
 */

#include "nomic_internal.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================================
// Konstanten
// ============================================================================

// ImageNet Normalisierung (Nomic verwendet diese Werte)
static const float IMG_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMG_STD[3]  = {0.229f, 0.224f, 0.225f};

// ============================================================================
// Bilineare Interpolation
// ============================================================================

/**
 * Berechnet bilinear interpolierten Pixelwert
 *
 * @param data RGB-Daten (HWC)
 * @param width Bildbreite
 * @param height Bildhoehe
 * @param channels Kanalanzahl (3)
 * @param c Kanal-Index
 * @param x Sub-Pixel X-Koordinate
 * @param y Sub-Pixel Y-Koordinate
 * @return Interpolierter Wert [0, 255]
 */
static float bilinear_sample(
    const uint8_t * data,
    int width, int height, int channels,
    int c, float x, float y
) {
    // Bounding
    x = std::max(0.0f, std::min(x, (float)(width - 1)));
    y = std::max(0.0f, std::min(y, (float)(height - 1)));

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    float fx = x - x0;
    float fy = y - y0;

    // HWC Layout: pixel = data[(y * width + x) * channels + c]
    float p00 = data[(y0 * width + x0) * channels + c];
    float p10 = data[(y0 * width + x1) * channels + c];
    float p01 = data[(y1 * width + x0) * channels + c];
    float p11 = data[(y1 * width + x1) * channels + c];

    // Bilinear
    return p00 * (1-fx) * (1-fy) +
           p10 * fx * (1-fy) +
           p01 * (1-fx) * fy +
           p11 * fx * fy;
}

// ============================================================================
// Preprocessing Funktionen
// ============================================================================

/**
 * Preprocessed ein Bild fuer Nomic Vision
 *
 * @param data RGB-Daten (HWC, uint8)
 * @param width Input-Breite
 * @param height Input-Hoehe
 * @param target_size Zielgroesse (384)
 * @return Float-Array [3, target_size, target_size] oder nullptr bei Fehler
 *
 * Der Aufrufer muss den Speicher mit delete[] freigeben!
 */
float * nomic_preprocess_image(
    const uint8_t * data,
    int width, int height,
    int target_size
) {
    if (!data || width <= 0 || height <= 0 || target_size <= 0) {
        return nullptr;
    }

    int channels = 3;
    size_t out_size = channels * target_size * target_size;
    float * output = new float[out_size];

    // Skalierungsfaktoren
    float scale_x = (float)width / target_size;
    float scale_y = (float)height / target_size;

    // Resize + Normalize in einem Durchgang
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                // Quellposition
                float src_x = (x + 0.5f) * scale_x - 0.5f;
                float src_y = (y + 0.5f) * scale_y - 0.5f;

                // Bilinear interpolieren
                float val = bilinear_sample(data, width, height, channels, c, src_x, src_y);

                // [0,255] -> [0,1]
                val /= 255.0f;

                // ImageNet Normalisierung
                val = (val - IMG_MEAN[c]) / IMG_STD[c];

                // CHW Layout: output[c * H * W + y * W + x]
                output[c * target_size * target_size + y * target_size + x] = val;
            }
        }
    }

    return output;
}

/**
 * Preprocessed mit Center-Crop
 * Crop das groesste zentrierte Quadrat, dann Resize
 */
float * nomic_preprocess_image_center_crop(
    const uint8_t * data,
    int width, int height,
    int target_size
) {
    if (!data || width <= 0 || height <= 0 || target_size <= 0) {
        return nullptr;
    }

    // Crop auf Quadrat (kleinere Dimension)
    int crop_size = std::min(width, height);
    int crop_x = (width - crop_size) / 2;
    int crop_y = (height - crop_size) / 2;

    int channels = 3;
    size_t out_size = channels * target_size * target_size;
    float * output = new float[out_size];

    float scale = (float)crop_size / target_size;

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                // Quellposition im gecropten Bereich
                float src_x = crop_x + (x + 0.5f) * scale - 0.5f;
                float src_y = crop_y + (y + 0.5f) * scale - 0.5f;

                float val = bilinear_sample(data, width, height, channels, c, src_x, src_y);
                val = (val / 255.0f - IMG_MEAN[c]) / IMG_STD[c];
                output[c * target_size * target_size + y * target_size + x] = val;
            }
        }
    }

    return output;
}

/**
 * Denormalisiert ein Bild (fuer Debugging/Visualisierung)
 * Konvertiert von CHW float zu HWC uint8
 */
void nomic_denormalize_image(
    const float * input,
    uint8_t * output,
    int size
) {
    if (!input || !output || size <= 0) return;

    int channels = 3;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            for (int c = 0; c < channels; c++) {
                // CHW -> HWC
                float val = input[c * size * size + y * size + x];

                // Denormalisieren
                val = val * IMG_STD[c] + IMG_MEAN[c];

                // [0,1] -> [0,255]
                val *= 255.0f;
                val = std::max(0.0f, std::min(255.0f, val));

                output[(y * size + x) * channels + c] = (uint8_t)val;
            }
        }
    }
}

// ============================================================================
// Speicher-Freigabe
// ============================================================================

void nomic_preprocess_free(float * preprocessed) {
    delete[] preprocessed;
}
