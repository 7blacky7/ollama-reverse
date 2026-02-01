// quant_c.h - C-Definitionen fuer Quantisierungs-Extraktion
//
// Dieses Modul enthaelt:
// - unpack_32_4: Entpackt 32 4-bit Gewichte
// - extract_q4_0_data: Extrahiert Q4_0 Daten (4-bit mit Scale)
// - extract_q4_1_data: Extrahiert Q4_1 Daten (4-bit mit Scale+Bias)
// - extract_q8_0_data: Extrahiert Q8_0 Daten (8-bit mit Scale)
// - dequant_row_q6_K: Dequantisiert Q6_K Zeile
// - dequant_row_q4_K: Dequantisiert Q4_K Zeile
//
// Abgeleitet von: https://github.com/ml-explore/mlx/blob/main/mlx/io/gguf_quants.cpp
// und ggml-quants.c

#ifndef QUANT_C_H
#define QUANT_C_H

#include <stdio.h>
#include <string.h>
#include "mlx/c/array.h"
#include "mlx/c/ops.h"

// =============================================================================
// Q4/Q8 Basis-Quantisierung
// =============================================================================

// unpack_32_4 entpackt 32 4-bit Gewichte aus komprimierten Daten
// Die unteren 16 Gewichte sind in den unteren 4 Bits
// Die oberen 16 Gewichte sind in den oberen 4 Bits
void unpack_32_4(uint8_t* data, int8_t* dst) {
    memset(dst, 0, 16);
    // Erste 16 Gewichte aus unteren 4 Bits
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j + 2] & 0x0F); // j+2 um Scale-Bytes zu ueberspringen
        if (j % 2 != 0) {
            x <<= 4;
        }
        dst[j / 2] += x;
    }
    // Letzte 16 Gewichte aus oberen 4 Bits
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j + 2] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
        }
        dst[8 + j / 2] += x;
    }
}

// extract_q4_0_data extrahiert Daten aus Q4_0 Tensoren
// Layout: |16 bit scale|32 x 4bit weights|
void extract_q4_0_data(
        uint8_t* data,
        mlx_array* weights_arr,
        mlx_array* scales_arr,
        mlx_array* biases_arr) {
    const uint64_t bytes_per_block = 18; // 2 Bytes Scale, 32x0.5 Byte Gewichte
    uint8_t* weights = mlx_array_data_uint8(*weights_arr);
    float16_t* scales = mlx_array_data_float16(*scales_arr);
    float16_t* biases = mlx_array_data_float16(*biases_arr);

    for (int64_t i = 0; i < mlx_array_size(*scales_arr); i++) {
        scales[i] = *((float16_t*)data);
        biases[i] = -8 * scales[i];
        unpack_32_4(data, weights);
        weights += 16;
        data += bytes_per_block;
    }
}

// extract_q4_1_data extrahiert Daten aus Q4_1 Tensoren
// Layout: |16 bit scale|16 bit bias|32 x 4bit weights|
void extract_q4_1_data(
        uint8_t* data,
        mlx_array* weights_arr,
        mlx_array* scales_arr,
        mlx_array* biases_arr) {
    const uint64_t bytes_per_block = 20; // 2 Bytes Scale, 2 Bytes Bias, 32x0.5 Byte Gewichte
    uint8_t* weights = mlx_array_data_uint8(*weights_arr);
    float16_t* scales = mlx_array_data_float16(*scales_arr);
    float16_t* biases = mlx_array_data_float16(*biases_arr);

    for (int64_t i = 0; i < mlx_array_size(*scales_arr); i++) {
        scales[i] = *((float16_t*)data);
        biases[i] = *((float16_t*)(data) + 1);
        unpack_32_4(data, weights);
        weights += 16;
        data += bytes_per_block;
    }
}

// extract_q8_0_data extrahiert Daten aus Q8_0 Tensoren
// Layout: |16 bit scale|32 x 8bit weights|
void extract_q8_0_data(
        uint8_t* data,
        mlx_array* weights_arr,
        mlx_array* scales_arr,
        mlx_array* biases_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34; // 2 Bytes Scale, 32x1 Byte Gewichte
    uint8_t* weights = mlx_array_data_uint8(*weights_arr);
    float16_t* scales = mlx_array_data_float16(*scales_arr);
    float16_t* biases = mlx_array_data_float16(*biases_arr);

    for (int64_t i = 0; i < mlx_array_size(*scales_arr); i++) {
        uint8_t* block_data = data + i * bytes_per_block;
        scales[i] = *((float16_t*)block_data);
        biases[i] = -128 * scales[i];

        for (int64_t j = 0; j < weights_per_block; ++j) {
            uint8_t x = block_data[j + 2]; // j+2 um Scale-Bytes zu ueberspringen
            // Originaldaten sind int8_t, daher Bias von -128 und erstes Bit invertieren
            x ^= 1 << 7;
            weights[i * weights_per_block + j] = x;
        }
    }
}

// =============================================================================
// K-Quantisierung (Q4_K, Q6_K)
// =============================================================================

#define QK_K 256
#define K_SCALE_SIZE 12
#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

// block_q6_K - 6-bit Quantisierung
// 16 Bloecke mit je 16 Elementen
// Effektiv 6.5625 Bits pro Gewicht
typedef struct {
    uint8_t ql[QK_K/2];      // Quants, untere 4 Bits
    uint8_t qh[QK_K/4];      // Quants, obere 2 Bits
    int8_t  scales[QK_K/16]; // Scales, 8-bit quantisiert
    uint16_t d;              // Super-Block Scale
} block_q6_K;

// block_q4_K - 4-bit Quantisierung
// 8 Bloecke mit je 32 Elementen
// Effektiv 4.5 Bits pro Gewicht
typedef struct {
    union {
        struct {
            uint16_t d;    // Super-Block Scale fuer quantisierte Scales
            uint16_t dmin; // Super-Block Scale fuer quantisierte Mins
        } GGML_COMMON_AGGR_S;
        uint16_t dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // Scales und Mins, 6-bit quantisiert
    uint8_t qs[QK_K/2];           // 4-bit Quants
} block_q4_K;

// dequant_row_q6_K dequantisiert eine Zeile Q6_K Daten
void dequant_row_q6_K(const void * restrict vx, void * restrict vy, int k) {
    const int64_t nb = k / QK_K;
    block_q6_K *x = (block_q6_K *)vx;
    float16_t* y = (float16_t *)vy;

    for (int i = 0; i < nb; i++) {
        float16_t d = 0.0;
        memcpy(&d, &x[i].d, sizeof(d));

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// get_scale_min_k4 holt Scale und Min fuer Q4_K
static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// dequant_row_q4_K dequantisiert eine Zeile Q4_K Daten
void dequant_row_q4_K(const void * restrict vx, void * restrict vy, int k) {
    block_q4_K *x = (block_q4_K *)vx;
    float16_t* y = (float16_t *)vy;
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        float16_t d = 0.0;
        memcpy(&d, &x[i].d, sizeof(d));
        float16_t min = 0.0;
        memcpy(&min, &x[i].dmin, sizeof(d));

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float16_t d1 = d * sc;
            const float16_t m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float16_t d2 = d * sc;
            const float16_t m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

#endif // QUANT_C_H
