#!/usr/bin/env python3
"""
MODUL: quantization.py
ZWECK: Quantisierung von Float32-Tensoren in niedrigere Praezisionsformate (Q8_0, Q4_K)
INPUT: NumPy-Arrays mit Float32-Werten
OUTPUT: Quantisierte Bytes oder dequantisierte NumPy-Arrays
NEBENEFFEKTE: Keine
ABHAENGIGKEITEN: numpy, struct (stdlib)
HINWEISE: Block-basierte Quantisierung - Datenlaenge muss auf Blockgroesse aufgefuellt werden

Unterstuetzte Formate:
- Q8_0:   8-bit mit Block-Scale (32 Elemente pro Block, 34 Bytes/Block)
- Q4_K_M: 4-bit mit K-Quants (256 Elemente pro Super-Block, 144 Bytes/Block)
"""

import struct
import numpy as np


def quantize_q8_0(data: np.ndarray) -> bytes:
    """
    Quantisiert zu Q8_0 (8-bit mit Block-Scale)

    Q8_0 verwendet 32-Element-Blöcke mit einem gemeinsamen Scale-Faktor.
    Jeder Block besteht aus:
    - 2 Bytes: Scale (float16)
    - 32 Bytes: Quantisierte Werte (int8)

    Args:
        data: NumPy-Array mit Float32-Werten

    Returns:
        Quantisierte Daten als Bytes
    """
    data = data.astype(np.float32).flatten()
    n_blocks = (len(data) + 31) // 32
    data = np.pad(data, (0, n_blocks * 32 - len(data)))
    data = data.reshape(-1, 32)

    result = bytearray()
    for block in data:
        # Scale berechnen
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax != 0 else 1.0

        # Quantisieren
        quantized = np.round(block / scale).astype(np.int8)

        # Block schreiben: scale (f16) + 32 int8
        result.extend(struct.pack('<e', np.float16(scale)))
        result.extend(quantized.tobytes())

    return bytes(result)


def quantize_q4_k(data: np.ndarray) -> bytes:
    """
    Quantisiert zu Q4_K_M (4-bit mit K-Quants)

    Q4_K_M (K-Quants, Medium) verwendet 256-Element Super-Blöcke
    mit 8 Sub-Blöcken zu je 32 Elementen. Jeder Sub-Block hat
    eigene Scale- und Min-Werte für höhere Genauigkeit.

    Super-Block Struktur:
    - 2 Bytes: d (Super-Scale, float16)
    - 2 Bytes: dmin (Super-Min, float16)
    - 8 Bytes: Sub-Block Scales (6-bit gepackt)
    - 128 Bytes: Quantisierte Werte (4-bit gepackt, 2 pro Byte)

    Args:
        data: NumPy-Array mit Float32-Werten

    Returns:
        Quantisierte Daten als Bytes

    Hinweis:
        Dies ist eine vereinfachte Implementation von Q4_K_M.
        Die echte llama.cpp Implementation hat komplexere
        Optimierungen für SIMD-Vektorisierung.
    """
    data = data.astype(np.float32).flatten()
    n_blocks = (len(data) + 255) // 256
    data = np.pad(data, (0, n_blocks * 256 - len(data)))
    data = data.reshape(-1, 256)

    result = bytearray()
    for block in data:
        # Super-Block Scale
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax != 0 else 1.0

        # Sub-Blocks (8x32)
        sub_blocks = block.reshape(8, 32)
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)

        for i, sub in enumerate(sub_blocks):
            scales[i] = (np.max(sub) - np.min(sub)) / 15.0
            mins[i] = np.min(sub)

        # Header schreiben
        result.extend(struct.pack('<e', np.float16(d)))  # d
        result.extend(struct.pack('<e', np.float16(np.min(mins))))  # dmin

        # Scales (6 bits each, packed)
        for s in scales:
            result.extend(struct.pack('<B', min(63, int(s / d * 63))))

        # Quantisierte Daten (4-bit packed)
        for sub_idx, sub in enumerate(sub_blocks):
            scale = scales[sub_idx] if scales[sub_idx] != 0 else 1.0
            quantized = np.clip(np.round((sub - mins[sub_idx]) / scale), 0, 15).astype(np.uint8)
            # Pack 2 values per byte
            for j in range(0, 32, 2):
                packed = (quantized[j] & 0xF) | ((quantized[j+1] & 0xF) << 4)
                result.append(packed)

    return bytes(result)


def dequantize_q8_0(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantisiert Q8_0 zurück zu Float32

    Args:
        data: Quantisierte Bytes
        shape: Original-Shape des Tensors

    Returns:
        NumPy-Array mit Float32-Werten
    """
    n_elements = np.prod(shape)
    n_blocks = (n_elements + 31) // 32

    result = np.zeros(n_blocks * 32, dtype=np.float32)

    offset = 0
    for i in range(n_blocks):
        # Scale lesen (float16)
        scale = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        offset += 2

        # Quantisierte Werte lesen (int8)
        quantized = np.frombuffer(data[offset:offset+32], dtype=np.int8)
        offset += 32

        # Dequantisieren
        result[i*32:(i+1)*32] = quantized.astype(np.float32) * scale

    return result[:n_elements].reshape(shape)


def dequantize_q4_k(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantisiert Q4_K_M zurück zu Float32

    Args:
        data: Quantisierte Bytes
        shape: Original-Shape des Tensors

    Returns:
        NumPy-Array mit Float32-Werten

    Hinweis:
        Vereinfachte Dequantisierung - kann leichte Ungenauigkeiten
        gegenüber der originalen llama.cpp Implementation haben.
    """
    n_elements = np.prod(shape)
    n_blocks = (n_elements + 255) // 256

    result = np.zeros(n_blocks * 256, dtype=np.float32)

    offset = 0
    for block_idx in range(n_blocks):
        # Header lesen
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        offset += 2
        dmin = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        offset += 2

        # Scales lesen
        scales_raw = np.frombuffer(data[offset:offset+8], dtype=np.uint8)
        offset += 8
        scales = scales_raw.astype(np.float32) / 63.0 * d

        # Quantisierte Daten lesen und dequantisieren
        for sub_idx in range(8):
            packed = np.frombuffer(data[offset:offset+16], dtype=np.uint8)
            offset += 16

            # Unpack 4-bit values
            low = packed & 0xF
            high = (packed >> 4) & 0xF
            quantized = np.zeros(32, dtype=np.float32)
            quantized[0::2] = low
            quantized[1::2] = high

            # Dequantisieren
            block_start = block_idx * 256 + sub_idx * 32
            result[block_start:block_start+32] = quantized * scales[sub_idx] + dmin

    return result[:n_elements].reshape(shape)
