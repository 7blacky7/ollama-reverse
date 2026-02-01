#!/usr/bin/env python3
"""
GGUF Writer - Serialisierung für GGUF-Format

Dieses Modul enthält die GGUFWriter-Klasse und alle GGUF/GGML Konstanten
für die Serialisierung von Machine Learning Modellen im GGUF-Format.

Das GGUF-Format (GGML Universal Format) ist ein Dateiformat für
quantisierte ML-Modelle, entwickelt für llama.cpp und kompatible Tools.
"""

import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from quantization import quantize_q8_0, quantize_q4_k


# GGUF Konstanten
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGML Tensor-Typen
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 29

# Bytes pro Element für jeden Typ
GGML_TYPE_SIZE = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_Q8_0: 34,  # Block size: 32 elements
    GGML_TYPE_Q4_K: 144,  # Block size: 256 elements (Q4_K_M)
}

GGML_BLCK_SIZE = {
    GGML_TYPE_F32: 1,
    GGML_TYPE_F16: 1,
    GGML_TYPE_Q8_0: 32,
    GGML_TYPE_Q4_K: 256,
}

# GGUF Metadaten-Typen
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12


class GGUFWriter:
    """
    GGUF Datei-Writer

    Schreibt Machine Learning Modelle im GGUF-Format.
    Unterstützt verschiedene Datentypen und Quantisierungsstufen.

    Verwendung:
        writer = GGUFWriter(Path("model.gguf"))
        writer.add_metadata("general.architecture", "siglip")
        writer.add_tensor("weight", numpy_array, GGML_TYPE_F16)
        writer.write()
    """

    def __init__(self, path: Path):
        """
        Initialisiert den GGUF-Writer

        Args:
            path: Pfad zur Output-Datei
        """
        self.path = path
        self.metadata: dict[str, Any] = {}
        self.tensors: list[tuple[str, np.ndarray, int]] = []

    def add_metadata(self, key: str, value: Any):
        """
        Fügt Metadaten hinzu

        Args:
            key: Metadaten-Schlüssel (z.B. "general.architecture")
            value: Wert (str, int, float, bool, list)
        """
        self.metadata[key] = value

    def add_tensor(self, name: str, data: np.ndarray, dtype: int = GGML_TYPE_F32):
        """
        Fügt einen Tensor hinzu

        Args:
            name: Tensor-Name im GGUF-Format
            data: NumPy-Array mit den Daten
            dtype: GGML Datentyp (z.B. GGML_TYPE_F16)
        """
        self.tensors.append((name, data, dtype))

    def _write_string(self, f: BinaryIO, s: str):
        """Schreibt einen String im GGUF-Format (length-prefixed UTF-8)"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f: BinaryIO, value: Any):
        """Schreibt einen Metadaten-Wert mit Typ-Tag"""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUF_TYPE_BOOL))
            f.write(struct.pack('<B', 1 if value else 0))
        elif isinstance(value, int):
            if value >= 0:
                f.write(struct.pack('<I', GGUF_TYPE_UINT64))
                f.write(struct.pack('<Q', value))
            else:
                f.write(struct.pack('<I', GGUF_TYPE_INT64))
                f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUF_TYPE_STRING))
            self._write_string(f, value)
        elif isinstance(value, (list, tuple)):
            f.write(struct.pack('<I', GGUF_TYPE_ARRAY))
            if len(value) == 0:
                f.write(struct.pack('<I', GGUF_TYPE_UINT32))
                f.write(struct.pack('<Q', 0))
            elif isinstance(value[0], int):
                f.write(struct.pack('<I', GGUF_TYPE_INT64))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    f.write(struct.pack('<q', v))
            elif isinstance(value[0], float):
                f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    f.write(struct.pack('<f', v))
            elif isinstance(value[0], str):
                f.write(struct.pack('<I', GGUF_TYPE_STRING))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    self._write_string(f, v)
            else:
                raise ValueError(f"Nicht unterstützter Array-Typ: {type(value[0])}")
        else:
            raise ValueError(f"Nicht unterstützter Metadaten-Typ: {type(value)}")

    def write(self):
        """
        Schreibt die GGUF-Datei

        Erstellt die komplette GGUF-Datei mit Header, Metadaten,
        Tensor-Infos und quantisierten Tensor-Daten.
        """
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))

            # Metadaten
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, value)

            # Tensor-Infos berechnen
            alignment = 32

            # Tensor-Infos und Offsets
            tensor_infos = []
            current_offset = 0

            for name, data, dtype in self.tensors:
                # Tensor-Info schreiben
                self._write_string(f, name)

                # Dimensionen
                shape = data.shape
                n_dims = len(shape)
                f.write(struct.pack('<I', n_dims))
                for dim in shape:
                    f.write(struct.pack('<Q', dim))

                # Typ
                f.write(struct.pack('<I', dtype))

                # Offset
                f.write(struct.pack('<Q', current_offset))

                # Größe berechnen
                n_elements = np.prod(shape)
                if dtype == GGML_TYPE_F32:
                    size = n_elements * 4
                elif dtype == GGML_TYPE_F16:
                    size = n_elements * 2
                elif dtype == GGML_TYPE_Q8_0:
                    n_blocks = (n_elements + 31) // 32
                    size = n_blocks * 34  # 2 (scale) + 32 (data)
                elif dtype == GGML_TYPE_Q4_K:
                    n_blocks = (n_elements + 255) // 256
                    size = n_blocks * 144
                else:
                    size = n_elements * 4  # Fallback zu F32

                tensor_infos.append((name, data, dtype, current_offset, size))
                current_offset += size
                # Alignment
                current_offset = (current_offset + alignment - 1) // alignment * alignment

            # Padding bis Alignment
            current_pos = f.tell()
            padding = (alignment - (current_pos % alignment)) % alignment
            f.write(b'\x00' * padding)

            # Tensor-Daten schreiben
            data_start = f.tell()
            for name, data, dtype, offset, size in tensor_infos:
                # Zur korrekten Position
                target_pos = data_start + offset
                current_pos = f.tell()
                if target_pos > current_pos:
                    f.write(b'\x00' * (target_pos - current_pos))

                # Daten schreiben
                if dtype == GGML_TYPE_F32:
                    f.write(data.astype(np.float32).tobytes())
                elif dtype == GGML_TYPE_F16:
                    f.write(data.astype(np.float16).tobytes())
                elif dtype == GGML_TYPE_Q8_0:
                    f.write(quantize_q8_0(data))
                elif dtype == GGML_TYPE_Q4_K:
                    f.write(quantize_q4_k(data))
                else:
                    f.write(data.astype(np.float32).tobytes())

        print(f"GGUF geschrieben: {self.path} ({self.path.stat().st_size / 1024 / 1024:.1f} MB)")
