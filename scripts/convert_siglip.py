#!/usr/bin/env python3
"""
SigLIP HuggingFace → GGUF Konverter

Konvertiert SigLIP Vision Encoder Modelle von HuggingFace in GGUF-Format
für die Verwendung mit llama.cpp/ollama-reverse.

Unterstützte Modelle:
- google/siglip-base-patch16-224   (ViT-B/16, 86M params)
- google/siglip-large-patch16-256  (ViT-L/16, 303M params)
- google/siglip-so400m-patch14-384 (ViT-SO400M, 400M params)

Quantisierungsoptionen:
- f32:    Full precision (4 bytes/param)
- f16:    Half precision (2 bytes/param)
- q8_0:   8-bit quantization (~1 byte/param)
- q4_k_m: 4-bit quantization (~0.5 bytes/param)

Verwendung:
    python convert_siglip.py --model google/siglip-base-patch16-224 --output siglip-vit-b.gguf
    python convert_siglip.py --model google/siglip-large-patch16-256 --output siglip-vit-l.gguf --quantize q8_0
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

try:
    import torch
    from transformers import SiglipVisionModel, SiglipImageProcessor, AutoConfig
except ImportError:
    print("Fehler: Bitte installiere die benötigten Pakete:")
    print("  pip install torch transformers safetensors")
    sys.exit(1)


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
    """GGUF Datei-Writer"""

    def __init__(self, path: Path):
        self.path = path
        self.metadata: dict[str, Any] = {}
        self.tensors: list[tuple[str, np.ndarray, int]] = []

    def add_metadata(self, key: str, value: Any):
        """Fügt Metadaten hinzu"""
        self.metadata[key] = value

    def add_tensor(self, name: str, data: np.ndarray, dtype: int = GGML_TYPE_F32):
        """Fügt einen Tensor hinzu"""
        self.tensors.append((name, data, dtype))

    def _write_string(self, f: BinaryIO, s: str):
        """Schreibt einen String im GGUF-Format"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f: BinaryIO, value: Any):
        """Schreibt einen Metadaten-Wert"""
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
                raise ValueError(f"Unsupported array type: {type(value[0])}")
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")

    def _quantize_q8_0(self, data: np.ndarray) -> bytes:
        """Quantisiert zu Q8_0 (8-bit mit Block-Scale)"""
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

    def _quantize_q4_k(self, data: np.ndarray) -> bytes:
        """Quantisiert zu Q4_K_M (4-bit mit K-Quants)"""
        # Vereinfachte Q4_K Implementation
        # Echte Q4_K_M ist komplexer mit Super-Blocks
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

    def write(self):
        """Schreibt die GGUF-Datei"""
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
            tensor_data_offset = f.tell()
            # Alignment auf 32 bytes
            alignment = 32

            # Platzhalter für Tensor-Infos
            tensor_info_start = f.tell()
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

                # Offset (wird später korrigiert)
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
                    f.write(self._quantize_q8_0(data))
                elif dtype == GGML_TYPE_Q4_K:
                    f.write(self._quantize_q4_k(data))
                else:
                    f.write(data.astype(np.float32).tobytes())

        print(f"GGUF geschrieben: {self.path} ({self.path.stat().st_size / 1024 / 1024:.1f} MB)")


def get_tensor_name(hf_name: str) -> str:
    """Konvertiert HuggingFace Tensor-Namen zu GGUF-Namen"""
    # SigLIP Vision Encoder Mapping
    mappings = {
        # Patch Embedding
        'embeddings.patch_embedding.weight': 'siglip.patch_embed.weight',
        'embeddings.patch_embedding.bias': 'siglip.patch_embed.bias',
        'embeddings.position_embedding.weight': 'siglip.pos_embed',

        # Layer Norm
        'post_layernorm.weight': 'siglip.norm.weight',
        'post_layernorm.bias': 'siglip.norm.bias',

        # Head (falls vorhanden)
        'head.weight': 'siglip.head.weight',
        'head.bias': 'siglip.head.bias',
    }

    # Direkte Mappings
    if hf_name in mappings:
        return mappings[hf_name]

    # Encoder Layers
    # encoder.layers.{i}.self_attn.* → siglip.blocks.{i}.attn.*
    # encoder.layers.{i}.mlp.* → siglip.blocks.{i}.mlp.*
    # encoder.layers.{i}.layer_norm1.* → siglip.blocks.{i}.ln1.*
    # encoder.layers.{i}.layer_norm2.* → siglip.blocks.{i}.ln2.*

    parts = hf_name.split('.')

    if len(parts) >= 3 and parts[0] == 'encoder' and parts[1] == 'layers':
        layer_idx = parts[2]
        rest = '.'.join(parts[3:])

        # Self-Attention
        if rest.startswith('self_attn.'):
            attn_part = rest.replace('self_attn.', '')
            # q_proj, k_proj, v_proj, out_proj
            attn_mappings = {
                'q_proj.weight': 'attn.q.weight',
                'q_proj.bias': 'attn.q.bias',
                'k_proj.weight': 'attn.k.weight',
                'k_proj.bias': 'attn.k.bias',
                'v_proj.weight': 'attn.v.weight',
                'v_proj.bias': 'attn.v.bias',
                'out_proj.weight': 'attn.out.weight',
                'out_proj.bias': 'attn.out.bias',
            }
            if attn_part in attn_mappings:
                return f'siglip.blocks.{layer_idx}.{attn_mappings[attn_part]}'

        # MLP
        elif rest.startswith('mlp.'):
            mlp_part = rest.replace('mlp.', '')
            mlp_mappings = {
                'fc1.weight': 'mlp.fc1.weight',
                'fc1.bias': 'mlp.fc1.bias',
                'fc2.weight': 'mlp.fc2.weight',
                'fc2.bias': 'mlp.fc2.bias',
            }
            if mlp_part in mlp_mappings:
                return f'siglip.blocks.{layer_idx}.{mlp_mappings[mlp_part]}'

        # Layer Norms
        elif rest.startswith('layer_norm1'):
            suffix = rest.replace('layer_norm1', '')
            return f'siglip.blocks.{layer_idx}.ln1{suffix}'
        elif rest.startswith('layer_norm2'):
            suffix = rest.replace('layer_norm2', '')
            return f'siglip.blocks.{layer_idx}.ln2{suffix}'

    # Fallback: Original-Name mit Präfix
    return f'siglip.{hf_name}'


def convert_siglip(model_name: str, output_path: Path, quantize: str = 'f16'):
    """Konvertiert SigLIP Modell zu GGUF"""

    print(f"Lade Modell: {model_name}")

    # Modell laden
    config = AutoConfig.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    processor = SiglipImageProcessor.from_pretrained(model_name)

    model.eval()

    # Modell-Info extrahieren
    vision_config = config.vision_config if hasattr(config, 'vision_config') else config

    hidden_size = vision_config.hidden_size
    num_attention_heads = vision_config.num_attention_heads
    num_hidden_layers = vision_config.num_hidden_layers
    intermediate_size = vision_config.intermediate_size
    image_size = vision_config.image_size
    patch_size = vision_config.patch_size

    print(f"  Hidden Size: {hidden_size}")
    print(f"  Attention Heads: {num_attention_heads}")
    print(f"  Layers: {num_hidden_layers}")
    print(f"  MLP Size: {intermediate_size}")
    print(f"  Image Size: {image_size}")
    print(f"  Patch Size: {patch_size}")

    # GGUF Writer erstellen
    writer = GGUFWriter(output_path)

    # Metadaten
    writer.add_metadata('general.architecture', 'siglip')
    writer.add_metadata('general.name', model_name.split('/')[-1])
    writer.add_metadata('general.file_type', quantize)

    writer.add_metadata('siglip.hidden_size', hidden_size)
    writer.add_metadata('siglip.num_attention_heads', num_attention_heads)
    writer.add_metadata('siglip.num_hidden_layers', num_hidden_layers)
    writer.add_metadata('siglip.intermediate_size', intermediate_size)
    writer.add_metadata('siglip.image_size', image_size)
    writer.add_metadata('siglip.patch_size', patch_size)
    writer.add_metadata('siglip.num_patches', (image_size // patch_size) ** 2)

    # Preprocessing-Parameter
    writer.add_metadata('siglip.image_mean', list(processor.image_mean))
    writer.add_metadata('siglip.image_std', list(processor.image_std))

    # Quantisierungs-Typ bestimmen
    quant_map = {
        'f32': GGML_TYPE_F32,
        'f16': GGML_TYPE_F16,
        'q8_0': GGML_TYPE_Q8_0,
        'q4_k_m': GGML_TYPE_Q4_K,
    }
    quant_type = quant_map.get(quantize.lower(), GGML_TYPE_F16)

    print(f"\nKonvertiere Tensoren (Quantisierung: {quantize})...")

    # Tensoren konvertieren
    state_dict = model.state_dict()

    for hf_name, tensor in state_dict.items():
        gguf_name = get_tensor_name(hf_name)
        data = tensor.cpu().numpy()

        # Embedding und Norm-Layer nicht quantisieren
        is_embedding = 'embed' in gguf_name.lower() or 'pos' in gguf_name.lower()
        is_norm = 'norm' in gguf_name.lower() or 'ln' in gguf_name.lower()
        is_bias = 'bias' in gguf_name.lower()

        if is_embedding or is_norm or is_bias:
            dtype = GGML_TYPE_F32
        else:
            dtype = quant_type

        writer.add_tensor(gguf_name, data, dtype)

        dtype_name = {
            GGML_TYPE_F32: 'F32',
            GGML_TYPE_F16: 'F16',
            GGML_TYPE_Q8_0: 'Q8_0',
            GGML_TYPE_Q4_K: 'Q4_K',
        }.get(dtype, 'F32')

        print(f"  {gguf_name}: {data.shape} -> {dtype_name}")

    # GGUF schreiben
    writer.write()

    print(f"\nKonvertierung abgeschlossen!")
    print(f"Output: {output_path}")

    return output_path


def validate_model(gguf_path: Path, model_name: str):
    """Validiert das konvertierte Modell gegen PyTorch-Referenz"""
    print(f"\nValidierung gegen PyTorch-Referenz...")

    # Hier würde die Validierung implementiert werden
    # - GGUF laden
    # - Test-Bild durch beide Modelle
    # - Cosine Similarity vergleichen

    print("  (Validierung noch nicht implementiert - nutze benchmark.py)")


def main():
    parser = argparse.ArgumentParser(
        description='SigLIP HuggingFace → GGUF Konverter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --model google/siglip-base-patch16-224 --output siglip-vit-b.gguf
  %(prog)s --model google/siglip-large-patch16-256 --output siglip-vit-l-q8.gguf --quantize q8_0
  %(prog)s --model google/siglip-so400m-patch14-384 --output siglip-so400m-q4.gguf --quantize q4_k_m
        """
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='HuggingFace Modell-Name (z.B. google/siglip-base-patch16-224)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output GGUF-Datei'
    )

    parser.add_argument(
        '--quantize', '-q',
        choices=['f32', 'f16', 'q8_0', 'q4_k_m'],
        default='f16',
        help='Quantisierungs-Typ (default: f16)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validiere gegen PyTorch-Referenz'
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Konvertieren
    gguf_path = convert_siglip(args.model, output_path, args.quantize)

    # Validieren
    if args.validate:
        validate_model(gguf_path, args.model)


if __name__ == '__main__':
    main()
