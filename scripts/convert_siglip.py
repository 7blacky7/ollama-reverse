#!/usr/bin/env python3
"""
SigLIP HuggingFace -> GGUF Konverter

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
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import SiglipVisionModel, SiglipImageProcessor, AutoConfig
except ImportError:
    print("Fehler: Bitte installiere die benötigten Pakete:")
    print("  pip install torch transformers safetensors")
    sys.exit(1)

from gguf_writer import (
    GGUFWriter,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K,
)


# HuggingFace -> GGUF Tensor-Name Mapping
TENSOR_MAPPINGS = {
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

# Attention-Layer Mapping
ATTN_MAPPINGS = {
    'q_proj.weight': 'attn.q.weight',
    'q_proj.bias': 'attn.q.bias',
    'k_proj.weight': 'attn.k.weight',
    'k_proj.bias': 'attn.k.bias',
    'v_proj.weight': 'attn.v.weight',
    'v_proj.bias': 'attn.v.bias',
    'out_proj.weight': 'attn.out.weight',
    'out_proj.bias': 'attn.out.bias',
}

# MLP-Layer Mapping
MLP_MAPPINGS = {
    'fc1.weight': 'mlp.fc1.weight',
    'fc1.bias': 'mlp.fc1.bias',
    'fc2.weight': 'mlp.fc2.weight',
    'fc2.bias': 'mlp.fc2.bias',
}


def get_tensor_name(hf_name: str) -> str:
    """
    Konvertiert HuggingFace Tensor-Namen zu GGUF-Namen

    Das Mapping folgt der Konvention:
    - encoder.layers.{i}.self_attn.* -> siglip.blocks.{i}.attn.*
    - encoder.layers.{i}.mlp.* -> siglip.blocks.{i}.mlp.*
    - encoder.layers.{i}.layer_norm1.* -> siglip.blocks.{i}.ln1.*
    - encoder.layers.{i}.layer_norm2.* -> siglip.blocks.{i}.ln2.*

    Args:
        hf_name: Original HuggingFace Tensor-Name

    Returns:
        GGUF-konformer Tensor-Name
    """
    # Direkte Mappings
    if hf_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[hf_name]

    # Encoder Layers parsen
    parts = hf_name.split('.')

    if len(parts) >= 3 and parts[0] == 'encoder' and parts[1] == 'layers':
        layer_idx = parts[2]
        rest = '.'.join(parts[3:])

        # Self-Attention
        if rest.startswith('self_attn.'):
            attn_part = rest.replace('self_attn.', '')
            if attn_part in ATTN_MAPPINGS:
                return f'siglip.blocks.{layer_idx}.{ATTN_MAPPINGS[attn_part]}'

        # MLP
        elif rest.startswith('mlp.'):
            mlp_part = rest.replace('mlp.', '')
            if mlp_part in MLP_MAPPINGS:
                return f'siglip.blocks.{layer_idx}.{MLP_MAPPINGS[mlp_part]}'

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
    """
    Konvertiert SigLIP Modell zu GGUF

    Lädt das HuggingFace-Modell, extrahiert alle Gewichte,
    führt die gewünschte Quantisierung durch und speichert
    im GGUF-Format.

    Args:
        model_name: HuggingFace Modell-ID (z.B. "google/siglip-base-patch16-224")
        output_path: Pfad für die Output-GGUF-Datei
        quantize: Quantisierungs-Typ ("f32", "f16", "q8_0", "q4_k_m")

    Returns:
        Pfad zur erstellten GGUF-Datei
    """
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

    # Metadaten hinzufügen
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

        # Embedding und Norm-Layer nicht quantisieren (bleiben F32)
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
    """
    Validiert das konvertierte Modell gegen PyTorch-Referenz

    Args:
        gguf_path: Pfad zur GGUF-Datei
        model_name: HuggingFace Modell-ID für Referenz

    Hinweis:
        Validierung noch nicht implementiert - nutze benchmark.py
    """
    print(f"\nValidierung gegen PyTorch-Referenz...")
    print("  (Validierung noch nicht implementiert - nutze benchmark.py)")


def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(
        description='SigLIP HuggingFace -> GGUF Konverter',
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
