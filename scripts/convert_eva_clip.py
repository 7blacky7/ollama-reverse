#!/usr/bin/env python3
"""
MODUL: convert_eva_clip
ZWECK: Konvertiert EVA-CLIP Modelle von HuggingFace zu GGUF
INPUT: HuggingFace Modell-ID, Output-Pfad, Quantisierungs-Typ
OUTPUT: GGUF-Datei mit EVA-CLIP Gewichten
NEBENEFFEKTE: Datei-Schreiboperationen, HuggingFace Download
ABHAENGIGKEITEN: torch (extern), transformers (extern), gguf_writer (intern)
HINWEISE: EVA-CLIP nutzt EVA-Initialisierung, bessere Skalierung

Unterstuetzte Modelle:
- BAAI/EVA02-CLIP-B-16        (ViT-B/16, 86M params)
- BAAI/EVA02-CLIP-L-14        (ViT-L/14, 304M params)
- BAAI/EVA02-CLIP-L-14-336    (ViT-L/14, 336px, 304M params)
- BAAI/EVA02-CLIP-bigE-14     (ViT-bigE/14, 4B params)
- BAAI/EVA02-CLIP-bigE-14-plus (ViT-bigE/14+, 4B params)

Verwendung:
    python convert_eva_clip.py --model BAAI/EVA02-CLIP-L-14-336 --output eva-clip.gguf
    python convert_eva_clip.py --model BAAI/EVA02-CLIP-L-14-336 --output eva-clip-q8.gguf --quantize q8_0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoConfig, AutoProcessor
except ImportError:
    print("Fehler: Bitte installiere die benoetigten Pakete:")
    print("  pip install torch transformers safetensors")
    sys.exit(1)

from gguf_writer import (
    GGUFWriter,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K,
)


# ============================================================================
# Tensor-Mappings: HuggingFace -> GGUF
# ============================================================================

# Globale Tensor-Mappings (Embeddings, Norms)
TENSOR_MAPPINGS = {
    # Patch Embedding
    'visual.patch_embed.proj.weight': 'evaclip.patch_embed.weight',
    'visual.patch_embed.proj.bias': 'evaclip.patch_embed.bias',
    # Positional Embedding
    'visual.pos_embed': 'evaclip.pos_embed',
    # CLS Token
    'visual.cls_token': 'evaclip.cls_token',
    # Final LayerNorm
    'visual.norm.weight': 'evaclip.norm.weight',
    'visual.norm.bias': 'evaclip.norm.bias',
    # Head (optional)
    'visual.head.weight': 'evaclip.head.weight',
    'visual.head.bias': 'evaclip.head.bias',
}

# Attention-Mapping
ATTN_MAPPINGS = {
    'attn.q_proj.weight': 'attn.q.weight',
    'attn.q_proj.bias': 'attn.q.bias',
    'attn.k_proj.weight': 'attn.k.weight',
    'attn.k_proj.bias': 'attn.k.bias',
    'attn.v_proj.weight': 'attn.v.weight',
    'attn.v_proj.bias': 'attn.v.bias',
    'attn.proj.weight': 'attn.out.weight',
    'attn.proj.bias': 'attn.out.bias',
}

# MLP-Mapping (EVA-CLIP nutzt SwiGLU-aehnliche Struktur)
MLP_MAPPINGS = {
    'mlp.fc1.weight': 'mlp.fc1.weight',
    'mlp.fc1.bias': 'mlp.fc1.bias',
    'mlp.fc2.weight': 'mlp.fc2.weight',
    'mlp.fc2.bias': 'mlp.fc2.bias',
}


def get_tensor_name(hf_name: str) -> str:
    """
    Konvertiert HuggingFace Tensor-Namen zu GGUF-Namen

    EVA-CLIP Struktur:
    - visual.patch_embed.* -> evaclip.patch_embed.*
    - visual.blocks.{i}.* -> evaclip.blocks.{i}.*
    - visual.norm.* -> evaclip.norm.*
    """
    # Direkte Mappings pruefen
    if hf_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[hf_name]

    # Encoder Blocks parsen
    parts = hf_name.split('.')

    # visual.blocks.{i}.* Pattern
    if len(parts) >= 4 and parts[0] == 'visual' and parts[1] == 'blocks':
        layer_idx = parts[2]
        rest = '.'.join(parts[3:])

        # Attention
        for hf_pattern, gguf_pattern in ATTN_MAPPINGS.items():
            if rest == hf_pattern:
                return f'evaclip.blocks.{layer_idx}.{gguf_pattern}'

        # MLP
        for hf_pattern, gguf_pattern in MLP_MAPPINGS.items():
            if rest == hf_pattern:
                return f'evaclip.blocks.{layer_idx}.{gguf_pattern}'

        # Layer Norms (EVA nutzt norm1/norm2)
        if rest == 'norm1.weight':
            return f'evaclip.blocks.{layer_idx}.ln1.weight'
        if rest == 'norm1.bias':
            return f'evaclip.blocks.{layer_idx}.ln1.bias'
        if rest == 'norm2.weight':
            return f'evaclip.blocks.{layer_idx}.ln2.weight'
        if rest == 'norm2.bias':
            return f'evaclip.blocks.{layer_idx}.ln2.bias'

    # Fallback: Original-Name mit Praefix
    return f'evaclip.{hf_name}'


def extract_config(model, config) -> dict:
    """
    Extrahiert Modell-Konfiguration aus verschiedenen Config-Formaten

    EVA-CLIP kann verschiedene Config-Strukturen haben.
    Unterstuetzt: vision_config, visual_config, direkte Attribute
    """
    # Versuche verschiedene Config-Strukturen
    if hasattr(config, 'vision_config'):
        vision_cfg = config.vision_config
    elif hasattr(config, 'visual'):
        vision_cfg = config.visual
    else:
        vision_cfg = config

    # Extrahiere Werte mit Fallbacks
    result = {
        'hidden_size': getattr(vision_cfg, 'hidden_size', 1024),
        'num_attention_heads': getattr(vision_cfg, 'num_attention_heads', 16),
        'num_hidden_layers': getattr(vision_cfg, 'num_hidden_layers', 24),
        'intermediate_size': getattr(vision_cfg, 'intermediate_size', 4096),
        'image_size': getattr(vision_cfg, 'image_size', 336),
        'patch_size': getattr(vision_cfg, 'patch_size', 14),
    }

    return result


def convert_eva_clip(model_name: str, output_path: Path, quantize: str = 'f16'):
    """
    Konvertiert EVA-CLIP Modell zu GGUF

    Laedt das HuggingFace-Modell, extrahiert Gewichte,
    fuehrt Quantisierung durch und speichert im GGUF-Format.
    """
    print(f"Lade Modell: {model_name}")

    # Modell und Config laden
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # Konfiguration extrahieren
    cfg = extract_config(model, config)
    hidden_size = cfg['hidden_size']
    num_attention_heads = cfg['num_attention_heads']
    num_hidden_layers = cfg['num_hidden_layers']
    intermediate_size = cfg['intermediate_size']
    image_size = cfg['image_size']
    patch_size = cfg['patch_size']
    num_patches = (image_size // patch_size) ** 2

    print(f"  Hidden Size: {hidden_size}")
    print(f"  Attention Heads: {num_attention_heads}")
    print(f"  Layers: {num_hidden_layers}")
    print(f"  MLP Size: {intermediate_size}")
    print(f"  Image Size: {image_size}")
    print(f"  Patch Size: {patch_size}")

    # GGUF Writer erstellen
    writer = GGUFWriter(output_path)

    # Metadaten hinzufuegen
    writer.add_metadata('general.architecture', 'evaclip')
    writer.add_metadata('general.name', model_name.split('/')[-1])
    writer.add_metadata('general.file_type', quantize)

    writer.add_metadata('evaclip.hidden_size', hidden_size)
    writer.add_metadata('evaclip.num_attention_heads', num_attention_heads)
    writer.add_metadata('evaclip.num_hidden_layers', num_hidden_layers)
    writer.add_metadata('evaclip.intermediate_size', intermediate_size)
    writer.add_metadata('evaclip.image_size', image_size)
    writer.add_metadata('evaclip.patch_size', patch_size)
    writer.add_metadata('evaclip.num_patches', num_patches)

    # EVA-CLIP Preprocessing-Parameter (ImageNet-Standard)
    writer.add_metadata('evaclip.image_mean', [0.48145466, 0.4578275, 0.40821073])
    writer.add_metadata('evaclip.image_std', [0.26862954, 0.26130258, 0.27577711])

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
    tensor_count = 0

    for hf_name, tensor in state_dict.items():
        # Nur Vision-Tensoren (visual.*) konvertieren
        if not hf_name.startswith('visual.'):
            continue

        gguf_name = get_tensor_name(hf_name)
        data = tensor.cpu().numpy()

        # Embedding/Norm/Bias Tensoren nicht quantisieren
        is_embedding = 'embed' in gguf_name.lower() or 'pos' in gguf_name.lower()
        is_norm = 'norm' in gguf_name.lower() or 'ln' in gguf_name.lower()
        is_bias = 'bias' in gguf_name.lower()
        is_cls = 'cls' in gguf_name.lower()

        if is_embedding or is_norm or is_bias or is_cls:
            dtype = GGML_TYPE_F32
        else:
            dtype = quant_type

        writer.add_tensor(gguf_name, data, dtype)
        tensor_count += 1

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
    print(f"  Tensoren: {tensor_count}")
    print(f"  Output: {output_path}")

    return output_path


def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(
        description='EVA-CLIP HuggingFace -> GGUF Konverter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --model BAAI/EVA02-CLIP-L-14-336 --output eva-clip.gguf
  %(prog)s --model BAAI/EVA02-CLIP-L-14 --output eva-clip-l14-q8.gguf --quantize q8_0
  %(prog)s --model BAAI/EVA02-CLIP-B-16 --output eva-clip-b16-q4.gguf --quantize q4_k_m
        """
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='HuggingFace Modell-Name (z.B. BAAI/EVA02-CLIP-L-14-336)'
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

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_eva_clip(args.model, output_path, args.quantize)


if __name__ == '__main__':
    main()
