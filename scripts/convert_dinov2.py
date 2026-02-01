#!/usr/bin/env python3
"""
MODUL: convert_dinov2
ZWECK: Konvertiert DINOv2 Vision-Modelle von HuggingFace nach GGUF
INPUT: HuggingFace Modell-ID, Output-Pfad, Quantisierungs-Typ
OUTPUT: GGUF-Datei mit DINOv2 Vision Encoder Gewichten
NEBENEFFEKTE: Netzwerk-Download von HuggingFace, Datei-Schreiboperationen
ABHAENGIGKEITEN: torch, transformers (extern), gguf_writer (intern)
HINWEISE: DINOv2 ist ein reines Vision-Modell ohne Text-Encoder

Unterstuetzte Modelle:
- facebook/dinov2-small  (ViT-S/14, 22M params)
- facebook/dinov2-base   (ViT-B/14, 86M params)
- facebook/dinov2-large  (ViT-L/14, 300M params)
- facebook/dinov2-giant  (ViT-G/14, 1.1B params)

Architektur-Besonderheiten:
- Kein Text Encoder (reines Vision Self-Supervised Learning)
- ViT mit optionalen Register Tokens
- CLS Token + Patch Tokens als Output
- LayerScale fuer stabiles Training

Quantisierungsoptionen:
- f32:    Full precision (4 bytes/param)
- f16:    Half precision (2 bytes/param)
- q8_0:   8-bit quantization (~1 byte/param)
- q4_k_m: 4-bit quantization (~0.5 bytes/param)

Verwendung:
    python convert_dinov2.py --model facebook/dinov2-large --output dinov2-large.gguf
    python convert_dinov2.py -m facebook/dinov2-base -o dinov2-q8.gguf -q q8_0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoConfig, AutoImageProcessor
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
# Konfiguration: Tensor-Mappings
# ============================================================================

# Direkte Tensor-Mappings (Embeddings, Norms, Tokens)
TENSOR_MAPPINGS = {
    # Patch Embedding - konvertiert Bild-Patches zu Tokens
    'dinov2.embeddings.patch_embeddings.projection.weight': 'dinov2.patch_embed.weight',
    'dinov2.embeddings.patch_embeddings.projection.bias': 'dinov2.patch_embed.bias',
    # CLS Token - aggregiert globale Information
    'dinov2.embeddings.cls_token': 'dinov2.cls_token',
    # Position Embedding - lernt Positionsinformation
    'dinov2.embeddings.position_embeddings': 'dinov2.pos_embed',
    # Mask Token (fuer MAE-Training, optional)
    'dinov2.embeddings.mask_token': 'dinov2.mask_token',
    # Final LayerNorm
    'dinov2.layernorm.weight': 'dinov2.norm.weight',
    'dinov2.layernorm.bias': 'dinov2.norm.bias',
}

# Attention-Komponenten Mapping
ATTN_MAPPINGS = {
    'attention.attention.query.weight': 'attn.q.weight',
    'attention.attention.query.bias': 'attn.q.bias',
    'attention.attention.key.weight': 'attn.k.weight',
    'attention.attention.key.bias': 'attn.k.bias',
    'attention.attention.value.weight': 'attn.v.weight',
    'attention.attention.value.bias': 'attn.v.bias',
    'attention.output.dense.weight': 'attn.out.weight',
    'attention.output.dense.bias': 'attn.out.bias',
}

# MLP-Komponenten Mapping (DINOv2 nutzt Standard MLP)
MLP_MAPPINGS = {
    'mlp.fc1.weight': 'mlp.fc1.weight',
    'mlp.fc1.bias': 'mlp.fc1.bias',
    'mlp.fc2.weight': 'mlp.fc2.weight',
    'mlp.fc2.bias': 'mlp.fc2.bias',
}

# LayerNorm Mappings
NORM_MAPPINGS = {
    'norm1.weight': 'ln1.weight',
    'norm1.bias': 'ln1.bias',
    'norm2.weight': 'ln2.weight',
    'norm2.bias': 'ln2.bias',
}

# LayerScale Mappings (DINOv2 spezifisch)
LAYERSCALE_MAPPINGS = {
    'layer_scale1.lambda1': 'ls1.gamma',
    'layer_scale2.lambda1': 'ls2.gamma',
}


# ============================================================================
# Tensor-Name Konvertierung
# ============================================================================

def get_tensor_name(hf_name: str) -> str:
    """
    Konvertiert HuggingFace Tensor-Namen zu GGUF-Namen.

    DINOv2 Struktur:
    - dinov2.embeddings.* -> dinov2.*
    - dinov2.encoder.layer.{i}.* -> dinov2.blocks.{i}.*
    - dinov2.layernorm.* -> dinov2.norm.*
    """
    # Direkte Mappings pruefen
    if hf_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[hf_name]

    # Encoder Layer parsen (dinov2.encoder.layer.{i}.*)
    if 'encoder.layer.' in hf_name:
        return _convert_encoder_layer(hf_name)

    # Fallback: Original-Name mit dinov2 Praefix
    return f'dinov2.{hf_name}'


def _convert_encoder_layer(hf_name: str) -> str:
    """
    Konvertiert Encoder-Layer Tensor-Namen.

    Input:  dinov2.encoder.layer.{i}.{component}
    Output: dinov2.blocks.{i}.{mapped_component}
    """
    parts = hf_name.split('.')

    # Layer Index extrahieren (index 3 in dinov2.encoder.layer.{i})
    layer_idx = parts[3]
    rest = '.'.join(parts[4:])

    # Attention-Komponenten
    for hf_pattern, gguf_pattern in ATTN_MAPPINGS.items():
        if rest == hf_pattern:
            return f'dinov2.blocks.{layer_idx}.{gguf_pattern}'

    # MLP-Komponenten
    for hf_pattern, gguf_pattern in MLP_MAPPINGS.items():
        if rest == hf_pattern:
            return f'dinov2.blocks.{layer_idx}.{gguf_pattern}'

    # LayerNorm-Komponenten
    for hf_pattern, gguf_pattern in NORM_MAPPINGS.items():
        if rest == hf_pattern:
            return f'dinov2.blocks.{layer_idx}.{gguf_pattern}'

    # LayerScale-Komponenten (DINOv2 spezifisch)
    for hf_pattern, gguf_pattern in LAYERSCALE_MAPPINGS.items():
        if rest == hf_pattern:
            return f'dinov2.blocks.{layer_idx}.{gguf_pattern}'

    # Fallback
    return f'dinov2.blocks.{layer_idx}.{rest}'


# ============================================================================
# Modell-Laden und Konfiguration
# ============================================================================

def load_dinov2_model(model_name: str):
    """
    Laedt DINOv2 Modell von HuggingFace.

    Args:
        model_name: HuggingFace Modell-ID (z.B. facebook/dinov2-large)

    Returns:
        Tuple aus (model, config, processor)
    """
    print(f"Lade Modell: {model_name}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Image Processor laden (fuer Preprocessing-Parameter)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        processor = None

    model.eval()
    return model, config, processor


def extract_dinov2_config(config) -> dict:
    """
    Extrahiert DINOv2-Konfiguration aus dem Config-Objekt.

    Args:
        config: HuggingFace AutoConfig Objekt

    Returns:
        Dict mit Vision-Parametern
    """
    return {
        'hidden_size': getattr(config, 'hidden_size', 768),
        'num_attention_heads': getattr(config, 'num_attention_heads', 12),
        'num_hidden_layers': getattr(config, 'num_hidden_layers', 12),
        'intermediate_size': getattr(config, 'intermediate_size', 3072),
        'image_size': getattr(config, 'image_size', 518),
        'patch_size': getattr(config, 'patch_size', 14),
        'num_channels': getattr(config, 'num_channels', 3),
        'layerscale_init_value': getattr(config, 'layerscale_init_value', 1.0),
        'use_swiglu_ffn': getattr(config, 'use_swiglu_ffn', False),
    }


# ============================================================================
# Quantisierung und Hilfsfunktionen
# ============================================================================

def get_quantization_type(quantize: str) -> int:
    """
    Gibt den GGML-Typ fuer den Quantisierungs-String zurueck.
    """
    quant_map = {
        'f32': GGML_TYPE_F32,
        'f16': GGML_TYPE_F16,
        'q8_0': GGML_TYPE_Q8_0,
        'q4_k_m': GGML_TYPE_Q4_K,
    }
    return quant_map.get(quantize.lower(), GGML_TYPE_F16)


def should_skip_quantization(tensor_name: str) -> bool:
    """
    Prueft ob ein Tensor von Quantisierung ausgenommen werden soll.

    Embeddings, Norm-Layer, LayerScale und Biases bleiben in F32.
    """
    name_lower = tensor_name.lower()
    skip_patterns = (
        'embed', 'pos', 'cls', 'mask',  # Embeddings/Tokens
        'norm', 'ln',                    # LayerNorms
        'bias',                          # Biases
        'ls1', 'ls2', 'gamma',           # LayerScale
    )
    return any(pattern in name_lower for pattern in skip_patterns)


def get_dtype_name(dtype: int) -> str:
    """Gibt lesbaren Namen fuer GGML-Typ zurueck."""
    dtype_names = {
        GGML_TYPE_F32: 'F32',
        GGML_TYPE_F16: 'F16',
        GGML_TYPE_Q8_0: 'Q8_0',
        GGML_TYPE_Q4_K: 'Q4_K',
    }
    return dtype_names.get(dtype, 'F32')


# ============================================================================
# Metadaten-Schreibung
# ============================================================================

def write_metadata(writer: GGUFWriter, model_name: str, cfg: dict, processor):
    """
    Schreibt GGUF Metadaten fuer DINOv2.

    Args:
        writer: GGUFWriter Instanz
        model_name: Modell-Name fuer Metadaten
        cfg: DINOv2-Konfiguration
        processor: Image Processor (optional)
    """
    # Allgemeine Metadaten
    writer.add_metadata('general.architecture', 'dinov2')
    writer.add_metadata('general.name', model_name.split('/')[-1])

    # Vision-spezifische Metadaten
    writer.add_metadata('dinov2.hidden_size', cfg['hidden_size'])
    writer.add_metadata('dinov2.num_attention_heads', cfg['num_attention_heads'])
    writer.add_metadata('dinov2.num_hidden_layers', cfg['num_hidden_layers'])
    writer.add_metadata('dinov2.intermediate_size', cfg['intermediate_size'])
    writer.add_metadata('dinov2.image_size', cfg['image_size'])
    writer.add_metadata('dinov2.patch_size', cfg['patch_size'])

    # Berechnete Werte
    num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
    writer.add_metadata('dinov2.num_patches', num_patches)

    # DINOv2-spezifische Parameter
    writer.add_metadata('dinov2.layerscale_init_value', cfg['layerscale_init_value'])
    writer.add_metadata('dinov2.use_swiglu_ffn', cfg['use_swiglu_ffn'])

    # Preprocessing-Parameter (ImageNet-Standard)
    if processor and hasattr(processor, 'image_mean'):
        writer.add_metadata('dinov2.image_mean', list(processor.image_mean))
    else:
        writer.add_metadata('dinov2.image_mean', [0.485, 0.456, 0.406])

    if processor and hasattr(processor, 'image_std'):
        writer.add_metadata('dinov2.image_std', list(processor.image_std))
    else:
        writer.add_metadata('dinov2.image_std', [0.229, 0.224, 0.225])


# ============================================================================
# Tensor-Konvertierung
# ============================================================================

def convert_tensors(writer: GGUFWriter, model, quant_type: int) -> int:
    """
    Konvertiert alle Modell-Tensoren und fuegt sie zum Writer hinzu.

    Args:
        writer: GGUFWriter Instanz
        model: PyTorch Modell
        quant_type: GGML Quantisierungs-Typ

    Returns:
        Anzahl konvertierter Tensoren
    """
    state_dict = model.state_dict()
    tensor_count = 0

    for hf_name, tensor in state_dict.items():
        gguf_name = get_tensor_name(hf_name)
        data = tensor.cpu().numpy()

        # Quantisierungs-Typ bestimmen
        if should_skip_quantization(gguf_name):
            dtype = GGML_TYPE_F32
        else:
            dtype = quant_type

        writer.add_tensor(gguf_name, data, dtype)
        tensor_count += 1

        dtype_str = get_dtype_name(dtype)
        print(f"  {gguf_name}: {data.shape} -> {dtype_str}")

    return tensor_count


# ============================================================================
# Hauptkonvertierung
# ============================================================================

def convert_dinov2(model_name: str, output_path: Path, quantize: str = 'f16'):
    """
    Hauptfunktion: Konvertiert DINOv2 Modell zu GGUF.

    Args:
        model_name: HuggingFace Modell-ID
        output_path: Pfad fuer Output-GGUF
        quantize: Quantisierungs-Typ (f32, f16, q8_0, q4_k_m)

    Returns:
        Pfad zur erstellten GGUF-Datei
    """
    # Modell laden
    model, config, processor = load_dinov2_model(model_name)
    cfg = extract_dinov2_config(config)

    # Konfiguration ausgeben
    print(f"\nDINOv2 Konfiguration:")
    print(f"  Hidden Size: {cfg['hidden_size']}")
    print(f"  Attention Heads: {cfg['num_attention_heads']}")
    print(f"  Layers: {cfg['num_hidden_layers']}")
    print(f"  MLP Size: {cfg['intermediate_size']}")
    print(f"  Image Size: {cfg['image_size']}")
    print(f"  Patch Size: {cfg['patch_size']}")
    print(f"  LayerScale: {cfg['layerscale_init_value']}")
    print(f"  SwiGLU FFN: {cfg['use_swiglu_ffn']}")

    # GGUF Writer initialisieren
    writer = GGUFWriter(output_path)

    # Metadaten schreiben
    write_metadata(writer, model_name, cfg, processor)
    writer.add_metadata('general.file_type', quantize)

    # Tensoren konvertieren
    quant_type = get_quantization_type(quantize)
    print(f"\nKonvertiere Tensoren (Quantisierung: {quantize})...")

    tensor_count = convert_tensors(writer, model, quant_type)

    # GGUF schreiben
    writer.write()

    print(f"\nKonvertierung abgeschlossen!")
    print(f"  Tensoren: {tensor_count}")
    print(f"  Output: {output_path}")

    return output_path


# ============================================================================
# CLI Interface
# ============================================================================

def parse_arguments():
    """Parst Kommandozeilen-Argumente."""
    parser = argparse.ArgumentParser(
        description='DINOv2 HuggingFace -> GGUF Konverter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --model facebook/dinov2-large --output dinov2-large.gguf
  %(prog)s -m facebook/dinov2-base -o dinov2-base-q8.gguf -q q8_0
  %(prog)s -m facebook/dinov2-small -o dinov2-small-q4.gguf -q q4_k_m
  %(prog)s -m facebook/dinov2-giant -o dinov2-giant-f16.gguf -q f16

Unterstuetzte Modelle:
  - facebook/dinov2-small  (ViT-S/14, 22M params)
  - facebook/dinov2-base   (ViT-B/14, 86M params)
  - facebook/dinov2-large  (ViT-L/14, 300M params)
  - facebook/dinov2-giant  (ViT-G/14, 1.1B params)
        """
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='HuggingFace Modell-Name (z.B. facebook/dinov2-large)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output GGUF-Datei'
    )

    parser.add_argument(
        '--quantization', '-q',
        choices=['f32', 'f16', 'q8_0', 'q4_k_m'],
        default='f16',
        help='Quantisierungs-Typ (default: f16)'
    )

    return parser.parse_args()


def main():
    """CLI Entry Point"""
    args = parse_arguments()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_dinov2(args.model, output_path, args.quantization)


if __name__ == '__main__':
    main()
