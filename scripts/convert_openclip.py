#!/usr/bin/env python3
"""
MODUL: convert_openclip
ZWECK: Konvertiert OpenCLIP Modelle von HuggingFace/LAION nach GGUF
INPUT: HuggingFace Modell-ID oder open_clip Modell-Name, Output-Pfad, Quantisierung
OUTPUT: GGUF-Datei mit Vision Encoder Gewichten
NEBENEFFEKTE: Netzwerk-Download, Datei-Schreiboperationen
ABHAENGIGKEITEN: torch, open_clip_torch (extern), gguf_writer (intern)
HINWEISE: OpenCLIP Modelle sind groesser als Standard-CLIP
          ViT-bigG-14: 1.8B Parameter, 1280-dim, 40 Layers

Unterstuetzte Modelle:
- laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (ViT-bigG-14, 1280 dim, 40 layers)
- laion/CLIP-ViT-H-14-laion2B-s32B-b79K   (ViT-H-14, 1024 dim, 32 layers)
- laion/CLIP-ViT-L-14-laion2B-s32B-b82K   (ViT-L-14, 1024 dim, 24 layers)
- laion/CLIP-ViT-B-32-laion2B-s34B-b79K   (ViT-B-32, 768 dim, 12 layers)

Quantisierungsoptionen:
- f32:    Full precision (4 bytes/param)
- f16:    Half precision (2 bytes/param)
- q8_0:   8-bit quantization (~1 byte/param)
- q4_k_m: 4-bit quantization (~0.5 bytes/param)

Verwendung:
    python convert_openclip.py --model laion/CLIP-ViT-bigG-14-laion2B-39B-b160k --output openclip.gguf
    python convert_openclip.py -m ViT-bigG-14 --pretrained laion2b_s39b_b160k -o bigG.gguf -q q8_0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("Fehler: Bitte installiere torch:")
    print("  pip install torch")
    sys.exit(1)

try:
    import open_clip
except ImportError:
    print("Fehler: Bitte installiere open_clip_torch:")
    print("  pip install open_clip_torch")
    sys.exit(1)

from gguf_writer import (
    GGUFWriter,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K,
)


# ============================================================================
# Tensor-Mapping Konfiguration
# ============================================================================

# OpenCLIP verwendet aehnliche Namen wie Standard-CLIP,
# aber mit visual. Praefix fuer Vision Encoder
TENSOR_MAPPINGS = {
    # Patch Embedding
    'visual.conv1.weight': 'openclip.patch_embed.weight',
    'visual.class_embedding': 'openclip.cls_token',
    'visual.positional_embedding': 'openclip.pos_embed',

    # Pre-LayerNorm (vor Transformer Blocks)
    'visual.ln_pre.weight': 'openclip.pre_ln.weight',
    'visual.ln_pre.bias': 'openclip.pre_ln.bias',

    # Post-LayerNorm
    'visual.ln_post.weight': 'openclip.norm.weight',
    'visual.ln_post.bias': 'openclip.norm.bias',

    # Projection (optional)
    'visual.proj': 'openclip.proj.weight',
}

# Attention-Layer Mapping (in_proj ist Q/K/V kombiniert)
ATTN_MAPPINGS = {
    'attn.in_proj_weight': 'attn.qkv.weight',
    'attn.in_proj_bias': 'attn.qkv.bias',
    'attn.out_proj.weight': 'attn.proj.weight',
    'attn.out_proj.bias': 'attn.proj.bias',
}

# MLP-Layer Mapping (c_fc = up, c_proj = down)
MLP_MAPPINGS = {
    'mlp.c_fc.weight': 'mlp.fc1.weight',
    'mlp.c_fc.bias': 'mlp.fc1.bias',
    'mlp.c_proj.weight': 'mlp.fc2.weight',
    'mlp.c_proj.bias': 'mlp.fc2.bias',
}


# ============================================================================
# Tensor-Name Konvertierung
# ============================================================================

def get_tensor_name(hf_name: str) -> str:
    """
    Konvertiert OpenCLIP Tensor-Namen zu GGUF-Namen.

    Mapping-Strategie:
    - visual.transformer.resblocks.{i}.attn.* -> openclip.blocks.{i}.attn.*
    - visual.transformer.resblocks.{i}.mlp.* -> openclip.blocks.{i}.mlp.*
    - visual.transformer.resblocks.{i}.ln_1.* -> openclip.blocks.{i}.ln1.*
    - visual.transformer.resblocks.{i}.ln_2.* -> openclip.blocks.{i}.ln2.*

    Args:
        hf_name: Original OpenCLIP Tensor-Name

    Returns:
        GGUF-konformer Tensor-Name
    """
    # Direkte Mappings pruefen
    if hf_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[hf_name]

    # Transformer Blocks parsen (visual.transformer.resblocks.{i}.*)
    parts = hf_name.split('.')

    if len(parts) >= 5 and parts[0] == 'visual' and parts[1] == 'transformer':
        if parts[2] == 'resblocks' and parts[3].isdigit():
            layer_idx = parts[3]
            rest = '.'.join(parts[4:])

            # Attention-Komponenten
            if rest.startswith('attn.'):
                attn_part = rest
                if attn_part in ATTN_MAPPINGS:
                    return f'openclip.blocks.{layer_idx}.{ATTN_MAPPINGS[attn_part]}'

            # MLP-Komponenten
            elif rest.startswith('mlp.'):
                mlp_part = rest
                if mlp_part in MLP_MAPPINGS:
                    return f'openclip.blocks.{layer_idx}.{MLP_MAPPINGS[mlp_part]}'

            # Layer Norms
            elif rest.startswith('ln_1'):
                suffix = rest.replace('ln_1', '')
                return f'openclip.blocks.{layer_idx}.ln1{suffix}'
            elif rest.startswith('ln_2'):
                suffix = rest.replace('ln_2', '')
                return f'openclip.blocks.{layer_idx}.ln2{suffix}'

    # Fallback: Prefix mit openclip.
    return f'openclip.{hf_name}'


# ============================================================================
# Modell-Laden
# ============================================================================

def load_openclip_model(model_name: str, pretrained: str = None):
    """
    Laedt OpenCLIP Modell.

    Args:
        model_name: Modell-Architektur (z.B. "ViT-bigG-14") oder HF-ID
        pretrained: Pretrained Checkpoint Name (z.B. "laion2b_s39b_b160k")

    Returns:
        Tuple aus (model, preprocess, config_dict)
    """
    print(f"Lade OpenCLIP Modell: {model_name}")

    # HuggingFace-Format erkennen (laion/CLIP-*)
    if '/' in model_name and model_name.startswith('laion/'):
        # Parse HuggingFace Model ID
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name.split('/')[-1].replace('CLIP-', ''),
            pretrained='laion2b_s39b_b160k'
        )
    else:
        # Direktes Modell-Format
        if pretrained:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained='openai'
            )

    model.eval()

    # Config extrahieren
    config = extract_model_config(model)

    return model, preprocess, config


def extract_model_config(model) -> dict:
    """
    Extrahiert Modell-Konfiguration aus dem OpenCLIP Modell.

    Args:
        model: OpenCLIP Modell

    Returns:
        Dict mit Vision-Parametern
    """
    visual = model.visual

    # Embedding Dimension aus Projection oder ln_post
    if hasattr(visual, 'proj') and visual.proj is not None:
        embed_dim = visual.proj.shape[1]
    elif hasattr(visual, 'ln_post'):
        embed_dim = visual.ln_post.weight.shape[0]
    else:
        embed_dim = 768

    # Anzahl Transformer Blocks
    if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
        num_layers = len(visual.transformer.resblocks)
    else:
        num_layers = 12

    # Attention Heads (aus in_proj Groesse schaetzen)
    if num_layers > 0:
        block = visual.transformer.resblocks[0]
        if hasattr(block, 'attn'):
            num_heads = block.attn.num_heads
        else:
            num_heads = embed_dim // 64
    else:
        num_heads = embed_dim // 64

    # Image Size aus positional_embedding
    if hasattr(visual, 'positional_embedding'):
        num_patches = visual.positional_embedding.shape[0] - 1
        grid_size = int(num_patches ** 0.5)
        patch_size = 224 // grid_size  # Annahme: 224 ist Standard
        image_size = grid_size * patch_size
    else:
        image_size = 224
        patch_size = 14

    # MLP Ratio (intermediate_size / hidden_size)
    if num_layers > 0:
        block = visual.transformer.resblocks[0]
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'c_fc'):
            intermediate = block.mlp.c_fc.weight.shape[0]
            mlp_ratio = intermediate / embed_dim
        else:
            mlp_ratio = 4.0
    else:
        mlp_ratio = 4.0

    return {
        'hidden_size': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'mlp_ratio': mlp_ratio,
        'image_size': image_size,
        'patch_size': patch_size,
    }


# ============================================================================
# Quantisierung
# ============================================================================

def get_quantization_type(quantize: str) -> int:
    """Gibt den GGML-Typ fuer den Quantisierungs-String zurueck."""
    quant_map = {
        'f32': GGML_TYPE_F32,
        'f16': GGML_TYPE_F16,
        'q8_0': GGML_TYPE_Q8_0,
        'q4_k_m': GGML_TYPE_Q4_K,
    }
    return quant_map.get(quantize.lower(), GGML_TYPE_F16)


def should_skip_quantization(tensor_name: str) -> bool:
    """Prueft ob ein Tensor von Quantisierung ausgenommen werden soll."""
    name_lower = tensor_name.lower()
    skip_patterns = ('embed', 'pos', 'cls', 'norm', 'ln', 'bias', 'proj')
    return any(pattern in name_lower for pattern in skip_patterns)


# ============================================================================
# Konvertierung
# ============================================================================

def convert_openclip(model_name: str, output_path: Path, quantize: str = 'f16',
                     pretrained: str = None):
    """
    Hauptfunktion: Konvertiert OpenCLIP Modell zu GGUF.

    Args:
        model_name: OpenCLIP Modell-Name oder HuggingFace ID
        output_path: Pfad fuer Output-GGUF
        quantize: Quantisierungs-Typ
        pretrained: Pretrained Checkpoint

    Returns:
        Pfad zur erstellten GGUF-Datei
    """
    model, preprocess, cfg = load_openclip_model(model_name, pretrained)

    # Konfiguration ausgeben
    print(f"  Hidden Size: {cfg['hidden_size']}")
    print(f"  Attention Heads: {cfg['num_heads']}")
    print(f"  Layers: {cfg['num_layers']}")
    print(f"  MLP Ratio: {cfg['mlp_ratio']}")
    print(f"  Image Size: {cfg['image_size']}")
    print(f"  Patch Size: {cfg['patch_size']}")

    # GGUF Writer initialisieren
    writer = GGUFWriter(output_path)

    # Metadaten schreiben
    write_metadata(writer, model_name, cfg, preprocess, quantize)

    # Tensoren konvertieren
    quant_type = get_quantization_type(quantize)
    print(f"\nKonvertiere Tensoren (Quantisierung: {quantize})...")

    convert_tensors(writer, model, quant_type)

    # GGUF schreiben
    writer.write()

    print(f"\nKonvertierung abgeschlossen!")
    print(f"Output: {output_path}")

    return output_path


def write_metadata(writer: GGUFWriter, model_name: str, cfg: dict, preprocess, quantize: str):
    """Schreibt GGUF Metadaten."""
    # Allgemeine Metadaten
    writer.add_metadata('general.architecture', 'openclip')
    writer.add_metadata('general.name', model_name.split('/')[-1] if '/' in model_name else model_name)
    writer.add_metadata('general.file_type', quantize)

    # Vision-spezifische Metadaten
    writer.add_metadata('openclip.hidden_size', cfg['hidden_size'])
    writer.add_metadata('openclip.num_attention_heads', cfg['num_heads'])
    writer.add_metadata('openclip.num_hidden_layers', cfg['num_layers'])
    writer.add_metadata('openclip.intermediate_size', int(cfg['hidden_size'] * cfg['mlp_ratio']))
    writer.add_metadata('openclip.image_size', cfg['image_size'])
    writer.add_metadata('openclip.patch_size', cfg['patch_size'])

    num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
    writer.add_metadata('openclip.num_patches', num_patches)

    # CLIP Normalisierungswerte
    writer.add_metadata('openclip.image_mean', [0.48145466, 0.4578275, 0.40821073])
    writer.add_metadata('openclip.image_std', [0.26862954, 0.26130258, 0.27577711])


def convert_tensors(writer: GGUFWriter, model, quant_type: int):
    """Konvertiert alle Modell-Tensoren."""
    state_dict = model.state_dict()
    dtype_names = {
        GGML_TYPE_F32: 'F32',
        GGML_TYPE_F16: 'F16',
        GGML_TYPE_Q8_0: 'Q8_0',
        GGML_TYPE_Q4_K: 'Q4_K',
    }

    # Nur visual.* Tensoren konvertieren
    for hf_name, tensor in state_dict.items():
        if not hf_name.startswith('visual.'):
            continue

        gguf_name = get_tensor_name(hf_name)
        data = tensor.cpu().numpy()

        # Quantisierungs-Typ bestimmen
        if should_skip_quantization(gguf_name):
            dtype = GGML_TYPE_F32
        else:
            dtype = quant_type

        writer.add_tensor(gguf_name, data, dtype)

        dtype_str = dtype_names.get(dtype, 'F32')
        print(f"  {gguf_name}: {data.shape} -> {dtype_str}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_arguments():
    """Parst Kommandozeilen-Argumente."""
    parser = argparse.ArgumentParser(
        description='OpenCLIP HuggingFace/LAION -> GGUF Konverter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --model ViT-bigG-14 --pretrained laion2b_s39b_b160k --output openclip-bigG.gguf
  %(prog)s -m ViT-H-14 --pretrained laion2b_s32b_b79k -o openclip-H.gguf -q q8_0
  %(prog)s -m ViT-L-14 --pretrained laion2b_s32b_b82k -o openclip-L-q4.gguf -q q4_k_m
        """
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='OpenCLIP Modell-Name (z.B. ViT-bigG-14) oder HuggingFace ID'
    )

    parser.add_argument(
        '--pretrained', '-p',
        default=None,
        help='Pretrained Checkpoint (z.B. laion2b_s39b_b160k)'
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
        '--list-models',
        action='store_true',
        help='Zeigt verfuegbare OpenCLIP Modelle an'
    )

    return parser.parse_args()


def list_available_models():
    """Zeigt verfuegbare OpenCLIP Modelle an."""
    print("Verfuegbare OpenCLIP Modelle:\n")
    pretrained = open_clip.list_pretrained()

    # Nach Groesse sortieren
    models_by_size = {}
    for model_name, checkpoint in pretrained:
        if model_name not in models_by_size:
            models_by_size[model_name] = []
        models_by_size[model_name].append(checkpoint)

    for model_name in sorted(models_by_size.keys()):
        checkpoints = models_by_size[model_name]
        print(f"  {model_name}:")
        for cp in checkpoints[:3]:  # Max 3 pro Modell
            print(f"    --pretrained {cp}")
        if len(checkpoints) > 3:
            print(f"    ... und {len(checkpoints) - 3} weitere")
        print()


def main():
    """CLI Entry Point"""
    args = parse_arguments()

    if args.list_models:
        list_available_models()
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Konvertierung starten
    convert_openclip(args.model, output_path, args.quantize, args.pretrained)


if __name__ == '__main__':
    main()
