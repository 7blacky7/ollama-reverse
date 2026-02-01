#!/usr/bin/env python3
"""
MODUL: convert_nomic_vision
ZWECK: Konvertiert Nomic Embed Vision Modelle von HuggingFace nach GGUF
INPUT: HuggingFace Modell-ID, Output-Pfad, Quantisierungs-Typ
OUTPUT: GGUF-Datei mit Vision Encoder Gewichten
NEBENEFFEKTE: Netzwerk-Download von HuggingFace, Datei-Schreiboperationen
ABHAENGIGKEITEN: torch, transformers (extern), gguf_writer (intern)
HINWEISE: Nomic Vision nutzt ViT-Architektur mit 768 hidden_size

Unterstuetzte Modelle:
- nomic-ai/nomic-embed-vision-v1.5 (ViT-B/14, 768 dim)
- nomic-ai/nomic-embed-vision-v1   (ViT-B/16, 768 dim)

Quantisierungsoptionen:
- f32:    Full precision (4 bytes/param)
- f16:    Half precision (2 bytes/param)
- q8_0:   8-bit quantization (~1 byte/param)
- q4_k_m: 4-bit quantization (~0.5 bytes/param)

Verwendung:
    python convert_nomic_vision.py --model nomic-ai/nomic-embed-vision-v1.5 --output nomic.gguf
    python convert_nomic_vision.py -m nomic-ai/nomic-embed-vision-v1.5 -o nomic-q8.gguf -q q8_0
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
# Tensor-Mapping Konfiguration
# ============================================================================

# Nomic Vision nutzt einen modifizierten ViT mit DINOv2-aehnlicher Architektur
# Das Mapping konvertiert HuggingFace-Namen zu GGUF-konformen Namen

TENSOR_MAPPINGS = {
    # Patch Embedding - konvertiert Bild-Patches zu Tokens
    'visual.patch_embed.proj.weight': 'nomic.patch_embed.weight',
    'visual.patch_embed.proj.bias': 'nomic.patch_embed.bias',

    # Positional Embedding - lernt Positionsinformation
    'visual.pos_embed': 'nomic.pos_embed',

    # CLS Token - aggregiert globale Information
    'visual.cls_token': 'nomic.cls_token',

    # Output Norm - finale Normalisierung
    'visual.norm.weight': 'nomic.norm.weight',
    'visual.norm.bias': 'nomic.norm.bias',

    # Head/Projection - optional
    'visual.head.weight': 'nomic.head.weight',
    'visual.head.bias': 'nomic.head.bias',
}

# Attention-Layer Komponenten
ATTN_MAPPINGS = {
    'qkv.weight': 'attn.qkv.weight',
    'qkv.bias': 'attn.qkv.bias',
    'proj.weight': 'attn.proj.weight',
    'proj.bias': 'attn.proj.bias',
}

# MLP-Layer Komponenten
MLP_MAPPINGS = {
    'fc1.weight': 'mlp.fc1.weight',
    'fc1.bias': 'mlp.fc1.bias',
    'fc2.weight': 'mlp.fc2.weight',
    'fc2.bias': 'mlp.fc2.bias',
}


# ============================================================================
# Tensor-Name Konvertierung
# ============================================================================

def get_tensor_name(hf_name: str) -> str:
    """
    Konvertiert HuggingFace Tensor-Namen zu GGUF-Namen.

    Mapping-Strategie:
    - visual.blocks.{i}.attn.* -> nomic.blocks.{i}.attn.*
    - visual.blocks.{i}.mlp.* -> nomic.blocks.{i}.mlp.*
    - visual.blocks.{i}.norm1.* -> nomic.blocks.{i}.ln1.*
    - visual.blocks.{i}.norm2.* -> nomic.blocks.{i}.ln2.*

    Args:
        hf_name: Original HuggingFace Tensor-Name

    Returns:
        GGUF-konformer Tensor-Name
    """
    # Direkte Mappings pruefen
    if hf_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[hf_name]

    # Block-Layer parsen (visual.blocks.{i}.*)
    parts = hf_name.split('.')

    if len(parts) >= 4 and parts[0] == 'visual' and parts[1] == 'blocks':
        layer_idx = parts[2]
        rest = '.'.join(parts[3:])

        # Attention-Komponenten
        if rest.startswith('attn.'):
            attn_part = rest.replace('attn.', '')
            if attn_part in ATTN_MAPPINGS:
                return f'nomic.blocks.{layer_idx}.{ATTN_MAPPINGS[attn_part]}'

        # MLP-Komponenten
        elif rest.startswith('mlp.'):
            mlp_part = rest.replace('mlp.', '')
            if mlp_part in MLP_MAPPINGS:
                return f'nomic.blocks.{layer_idx}.{MLP_MAPPINGS[mlp_part]}'

        # Layer Norms
        elif rest.startswith('norm1'):
            suffix = rest.replace('norm1', '')
            return f'nomic.blocks.{layer_idx}.ln1{suffix}'
        elif rest.startswith('norm2'):
            suffix = rest.replace('norm2', '')
            return f'nomic.blocks.{layer_idx}.ln2{suffix}'

    # Fallback: Prefix mit nomic.
    return f'nomic.{hf_name}'


# ============================================================================
# Modell-Konvertierung
# ============================================================================

def load_nomic_model(model_name: str):
    """
    Laedt Nomic Vision Modell von HuggingFace.

    Args:
        model_name: HuggingFace Modell-ID

    Returns:
        Tuple aus (model, config, processor)
    """
    print(f"Lade Modell: {model_name}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_name)

    model.eval()
    return model, config, processor


def extract_vision_config(config) -> dict:
    """
    Extrahiert Vision-Konfiguration aus dem Modell-Config.

    Args:
        config: HuggingFace AutoConfig Objekt

    Returns:
        Dict mit Vision-Parametern
    """
    # Nomic speichert Vision-Config unterschiedlich
    if hasattr(config, 'vision_config'):
        vc = config.vision_config
    elif hasattr(config, 'visual'):
        vc = config.visual
    else:
        vc = config

    return {
        'hidden_size': getattr(vc, 'hidden_size', getattr(vc, 'embed_dim', 768)),
        'num_heads': getattr(vc, 'num_attention_heads', getattr(vc, 'num_heads', 12)),
        'num_layers': getattr(vc, 'num_hidden_layers', getattr(vc, 'depth', 12)),
        'mlp_ratio': getattr(vc, 'mlp_ratio', 4.0),
        'image_size': getattr(vc, 'image_size', 224),
        'patch_size': getattr(vc, 'patch_size', 14),
    }


def get_quantization_type(quantize: str) -> int:
    """
    Gibt den GGML-Typ fuer den Quantisierungs-String zurueck.

    Args:
        quantize: Quantisierungs-Typ als String

    Returns:
        GGML Typ-Konstante
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

    Embeddings, Norm-Layer und Biases bleiben in F32 fuer Stabilitaet.

    Args:
        tensor_name: GGUF Tensor-Name

    Returns:
        True wenn Tensor nicht quantisiert werden soll
    """
    name_lower = tensor_name.lower()
    skip_patterns = ('embed', 'pos', 'cls', 'norm', 'ln', 'bias')
    return any(pattern in name_lower for pattern in skip_patterns)


def convert_nomic_vision(model_name: str, output_path: Path, quantize: str = 'f16'):
    """
    Hauptfunktion: Konvertiert Nomic Vision Modell zu GGUF.

    Args:
        model_name: HuggingFace Modell-ID
        output_path: Pfad fuer Output-GGUF
        quantize: Quantisierungs-Typ

    Returns:
        Pfad zur erstellten GGUF-Datei
    """
    model, config, processor = load_nomic_model(model_name)
    vision_cfg = extract_vision_config(config)

    # Konfiguration ausgeben
    print(f"  Hidden Size: {vision_cfg['hidden_size']}")
    print(f"  Attention Heads: {vision_cfg['num_heads']}")
    print(f"  Layers: {vision_cfg['num_layers']}")
    print(f"  Image Size: {vision_cfg['image_size']}")
    print(f"  Patch Size: {vision_cfg['patch_size']}")

    # GGUF Writer initialisieren
    writer = GGUFWriter(output_path)

    # Metadaten schreiben
    write_metadata(writer, model_name, vision_cfg, processor, quantize)

    # Tensoren konvertieren
    quant_type = get_quantization_type(quantize)
    print(f"\nKonvertiere Tensoren (Quantisierung: {quantize})...")

    convert_tensors(writer, model, quant_type)

    # GGUF schreiben
    writer.write()

    print(f"\nKonvertierung abgeschlossen!")
    print(f"Output: {output_path}")

    return output_path


def write_metadata(writer: GGUFWriter, model_name: str, cfg: dict, processor, quantize: str):
    """
    Schreibt GGUF Metadaten.

    Args:
        writer: GGUFWriter Instanz
        model_name: Modell-Name fuer Metadaten
        cfg: Vision-Konfiguration
        processor: Image Processor fuer Preprocessing-Params
        quantize: Quantisierungs-Typ
    """
    # Allgemeine Metadaten
    writer.add_metadata('general.architecture', 'nomic')
    writer.add_metadata('general.name', model_name.split('/')[-1])
    writer.add_metadata('general.file_type', quantize)

    # Vision-spezifische Metadaten
    writer.add_metadata('nomic.hidden_size', cfg['hidden_size'])
    writer.add_metadata('nomic.num_attention_heads', cfg['num_heads'])
    writer.add_metadata('nomic.num_hidden_layers', cfg['num_layers'])
    writer.add_metadata('nomic.intermediate_size', int(cfg['hidden_size'] * cfg['mlp_ratio']))
    writer.add_metadata('nomic.image_size', cfg['image_size'])
    writer.add_metadata('nomic.patch_size', cfg['patch_size'])

    num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
    writer.add_metadata('nomic.num_patches', num_patches)

    # Preprocessing-Parameter
    if hasattr(processor, 'image_mean'):
        writer.add_metadata('nomic.image_mean', list(processor.image_mean))
    if hasattr(processor, 'image_std'):
        writer.add_metadata('nomic.image_std', list(processor.image_std))


def convert_tensors(writer: GGUFWriter, model, quant_type: int):
    """
    Konvertiert alle Modell-Tensoren und fuegt sie zum Writer hinzu.

    Args:
        writer: GGUFWriter Instanz
        model: PyTorch Modell
        quant_type: GGML Quantisierungs-Typ
    """
    state_dict = model.state_dict()
    dtype_names = {
        GGML_TYPE_F32: 'F32',
        GGML_TYPE_F16: 'F16',
        GGML_TYPE_Q8_0: 'Q8_0',
        GGML_TYPE_Q4_K: 'Q4_K',
    }

    for hf_name, tensor in state_dict.items():
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
    """
    Parst Kommandozeilen-Argumente.

    Returns:
        Namespace mit geparseten Argumenten
    """
    parser = argparse.ArgumentParser(
        description='Nomic Embed Vision HuggingFace -> GGUF Konverter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --model nomic-ai/nomic-embed-vision-v1.5 --output nomic-vision.gguf
  %(prog)s -m nomic-ai/nomic-embed-vision-v1.5 -o nomic-q8.gguf -q q8_0
  %(prog)s -m nomic-ai/nomic-embed-vision-v1 -o nomic-v1-q4.gguf -q q4_k_m
        """
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='HuggingFace Modell-Name (z.B. nomic-ai/nomic-embed-vision-v1.5)'
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

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validiere gegen PyTorch-Referenz (noch nicht implementiert)'
    )

    return parser.parse_args()


def main():
    """CLI Entry Point"""
    args = parse_arguments()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Konvertierung starten
    gguf_path = convert_nomic_vision(args.model, output_path, args.quantization)

    # Validierung (Platzhalter)
    if args.validate:
        print("\nValidierung gegen PyTorch-Referenz...")
        print("  (Validierung noch nicht implementiert - nutze benchmark.py)")


if __name__ == '__main__':
    main()
