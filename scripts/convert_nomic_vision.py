#!/usr/bin/env python3
"""
MODUL: convert_nomic_vision
ZWECK: Konvertiert nomic-embed-vision Modelle von HuggingFace SafeTensors zu GGUF
INPUT: HuggingFace Modell-ID (z.B. nomic-ai/nomic-embed-vision-v1.5), Output-Pfad, Quantisierung
OUTPUT: GGUF-Datei mit Vision Encoder Gewichten
NEBENEFFEKTE: Netzwerk-Download von HuggingFace, Datei-Schreiboperationen
ABHAENGIGKEITEN: torch, transformers, safetensors (extern), gguf_writer (intern)
HINWEISE: nomic-embed-vision nutzt modifizierten NomicBert mit Vision-Encoder

Architektur nomic-embed-vision-v1.5:
- model_type: nomic_bert (Vision-Variante)
- n_embd: 768, n_head: 12, n_layer: 12
- img_size: 224, patch_size: 16
- Aktivierung: SwiGLU
- Besonderheit: rotary_pos_emb, flash_attn

Tensor-Mapping (HuggingFace -> GGUF):
- vision_model.embeddings.patch_embeddings.weight -> v.patch_emb.weight
- vision_model.embeddings.position_embeddings -> v.pos_emb
- vision_model.encoder.layer.{i}.* -> v.blk.{i}.*
- vision_model.post_layernorm.* -> v.post_ln.*

Verwendung:
    python convert_nomic_vision.py --model nomic-ai/nomic-embed-vision-v1.5 --output model.gguf
    python convert_nomic_vision.py -m nomic-ai/nomic-embed-vision-v1.5 -o model-q8.gguf -q q8_0
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
# Tensor-Mapping: HuggingFace -> GGUF (v.* Konvention)
# ============================================================================

# Direkte Mappings fuer globale Tensoren
TENSOR_MAPPINGS = {
    # Patch Embedding - konvertiert 16x16 Patches zu 768-dim Tokens
    'vision_model.embeddings.patch_embeddings.projection.weight': 'v.patch_emb.weight',
    'vision_model.embeddings.patch_embeddings.projection.bias': 'v.patch_emb.bias',

    # Position Embedding - 196+1 Positionen (14x14 Patches + CLS)
    'vision_model.embeddings.position_embeddings': 'v.pos_emb',

    # CLS Token - [CLS] wird fuer finale Repraesentation genutzt
    'vision_model.embeddings.cls_token': 'v.cls_token',

    # Post Layer Norm - finale Normalisierung nach allen Blocks
    'vision_model.post_layernorm.weight': 'v.post_ln.weight',
    'vision_model.post_layernorm.bias': 'v.post_ln.bias',

    # Alternative Namen (falls Modell anders strukturiert)
    'visual.patch_embed.proj.weight': 'v.patch_emb.weight',
    'visual.patch_embed.proj.bias': 'v.patch_emb.bias',
    'visual.pos_embed': 'v.pos_emb',
    'visual.cls_token': 'v.cls_token',
    'visual.norm.weight': 'v.post_ln.weight',
    'visual.norm.bias': 'v.post_ln.bias',
}

# Attention-Komponenten innerhalb eines Blocks
ATTN_MAPPINGS = {
    'attention.self.query.weight': 'attn.q.weight',
    'attention.self.query.bias': 'attn.q.bias',
    'attention.self.key.weight': 'attn.k.weight',
    'attention.self.key.bias': 'attn.k.bias',
    'attention.self.value.weight': 'attn.v.weight',
    'attention.self.value.bias': 'attn.v.bias',
    'attention.output.dense.weight': 'attn.out.weight',
    'attention.output.dense.bias': 'attn.out.bias',
    # QKV kombiniert (falls vorhanden)
    'attn.qkv.weight': 'attn.qkv.weight',
    'attn.qkv.bias': 'attn.qkv.bias',
    'attn.proj.weight': 'attn.out.weight',
    'attn.proj.bias': 'attn.out.bias',
}

# MLP/FFN-Komponenten (SwiGLU: gate + up -> down)
MLP_MAPPINGS = {
    'mlp.fc1.weight': 'ffn.up.weight',
    'mlp.fc1.bias': 'ffn.up.bias',
    'mlp.fc2.weight': 'ffn.down.weight',
    'mlp.fc2.bias': 'ffn.down.bias',
    # Alternative: gate/up/down Struktur
    'intermediate.dense.weight': 'ffn.up.weight',
    'intermediate.dense.bias': 'ffn.up.bias',
    'output.dense.weight': 'ffn.down.weight',
    'output.dense.bias': 'ffn.down.bias',
}


def get_tensor_name(hf_name: str) -> str:
    """
    Konvertiert HuggingFace Tensor-Namen zu GGUF-Namen (v.* Schema).

    Mapping-Strategie:
    - vision_model.encoder.layer.{i}.attention.* -> v.blk.{i}.attn.*
    - vision_model.encoder.layer.{i}.mlp.* -> v.blk.{i}.ffn.*
    - vision_model.encoder.layer.{i}.layernorm_before.* -> v.blk.{i}.ln1.*
    - vision_model.encoder.layer.{i}.layernorm_after.* -> v.blk.{i}.ln2.*
    """
    # Direkte Mappings pruefen
    if hf_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[hf_name]

    parts = hf_name.split('.')

    # Encoder Layer parsen: vision_model.encoder.layer.{i}.*
    if 'encoder' in parts and 'layer' in parts:
        try:
            layer_idx = parts[parts.index('layer') + 1]
            rest_start = parts.index('layer') + 2
            rest = '.'.join(parts[rest_start:])

            # Attention-Komponenten
            for hf_pattern, gguf_pattern in ATTN_MAPPINGS.items():
                if rest.startswith(hf_pattern.split('.')[0]) and hf_pattern in rest:
                    return f'v.blk.{layer_idx}.{gguf_pattern}'

            # MLP-Komponenten
            for hf_pattern, gguf_pattern in MLP_MAPPINGS.items():
                if hf_pattern in rest:
                    return f'v.blk.{layer_idx}.{gguf_pattern}'

            # Layer Norms
            if 'layernorm_before' in rest or 'norm1' in rest:
                suffix = '.weight' if 'weight' in rest else '.bias'
                return f'v.blk.{layer_idx}.ln1{suffix}'
            if 'layernorm_after' in rest or 'norm2' in rest:
                suffix = '.weight' if 'weight' in rest else '.bias'
                return f'v.blk.{layer_idx}.ln2{suffix}'

        except (ValueError, IndexError):
            pass

    # Visual Blocks (alternatives Format): visual.blocks.{i}.*
    if 'blocks' in parts:
        try:
            layer_idx = parts[parts.index('blocks') + 1]
            rest = '.'.join(parts[parts.index('blocks') + 2:])

            if rest.startswith('attn.'):
                attn_rest = rest.replace('attn.', '')
                return f'v.blk.{layer_idx}.attn.{attn_rest}'
            if rest.startswith('mlp.'):
                return f'v.blk.{layer_idx}.ffn.{rest.replace("mlp.", "")}'
            if 'norm1' in rest:
                suffix = '.weight' if 'weight' in rest else '.bias'
                return f'v.blk.{layer_idx}.ln1{suffix}'
            if 'norm2' in rest:
                suffix = '.weight' if 'weight' in rest else '.bias'
                return f'v.blk.{layer_idx}.ln2{suffix}'
        except (ValueError, IndexError):
            pass

    # Fallback: v. Prefix
    return f'v.{hf_name}'


def should_skip_quantization(name: str) -> bool:
    """Prueft ob Tensor von Quantisierung ausgenommen werden soll."""
    skip_patterns = ('embed', 'pos', 'cls', 'norm', 'ln', 'bias')
    return any(p in name.lower() for p in skip_patterns)


def load_model(model_name: str):
    """Laedt Modell von HuggingFace mit trust_remote_code."""
    print(f"Lade Modell: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, config, processor


def extract_config(config) -> dict:
    """Extrahiert Vision-Konfiguration aus verschiedenen Config-Formaten."""
    vc = getattr(config, 'vision_config', getattr(config, 'visual', config))
    return {
        'hidden_size': getattr(vc, 'hidden_size', getattr(vc, 'embed_dim', 768)),
        'num_heads': getattr(vc, 'num_attention_heads', getattr(vc, 'num_heads', 12)),
        'num_layers': getattr(vc, 'num_hidden_layers', getattr(vc, 'depth', 12)),
        'mlp_ratio': getattr(vc, 'mlp_ratio', 4.0),
        'image_size': getattr(vc, 'image_size', 224),
        'patch_size': getattr(vc, 'patch_size', 16),
    }


def convert_nomic_vision(model_name: str, output_path: Path, quantize: str = 'f16'):
    """Hauptfunktion: Konvertiert nomic-embed-vision zu GGUF."""
    model, config, processor = load_model(model_name)
    cfg = extract_config(config)

    print(f"  Hidden Size: {cfg['hidden_size']}, Heads: {cfg['num_heads']}, Layers: {cfg['num_layers']}")
    print(f"  Image: {cfg['image_size']}x{cfg['image_size']}, Patch: {cfg['patch_size']}")

    # GGUF Writer
    writer = GGUFWriter(output_path)

    # Metadaten
    writer.add_metadata('general.architecture', 'nomic-vision')
    writer.add_metadata('general.name', model_name.split('/')[-1])
    writer.add_metadata('general.file_type', quantize)
    writer.add_metadata('nomic.hidden_size', cfg['hidden_size'])
    writer.add_metadata('nomic.num_attention_heads', cfg['num_heads'])
    writer.add_metadata('nomic.num_hidden_layers', cfg['num_layers'])
    writer.add_metadata('nomic.intermediate_size', int(cfg['hidden_size'] * cfg['mlp_ratio']))
    writer.add_metadata('nomic.image_size', cfg['image_size'])
    writer.add_metadata('nomic.patch_size', cfg['patch_size'])
    writer.add_metadata('nomic.num_patches', (cfg['image_size'] // cfg['patch_size']) ** 2)

    if hasattr(processor, 'image_mean'):
        writer.add_metadata('nomic.image_mean', list(processor.image_mean))
    if hasattr(processor, 'image_std'):
        writer.add_metadata('nomic.image_std', list(processor.image_std))

    # Quantisierungs-Typ
    quant_map = {'f32': GGML_TYPE_F32, 'f16': GGML_TYPE_F16, 'q8_0': GGML_TYPE_Q8_0, 'q4_k_m': GGML_TYPE_Q4_K}
    quant_type = quant_map.get(quantize.lower(), GGML_TYPE_F16)
    dtype_names = {GGML_TYPE_F32: 'F32', GGML_TYPE_F16: 'F16', GGML_TYPE_Q8_0: 'Q8_0', GGML_TYPE_Q4_K: 'Q4_K'}

    print(f"\nKonvertiere Tensoren ({quantize})...")

    # Tensoren konvertieren
    for hf_name, tensor in model.state_dict().items():
        gguf_name = get_tensor_name(hf_name)
        data = tensor.cpu().numpy()
        dtype = GGML_TYPE_F32 if should_skip_quantization(gguf_name) else quant_type
        writer.add_tensor(gguf_name, data, dtype)
        print(f"  {gguf_name}: {data.shape} -> {dtype_names.get(dtype, 'F32')}")

    writer.write()
    print(f"\nFertig: {output_path}")
    return output_path


def main():
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description='nomic-embed-vision HuggingFace -> GGUF Konverter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --model nomic-ai/nomic-embed-vision-v1.5 --output nomic-vision.gguf
  %(prog)s -m nomic-ai/nomic-embed-vision-v1.5 -o nomic-q8.gguf -q q8_0
  %(prog)s -m nomic-ai/nomic-embed-vision-v1 -o nomic-v1-q4.gguf -q q4_k_m
        """
    )
    parser.add_argument('--model', '-m', required=True, help='HuggingFace Modell-ID')
    parser.add_argument('--output', '-o', required=True, help='Output GGUF-Datei')
    parser.add_argument('--quantization', '-q', choices=['f32', 'f16', 'q8_0', 'q4_k_m'], default='f16')
    parser.add_argument('--validate', action='store_true', help='Validiere (noch nicht implementiert)')

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_nomic_vision(args.model, output_path, args.quantization)

    if args.validate:
        print("\nValidierung noch nicht implementiert - nutze benchmark.py")


if __name__ == '__main__':
    main()
