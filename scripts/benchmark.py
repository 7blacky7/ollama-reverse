#!/usr/bin/env python3
"""
SigLIP Benchmark & Validierung

Vergleicht GGUF-Modell gegen PyTorch-Referenz und misst Performance.

Verwendung:
    python benchmark.py --gguf siglip-vit-b.gguf --hf google/siglip-base-patch16-224
    python benchmark.py --gguf siglip-vit-b.gguf --dir ./test_images/ --benchmark
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import SiglipVisionModel, SiglipImageProcessor
except ImportError:
    print("Fehler: Bitte installiere die benötigten Pakete:")
    print("  pip install torch transformers pillow")
    exit(1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Berechnet Cosine Similarity zwischen zwei Vektoren"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_pytorch_embedding(model_name: str, image_path: Path) -> np.ndarray:
    """Generiert Embedding mit PyTorch/HuggingFace"""
    model = SiglipVisionModel.from_pretrained(model_name)
    processor = SiglipImageProcessor.from_pretrained(model_name)

    model.eval()

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # Pooled output verwenden
        embedding = outputs.pooler_output[0].numpy()

    return embedding


def get_gguf_embedding(gguf_path: Path, image_path: Path, cli_path: Optional[Path] = None) -> np.ndarray:
    """Generiert Embedding mit siglip-cli"""
    if cli_path is None:
        # Standard-Pfade versuchen
        possible_paths = [
            Path('./build/bin/siglip-cli'),
            Path('./build/siglip-cli'),
            Path('../build/bin/siglip-cli'),
        ]
        for p in possible_paths:
            if p.exists():
                cli_path = p
                break

    if cli_path is None or not cli_path.exists():
        raise FileNotFoundError("siglip-cli nicht gefunden. Bitte --cli angeben oder zuerst bauen.")

    # CLI aufrufen
    result = subprocess.run(
        [str(cli_path), '-m', str(gguf_path), '-i', str(image_path), '--format', 'json'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"siglip-cli Fehler: {result.stderr}")

    # JSON parsen
    data = json.loads(result.stdout)
    return np.array(data['embedding'])


def validate_model(gguf_path: Path, model_name: str, test_images: list[Path], cli_path: Optional[Path] = None):
    """Validiert GGUF gegen PyTorch-Referenz"""
    print(f"Validierung: {gguf_path.name} vs {model_name}")
    print("=" * 60)

    similarities = []

    for img_path in test_images:
        print(f"\n  Bild: {img_path.name}")

        try:
            # PyTorch Embedding
            start = time.perf_counter()
            pytorch_emb = get_pytorch_embedding(model_name, img_path)
            pytorch_time = (time.perf_counter() - start) * 1000

            # GGUF Embedding
            start = time.perf_counter()
            gguf_emb = get_gguf_embedding(gguf_path, img_path, cli_path)
            gguf_time = (time.perf_counter() - start) * 1000

            # Vergleich
            sim = cosine_similarity(pytorch_emb, gguf_emb)
            similarities.append(sim)

            print(f"    PyTorch Zeit: {pytorch_time:.1f}ms")
            print(f"    GGUF Zeit:    {gguf_time:.1f}ms")
            print(f"    Cosine Sim:   {sim:.6f}")

        except Exception as e:
            print(f"    Fehler: {e}")

    if similarities:
        print("\n" + "=" * 60)
        print(f"Durchschnittliche Cosine Similarity: {np.mean(similarities):.6f}")
        print(f"Minimum Cosine Similarity:           {np.min(similarities):.6f}")

        if np.min(similarities) >= 0.99:
            print("\n[PASS] Validierung erfolgreich!")
        else:
            print("\n[WARN] Similarity unter 0.99 - mögliche Quantisierungsfehler")


def benchmark_performance(gguf_path: Path, test_images: list[Path], cli_path: Optional[Path] = None, iterations: int = 10):
    """Misst Performance des GGUF-Modells"""
    print(f"Performance-Benchmark: {gguf_path.name}")
    print(f"Bilder: {len(test_images)}, Iterationen: {iterations}")
    print("=" * 60)

    times = []

    for i in range(iterations):
        for img_path in test_images:
            start = time.perf_counter()
            _ = get_gguf_embedding(gguf_path, img_path, cli_path)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        print(f"  Iteration {i+1}/{iterations}: {np.mean(times[-len(test_images):]):.1f}ms avg")

    print("\n" + "=" * 60)
    print(f"Durchschnitt:  {np.mean(times):.1f}ms")
    print(f"Median:        {np.median(times):.1f}ms")
    print(f"Min:           {np.min(times):.1f}ms")
    print(f"Max:           {np.max(times):.1f}ms")
    print(f"Std:           {np.std(times):.1f}ms")
    print(f"Throughput:    {1000 / np.mean(times):.1f} images/sec")


def create_test_images(output_dir: Path, count: int = 5):
    """Erstellt Test-Bilder für Benchmarks"""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for i in range(count):
        # Zufälliges Bild erstellen
        img = Image.new('RGB', (224, 224))
        pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)

        path = output_dir / f"test_{i:03d}.jpg"
        img.save(path)
        images.append(path)

    return images


def main():
    parser = argparse.ArgumentParser(
        description='SigLIP Benchmark & Validierung'
    )

    parser.add_argument(
        '--gguf', '-g',
        type=Path,
        required=True,
        help='GGUF Modell-Pfad'
    )

    parser.add_argument(
        '--hf', '-m',
        default='google/siglip-base-patch16-224',
        help='HuggingFace Modell für Validierung'
    )

    parser.add_argument(
        '--dir', '-d',
        type=Path,
        help='Verzeichnis mit Test-Bildern'
    )

    parser.add_argument(
        '--image', '-i',
        type=Path,
        help='Einzelnes Test-Bild'
    )

    parser.add_argument(
        '--cli',
        type=Path,
        help='Pfad zu siglip-cli'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Performance-Benchmark ausführen'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validierung gegen PyTorch'
    )

    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=10,
        help='Benchmark-Iterationen'
    )

    parser.add_argument(
        '--create-test-images',
        type=int,
        metavar='N',
        help='Erstelle N Test-Bilder'
    )

    args = parser.parse_args()

    # Test-Bilder sammeln
    test_images = []

    if args.create_test_images:
        test_dir = Path('./test_images')
        test_images = create_test_images(test_dir, args.create_test_images)
        print(f"Erstellt {len(test_images)} Test-Bilder in {test_dir}")

    if args.image:
        test_images.append(args.image)

    if args.dir:
        test_images.extend(args.dir.glob('*.jpg'))
        test_images.extend(args.dir.glob('*.jpeg'))
        test_images.extend(args.dir.glob('*.png'))

    if not test_images:
        print("Keine Test-Bilder angegeben. Verwende --image, --dir oder --create-test-images")
        return

    print(f"Test-Bilder: {len(test_images)}")

    # Validierung
    if args.validate or not args.benchmark:
        validate_model(args.gguf, args.hf, test_images, args.cli)

    # Benchmark
    if args.benchmark:
        print("\n")
        benchmark_performance(args.gguf, test_images, args.cli, args.iterations)


if __name__ == '__main__':
    main()
