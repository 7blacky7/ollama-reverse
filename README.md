<div align="center">
  <img alt="ollama-vision" width="240" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
  <h1>Ollama Vision</h1>
  <p><strong>Multi-Model Vision Embedding System</strong></p>
  <p>Fork von Ollama mit erweiterter Vision-Encoder Unterstuetzung</p>
</div>

---

## Uebersicht

Ollama Vision erweitert das Standard-Ollama um ein vollstaendiges **Multi-Model Vision Embedding System**. Es unterstuetzt verschiedene Vision-Encoder fuer Image-Embeddings mit HuggingFace-Integration und GPU-Beschleunigung.

### Unterstuetzte Vision-Encoder

| Modell | Parameter | Embedding-Dim | Use-Case |
|--------|-----------|---------------|----------|
| **SigLIP** | 86M - 303M | 768 - 1024 | Text-Image Matching, Zero-Shot Classification |
| **CLIP** | 151M - 428M | 512 - 768 | Multimodal Search, Image Classification |
| **OpenCLIP** | 354M - 2.5B | 768 - 1280 | Large-Scale Image-Text Retrieval |
| **DINOv2** | 22M - 1.1B | 384 - 1536 | Feature Extraction, Segmentation |
| **Nomic Embed Vision** | 86M | 768 | Unified Text+Image Embedding Space |
| **EVA-CLIP** | 307M - 4.4B | 768 - 1024 | High-Performance Vision Tasks |

## Schnellstart

### Docker (Empfohlen)

```bash
# Image bauen
docker build -t ollama-vision .

# Container starten (mit GPU)
docker run -d \
  --name ollama-vision \
  --gpus all \
  -v /pfad/zu/modellen:/root/.ollama \
  -p 11435:11434 \
  --restart unless-stopped \
  ollama-vision:latest
```

### Von Source bauen

```bash
# Repository klonen
git clone https://github.com/moritzWa/ollama-reverse.git
cd ollama-reverse

# Go Build
go build -o ollama .

# Server starten
./ollama serve
```

## Vision API Endpoints

### Einzelbild encodieren

```bash
curl -X POST http://localhost:11435/api/vision/encode \
  -H "Content-Type: application/json" \
  -d '{
    "model": "siglip-base",
    "image": "/pfad/zum/bild.jpg"
  }'
```

### Batch-Encoding

```bash
curl -X POST http://localhost:11435/api/vision/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clip-vit-l-14",
    "images": ["/bild1.jpg", "/bild2.jpg", "/bild3.jpg"]
  }'
```

### Bild-Aehnlichkeit

```bash
curl -X POST http://localhost:11435/api/vision/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "model": "siglip-base",
    "image1": "/bild1.jpg",
    "image2": "/bild2.jpg"
  }'
```

### HuggingFace Modell laden

```bash
curl -X POST http://localhost:11435/api/vision/load/hf \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "google/siglip-base-patch16-224",
    "quantization": "q8_0"
  }'
```

### Verfuegbare Modelle

```bash
curl http://localhost:11435/api/vision/models
```

## CLI Tools

### siglip-cli

Standalone CLI fuer SigLIP Image-Embeddings:

```bash
# Einzelbild encodieren
siglip-cli --model model.gguf --encode image.jpg --format json

# Batch-Verarbeitung
siglip-cli --model model.gguf --batch images/ -o embeddings.bin

# Aehnlichkeit berechnen
siglip-cli --model model.gguf --similarity img1.jpg img2.jpg
```

### vision-bench

GPU-Backend Benchmark Suite:

```bash
# Alle Backends benchmarken
vision-bench --model siglip-base --backends cpu,cuda,metal

# Batch-Size Vergleich
vision-bench --model clip-vit-l-14 --batch-sizes 1,4,16,64
```

## HuggingFace Integration

Modelle koennen direkt von HuggingFace geladen werden:

```bash
# Ueber API
curl -X POST http://localhost:11435/api/vision/load/hf \
  -d '{"model_id": "nomic-ai/nomic-embed-vision-v1"}'

# Ueber Go Code
import "ollama/huggingface"

client := huggingface.NewClient()
model, _ := client.LoadModel("google/siglip-base-patch16-224")
```

### Unterstuetzte HuggingFace Modelle

- `google/siglip-base-patch16-224`
- `google/siglip-large-patch16-384`
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
- `facebook/dinov2-base`
- `facebook/dinov2-large`
- `nomic-ai/nomic-embed-vision-v1`
- `nomic-ai/nomic-embed-vision-v1.5`

## GPU-Backend Support

| Backend | Plattform | Status |
|---------|-----------|--------|
| **CPU** | Alle | Stabil |
| **CUDA** | Linux/Windows | Stabil |
| **Metal** | macOS | Stabil |
| **ROCm** | Linux (AMD) | Experimentell |
| **Vulkan** | Alle | Experimentell |

### Backend-Erkennung

```go
import "ollama/vision/backend"

backends := backend.DetectBackends()
best := backend.SelectBestBackend()
devices := backend.GetDevices()
```

## Projekt-Struktur

```
ollama-reverse/
├── siglip/           # SigLIP C++ Encoder
├── vision/           # Vision Framework
│   ├── clip/         # CLIP Encoder
│   ├── dinov2/       # DINOv2 Encoder
│   ├── openclip/     # OpenCLIP Encoder
│   ├── nomic/        # Nomic Vision Encoder
│   ├── backend/      # GPU Backend Management
│   └── benchmark/    # Benchmark Suite
├── huggingface/      # HuggingFace Hub Integration
├── server/           # REST API Server
├── scripts/          # Python Konverter
│   ├── convert_siglip.py
│   ├── convert_clip.py
│   ├── convert_dinov2.py
│   └── convert_openclip.py
└── tools/
    ├── siglip-cli/   # CLI Tool
    └── vision-bench/ # Benchmark Tool
```

## Konfiguration

### Umgebungsvariablen

| Variable | Beschreibung | Default |
|----------|--------------|---------|
| `OLLAMA_HOST` | Server-Adresse | `127.0.0.1:11434` |
| `OLLAMA_MODELS` | Modell-Verzeichnis | `~/.ollama/models` |
| `OLLAMA_VISION_CACHE` | Vision-Cache | `~/.ollama/vision/cache` |
| `OLLAMA_GPU_LAYERS` | GPU-Layer Anzahl | `auto` |
| `HF_TOKEN` | HuggingFace Token | - |

## Entwicklung

### Voraussetzungen

- Go 1.21+
- CMake 3.16+
- C++17 Compiler
- Python 3.10+ (fuer Konverter)

### Build

```bash
# Go Build
go build -o ollama .

# Mit CUDA
go build -tags cuda -o ollama .

# CMake Build (C++ Libraries)
cmake -B build -DSIGLIP_CUDA=ON
cmake --build build --parallel
```

### Tests

```bash
# Go Tests
go test ./...

# Vision Tests
go test ./vision/... -v

# Benchmark
go test ./vision/benchmark/... -bench=.
```

## Mitwirken

Beitraege sind willkommen! Bitte beachte:

- Code-Style: `gofmt` und deutsche Kommentare
- LOC-Limit: Max. 300 Zeilen pro Datei
- Tests: Fuer neue Features erforderlich

## Lizenz

MIT License - siehe [LICENSE](LICENSE)

## Danksagungen

- [Ollama](https://github.com/ollama/ollama) - Basis-Projekt
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGML Backend
- [SigLIP](https://github.com/google-research/big_vision) - Google Research
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - LAION
- [DINOv2](https://github.com/facebookresearch/dinov2) - Meta Research

---

<div align="center">
  <sub>Entwickelt mit Claude Code</sub>
</div>
