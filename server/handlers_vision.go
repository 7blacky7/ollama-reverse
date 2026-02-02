//go:build vision

// MODUL: handlers_vision - HTTP Handler fuer Vision Embedding Endpoints
// ABHAENGIGKEITEN: vision/onnx, encoding/base64, net/http

package server

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/ollama/ollama/vision"
	"github.com/ollama/ollama/vision/onnx"
)

// ONNX Encoder Konfiguration
const (
	onnxModelEnvVar      = "OLLAMA_VISION_ONNX_MODEL"
	onnxDefaultModelPath = "/root/.cache/huggingface/hub/models--nomic-ai--nomic-embed-vision-v1.5/snapshots/main/onnx/model_int8.onnx"
	onnxModelName        = "nomic-onnx"
)

// ONNX Encoder Singleton
var (
	onnxEncoder     vision.VisionEncoder
	onnxEncoderOnce sync.Once
	onnxEncoderErr  error
	onnxEncoderMu   sync.RWMutex
)

// getOnnxModelPath ermittelt ONNX Modell-Pfad (Environment oder Default)
func getOnnxModelPath() string {
	if path := os.Getenv(onnxModelEnvVar); path != "" {
		return path
	}
	return onnxDefaultModelPath
}

// loadOnnxEncoder laedt den ONNX Encoder bei Bedarf (Lazy Loading)
func loadOnnxEncoder() (vision.VisionEncoder, error) {
	onnxEncoderOnce.Do(func() {
		modelPath := getOnnxModelPath()
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			onnxEncoderErr = fmt.Errorf("ONNX modell nicht gefunden: %s", modelPath)
			return
		}
		opts := vision.LoadOptions{Threads: 0, Device: ""}
		encoder, err := onnx.NewOnnxEncoder(modelPath, opts)
		if err != nil {
			onnxEncoderErr = fmt.Errorf("ONNX encoder laden: %w", err)
			return
		}
		onnxEncoder = encoder
	})
	return onnxEncoder, onnxEncoderErr
}

func errModelNotLoaded(name string) error {
	return fmt.Errorf("model not loaded: %s", name)
}

// HandleEncode verarbeitet POST /api/vision/encode (ONNX Lazy Loading)
func (h *VisionHandler) HandleEncode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionEncodeRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	if err := h.validateEncodeRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Zeitmessung starten
	startTime := time.Now()

	embedding, err := h.encodeImage(req.Model, req.Image)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "ENCODING_ERROR")
		return
	}

	// Verarbeitungszeit berechnen
	processingMs := time.Since(startTime).Milliseconds()

	h.writeVisionJSON(w, http.StatusOK, VisionEncodeResponse{
		Embedding:        embedding,
		Dimensions:       len(embedding),
		Model:            req.Model,
		ProcessingTimeMs: processingMs,
	})
}

func (h *VisionHandler) validateEncodeRequest(req VisionEncodeRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if req.Image == "" {
		return fmt.Errorf("image is required")
	}
	return nil
}

// encodeImage dekodiert Base64 und generiert Embedding (ONNX oder Registry)
func (h *VisionHandler) encodeImage(modelName, imageBase64 string) ([]float32, error) {
	imageData, err := base64.StdEncoding.DecodeString(imageBase64)
	if err != nil {
		return nil, fmt.Errorf("invalid base64: %v", err)
	}

	// ONNX Encoder fuer spezifische Modellnamen verwenden
	if isOnnxModel(modelName) {
		return h.encodeWithOnnx(imageData)
	}

	encoder, err := h.getModel(modelName)
	if err != nil {
		return nil, err
	}

	return encoder.Encode(imageData)
}

func isOnnxModel(name string) bool {
	return name == onnxModelName || name == "nomic-embed-vision-onnx" || name == "onnx"
}

func (h *VisionHandler) encodeWithOnnx(imageData []byte) ([]float32, error) {
	encoder, err := loadOnnxEncoder()
	if err != nil {
		return nil, fmt.Errorf("ONNX encoder nicht verfuegbar: %w", err)
	}

	return encoder.Encode(imageData)
}

// HandleBatch verarbeitet POST /api/vision/batch (ONNX Batch-Verarbeitung)
func (h *VisionHandler) HandleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeVisionError(w, http.StatusMethodNotAllowed, "method not allowed", "METHOD_NOT_ALLOWED")
		return
	}

	var req VisionBatchRequest
	if err := decodeVisionJSON(r, &req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err), "INVALID_REQUEST")
		return
	}

	if err := h.validateBatchRequest(req); err != nil {
		h.writeVisionError(w, http.StatusBadRequest, err.Error(), "VALIDATION_ERROR")
		return
	}

	// Zeitmessung starten
	startTime := time.Now()

	embeddings, err := h.encodeBatch(req.Model, req.Images)
	if err != nil {
		h.writeVisionError(w, http.StatusInternalServerError, err.Error(), "ENCODING_ERROR")
		return
	}

	// Verarbeitungszeit berechnen
	processingMs := time.Since(startTime).Milliseconds()

	dimensions := 0
	if len(embeddings) > 0 {
		dimensions = len(embeddings[0])
	}

	h.writeVisionJSON(w, http.StatusOK, VisionBatchResponse{
		Embeddings:       embeddings,
		Dimensions:       dimensions,
		Model:            req.Model,
		Count:            len(embeddings),
		ProcessingTimeMs: processingMs,
	})
}

func (h *VisionHandler) validateBatchRequest(req VisionBatchRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if len(req.Images) == 0 {
		return fmt.Errorf("images are required")
	}
	return nil
}

func (h *VisionHandler) encodeBatch(modelName string, imagesBase64 []string) ([][]float32, error) {
	imagesData := make([][]byte, len(imagesBase64))
	for i, img := range imagesBase64 {
		data, err := base64.StdEncoding.DecodeString(img)
		if err != nil {
			return nil, fmt.Errorf("invalid base64 at index %d: %v", i, err)
		}
		imagesData[i] = data
	}

	// ONNX Encoder fuer spezifische Modellnamen verwenden
	if isOnnxModel(modelName) {
		return h.encodeBatchWithOnnx(imagesData)
	}

	encoder, err := h.getModel(modelName)
	if err != nil {
		return nil, err
	}

	return encoder.EncodeBatch(imagesData)
}

func (h *VisionHandler) encodeBatchWithOnnx(imagesData [][]byte) ([][]float32, error) {
	encoder, err := loadOnnxEncoder()
	if err != nil {
		return nil, fmt.Errorf("ONNX encoder nicht verfuegbar: %w", err)
	}
	return encoder.EncodeBatch(imagesData)
}

// CloseOnnxEncoder schliesst den globalen ONNX Encoder (Server-Shutdown)
func CloseOnnxEncoder() error {
	onnxEncoderMu.Lock()
	defer onnxEncoderMu.Unlock()
	if onnxEncoder != nil {
		err := onnxEncoder.Close()
		onnxEncoder = nil
		return err
	}
	return nil
}

// IsOnnxEncoderLoaded gibt zurueck ob der ONNX Encoder geladen ist
func IsOnnxEncoderLoaded() bool {
	onnxEncoderMu.RLock()
	defer onnxEncoderMu.RUnlock()
	return onnxEncoder != nil
}

// GetOnnxEncoderInfo gibt Modell-Infos zurueck wenn geladen
func GetOnnxEncoderInfo() *vision.ModelInfo {
	onnxEncoderMu.RLock()
	defer onnxEncoderMu.RUnlock()
	if onnxEncoder == nil {
		return nil
	}
	info := onnxEncoder.ModelInfo()
	return &info
}
