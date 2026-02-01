// stubs.go - Interface-Stubs und Speicher-Methoden fuer Image-Generation
// Dieses Modul enthaelt nicht-implementierte LlamaServer-Methoden und VRAM-Abfragen.
package imagegen

import (
	"context"
	"errors"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// VRAMSize returns the estimated VRAM usage.
func (s *Server) VRAMSize() uint64 {
	return s.vramSize
}

// TotalSize returns the total memory usage.
func (s *Server) TotalSize() uint64 {
	return s.vramSize
}

// VRAMByGPU returns VRAM usage for a specific GPU.
func (s *Server) VRAMByGPU(id ml.DeviceID) uint64 {
	return s.vramSize
}

// Embedding is not supported for image generation models.
func (s *Server) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("not supported")
}

// Tokenize is not supported for image generation models.
func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("not supported")
}

// Detokenize is not supported for image generation models.
func (s *Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("not supported")
}

// GetDeviceInfos returns device information (not implemented).
func (s *Server) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }

// Ensure Server implements llm.LlamaServer
var _ llm.LlamaServer = (*Server)(nil)
