// Package siglip provides Go bindings for the SigLIP Vision Encoder.
//
// SigLIP (Sigmoid Loss for Language Image Pre-Training) ist ein Vision Transformer
// zur Generierung von Image-Embeddings. Diese Bindings wrappen die C-Implementation
// aus llama.cpp/src/siglip.h.
//
// Verwendung:
//
//	model, err := siglip.LoadModel("siglip-vit-b.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	embedding, err := model.Encode(imageData)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	floats := embedding.ToFloat32()
package siglip

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/src
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build -lsiglip -lggml -lm -lstdc++
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lpthread

#include <stdlib.h>
#include "siglip.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Fehler-Definitionen
var (
	ErrModelNotLoaded    = errors.New("siglip: model not loaded")
	ErrInvalidImage      = errors.New("siglip: invalid image data")
	ErrEncodingFailed    = errors.New("siglip: encoding failed")
	ErrInvalidParameters = errors.New("siglip: invalid parameters")
	ErrDimensionMismatch = errors.New("siglip: embedding dimension mismatch")
)

// ============================================================================
// Enums
// ============================================================================

// ModelType definiert den Modell-Typ
type ModelType int

const (
	ModelVitB16    ModelType = C.SIGLIP_MODEL_VIT_B_16    // ViT-Base, Patch 16, 86M params
	ModelVitL16    ModelType = C.SIGLIP_MODEL_VIT_L_16    // ViT-Large, Patch 16, 303M params
	ModelVitSO400M ModelType = C.SIGLIP_MODEL_VIT_SO400M  // ViT-SO400M, Patch 14, 400M params
	ModelUnknown   ModelType = C.SIGLIP_MODEL_UNKNOWN
)

// String gibt den Namen des Modell-Typs zurueck
func (t ModelType) String() string {
	switch t {
	case ModelVitB16:
		return "ViT-B/16"
	case ModelVitL16:
		return "ViT-L/16"
	case ModelVitSO400M:
		return "ViT-SO400M"
	default:
		return "Unknown"
	}
}

// Backend definiert das Compute-Backend
type Backend int

const (
	BackendCPU    Backend = C.SIGLIP_BACKEND_CPU    // CPU (GGML)
	BackendCUDA   Backend = C.SIGLIP_BACKEND_CUDA   // NVIDIA CUDA
	BackendMetal  Backend = C.SIGLIP_BACKEND_METAL  // Apple Metal
	BackendVulkan Backend = C.SIGLIP_BACKEND_VULKAN // Vulkan (experimentell)
)

// String gibt den Namen des Backends zurueck
func (b Backend) String() string {
	switch b {
	case BackendCPU:
		return "CPU"
	case BackendCUDA:
		return "CUDA"
	case BackendMetal:
		return "Metal"
	case BackendVulkan:
		return "Vulkan"
	default:
		return "Unknown"
	}
}

// LogLevel definiert das Log-Level
type LogLevel int

const (
	LogNone  LogLevel = C.SIGLIP_LOG_NONE
	LogError LogLevel = C.SIGLIP_LOG_ERROR
	LogWarn  LogLevel = C.SIGLIP_LOG_WARN
	LogInfo  LogLevel = C.SIGLIP_LOG_INFO
	LogDebug LogLevel = C.SIGLIP_LOG_DEBUG
)

// EmbedFormat definiert das Embedding-Format
type EmbedFormat int

const (
	EmbedF32        EmbedFormat = C.SIGLIP_EMBED_F32        // float32 Array
	EmbedF16        EmbedFormat = C.SIGLIP_EMBED_F16        // float16 Array
	EmbedNormalized EmbedFormat = C.SIGLIP_EMBED_NORMALIZED // L2-normalisiert
)

// ============================================================================
// Options
// ============================================================================

// Option ist eine funktionale Option fuer LoadModel
type Option func(*options)

type options struct {
	backend     Backend
	logLevel    LogLevel
	embedFormat EmbedFormat
	nThreads    int
	nGPULayers  int
	mainGPU     int
	useMmap     bool
	useMlock    bool
	batchSize   int
}

func defaultOptions() *options {
	return &options{
		backend:     BackendCPU,
		logLevel:    LogInfo,
		embedFormat: EmbedF32,
		nThreads:    runtime.NumCPU(),
		nGPULayers:  -1, // alle
		mainGPU:     0,
		useMmap:     true,
		useMlock:    false,
		batchSize:   1,
	}
}

// WithBackend setzt das Compute-Backend
func WithBackend(backend Backend) Option {
	return func(o *options) {
		o.backend = backend
	}
}

// WithLogLevel setzt das Log-Level
func WithLogLevel(level LogLevel) Option {
	return func(o *options) {
		o.logLevel = level
	}
}

// WithEmbedFormat setzt das Embedding-Format
func WithEmbedFormat(format EmbedFormat) Option {
	return func(o *options) {
		o.embedFormat = format
	}
}

// WithThreads setzt die Anzahl der CPU-Threads
func WithThreads(n int) Option {
	return func(o *options) {
		o.nThreads = n
	}
}

// WithGPULayers setzt die Anzahl der GPU-Layers (-1 fuer alle)
func WithGPULayers(n int) Option {
	return func(o *options) {
		o.nGPULayers = n
	}
}

// WithMainGPU setzt den Haupt-GPU Index
func WithMainGPU(gpu int) Option {
	return func(o *options) {
		o.mainGPU = gpu
	}
}

// WithMmap aktiviert/deaktiviert Memory-Mapping
func WithMmap(enabled bool) Option {
	return func(o *options) {
		o.useMmap = enabled
	}
}

// WithMlock aktiviert/deaktiviert Memory-Locking
func WithMlock(enabled bool) Option {
	return func(o *options) {
		o.useMlock = enabled
	}
}

// WithBatchSize setzt die Batch-Groesse
func WithBatchSize(size int) Option {
	return func(o *options) {
		o.batchSize = size
	}
}

// ============================================================================
// Model
// ============================================================================

// Model repraesentiert ein geladenes SigLIP-Modell
type Model struct {
	ctx    *C.struct_siglip_ctx
	mu     sync.Mutex
	closed bool

	// Cached Info
	embeddingDim int
	imageSize    int
	modelType    ModelType
	modelName    string
}

// LoadModel laedt ein SigLIP-Modell aus einer GGUF-Datei
func LoadModel(path string, opts ...Option) (*Model, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(o)
	}

	// C-Parameter erstellen
	var params C.struct_siglip_params
	params.backend = C.enum_siglip_backend(o.backend)
	params.log_level = C.enum_siglip_log_level(o.logLevel)
	params.embed_format = C.enum_siglip_embed_format(o.embedFormat)
	params.n_threads = C.int(o.nThreads)
	params.n_gpu_layers = C.int(o.nGPULayers)
	params.main_gpu = C.int(o.mainGPU)
	params.use_mmap = C.bool(o.useMmap)
	params.use_mlock = C.bool(o.useMlock)
	params.batch_size = C.int(o.batchSize)

	// Pfad zu C-String konvertieren
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	// Modell laden
	ctx := C.siglip_load_model(cPath, params)
	if ctx == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrModelNotLoaded
	}

	// Model erstellen
	m := &Model{
		ctx:          ctx,
		embeddingDim: int(C.siglip_get_embedding_dim(ctx)),
		imageSize:    int(C.siglip_get_image_size(ctx)),
		modelType:    ModelType(C.siglip_get_model_type(ctx)),
	}

	// Model-Name holen
	cName := C.siglip_get_model_name(ctx)
	if cName != nil {
		m.modelName = C.GoString(cName)
	}

	// Finalizer setzen fuer automatisches Cleanup
	runtime.SetFinalizer(m, (*Model).Close)

	return m, nil
}

// Close gibt das Modell frei
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}

	if m.ctx != nil {
		C.siglip_free(m.ctx)
		m.ctx = nil
	}

	m.closed = true
	runtime.SetFinalizer(m, nil)

	return nil
}

// EmbeddingDim gibt die Embedding-Dimension zurueck
func (m *Model) EmbeddingDim() int {
	return m.embeddingDim
}

// ImageSize gibt die erwartete Bildgroesse zurueck
func (m *Model) ImageSize() int {
	return m.imageSize
}

// ModelType gibt den Modell-Typ zurueck
func (m *Model) ModelType() ModelType {
	return m.modelType
}

// ModelName gibt den Modell-Namen zurueck
func (m *Model) ModelName() string {
	return m.modelName
}

// Encode generiert ein Embedding fuer ein Bild (Rohdaten)
//
// Das Bild wird als Byte-Array uebergeben (JPG, PNG, etc.)
func (m *Model) Encode(image []byte) (*Embedding, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.ctx == nil {
		return nil, ErrModelNotLoaded
	}

	if len(image) == 0 {
		return nil, ErrInvalidImage
	}

	// Bild aus Speicher laden (siglip_image_from_raw erwartet RGB-Daten,
	// wir muessen stb_image nutzen - das passiert in C)
	// Da siglip.h keine direkte Funktion fuer Speicher-Bilder hat,
	// nutzen wir einen Workaround ueber Base64 oder temporaere Datei.
	// Hier implementieren wir es ueber direkte Raw-Daten nach Dekodierung.

	// Fuer jetzt: Wir erwarten RGB-Daten direkt oder implementieren
	// eine Hilfsfunktion. Da siglip_image_from_base64 existiert, nutzen wir das.
	// Besser: Wir fuegen eine neue C-Funktion hinzu oder nutzen stb_image direkt.

	// Temporaerer Workaround: Wir nehmen an, dass das Bild bereits dekodiert ist
	// als RGB uint8 Array mit bekannten Dimensionen.
	// In der Praxis wuerde man hier stb_image_load_from_memory nutzen.

	// Fuer eine vollstaendige Implementation fuegen wir eine Hilfsfunktion hinzu:
	cImg := m.loadImageFromMemory(image)
	if cImg == nil {
		return nil, ErrInvalidImage
	}
	defer C.siglip_image_free(cImg)

	// Encoding
	cEmb := C.siglip_encode(m.ctx, cImg)
	if cEmb == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrEncodingFailed
	}

	return newEmbeddingFromC(cEmb), nil
}

// EncodeRaw generiert ein Embedding fuer ein Bild aus RGB-Rohdaten
//
// data: RGB uint8 Array (HWC Format)
// width, height: Bildgroesse
func (m *Model) EncodeRaw(data []byte, width, height int) (*Embedding, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.ctx == nil {
		return nil, ErrModelNotLoaded
	}

	if len(data) == 0 || width <= 0 || height <= 0 {
		return nil, ErrInvalidImage
	}

	expectedSize := width * height * 3
	if len(data) < expectedSize {
		return nil, fmt.Errorf("siglip: image data too small (expected %d, got %d)", expectedSize, len(data))
	}

	// C-Bild erstellen
	cImg := C.siglip_image_from_raw(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.int(width),
		C.int(height),
		C.int(3), // channels
	)
	if cImg == nil {
		return nil, ErrInvalidImage
	}
	defer C.siglip_image_free(cImg)

	// Encoding
	cEmb := C.siglip_encode(m.ctx, cImg)
	if cEmb == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrEncodingFailed
	}

	return newEmbeddingFromC(cEmb), nil
}

// EncodeBatch generiert Embeddings fuer mehrere Bilder
func (m *Model) EncodeBatch(images [][]byte) ([]*Embedding, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed || m.ctx == nil {
		return nil, ErrModelNotLoaded
	}

	if len(images) == 0 {
		return nil, ErrInvalidParameters
	}

	// C-Bilder erstellen
	cImages := make([]*C.struct_siglip_image, len(images))
	for i, img := range images {
		cImg := m.loadImageFromMemory(img)
		if cImg == nil {
			// Cleanup bereits erstellte Bilder
			for j := 0; j < i; j++ {
				C.siglip_image_free(cImages[j])
			}
			return nil, fmt.Errorf("siglip: failed to load image %d", i)
		}
		cImages[i] = cImg
	}

	// Cleanup am Ende
	defer func() {
		for _, cImg := range cImages {
			if cImg != nil {
				C.siglip_image_free(cImg)
			}
		}
	}()

	// Batch erstellen
	var batch C.struct_siglip_batch
	batch.images = (**C.struct_siglip_image)(unsafe.Pointer(&cImages[0]))
	batch.n_images = C.int(len(images))

	// Batch-Encoding
	cEmb := C.siglip_encode_batch(m.ctx, &batch)
	if cEmb == nil {
		errStr := C.siglip_get_last_error()
		if errStr != nil {
			return nil, fmt.Errorf("siglip: %s", C.GoString(errStr))
		}
		return nil, ErrEncodingFailed
	}

	// Embeddings extrahieren
	embeddings := make([]*Embedding, len(images))
	batchSize := int(cEmb.batch_size)
	embSize := int(cEmb.size)

	for i := 0; i < batchSize && i < len(images); i++ {
		emb := &Embedding{
			data:       make([]float32, embSize),
			normalized: bool(cEmb.normalized),
		}

		// Daten kopieren
		srcPtr := unsafe.Pointer(uintptr(unsafe.Pointer(cEmb.data)) + uintptr(i*embSize*4))
		for j := 0; j < embSize; j++ {
			emb.data[j] = *(*float32)(unsafe.Pointer(uintptr(srcPtr) + uintptr(j*4)))
		}

		embeddings[i] = emb
	}

	// C-Embedding freigeben
	C.siglip_embedding_free(cEmb)

	return embeddings, nil
}

// loadImageFromMemory laedt ein Bild aus Speicher
// Dies ist eine Hilfsfunktion, die stb_image nutzt
func (m *Model) loadImageFromMemory(data []byte) *C.struct_siglip_image {
	// Da siglip.h keine direkte Funktion fuer Speicher-Bilder hat,
	// implementieren wir einen Workaround.
	// Option 1: Base64-Encoding (ineffizient)
	// Option 2: Temporaere Datei (langsam)
	// Option 3: Direkte stb_image Nutzung in CGO

	// Fuer die vollstaendige Implementation wuerde man eine neue C-Funktion
	// siglip_image_from_memory hinzufuegen. Hier nutzen wir einen Workaround
	// mit Base64.

	// Base64 encoding
	base64Data := base64Encode(data)
	cBase64 := C.CString(base64Data)
	defer C.free(unsafe.Pointer(cBase64))

	return C.siglip_image_from_base64(cBase64)
}

// ============================================================================
// Embedding
// ============================================================================

// Embedding repraesentiert ein SigLIP Image-Embedding
type Embedding struct {
	data       []float32
	normalized bool
}

// newEmbeddingFromC erstellt ein Go-Embedding aus einem C-Embedding
func newEmbeddingFromC(cEmb *C.struct_siglip_embedding) *Embedding {
	if cEmb == nil {
		return nil
	}

	size := int(cEmb.size)
	emb := &Embedding{
		data:       make([]float32, size),
		normalized: bool(cEmb.normalized),
	}

	// Daten kopieren
	for i := 0; i < size; i++ {
		emb.data[i] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cEmb.data)) + uintptr(i*4))))
	}

	// C-Embedding freigeben
	C.siglip_embedding_free(cEmb)

	return emb
}

// ToFloat32 gibt das Embedding als float32-Slice zurueck
func (e *Embedding) ToFloat32() []float32 {
	if e == nil {
		return nil
	}
	// Kopie zurueckgeben
	result := make([]float32, len(e.data))
	copy(result, e.data)
	return result
}

// Size gibt die Dimension des Embeddings zurueck
func (e *Embedding) Size() int {
	if e == nil {
		return 0
	}
	return len(e.data)
}

// IsNormalized gibt zurueck ob das Embedding L2-normalisiert ist
func (e *Embedding) IsNormalized() bool {
	if e == nil {
		return false
	}
	return e.normalized
}

// Normalize normalisiert das Embedding in-place (L2-Norm)
func (e *Embedding) Normalize() {
	if e == nil || len(e.data) == 0 {
		return
	}

	var norm float32
	for _, v := range e.data {
		norm += v * v
	}
	norm = float32(sqrt64(float64(norm)))

	if norm > 0 {
		for i := range e.data {
			e.data[i] /= norm
		}
	}
	e.normalized = true
}

// CosineSimilarity berechnet die Cosine Similarity zwischen zwei Embeddings
func (e *Embedding) CosineSimilarity(other *Embedding) float32 {
	if e == nil || other == nil {
		return 0
	}

	if len(e.data) != len(other.data) {
		return 0
	}

	var dot, normA, normB float32
	for i := range e.data {
		dot += e.data[i] * other.data[i]
		normA += e.data[i] * e.data[i]
		normB += other.data[i] * other.data[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (float32(sqrt64(float64(normA))) * float32(sqrt64(float64(normB))))
}

// DotProduct berechnet das Skalarprodukt zwischen zwei Embeddings
func (e *Embedding) DotProduct(other *Embedding) float32 {
	if e == nil || other == nil {
		return 0
	}

	if len(e.data) != len(other.data) {
		return 0
	}

	var dot float32
	for i := range e.data {
		dot += e.data[i] * other.data[i]
	}

	return dot
}

// Clone erstellt eine Kopie des Embeddings
func (e *Embedding) Clone() *Embedding {
	if e == nil {
		return nil
	}

	return &Embedding{
		data:       e.ToFloat32(),
		normalized: e.normalized,
	}
}

// ============================================================================
// Image
// ============================================================================

// Image repraesentiert ein Bild fuer SigLIP
type Image struct {
	Data     []byte
	Width    int
	Height   int
	Channels int
}

// NewImageFromRGB erstellt ein neues Image aus RGB-Daten
func NewImageFromRGB(data []byte, width, height int) *Image {
	return &Image{
		Data:     data,
		Width:    width,
		Height:   height,
		Channels: 3,
	}
}

// ============================================================================
// Utility Functions
// ============================================================================

// Version gibt die SigLIP-Version zurueck
func Version() string {
	return C.GoString(C.siglip_version())
}

// BuildInfo gibt Build-Informationen zurueck
func BuildInfo() string {
	return C.GoString(C.siglip_build_info())
}

// SystemInfo gibt System-Informationen zurueck
func SystemInfo() string {
	return C.GoString(C.siglip_system_info())
}

// BackendAvailable prueft ob ein Backend verfuegbar ist
func BackendAvailable(backend Backend) bool {
	return bool(C.siglip_backend_available(C.enum_siglip_backend(backend)))
}

// AvailableBackends gibt eine Liste verfuegbarer Backends zurueck
func AvailableBackends() []Backend {
	var cBackends [4]C.enum_siglip_backend
	n := C.siglip_get_available_backends(&cBackends[0], 4)

	backends := make([]Backend, int(n))
	for i := 0; i < int(n); i++ {
		backends[i] = Backend(cBackends[i])
	}

	return backends
}

// SetLogLevel setzt das globale Log-Level
func SetLogLevel(level LogLevel) {
	C.siglip_set_log_level(C.enum_siglip_log_level(level))
}

// GetLastError gibt den letzten Fehler zurueck
func GetLastError() string {
	errStr := C.siglip_get_last_error()
	if errStr == nil {
		return ""
	}
	return C.GoString(errStr)
}

// ClearError loescht den letzten Fehler
func ClearError() {
	C.siglip_clear_error()
}

// ============================================================================
// Helper Functions
// ============================================================================

// sqrt64 ist eine Hilfsfunktion fuer Quadratwurzel
func sqrt64(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}

	// Newton-Raphson
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// base64Encode kodiert Bytes zu Base64
func base64Encode(data []byte) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

	result := make([]byte, 0, (len(data)+2)/3*4)

	for i := 0; i < len(data); i += 3 {
		var n uint32

		remaining := len(data) - i
		switch remaining {
		case 1:
			n = uint32(data[i]) << 16
			result = append(result, alphabet[n>>18], alphabet[(n>>12)&0x3F], '=', '=')
		case 2:
			n = uint32(data[i])<<16 | uint32(data[i+1])<<8
			result = append(result, alphabet[n>>18], alphabet[(n>>12)&0x3F], alphabet[(n>>6)&0x3F], '=')
		default:
			n = uint32(data[i])<<16 | uint32(data[i+1])<<8 | uint32(data[i+2])
			result = append(result, alphabet[n>>18], alphabet[(n>>12)&0x3F], alphabet[(n>>6)&0x3F], alphabet[n&0x3F])
		}
	}

	return string(result)
}

// ============================================================================
// Batch Processing Helpers
// ============================================================================

// BatchEncode ist ein Convenience-Wrapper fuer Batch-Encoding mit Fehlerbehandlung
func (m *Model) BatchEncode(images [][]byte, onProgress func(current, total int)) ([]*Embedding, []error) {
	embeddings := make([]*Embedding, len(images))
	errors := make([]error, len(images))

	for i, img := range images {
		emb, err := m.Encode(img)
		embeddings[i] = emb
		errors[i] = err

		if onProgress != nil {
			onProgress(i+1, len(images))
		}
	}

	return embeddings, errors
}

// CosineSimilarityMatrix berechnet die Cosine Similarity Matrix fuer mehrere Embeddings
func CosineSimilarityMatrix(embeddings []*Embedding) [][]float32 {
	n := len(embeddings)
	matrix := make([][]float32, n)

	for i := range matrix {
		matrix[i] = make([]float32, n)
		for j := range matrix[i] {
			if i == j {
				matrix[i][j] = 1.0
			} else if i < j {
				matrix[i][j] = embeddings[i].CosineSimilarity(embeddings[j])
			} else {
				matrix[i][j] = matrix[j][i] // Symmetrie nutzen
			}
		}
	}

	return matrix
}

// FindMostSimilar findet die aehnlichsten Embeddings zu einem Query-Embedding
func FindMostSimilar(query *Embedding, candidates []*Embedding, topK int) []int {
	if query == nil || len(candidates) == 0 {
		return nil
	}

	// Similarities berechnen
	type scored struct {
		index int
		score float32
	}

	scores := make([]scored, len(candidates))
	for i, cand := range candidates {
		scores[i] = scored{
			index: i,
			score: query.CosineSimilarity(cand),
		}
	}

	// Sortieren (einfaches Bubble Sort fuer kleine Listen)
	for i := 0; i < len(scores)-1; i++ {
		for j := 0; j < len(scores)-i-1; j++ {
			if scores[j].score < scores[j+1].score {
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}

	// Top-K extrahieren
	if topK > len(scores) {
		topK = len(scores)
	}

	result := make([]int, topK)
	for i := 0; i < topK; i++ {
		result[i] = scores[i].index
	}

	return result
}
