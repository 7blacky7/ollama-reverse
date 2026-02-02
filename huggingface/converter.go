// MODUL: huggingface/converter
// ZWECK: Python Subprocess-Wrapper fuer GGUF-Konvertierung von HuggingFace-Modellen
// INPUT: Model-Pfad, Konvertierungsoptionen (Script, Quantisierung, Timeout)
// OUTPUT: ConvertResult mit GGUF-Pfad, Statistiken und Metadaten
// NEBENEFFEKTE: Startet Python-Subprocess, schreibt GGUF-Dateien
// ABHAENGIGKEITEN: os/exec, context (stdlib), scripts/*.py (extern)

package huggingface

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Timeout-Konstanten fuer Konvertierung
const (
	ConvertDefaultTimeout = 30 * time.Minute
	ConvertMinTimeout     = 1 * time.Minute
	ConvertMaxTimeout     = 24 * time.Hour
	PythonCommand         = "python"
	FallbackPythonCommand = "python3"
)

// ScriptType definiert das Konvertierungs-Script
type ScriptType string

// Bekannte Konvertierungs-Scripts
const (
	ScriptSigLIP      ScriptType = "convert_siglip.py"
	ScriptNomicVision ScriptType = "convert_nomic_vision.py"
	ScriptDINOv2      ScriptType = "convert_dinov2.py"
	ScriptOpenCLIP    ScriptType = "convert_openclip.py"
	ScriptEvaCLIP     ScriptType = "convert_eva_clip.py"
)

// QuantizeType definiert den Quantisierungs-Typ
type QuantizeType string

// Unterstuetzte Quantisierungs-Typen
const (
	QuantizeF32   QuantizeType = "f32"
	QuantizeF16   QuantizeType = "f16"
	QuantizeQ8_0  QuantizeType = "q8_0"
	QuantizeQ4_KM QuantizeType = "q4_k_m"
)

// Fehler-Typen
var (
	ErrPythonNotFound        = errors.New("python nicht gefunden")
	ErrScriptNotFound        = errors.New("script nicht gefunden")
	ErrConversionFailed      = errors.New("konvertierung fehlgeschlagen")
	ErrConversionTimeout     = errors.New("timeout ueberschritten")
	ErrInvalidConvertOptions = errors.New("ungueltige optionen")
)

// ConverterOptions enthaelt Konvertierungs-Optionen
type ConverterOptions struct {
	Script     ScriptType    // Konvertierungs-Script
	OutputPath string        // Ziel-Pfad fuer GGUF
	Quantize   QuantizeType  // Quantisierung (Standard: f16)
	Timeout    time.Duration // Max. Ausfuehrungszeit
	ImageMean  []float64     // Normalisierungs-Mittelwerte
	ImageStd   []float64     // Normalisierungs-Stdabw.
	Validate   bool          // PyTorch-Validierung
	ExtraArgs  []string      // Zusaetzliche CLI-Args
}

// ConvertResult enthaelt das Konvertierungs-Ergebnis
type ConvertResult struct {
	OutputPath   string        // Pfad zur GGUF-Datei
	ModelType    string        // Erkannter Modell-Typ
	Duration     time.Duration // Dauer
	OriginalSize int64         // Original-Groesse (Bytes)
	OutputSize   int64         // GGUF-Groesse (Bytes)
	Quantization string        // Quantisierungs-Typ
	TensorCount  int           // Anzahl Tensoren
	Stdout       string        // Python stdout
	Stderr       string        // Python stderr
}

// Converter konvertiert HuggingFace-Modelle zu GGUF
type Converter struct {
	scriptDir, pythonPath string
}

// NewConverter erstellt einen neuen Converter
func NewConverter(scriptDir string) *Converter { return &Converter{scriptDir: scriptDir} }

// Convert fuehrt die GGUF-Konvertierung durch
func (c *Converter) Convert(modelPath string, opts ConverterOptions) (*ConvertResult, error) {
	return c.ConvertWithProgress(modelPath, opts, nil)
}

// ConvertWithProgress fuehrt Konvertierung mit Progress-Callback durch
func (c *Converter) ConvertWithProgress(modelPath string, opts ConverterOptions, progress func(string)) (*ConvertResult, error) {
	startTime := time.Now()
	if err := c.validateOpts(&opts); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidConvertOptions, err)
	}
	pythonPath, err := c.findPython()
	if err != nil {
		return nil, err
	}
	scriptPath := filepath.Join(c.scriptDir, string(opts.Script))
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("%w: %s", ErrScriptNotFound, scriptPath)
	}
	modelInfo, err := os.Stat(modelPath)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("%w: %s", ErrModelNotFound, modelPath)
	}
	originalSize := c.calcSize(modelPath, modelInfo)
	if opts.OutputPath == "" {
		opts.OutputPath = c.genOutputPath(modelPath, opts)
	}
	ctx, cancel := context.WithTimeout(context.Background(), opts.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, pythonPath, c.buildArgs(scriptPath, modelPath, opts)...)
	cmd.Dir = c.scriptDir
	stdoutPipe, _ := cmd.StdoutPipe()
	stderrPipe, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("python start fehler: %w", err)
	}
	var stdout, stderr strings.Builder
	tensorCount := 0
	done1, done2 := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done1)
		s := bufio.NewScanner(stdoutPipe)
		for s.Scan() {
			line := s.Text()
			stdout.WriteString(line + "\n")
			if progress != nil {
				progress(line)
			}
			if strings.Contains(line, "->") && strings.Contains(line, ":") {
				tensorCount++
			}
		}
	}()
	go func() {
		defer close(done2)
		s := bufio.NewScanner(stderrPipe)
		for s.Scan() {
			line := s.Text()
			stderr.WriteString(line + "\n")
			if progress != nil {
				progress("[ERR] " + line)
			}
		}
	}()
	<-done1
	<-done2
	err = cmd.Wait()
	if ctx.Err() == context.DeadlineExceeded {
		return nil, fmt.Errorf("%w: %v", ErrConversionTimeout, opts.Timeout)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %v\n%s", ErrConversionFailed, err, stderr.String())
	}
	outInfo, err := os.Stat(opts.OutputPath)
	if err != nil {
		return nil, fmt.Errorf("output fehlt: %w", err)
	}
	return &ConvertResult{
		OutputPath: opts.OutputPath, ModelType: c.modelType(opts.Script), Duration: time.Since(startTime),
		OriginalSize: originalSize, OutputSize: outInfo.Size(), Quantization: string(opts.Quantize),
		TensorCount: tensorCount, Stdout: stdout.String(), Stderr: stderr.String(),
	}, nil
}

// ListAvailableScripts gibt verfuegbare Scripts zurueck
func (c *Converter) ListAvailableScripts() ([]ScriptType, error) {
	entries, err := os.ReadDir(c.scriptDir)
	if err != nil {
		return nil, err
	}
	var scripts []ScriptType
	for _, e := range entries {
		if n := e.Name(); strings.HasPrefix(n, "convert_") && strings.HasSuffix(n, ".py") {
			scripts = append(scripts, ScriptType(n))
		}
	}
	return scripts, nil
}

// DetectScript erkennt das passende Script fuer ein Modell
func (c *Converter) DetectScript(modelPath string) (ScriptType, error) {
	content, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return "", err
	}
	cfg := strings.ToLower(string(content))
	switch {
	case strings.Contains(cfg, "siglip"):
		return ScriptSigLIP, nil
	case strings.Contains(cfg, "nomic"):
		return ScriptNomicVision, nil
	case strings.Contains(cfg, "dinov2"):
		return ScriptDINOv2, nil
	case strings.Contains(cfg, "openclip"), strings.Contains(cfg, "open_clip"):
		return ScriptOpenCLIP, nil
	case strings.Contains(cfg, "eva") && strings.Contains(cfg, "clip"):
		return ScriptEvaCLIP, nil
	}
	return "", fmt.Errorf("unbekannter modell-typ")
}

func (c *Converter) validateOpts(o *ConverterOptions) error {
	if o.Script == "" {
		return fmt.Errorf("script erforderlich")
	}
	if o.Quantize == "" {
		o.Quantize = QuantizeF16
	}
	switch o.Quantize {
	case QuantizeF32, QuantizeF16, QuantizeQ8_0, QuantizeQ4_KM:
	default:
		return fmt.Errorf("ungueltig: %s", o.Quantize)
	}
	if o.Timeout == 0 {
		o.Timeout = ConvertDefaultTimeout
	} else if o.Timeout < ConvertMinTimeout {
		o.Timeout = ConvertMinTimeout
	} else if o.Timeout > ConvertMaxTimeout {
		o.Timeout = ConvertMaxTimeout
	}
	return nil
}

func (c *Converter) findPython() (string, error) {
	if c.pythonPath != "" {
		return c.pythonPath, nil
	}
	for _, cmd := range []string{PythonCommand, FallbackPythonCommand} {
		if p, err := exec.LookPath(cmd); err == nil {
			c.pythonPath = p
			return p, nil
		}
	}
	return "", ErrPythonNotFound
}

func (c *Converter) buildArgs(script, model string, o ConverterOptions) []string {
	args := []string{script, "--model", model, "--output", o.OutputPath, "--quantize", string(o.Quantize)}
	if len(o.ImageMean) > 0 {
		args = append(args, "--image-mean")
		for _, v := range o.ImageMean {
			args = append(args, fmt.Sprintf("%g", v))
		}
	}
	if len(o.ImageStd) > 0 {
		args = append(args, "--image-std")
		for _, v := range o.ImageStd {
			args = append(args, fmt.Sprintf("%g", v))
		}
	}
	if o.Validate {
		args = append(args, "--validate")
	}
	return append(args, o.ExtraArgs...)
}

func (c *Converter) genOutputPath(model string, o ConverterOptions) string {
	base := regexp.MustCompile(`[^a-zA-Z0-9_-]`).ReplaceAllString(filepath.Base(model), "_")
	return filepath.Join(filepath.Dir(model), fmt.Sprintf("%s_%s.gguf", base, o.Quantize))
}

func (c *Converter) modelType(s ScriptType) string {
	m := map[ScriptType]string{ScriptSigLIP: "siglip", ScriptNomicVision: "nomic", ScriptDINOv2: "dinov2", ScriptOpenCLIP: "openclip", ScriptEvaCLIP: "eva_clip"}
	if t, ok := m[s]; ok {
		return t
	}
	return strings.TrimPrefix(strings.TrimSuffix(string(s), ".py"), "convert_")
}

func (c *Converter) calcSize(path string, info os.FileInfo) int64 {
	if !info.IsDir() {
		return info.Size()
	}
	var sz int64
	_ = filepath.Walk(path, func(_ string, fi os.FileInfo, _ error) error {
		if fi != nil && !fi.IsDir() {
			sz += fi.Size()
		}
		return nil
	})
	return sz
}
