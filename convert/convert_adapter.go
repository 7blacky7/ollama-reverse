// convert_adapter.go - Adapter-Konvertierung: Laedt und konvertiert LoRA-Adapter zu GGUF
// Hauptfunktionen: ConvertAdapter
package convert

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"strings"

	ofs "github.com/ollama/ollama/fs"
)

// ConvertAdapter - Konvertiert einen LoRA-Adapter zu GGUF Format
func ConvertAdapter(fsys fs.FS, f *os.File, baseKV ofs.Config) error {
	bts, err := fs.ReadFile(fsys, "adapter_config.json")
	if err != nil {
		return err
	}

	var p AdapterParameters
	if err := json.Unmarshal(bts, &p); err != nil {
		return err
	}

	arch := baseKV.Architecture()
	if arch == "" {
		return errors.New("architecture not set for the base model")
	}

	var conv AdapterConverter
	switch arch {
	case "llama":
		conv = &llamaAdapter{}
	case "gemma2":
		conv = &gemma2Adapter{}
	default:
		return errors.New("unsupported architecture")
	}

	ts, err := parseTensors(fsys, strings.NewReplacer(conv.Replacements()...))
	if err != nil {
		return err
	}

	if err := json.Unmarshal(bts, conv); err != nil {
		return err
	}

	return writeFile(f, conv.KV(baseKV), conv.Tensors(ts))
}
