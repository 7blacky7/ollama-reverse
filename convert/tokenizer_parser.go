// tokenizer_parser.go - Tokenizer-Parsing aus HuggingFace-Dateien
// Enthält: Tokenizer-Struct, parseTokenizer, Pre-Tokenizer-Erkennung

package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"slices"
	"strings"
)

// Tokenizer enthält alle Informationen zum Token-Parsing
type Tokenizer struct {
	*Vocabulary
	SpecialVocabulary []*SpecialVocabulary
	Merges            []string

	Pre      string
	Template string
}

// parseTokenizer parst Tokenizer-Dateien aus einem Dateisystem
func parseTokenizer(fsys fs.FS, specialTokenTypes []string) (*Tokenizer, error) {
	v, err := parseVocabulary(fsys)
	if err != nil {
		return nil, err
	}

	t := &Tokenizer{
		Vocabulary: v,
		Pre:        "default",
	}

	addedTokens := make(map[string]token)
	if f, err := fsys.Open("tokenizer.json"); errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		return nil, err
	} else {
		defer f.Close()

		var tt tokenizer
		if err := json.NewDecoder(f).Decode(&tt); err != nil {
			return nil, err
		}

		for _, t := range tt.AddedTokens {
			addedTokens[t.Content] = t
		}

		// Merges parsen (kann []string oder [][]string sein)
		if err := parseMerges(&tt, t); err != nil {
			return nil, err
		}

		// Pre-Tokenizer-Typ erkennen
		t.Pre = detectPreTokenizer(&tt)
	}

	// tokenizer_config.json parsen
	if err := parseTokenizerConfig(fsys, t, addedTokens, specialTokenTypes); err != nil {
		return nil, err
	}

	// generation_config.json parsen
	if err := parseGenerationConfig(fsys, t, specialTokenTypes); err != nil {
		return nil, err
	}

	return t, nil
}

// parseMerges parst die Merges aus tokenizer.json
func parseMerges(tt *tokenizer, t *Tokenizer) error {
	if len(tt.Model.Merges) == 0 {
		return nil
	}

	// Versuche als []string
	if err := json.Unmarshal(tt.Model.Merges, &t.Merges); err == nil {
		return nil
	}

	// Versuche als [][]string
	var merges [][]string
	if err := json.Unmarshal(tt.Model.Merges, &merges); err != nil {
		return errors.New("could not parse tokenizer merges. expected []string or [][]string")
	}

	t.Merges = make([]string, len(merges))
	for i := range merges {
		t.Merges[i] = strings.Join(merges[i], " ")
	}

	return nil
}

// detectPreTokenizer erkennt den Pre-Tokenizer-Typ anhand eines SHA256-Hashs
func detectPreTokenizer(tt *tokenizer) string {
	sha256sum := sha256.New()
	for _, pt := range tt.PreTokenizer.PreTokenizers {
		if pt.Type == "Split" && pt.Pattern.Regex != "" {
			sha256sum.Write([]byte(pt.Pattern.Regex))
		}
	}

	switch digest := hex.EncodeToString(sha256sum.Sum(nil)); digest {
	case "d98f9631be1e9607a9848c26c1f9eac1aa9fc21ac6ba82a2fc0741af9780a48f":
		return "llama-bpe"
	case "03df5c5863ad70781dcfdef491ead25140f895fe8010964be0daefe27be32b02":
		return "deepseek-llm"
	case "21cde974d587f0d54dc8d56b183cc1e6239600172035c68fbd6d4b9f8da0576e":
		return "deepseek-coder"
	case "1ff7f41064896984db5d1bb6ff64fa4bc29007d08c1b439e505b7392777a319e":
		return "qwen2"
	case "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
		// Leerer Pre-Tokenizer
		return "default"
	default:
		slog.Warn("unknown pretokenizer, using default", "digest", digest)
		return "default"
	}
}

// parseGenerationConfig parst generation_config.json für spezielle Token-IDs
func parseGenerationConfig(fsys fs.FS, t *Tokenizer, specialTokenTypes []string) error {
	f, err := fsys.Open("generation_config.json")
	if errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	defer f.Close()

	var p map[string]json.RawMessage
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return err
	}

	for _, st := range specialTokenTypes {
		if bts, ok := p[st+"_token_id"]; ok {
			var ids []int32
			if err := json.Unmarshal(bts, &ids); err != nil {
				// Wert ist keine Liste, existierende ID wird verwendet
				continue
			}

			if i := slices.IndexFunc(t.SpecialVocabulary, func(sv *SpecialVocabulary) bool {
				return sv.Type == st
			}); i >= 0 {
				t.SpecialVocabulary[i].IDs = ids
			}
		}
	}

	return nil
}
