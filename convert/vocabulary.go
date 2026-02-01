// vocabulary.go - Vokabular-Parsing für Tokenizer
// Enthält: Vocabulary, SpecialVocabulary, Token-Strukturen und Parser

package convert

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"slices"
)

// Token-Typen für das Vokabular
const (
	_ int32 = iota
	tokenTypeNormal
	tokenTypeUnknown
	tokenTypeControl
	tokenTypeUserDefined
	tokenTypeUnused
	tokenTypeByte
)

// tokenizer repräsentiert die tokenizer.json Struktur
type tokenizer struct {
	AddedTokens []token `json:"added_tokens"`
	Model       struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`

	PreTokenizer struct {
		PreTokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

// token repräsentiert ein einzelnes Token
type token struct {
	ID          int    `json:"id"`
	Content     string `json:"content"`
	Special     bool   `json:"special"`
	UserDefined bool
}

// Vocabulary enthält das Vokabular eines Modells
type Vocabulary struct {
	Model  string
	Tokens []string
	Scores []float32
	Types  []int32
}

// SpecialVocabulary enthält spezielle Token-Definitionen
type SpecialVocabulary struct {
	Type     string
	ID       int
	Content  string
	AddToken bool

	// IDs wird von generation_config.json befüllt
	IDs []int32
}

// Key gibt den GGUF-Schlüsselnamen für den Token-Typ zurück
func (sv SpecialVocabulary) Key() string {
	switch t := sv.Type; t {
	case "bos", "eos", "cls", "mask":
		return t
	case "unk":
		return "unknown"
	case "sep":
		//nolint:misspell // Upstream-Tippfehler
		return "seperator"
	case "pad":
		return "padding"
	}

	panic("unknown special vocabulary type")
}

// parseVocabulary parst das Vokabular aus dem Dateisystem
func parseVocabulary(fsys fs.FS) (*Vocabulary, error) {
	patterns := []struct {
		Pattern string
		Func    func(fs.FS) (*Vocabulary, error)
	}{
		{"tokenizer.model", parseSentencePiece},
		{"tokenizer.json", parseVocabularyFromTokenizer},
	}

	for _, pattern := range patterns {
		if _, err := fs.Stat(fsys, pattern.Pattern); errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return nil, err
		}

		return pattern.Func(fsys)
	}

	return nil, errors.New("unknown tokenizer format")
}

// parseVocabularyFromTokenizer parst das Vokabular aus tokenizer.json
func parseVocabularyFromTokenizer(fsys fs.FS) (*Vocabulary, error) {
	f, err := fsys.Open("tokenizer.json")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var t tokenizer
	if err := json.NewDecoder(f).Decode(&t); err != nil {
		return nil, err
	}

	tokens := make(map[int]token, len(t.Model.Vocab))
	for k, v := range t.Model.Vocab {
		tokens[v] = token{
			ID:      v,
			Content: k,
		}
	}

	for _, tok := range t.AddedTokens {
		tok.UserDefined = true
		tokens[tok.ID] = tok
	}

	v := Vocabulary{Model: "gpt2"}
	for _, k := range slices.Sorted(maps.Keys(tokens)) {
		tok := tokens[k]
		v.Tokens = append(v.Tokens, tok.Content)
		v.Scores = append(v.Scores, float32(tok.ID))

		switch {
		case tok.Special:
			v.Types = append(v.Types, tokenTypeControl)
		case tok.UserDefined:
			v.Types = append(v.Types, tokenTypeUserDefined)
		default:
			v.Types = append(v.Types, tokenTypeNormal)
		}
	}

	return &v, nil
}

// parseTokenizerConfig parst tokenizer_config.json
func parseTokenizerConfig(fsys fs.FS, t *Tokenizer, addedTokens map[string]token, specialTokenTypes []string) error {
	f, err := fsys.Open("tokenizer_config.json")
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

	// Chat-Template parsen
	if template, ok := p["chat_template"]; ok {
		if err := parseChatTemplate(template, t); err != nil {
			return err
		}
	}

	// Spezielle Token parsen
	for _, st := range specialTokenTypes {
		sv := SpecialVocabulary{Type: st}
		if bts, ok := p[fmt.Sprintf("add_%s_token", st)]; ok {
			if err := json.Unmarshal(bts, &sv.AddToken); err != nil {
				return err
			}
		}

		if bts, ok := p[fmt.Sprintf("%s_token", st)]; ok {
			content, err := parseTokenContent(bts)
			if err != nil || content == "" {
				continue
			}
			sv.Content = content
		}

		if id, ok := addedTokens[sv.Content]; ok {
			sv.ID = id.ID
			t.SpecialVocabulary = append(t.SpecialVocabulary, &sv)
		}
	}

	return nil
}

// parseChatTemplate parst das Chat-Template (kann String oder Array sein)
func parseChatTemplate(template json.RawMessage, t *Tokenizer) error {
	var s []struct {
		Name     string `json:"name"`
		Template string `json:"template"`
	}

	if err := json.Unmarshal(template, &t.Template); err == nil {
		return nil
	} else if err := json.Unmarshal(template, &s); err == nil {
		for _, e := range s {
			if e.Name == "default" {
				t.Template = e.Template
				break
			}
		}
		return nil
	}

	return fmt.Errorf("invalid chat_template format")
}

// parseTokenContent parst den Token-Inhalt (kann String oder Objekt sein)
func parseTokenContent(bts json.RawMessage) (string, error) {
	var content string
	if err := json.Unmarshal(bts, &content); err == nil {
		return content, nil
	}

	var mm map[string]any
	if err := json.Unmarshal(bts, &mm); err != nil {
		return "", err
	}

	content, _ = mm["content"].(string)
	return content, nil
}
