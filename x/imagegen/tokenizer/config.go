//go:build mlx

// config.go - Laden von Special Token Konfiguration
//
// Enthält:
// - loadSpecialTokenConfig: Lädt aus Dateien (generation_config.json, config.json, etc.)
// - loadSpecialTokenConfigFromBytes: Lädt aus Byte-Slices
// - extractTokenString: Extrahiert Token-Strings aus verschiedenen JSON-Formaten

package tokenizer

import (
	"encoding/json"
	"os"
)

// loadSpecialTokenConfig loads special token configuration from HuggingFace companion files.
//
// Loading priority for EOS tokens (can be single int or []int):
//  1. generation_config.json - eos_token_id (preferred, matches HuggingFace generation)
//  2. config.json - eos_token_id (model config fallback)
//  3. tokenizer_config.json - eos_token string + add_bos/add_eos flags
//  4. special_tokens_map.json - final fallback
func loadSpecialTokenConfig(dir string, t *Tokenizer) {
	// Helper to parse eos_token_id which can be int or []int
	parseTokenIDs := func(v interface{}) []int32 {
		switch val := v.(type) {
		case float64:
			return []int32{int32(val)}
		case []interface{}:
			ids := make([]int32, 0, len(val))
			for _, id := range val {
				if f, ok := id.(float64); ok {
					ids = append(ids, int32(f))
				}
			}
			return ids
		}
		return nil
	}

	// Priority 1: generation_config.json (eos_token_id can be int or []int)
	if data, err := os.ReadFile(dir + "generation_config.json"); err == nil {
		var config struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			if ids := parseTokenIDs(config.EOSTokenID); len(ids) > 0 {
				t.vocab.EOS = ids
			}
			if ids := parseTokenIDs(config.BOSTokenID); len(ids) > 0 {
				t.vocab.BOS = ids[0]
			}
		}
	}

	// Priority 2: config.json (model config, same format)
	if len(t.vocab.EOS) == 0 || t.vocab.BOS < 0 {
		if data, err := os.ReadFile(dir + "config.json"); err == nil {
			var config struct {
				EOSTokenID interface{} `json:"eos_token_id"`
				BOSTokenID interface{} `json:"bos_token_id"`
			}
			if err := json.Unmarshal(data, &config); err == nil {
				if len(t.vocab.EOS) == 0 {
					if ids := parseTokenIDs(config.EOSTokenID); len(ids) > 0 {
						t.vocab.EOS = ids
					}
				}
				if t.vocab.BOS < 0 {
					if ids := parseTokenIDs(config.BOSTokenID); len(ids) > 0 {
						t.vocab.BOS = ids[0]
					}
				}
			}
		}
	}

	// Priority 3: tokenizer_config.json (token strings + add_bos/add_eos flags)
	if data, err := os.ReadFile(dir + "tokenizer_config.json"); err == nil {
		var config struct {
			BOSToken    interface{} `json:"bos_token"`
			EOSToken    interface{} `json:"eos_token"`
			PADToken    interface{} `json:"pad_token"`
			AddBOSToken *bool       `json:"add_bos_token"`
			AddEOSToken *bool       `json:"add_eos_token"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(config.BOSToken); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(config.EOSToken); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(config.PADToken); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
			if config.AddBOSToken != nil {
				t.vocab.AddBOS = *config.AddBOSToken
			}
			if config.AddEOSToken != nil {
				t.vocab.AddEOS = *config.AddEOSToken
			}
		}
	}

	// Priority 4: special_tokens_map.json (final fallback)
	if data, err := os.ReadFile(dir + "special_tokens_map.json"); err == nil {
		var tokensMap map[string]interface{}
		if err := json.Unmarshal(data, &tokensMap); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(tokensMap["bos_token"]); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(tokensMap["eos_token"]); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(tokensMap["pad_token"]); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
		}
	}
}

// extractTokenString extracts the token string from various formats used in HuggingFace configs.
// Tokens can be represented as:
//   - string: "token"
//   - object: {"content": "token", ...}
func extractTokenString(v interface{}) string {
	if v == nil {
		return ""
	}
	// Direct string
	if s, ok := v.(string); ok {
		return s
	}
	// Object with content field
	if m, ok := v.(map[string]interface{}); ok {
		if content, ok := m["content"].(string); ok {
			return content
		}
	}
	return ""
}

// loadSpecialTokenConfigFromBytes loads special token configuration from byte slices.
func loadSpecialTokenConfigFromBytes(t *Tokenizer, config *TokenizerConfig) {
	// Helper to parse eos_token_id which can be int or []int
	parseTokenIDs := func(v interface{}) []int32 {
		switch val := v.(type) {
		case float64:
			return []int32{int32(val)}
		case []interface{}:
			ids := make([]int32, 0, len(val))
			for _, id := range val {
				if f, ok := id.(float64); ok {
					ids = append(ids, int32(f))
				}
			}
			return ids
		}
		return nil
	}

	// Priority 1: generation_config.json
	if len(config.GenerationConfigJSON) > 0 {
		var genConfig struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(config.GenerationConfigJSON, &genConfig); err == nil {
			if ids := parseTokenIDs(genConfig.EOSTokenID); len(ids) > 0 {
				t.vocab.EOS = ids
			}
			if ids := parseTokenIDs(genConfig.BOSTokenID); len(ids) > 0 {
				t.vocab.BOS = ids[0]
			}
		}
	}

	// Priority 2: config.json
	if len(config.ConfigJSON) > 0 && (len(t.vocab.EOS) == 0 || t.vocab.BOS < 0) {
		var modelConfig struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(config.ConfigJSON, &modelConfig); err == nil {
			if len(t.vocab.EOS) == 0 {
				if ids := parseTokenIDs(modelConfig.EOSTokenID); len(ids) > 0 {
					t.vocab.EOS = ids
				}
			}
			if t.vocab.BOS < 0 {
				if ids := parseTokenIDs(modelConfig.BOSTokenID); len(ids) > 0 {
					t.vocab.BOS = ids[0]
				}
			}
		}
	}

	// Priority 3: tokenizer_config.json
	if len(config.TokenizerConfigJSON) > 0 {
		var tokConfig struct {
			BOSToken    interface{} `json:"bos_token"`
			EOSToken    interface{} `json:"eos_token"`
			PADToken    interface{} `json:"pad_token"`
			AddBOSToken *bool       `json:"add_bos_token"`
			AddEOSToken *bool       `json:"add_eos_token"`
		}
		if err := json.Unmarshal(config.TokenizerConfigJSON, &tokConfig); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(tokConfig.BOSToken); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(tokConfig.EOSToken); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(tokConfig.PADToken); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
			if tokConfig.AddBOSToken != nil {
				t.vocab.AddBOS = *tokConfig.AddBOSToken
			}
			if tokConfig.AddEOSToken != nil {
				t.vocab.AddEOS = *tokConfig.AddEOSToken
			}
		}
	}

	// Priority 4: special_tokens_map.json
	if len(config.SpecialTokensMapJSON) > 0 {
		var tokensMap map[string]interface{}
		if err := json.Unmarshal(config.SpecialTokensMapJSON, &tokensMap); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(tokensMap["bos_token"]); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(tokensMap["eos_token"]); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(tokensMap["pad_token"]); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
		}
	}
}
