// openai_to.go - Konvertierungsfunktionen von API-Format zu OpenAI-Format
//
// Enthaelt:
// - ToUsage, ToUsageGenerate: Token-Verbrauch konvertieren
// - ToToolCalls: Tool-Aufrufe konvertieren
// - ToChatCompletion, ToChunk: Chat-Antworten konvertieren
// - ToCompletion, ToCompleteChunk: Text-Completion konvertieren
// - ToListCompletion, ToModel: Model-Listen konvertieren
// - ToEmbeddingList: Embedding-Antworten konvertieren
// - ToImageGenerationResponse: Bild-Generierung konvertieren
//
// Verwandte Dateien:
// - openai_types.go: Typdefinitionen
// - openai_from.go: Konvertierung OpenAI -> API Format
package openai

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// ToUsage konvertiert eine api.ChatResponse zu Usage
func ToUsage(r api.ChatResponse) Usage {
	return Usage{
		PromptTokens:     r.Metrics.PromptEvalCount,
		CompletionTokens: r.Metrics.EvalCount,
		TotalTokens:      r.Metrics.PromptEvalCount + r.Metrics.EvalCount,
	}
}

// ToToolCalls konvertiert api.ToolCall zu OpenAI ToolCall Format
func ToToolCalls(tc []api.ToolCall) []ToolCall {
	toolCalls := make([]ToolCall, len(tc))
	for i, tc := range tc {
		toolCalls[i].ID = tc.ID
		toolCalls[i].Type = "function"
		toolCalls[i].Function.Name = tc.Function.Name
		toolCalls[i].Index = tc.Function.Index

		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			slog.Error("could not marshall function arguments to json", "error", err)
			continue
		}

		toolCalls[i].Function.Arguments = string(args)
	}
	return toolCalls
}

// ToChatCompletion konvertiert eine api.ChatResponse zu ChatCompletion
func ToChatCompletion(id string, r api.ChatResponse) ChatCompletion {
	toolCalls := ToToolCalls(r.Message.ToolCalls)

	var logprobs *ChoiceLogprobs
	if len(r.Logprobs) > 0 {
		logprobs = &ChoiceLogprobs{Content: r.Logprobs}
	}

	return ChatCompletion{
		Id:                id,
		Object:            "chat.completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []Choice{{
			Index:   0,
			Message: Message{Role: r.Message.Role, Content: r.Message.Content, ToolCalls: toolCalls, Reasoning: r.Message.Thinking},
			FinishReason: func(reason string) *string {
				if len(toolCalls) > 0 {
					reason = "tool_calls"
				}
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
			Logprobs: logprobs,
		}}, Usage: ToUsage(r),
		DebugInfo: r.DebugInfo,
	}
}

// ToChunk konvertiert eine api.ChatResponse zu ChatCompletionChunk
func ToChunk(id string, r api.ChatResponse, toolCallSent bool) ChatCompletionChunk {
	toolCalls := ToToolCalls(r.Message.ToolCalls)

	var logprobs *ChoiceLogprobs
	if len(r.Logprobs) > 0 {
		logprobs = &ChoiceLogprobs{Content: r.Logprobs}
	}

	return ChatCompletionChunk{
		Id:                id,
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChunkChoice{{
			Index: 0,
			Delta: Message{Role: "assistant", Content: r.Message.Content, ToolCalls: toolCalls, Reasoning: r.Message.Thinking},
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					if toolCallSent || len(toolCalls) > 0 {
						return &finishReasonToolCalls
					}
					return &reason
				}
				return nil
			}(r.DoneReason),
			Logprobs: logprobs,
		}},
	}
}

// ToUsageGenerate konvertiert eine api.GenerateResponse zu Usage
func ToUsageGenerate(r api.GenerateResponse) Usage {
	return Usage{
		PromptTokens:     r.Metrics.PromptEvalCount,
		CompletionTokens: r.Metrics.EvalCount,
		TotalTokens:      r.Metrics.PromptEvalCount + r.Metrics.EvalCount,
	}
}

// ToCompletion konvertiert eine api.GenerateResponse zu Completion
func ToCompletion(id string, r api.GenerateResponse) Completion {
	return Completion{
		Id:                id,
		Object:            "text_completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []CompleteChunkChoice{{
			Text:  r.Response,
			Index: 0,
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
		}},
		Usage: ToUsageGenerate(r),
	}
}

// ToCompleteChunk konvertiert eine api.GenerateResponse zu CompletionChunk
func ToCompleteChunk(id string, r api.GenerateResponse) CompletionChunk {
	return CompletionChunk{
		Id:                id,
		Object:            "text_completion",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []CompleteChunkChoice{{
			Text:  r.Response,
			Index: 0,
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
		}},
	}
}

// ToListCompletion konvertiert eine api.ListResponse zu ListCompletion
func ToListCompletion(r api.ListResponse) ListCompletion {
	var data []Model
	for _, m := range r.Models {
		data = append(data, Model{
			Id:      m.Name,
			Object:  "model",
			Created: m.ModifiedAt.Unix(),
			OwnedBy: model.ParseName(m.Name).Namespace,
		})
	}

	return ListCompletion{
		Object: "list",
		Data:   data,
	}
}

// ToEmbeddingList konvertiert eine api.EmbedResponse zu EmbeddingList
// encodingFormat kann "float", "base64" oder leer sein (Standard: "float")
func ToEmbeddingList(modelName string, r api.EmbedResponse, encodingFormat string) EmbeddingList {
	if r.Embeddings != nil {
		var data []Embedding
		for i, e := range r.Embeddings {
			var embedding any
			if strings.EqualFold(encodingFormat, "base64") {
				embedding = floatsToBase64(e)
			} else {
				embedding = e
			}

			data = append(data, Embedding{
				Object:    "embedding",
				Embedding: embedding,
				Index:     i,
			})
		}

		return EmbeddingList{
			Object: "list",
			Data:   data,
			Model:  modelName,
			Usage: EmbeddingUsage{
				PromptTokens: r.PromptEvalCount,
				TotalTokens:  r.PromptEvalCount,
			},
		}
	}

	return EmbeddingList{}
}

// floatsToBase64 kodiert einen []float32 als base64 String
func floatsToBase64(floats []float32) string {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, floats)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// ToModel konvertiert eine api.ShowResponse zu Model
func ToModel(r api.ShowResponse, m string) Model {
	return Model{
		Id:      m,
		Object:  "model",
		Created: r.ModifiedAt.Unix(),
		OwnedBy: model.ParseName(m).Namespace,
	}
}

// ToImageGenerationResponse konvertiert eine Ollama GenerateResponse zu OpenAI ImageGenerationResponse
func ToImageGenerationResponse(resp api.GenerateResponse) ImageGenerationResponse {
	var data []ImageURLOrData
	if resp.Image != "" {
		data = []ImageURLOrData{{B64JSON: resp.Image}}
	}
	return ImageGenerationResponse{
		Created: resp.CreatedAt.Unix(),
		Data:    data,
	}
}
