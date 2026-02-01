// types_model.go - Model Management API Types
// Enthaelt: CreateRequest, DeleteRequest, ShowRequest, ShowResponse, CopyRequest,
//           PullRequest, PushRequest, ListResponse, ProcessResponse, ModelDetails, Tensor
package api

import (
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/types/model"
)

// CreateRequest is the request passed to [Client.Create].
type CreateRequest struct {
	// Model is the model name to create.
	Model string `json:"model"`

	// Stream specifies whether the response is streaming; it is true by default.
	Stream *bool `json:"stream,omitempty"`

	// Quantize is the quantization format for the model; leave blank to not change the quantization level.
	Quantize string `json:"quantize,omitempty"`

	// From is the name of the model or file to use as the source.
	From string `json:"from,omitempty"`

	// RemoteHost is the URL of the upstream ollama API for the model (if any).
	RemoteHost string `json:"remote_host,omitempty"`

	// Files is a map of files include when creating the model.
	Files map[string]string `json:"files,omitempty"`

	// Adapters is a map of LoRA adapters to include when creating the model.
	Adapters map[string]string `json:"adapters,omitempty"`

	// Template is the template used when constructing a request to the model.
	Template string `json:"template,omitempty"`

	// License is a string or list of strings for licenses.
	License any `json:"license,omitempty"`

	// System is the system prompt for the model.
	System string `json:"system,omitempty"`

	// Parameters is a map of hyper-parameters which are applied to the model.
	Parameters map[string]any `json:"parameters,omitempty"`

	// Messages is a list of messages added to the model before chat and generation requests.
	Messages []Message `json:"messages,omitempty"`

	Renderer string `json:"renderer,omitempty"`
	Parser   string `json:"parser,omitempty"`

	// Requires is the minimum version of Ollama required by the model.
	Requires string `json:"requires,omitempty"`

	// Info is a map of additional information for the model
	Info map[string]any `json:"info,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
	// Deprecated: use Quantize instead
	Quantization string `json:"quantization,omitempty"`
}

// DeleteRequest is the request passed to [Client.Delete].
type DeleteRequest struct {
	Model string `json:"model"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowRequest is the request passed to [Client.Show].
type ShowRequest struct {
	Model  string `json:"model"`
	System string `json:"system"`

	// Template is deprecated
	Template string `json:"template"`
	Verbose  bool   `json:"verbose"`

	Options map[string]any `json:"options"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ShowResponse is the response returned from [Client.Show].
type ShowResponse struct {
	License       string             `json:"license,omitempty"`
	Modelfile     string             `json:"modelfile,omitempty"`
	Parameters    string             `json:"parameters,omitempty"`
	Template      string             `json:"template,omitempty"`
	System        string             `json:"system,omitempty"`
	Renderer      string             `json:"renderer,omitempty"`
	Parser        string             `json:"parser,omitempty"`
	Details       ModelDetails       `json:"details,omitempty"`
	Messages      []Message          `json:"messages,omitempty"`
	RemoteModel   string             `json:"remote_model,omitempty"`
	RemoteHost    string             `json:"remote_host,omitempty"`
	ModelInfo     map[string]any     `json:"model_info"`
	ProjectorInfo map[string]any     `json:"projector_info,omitempty"`
	Tensors       []Tensor           `json:"tensors,omitempty"`
	Capabilities  []model.Capability `json:"capabilities,omitempty"`
	ModifiedAt    time.Time          `json:"modified_at,omitempty"`
	Requires      string             `json:"requires,omitempty"`
}

// CopyRequest is the request passed to [Client.Copy].
type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

// PullRequest is the request passed to [Client.Pull].
type PullRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"` // Deprecated: ignored
	Username string `json:"username"`           // Deprecated: ignored
	Password string `json:"password"`           // Deprecated: ignored
	Stream   *bool  `json:"stream,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ProgressResponse is the response passed to progress functions like
// [PullProgressFunc] and [PushProgressFunc].
type ProgressResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// PushRequest is the request passed to [Client.Push].
type PushRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
	Stream   *bool  `json:"stream,omitempty"`

	// Deprecated: set the model name with Model instead
	Name string `json:"name"`
}

// ListResponse is the response from [Client.List].
type ListResponse struct {
	Models []ListModelResponse `json:"models"`
}

// ProcessResponse is the response from [Client.Process].
type ProcessResponse struct {
	Models []ProcessModelResponse `json:"models"`
}

// ListModelResponse is a single model description in [ListResponse].
type ListModelResponse struct {
	Name        string       `json:"name"`
	Model       string       `json:"model"`
	RemoteModel string       `json:"remote_model,omitempty"`
	RemoteHost  string       `json:"remote_host,omitempty"`
	ModifiedAt  time.Time    `json:"modified_at"`
	Size        int64        `json:"size"`
	Digest      string       `json:"digest"`
	Details     ModelDetails `json:"details,omitempty"`
}

// ProcessModelResponse is a single model description in [ProcessResponse].
type ProcessModelResponse struct {
	Name          string       `json:"name"`
	Model         string       `json:"model"`
	Size          int64        `json:"size"`
	Digest        string       `json:"digest"`
	Details       ModelDetails `json:"details,omitempty"`
	ExpiresAt     time.Time    `json:"expires_at"`
	SizeVRAM      int64        `json:"size_vram"`
	ContextLength int          `json:"context_length"`
}

type TokenResponse struct {
	Token string `json:"token"`
}

// ModelDetails provides details about a model.
type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// UserResponse provides information about a user.
type UserResponse struct {
	ID        uuid.UUID `json:"id"`
	Email     string    `json:"email"`
	Name      string    `json:"name"`
	Bio       string    `json:"bio,omitempty"`
	AvatarURL string    `json:"avatarurl,omitempty"`
	FirstName string    `json:"firstname,omitempty"`
	LastName  string    `json:"lastname,omitempty"`
	Plan      string    `json:"plan,omitempty"`
}

// Tensor describes the metadata for a given tensor.
type Tensor struct {
	Name  string   `json:"name"`
	Type  string   `json:"type"`
	Shape []uint64 `json:"shape"`
}
