//go:build windows || darwin

// Modul: store_types.go
// Beschreibung: Datentypen und Strukturen fuer den Store.
// Enthaelt Message, Chat, Settings und andere Modelle.

package store

import (
	"encoding/json"
	"time"
)

type File struct {
	Filename string `json:"filename"`
	Data     []byte `json:"data"`
}

type User struct {
	Name     string    `json:"name"`
	Email    string    `json:"email"`
	Plan     string    `json:"plan"`
	CachedAt time.Time `json:"cachedAt"`
}

type Message struct {
	Role              string           `json:"role"`
	Content           string           `json:"content"`
	Thinking          string           `json:"thinking"`
	Stream            bool             `json:"stream"`
	Model             string           `json:"model,omitempty"`
	Attachments       []File           `json:"attachments,omitempty"`
	ToolCalls         []ToolCall       `json:"tool_calls,omitempty"`
	ToolCall          *ToolCall        `json:"tool_call,omitempty"`
	ToolName          string           `json:"tool_name,omitempty"`
	ToolResult        *json.RawMessage `json:"tool_result,omitempty"`
	CreatedAt         time.Time        `json:"created_at"`
	UpdatedAt         time.Time        `json:"updated_at"`
	ThinkingTimeStart *time.Time       `json:"thinkingTimeStart,omitempty" ts_type:"Date | undefined" ts_transform:"__VALUE__ && new Date(__VALUE__)"`
	ThinkingTimeEnd   *time.Time       `json:"thinkingTimeEnd,omitempty" ts_type:"Date | undefined" ts_transform:"__VALUE__ && new Date(__VALUE__)"`
}

// MessageOptions contains optional parameters for creating a Message
type MessageOptions struct {
	Model             string
	Attachments       []File
	Stream            bool
	Thinking          string
	ToolCalls         []ToolCall
	ToolCall          *ToolCall
	ToolResult        *json.RawMessage
	ThinkingTimeStart *time.Time
	ThinkingTimeEnd   *time.Time
}

// NewMessage creates a new Message with the given options
func NewMessage(role, content string, opts *MessageOptions) Message {
	now := time.Now()
	msg := Message{
		Role:      role,
		Content:   content,
		CreatedAt: now,
		UpdatedAt: now,
	}

	if opts != nil {
		msg.Model = opts.Model
		msg.Attachments = opts.Attachments
		msg.Stream = opts.Stream
		msg.Thinking = opts.Thinking
		msg.ToolCalls = opts.ToolCalls
		msg.ToolCall = opts.ToolCall
		msg.ToolResult = opts.ToolResult
		msg.ThinkingTimeStart = opts.ThinkingTimeStart
		msg.ThinkingTimeEnd = opts.ThinkingTimeEnd
	}

	return msg
}

type ToolCall struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Result    any    `json:"result,omitempty"`
}

type Model struct {
	Model      string     `json:"model"`                 // Model name
	Digest     string     `json:"digest,omitempty"`      // Model digest from the registry
	ModifiedAt *time.Time `json:"modified_at,omitempty"` // When the model was last modified locally
}

type Chat struct {
	ID           string          `json:"id"`
	Messages     []Message       `json:"messages"`
	Title        string          `json:"title"`
	CreatedAt    time.Time       `json:"created_at"`
	BrowserState json.RawMessage `json:"browser_state,omitempty" ts_type:"BrowserStateData"`
}

// NewChat creates a new Chat with the ID, with CreatedAt timestamp initialized
func NewChat(id string) *Chat {
	return &Chat{
		ID:        id,
		Messages:  []Message{},
		CreatedAt: time.Now(),
	}
}

type Settings struct {
	// Expose is a boolean that indicates if the ollama server should
	// be exposed to the network
	Expose bool

	// Browser is a boolean that indicates if the ollama server should
	// be exposed to browser windows (e.g. CORS set to allow all origins)
	Browser bool

	// Survey is a boolean that indicates if the user allows anonymous
	// inference information to be shared with Ollama
	Survey bool

	// Models is a string that contains the models to load on startup
	Models string

	// TODO(parthsareen): temporary for experimentation
	// Agent indicates if the app should use multi-turn tools to fulfill user requests
	Agent bool

	// Tools indicates if the app should use single-turn tools to fulfill user requests
	Tools bool

	// WorkingDir specifies the working directory for all agent operations
	WorkingDir string

	// ContextLength specifies the context length for the ollama server (using OLLAMA_CONTEXT_LENGTH)
	ContextLength int

	// AirplaneMode when true, turns off Ollama Turbo features and only uses local models
	AirplaneMode bool

	// TurboEnabled indicates if Ollama Turbo features are enabled
	TurboEnabled bool

	// Maps gpt-oss specific frontend name' BrowserToolEnabled' to db field 'websearch_enabled'
	WebSearchEnabled bool

	// ThinkEnabled indicates if thinking is enabled
	ThinkEnabled bool

	// ThinkLevel indicates the level of thinking to use for models that support multiple levels
	ThinkLevel string

	// SelectedModel stores the last model that the user selected
	SelectedModel string

	// SidebarOpen indicates if the chat sidebar is open
	SidebarOpen bool
}
