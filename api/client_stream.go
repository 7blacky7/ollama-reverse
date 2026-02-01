// Package api - Stream-basierte Client-Methoden.
// Dieses Modul enthaelt alle Methoden, die Streaming-Responses verwenden.

package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"strconv"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/version"
)

const maxBufferSize = 8 * format.MegaByte

func (c *Client) stream(ctx context.Context, method, path string, data any, fn func([]byte) error) error {
	var buf *bytes.Buffer
	if data != nil {
		bts, err := json.Marshal(data)
		if err != nil {
			return err
		}

		buf = bytes.NewBuffer(bts)
	}

	requestURL := c.base.JoinPath(path)

	var token string
	if envconfig.UseAuth() || c.base.Hostname() == "ollama.com" {
		var err error
		now := strconv.FormatInt(time.Now().Unix(), 10)
		chal := fmt.Sprintf("%s,%s?ts=%s", method, path, now)
		token, err = getAuthorizationToken(ctx, chal)
		if err != nil {
			return err
		}

		q := requestURL.Query()
		q.Set("ts", now)
		requestURL.RawQuery = q.Encode()
	}

	var reqBody *bytes.Buffer
	if buf != nil {
		reqBody = buf
	}

	request, err := http.NewRequestWithContext(ctx, method, requestURL.String(), reqBody)
	if err != nil {
		return err
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/x-ndjson")
	request.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if token != "" {
		request.Header.Set("Authorization", token)
	}

	response, err := c.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	scanner := bufio.NewScanner(response.Body)
	// increase the buffer size to avoid running out of space
	scanBuf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(scanBuf, maxBufferSize)
	for scanner.Scan() {
		var errorResponse struct {
			Error     string `json:"error,omitempty"`
			SigninURL string `json:"signin_url,omitempty"`
		}

		bts := scanner.Bytes()
		if err := json.Unmarshal(bts, &errorResponse); err != nil {
			if response.StatusCode >= http.StatusBadRequest {
				return StatusError{
					StatusCode:   response.StatusCode,
					Status:       response.Status,
					ErrorMessage: string(bts),
				}
			}
			return errors.New(string(bts))
		}

		if response.StatusCode == http.StatusUnauthorized {
			return AuthorizationError{
				StatusCode: response.StatusCode,
				Status:     response.Status,
				SigninURL:  errorResponse.SigninURL,
			}
		} else if response.StatusCode >= http.StatusBadRequest {
			return StatusError{
				StatusCode:   response.StatusCode,
				Status:       response.Status,
				ErrorMessage: errorResponse.Error,
			}
		}

		if errorResponse.Error != "" {
			return errors.New(errorResponse.Error)
		}

		if err := fn(bts); err != nil {
			return err
		}
	}

	return nil
}

// GenerateResponseFunc is a function that [Client.Generate] invokes every time
// a response is received from the service. If this function returns an error,
// [Client.Generate] will stop generating and return this error.
type GenerateResponseFunc func(GenerateResponse) error

// Generate generates a response for a given prompt. The req parameter should
// be populated with prompt details. fn is called for each response (there may
// be multiple responses, e.g. in case streaming is enabled).
func (c *Client) Generate(ctx context.Context, req *GenerateRequest, fn GenerateResponseFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/generate", req, func(bts []byte) error {
		var resp GenerateResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

// ChatResponseFunc is a function that [Client.Chat] invokes every time
// a response is received from the service. If this function returns an error,
// [Client.Chat] will stop generating and return this error.
type ChatResponseFunc func(ChatResponse) error

// Chat generates the next message in a chat. [ChatRequest] may contain a
// sequence of messages which can be used to maintain chat history with a model.
// fn is called for each response (there may be multiple responses, e.g. if case
// streaming is enabled).
func (c *Client) Chat(ctx context.Context, req *ChatRequest, fn ChatResponseFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/chat", req, func(bts []byte) error {
		var resp ChatResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

// PullProgressFunc is a function that [Client.Pull] invokes every time there
// is progress with a "pull" request sent to the service. If this function
// returns an error, [Client.Pull] will stop the process and return this error.
type PullProgressFunc func(ProgressResponse) error

// Pull downloads a model from the ollama library. fn is called each time
// progress is made on the request and can be used to display a progress bar,
// etc.
func (c *Client) Pull(ctx context.Context, req *PullRequest, fn PullProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/pull", req, func(bts []byte) error {
		var resp ProgressResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

// PushProgressFunc is a function that [Client.Push] invokes when progress is
// made.
// It's similar to other progress function types like [PullProgressFunc].
type PushProgressFunc func(ProgressResponse) error

// Push uploads a model to the model library; requires registering for ollama.ai
// and adding a public key first. fn is called each time progress is made on
// the request and can be used to display a progress bar, etc.
func (c *Client) Push(ctx context.Context, req *PushRequest, fn PushProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/push", req, func(bts []byte) error {
		var resp ProgressResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

// CreateProgressFunc is a function that [Client.Create] invokes when progress
// is made.
// It's similar to other progress function types like [PullProgressFunc].
type CreateProgressFunc func(ProgressResponse) error

// Create creates a model from a [Modelfile]. fn is a progress function that
// behaves similarly to other methods (see [Client.Pull]).
//
// [Modelfile]: https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx
func (c *Client) Create(ctx context.Context, req *CreateRequest, fn CreateProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/create", req, func(bts []byte) error {
		var resp ProgressResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}
