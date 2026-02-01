// Package api - Einfache API-Methoden des Clients.
// Dieses Modul enthaelt alle nicht-streaming API-Methoden.

package api

import (
	"context"
	"fmt"
	"io"
	"net/http"
)

// List lists models that are available locally.
func (c *Client) List(ctx context.Context) (*ListResponse, error) {
	var lr ListResponse
	if err := c.do(ctx, http.MethodGet, "/api/tags", nil, &lr); err != nil {
		return nil, err
	}
	return &lr, nil
}

// ListRunning lists running models.
func (c *Client) ListRunning(ctx context.Context) (*ProcessResponse, error) {
	var lr ProcessResponse
	if err := c.do(ctx, http.MethodGet, "/api/ps", nil, &lr); err != nil {
		return nil, err
	}
	return &lr, nil
}

// Copy copies a model - creating a model with another name from an existing
// model.
func (c *Client) Copy(ctx context.Context, req *CopyRequest) error {
	if err := c.do(ctx, http.MethodPost, "/api/copy", req, nil); err != nil {
		return err
	}
	return nil
}

// Delete deletes a model and its data.
func (c *Client) Delete(ctx context.Context, req *DeleteRequest) error {
	if err := c.do(ctx, http.MethodDelete, "/api/delete", req, nil); err != nil {
		return err
	}
	return nil
}

// Show obtains model information, including details, modelfile, license etc.
func (c *Client) Show(ctx context.Context, req *ShowRequest) (*ShowResponse, error) {
	var resp ShowResponse
	if err := c.do(ctx, http.MethodPost, "/api/show", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Heartbeat checks if the server has started and is responsive; if yes, it
// returns nil, otherwise an error.
func (c *Client) Heartbeat(ctx context.Context) error {
	if err := c.do(ctx, http.MethodHead, "/", nil, nil); err != nil {
		return err
	}
	return nil
}

// Embed generates embeddings from a model.
func (c *Client) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
	var resp EmbedResponse
	if err := c.do(ctx, http.MethodPost, "/api/embed", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Embeddings generates an embedding from a model.
func (c *Client) Embeddings(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	var resp EmbeddingResponse
	if err := c.do(ctx, http.MethodPost, "/api/embeddings", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// CreateBlob creates a blob from a file on the server. digest is the
// expected SHA256 digest of the file, and r represents the file.
func (c *Client) CreateBlob(ctx context.Context, digest string, r io.Reader) error {
	return c.do(ctx, http.MethodPost, fmt.Sprintf("/api/blobs/%s", digest), r, nil)
}

// Version returns the Ollama server version as a string.
func (c *Client) Version(ctx context.Context) (string, error) {
	var version struct {
		Version string `json:"version"`
	}

	if err := c.do(ctx, http.MethodGet, "/api/version", nil, &version); err != nil {
		return "", err
	}

	return version.Version, nil
}

// Signout will signout a client for a local ollama server.
func (c *Client) Signout(ctx context.Context) error {
	return c.do(ctx, http.MethodPost, "/api/signout", nil, nil)
}

// Disconnect will disconnect an ollama instance from ollama.com.
func (c *Client) Disconnect(ctx context.Context, encodedKey string) error {
	return c.do(ctx, http.MethodDelete, fmt.Sprintf("/api/user/keys/%s", encodedKey), nil, nil)
}

// Whoami gibt Informationen ueber den aktuellen Benutzer zurueck.
func (c *Client) Whoami(ctx context.Context) (*UserResponse, error) {
	var resp UserResponse
	if err := c.do(ctx, http.MethodPost, "/api/me", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}
