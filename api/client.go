// Package api - Hauptmodul des Ollama API-Clients.
// Dieses Modul enthaelt die Client-Struktur und Basis-Methoden.
// Stream-Methoden sind in client_stream.go, API-Methoden in client_api.go.
//
// Package api implements the client-side API for code wishing to interact
// with the ollama service. The methods of the [Client] type correspond to
// the ollama REST API as described in [the API documentation].
// The ollama command-line client itself uses this package to interact with
// the backend service.
//
// # Examples
//
// Several examples of using this package are available [in the GitHub
// repository].
//
// [the API documentation]: https://github.com/ollama/ollama/blob/main/docs/api.md
// [in the GitHub repository]: https://github.com/ollama/ollama/tree/main/api/examples
package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"runtime"
	"strconv"
	"time"

	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

// Client encapsulates client state for interacting with the ollama
// service. Use [ClientFromEnvironment] to create new Clients.
type Client struct {
	base *url.URL
	http *http.Client
}

func checkError(resp *http.Response, body []byte) error {
	if resp.StatusCode < http.StatusBadRequest {
		return nil
	}

	if resp.StatusCode == http.StatusUnauthorized {
		authError := AuthorizationError{StatusCode: resp.StatusCode}
		json.Unmarshal(body, &authError)
		return authError
	}

	apiError := StatusError{StatusCode: resp.StatusCode}

	err := json.Unmarshal(body, &apiError)
	if err != nil {
		// Use the full body as the message if we fail to decode a response.
		apiError.ErrorMessage = string(body)
	}

	return apiError
}

// ClientFromEnvironment creates a new [Client] using configuration from the
// environment variable OLLAMA_HOST, which points to the network host and
// port on which the ollama service is listening. The format of this variable
// is:
//
//	<scheme>://<host>:<port>
//
// If the variable is not specified, a default ollama host and port will be
// used.
func ClientFromEnvironment() (*Client, error) {
	return &Client{
		base: envconfig.Host(),
		http: http.DefaultClient,
	}, nil
}

func NewClient(base *url.URL, http *http.Client) *Client {
	return &Client{
		base: base,
		http: http,
	}
}

func getAuthorizationToken(ctx context.Context, challenge string) (string, error) {
	token, err := auth.Sign(ctx, []byte(challenge))
	if err != nil {
		return "", err
	}
	return token, nil
}

func (c *Client) do(ctx context.Context, method, path string, reqData, respData any) error {
	var reqBody io.Reader
	var data []byte
	var err error

	switch reqData := reqData.(type) {
	case io.Reader:
		// reqData is already an io.Reader
		reqBody = reqData
	case nil:
		// noop
	default:
		data, err = json.Marshal(reqData)
		if err != nil {
			return err
		}

		reqBody = bytes.NewReader(data)
	}

	requestURL := c.base.JoinPath(path)

	var token string
	if envconfig.UseAuth() || c.base.Hostname() == "ollama.com" {
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

	request, err := http.NewRequestWithContext(ctx, method, requestURL.String(), reqBody)
	if err != nil {
		return err
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	request.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if token != "" {
		request.Header.Set("Authorization", token)
	}

	respObj, err := c.http.Do(request)
	if err != nil {
		return err
	}
	defer respObj.Body.Close()

	respBody, err := io.ReadAll(respObj.Body)
	if err != nil {
		return err
	}

	if err := checkError(respObj, respBody); err != nil {
		return err
	}

	if len(respBody) > 0 && respData != nil {
		if err := json.Unmarshal(respBody, respData); err != nil {
			return err
		}
	}
	return nil
}
