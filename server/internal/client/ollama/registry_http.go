// Package ollama - HTTP und Authentifizierung
//
// Diese Datei enthält:
// - newRequest() für Request-Erstellung mit Auth
// - sendRequest() für Request-Ausführung mit Error-Handling
// - makeAuthToken() für Ollama-Auth-Tokens
// - parseNameExtended() und splitExtended() für URL-Parsing
// - parseChunk() für Chunk-Range-Parsing
// - publicError für benutzerfreundliche Fehlermeldungen
package ollama

import (
	"bytes"
	"cmp"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"golang.org/x/crypto/ssh"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/internal/names"
)

// newRequest erstellt einen neuen Request mit UserAgent und Auth-Header
func (r *Registry) newRequest(ctx context.Context, method, url string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, err
	}
	if r.UserAgent != "" {
		req.Header.Set("User-Agent", r.UserAgent)
	}
	if r.Key != nil {
		token, err := makeAuthToken(r.Key)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req, nil
}

// sendRequest führt einen Request aus und behandelt Fehler-Responses
// Status 2xx gibt Response zurück, sonst wird Error geparst
func sendRequest(c *http.Client, r *http.Request) (_ *http.Response, err error) {
	if r.URL.Scheme == "https+insecure" {
		type cloner interface {
			Clone() *http.Transport
		}

		x, ok := cmp.Or(c.Transport, http.DefaultTransport).(cloner)
		if ok {
			tr := x.Clone()
			tr.TLSClientConfig = cmp.Or(tr.TLSClientConfig, &tls.Config{})
			tr.TLSClientConfig.InsecureSkipVerify = true

			cc := *c
			cc.Transport = tr
			c = &cc

			r = r.Clone(r.Context())
			r.URL.Scheme = "https"
		}
	}

	res, err := c.Do(r)
	if err != nil {
		return nil, err
	}
	if res.StatusCode/100 != 2 {
		out, err := io.ReadAll(res.Body)
		if err != nil {
			return nil, err
		}
		var re Error
		if err := json.Unmarshal(out, &re); err != nil {
			re.Message = string(out)
		}

		// MANIFEST_UNKNOWN zu ErrManifestNotFound konvertieren
		if strings.EqualFold(re.Code, "MANIFEST_UNKNOWN") {
			return nil, ErrModelNotFound
		}

		re.status = res.StatusCode
		return nil, &re
	}
	return res, nil
}

// send ist eine Convenience-Methode für newRequest + sendRequest
func (r *Registry) send(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	req, err := r.newRequest(ctx, method, path, body)
	if err != nil {
		return nil, err
	}
	return sendRequest(r.client(), req)
}

// makeAuthToken erstellt ein Ollama-Auth-Token für den Private Key
// Format: base64(url):pubKey:base64(signature)
func makeAuthToken(key crypto.PrivateKey) (string, error) {
	privKey, _ := key.(*ed25519.PrivateKey)
	if privKey == nil {
		return "", fmt.Errorf("unsupported private key type: %T", key)
	}

	url := fmt.Sprintf("https://ollama.com?ts=%d", time.Now().Unix())

	// Public Key extrahieren
	pubKeyShort, err := func() ([]byte, error) {
		sshPubKey, err := ssh.NewPublicKey(privKey.Public())
		if err != nil {
			return nil, err
		}
		pubKeyParts := bytes.Fields(ssh.MarshalAuthorizedKey(sshPubKey))
		if len(pubKeyParts) < 2 {
			return nil, fmt.Errorf("malformed public key: %q", pubKeyParts)
		}
		return pubKeyParts[1], nil
	}()
	if err != nil {
		return "", err
	}

	// Signatur erstellen
	sig := ed25519.Sign(*privKey, []byte(checkData(url)))

	// Token zusammenbauen: <checkData>:<pubKey>:<signature>
	var b strings.Builder
	io.WriteString(&b, base64.StdEncoding.EncodeToString([]byte(url)))
	b.WriteByte(':')
	b.Write(pubKeyShort)
	b.WriteByte(':')
	io.WriteString(&b, base64.StdEncoding.EncodeToString(sig))

	return b.String(), nil
}

// zeroSum ist der SHA256-Hash des leeren Strings (Legacy-Format)
var zeroSum = func() string {
	sha256sum := sha256.Sum256(nil)
	x := base64.StdEncoding.EncodeToString([]byte(hex.EncodeToString(sha256sum[:])))
	return x
}()

// checkData erstellt das Signatur-Format für Ollama-Tokens
func checkData(url string) string {
	return fmt.Sprintf("GET,%s,%s", url, zeroSum)
}

// publicError wraps einen Fehler mit benutzerfreundlicher Nachricht
type publicError struct {
	wrapped error
	message string
}

func withPublicMessagef(err error, message string, args ...any) error {
	return publicError{wrapped: err, message: fmt.Sprintf(message, args...)}
}

func (e publicError) Error() string { return e.message }
func (e publicError) Unwrap() error { return e.wrapped }

// Unterstützte URL-Schemes
var supportedSchemes = []string{
	"http",
	"https",
	"https+insecure",
}

var supportedSchemesMessage = fmt.Sprintf("supported schemes are %v", strings.Join(supportedSchemes, ", "))

// parseNameExtended parst und validiert einen erweiterten Namen
// Gibt Scheme, Name und Digest zurück
func (r *Registry) parseNameExtended(s string) (scheme string, _ names.Name, _ blob.Digest, _ error) {
	scheme, name, digest := splitExtended(s)
	scheme = cmp.Or(scheme, "https")
	if !slices.Contains(supportedSchemes, scheme) {
		err := withPublicMessagef(ErrNameInvalid, "unsupported scheme: %q: %s", scheme, supportedSchemesMessage)
		return "", names.Name{}, blob.Digest{}, err
	}

	var d blob.Digest
	if digest != "" {
		var err error
		d, err = blob.ParseDigest(digest)
		if err != nil {
			err = withPublicMessagef(ErrNameInvalid, "invalid digest: %q", digest)
			return "", names.Name{}, blob.Digest{}, err
		}
		if name == "" {
			return scheme, names.Name{}, d, nil
		}
	}

	n, err := r.parseName(name)
	if err != nil {
		return "", names.Name{}, blob.Digest{}, err
	}
	return scheme, n, d, nil
}

// splitExtended teilt einen erweiterten Namen in Scheme, Name und Digest
// Beispiele:
//   - http://ollama.com/user/model:latest@digest
//   - ollama.com/user/model:latest
//   - @digest
func splitExtended(s string) (scheme, name, digest string) {
	i := strings.Index(s, "://")
	if i >= 0 {
		scheme = s[:i]
		s = s[i+3:]
	}
	i = strings.LastIndex(s, "@")
	if i >= 0 {
		digest = s[i+1:]
		s = s[:i]
	}
	return scheme, s, digest
}

// parseChunk parst einen String "start-end" zu einem Chunk
func parseChunk[S ~string | ~[]byte](s S) (blob.Chunk, error) {
	startPart, endPart, found := strings.Cut(string(s), "-")
	if !found {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid range %q: missing '-'", s)
	}
	start, err := strconv.ParseInt(startPart, 10, 64)
	if err != nil {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid start to %q: %v", s, err)
	}
	end, err := strconv.ParseInt(endPart, 10, 64)
	if err != nil {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid end to %q: %v", s, err)
	}
	if start > end {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid range %q: start > end", s)
	}
	return blob.Chunk{Start: start, End: end}, nil
}
