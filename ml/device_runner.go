// device_runner.go
// Dieses Modul enthaelt die Runner-Interfaces und Funktionen fuer die
// Kommunikation mit dem Backend-Runner zur Geraete-Erkennung.
// Urspruenglich aus device.go extrahiert.

package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/ollama/ollama/logutil"
)

type BaseRunner interface {
	// GetPort returns the localhost port number the runner is running on
	GetPort() int

	// HasExited indicates if the runner is no longer running.  This can be used during
	// bootstrap to detect if a given filtered device is incompatible and triggered an assert
	HasExited() bool
}

type RunnerDiscovery interface {
	BaseRunner

	// GetDeviceInfos will perform a query of the underlying device libraries
	// for device identification and free VRAM information
	// During bootstrap scenarios, this routine may take seconds to complete
	GetDeviceInfos(ctx context.Context) []DeviceInfo
}

type FilteredRunnerDiscovery interface {
	RunnerDiscovery

	// GetActiveDeviceIDs returns the filtered set of devices actively in
	// use by this runner for running models.  If the runner is a bootstrap runner, no devices
	// will be active yet so no device IDs are returned.
	// This routine will not query the underlying device and will return immediately
	GetActiveDeviceIDs() []DeviceID
}

func GetDevicesFromRunner(ctx context.Context, runner BaseRunner) ([]DeviceInfo, error) {
	var moreDevices []DeviceInfo
	port := runner.GetPort()
	tick := time.Tick(10 * time.Millisecond)
	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("failed to finish discovery before timeout")
		case <-tick:
			r, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/info", port), nil)
			if err != nil {
				return nil, fmt.Errorf("failed to create request: %w", err)
			}
			r.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(r)
			if err != nil {
				// slog.Warn("failed to send request", "error", err)
				if runner.HasExited() {
					return nil, fmt.Errorf("runner crashed")
				}
				continue
			}
			defer resp.Body.Close()

			if resp.StatusCode == http.StatusNotFound {
				// old runner, fall back to bootstrapping model
				return nil, fmt.Errorf("llamarunner free vram reporting not supported")
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				slog.Warn("failed to read response", "error", err)
				continue
			}
			if resp.StatusCode != 200 {
				logutil.Trace("runner failed to discover free VRAM", "status", resp.StatusCode, "response", body)
				return nil, fmt.Errorf("runner error: %s", string(body))
			}

			if err := json.Unmarshal(body, &moreDevices); err != nil {
				slog.Warn("unmarshal encode response", "error", err)
				continue
			}
			return moreDevices, nil
		}
	}
}
