// Package updater - Upgrade-Logik fuer macOS.
// Dieses Modul enthaelt die DoUpgrade-Funktion und zugehoerige Hilfsfunktionen.

package updater

import (
	"archive/zip"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/user"
	"path/filepath"
	"strings"
)

// DoUpgrade fuehrt das eigentliche Upgrade durch.
func DoUpgrade(interactive bool) error {
	// TODO use UpgradeLogFile to record the upgrade details from->to version, etc.

	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	slog.Info("starting upgrade", "app", BundlePath, "update", bundle, "pid", os.Getpid(), "log", UpgradeLogFile)

	// TODO - in the future, consider shutting down the backend server now to give it
	// time to drain connections and stop allowing new connections while we perform the
	// actual upgrade to reduce the overall time to complete
	contentsName := filepath.Join(BundlePath, "Contents")
	appBackup := filepath.Join(appBackupDir, "Ollama.app")
	contentsOldName := filepath.Join(appBackup, "Contents")

	// Verify old doesn't exist yet
	if _, err := os.Stat(contentsOldName); err == nil {
		slog.Error("prior upgrade failed", "backup", contentsOldName)
		return fmt.Errorf("prior upgrade failed - please upgrade manually by installing the bundle")
	}
	if err := os.MkdirAll(appBackupDir, 0o755); err != nil {
		return fmt.Errorf("unable to create backup dir %s: %w", appBackupDir, err)
	}

	// Verify bundle loads before starting staging process
	r, err := zip.OpenReader(bundle)
	if err != nil {
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}
	defer r.Close()

	slog.Debug("temporarily staging old version", "staging", appBackup)
	if err := os.Rename(BundlePath, appBackup); err != nil {
		if !interactive {
			// We don't want to prompt for permission if we're attempting to upgrade at startup
			return fmt.Errorf("unable to upgrade in non-interactive mode with permission problems: %w", err)
		}
		// TODO actually inspect the error and look for permission problems before trying chown
		slog.Warn("unable to backup old version due to permission problems, changing ownership", "error", err)
		u, err := user.Current()
		if err != nil {
			return err
		}
		if !chownWithAuthorization(u.Username) {
			return fmt.Errorf("unable to change permissions to complete upgrade")
		}
		if err := os.Rename(BundlePath, appBackup); err != nil {
			return fmt.Errorf("unable to perform upgrade - failed to stage old version: %w", err)
		}
	}

	// Get ready to try to unwind a partial upgade failure during unzip
	// If something goes wrong, we attempt to put the old version back.
	anyFailures := false
	defer func() {
		if anyFailures {
			slog.Warn("upgrade failures detected, attempting to revert")
			if err := os.RemoveAll(BundlePath); err != nil {
				slog.Warn("failed to remove partial upgrade", "path", BundlePath, "error", err)
				// At this point, we're basically hosed and the user will need to re-install
				return
			}
			if err := os.Rename(appBackup, BundlePath); err != nil {
				slog.Error("failed to revert to prior version", "path", contentsName, "error", err)
			}
		}
	}()

	// Bundle contents Ollama.app/Contents/...
	links := []*zip.File{}
	for _, f := range r.File {
		s := strings.SplitN(f.Name, "/", 2)
		if len(s) < 2 || s[1] == "" {
			slog.Debug("skipping", "file", f.Name)
			continue
		}
		name := s[1]
		if strings.HasSuffix(name, "/") {
			d := filepath.Join(BundlePath, name)
			err := os.MkdirAll(d, 0o755)
			if err != nil {
				anyFailures = true
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}

		src, err := f.Open()
		if err != nil {
			anyFailures = true
			return fmt.Errorf("failed to open bundle file %s: %w", name, err)
		}
		destName := filepath.Join(BundlePath, name)
		// Verify directory first
		d := filepath.Dir(destName)
		if _, err := os.Stat(d); err != nil {
			err := os.MkdirAll(d, 0o755)
			if err != nil {
				anyFailures = true
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
		}
		destFile, err := os.OpenFile(destName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
		if err != nil {
			anyFailures = true
			return fmt.Errorf("failed to open output file %s: %w", destName, err)
		}
		defer destFile.Close()
		if _, err := io.Copy(destFile, src); err != nil {
			anyFailures = true
			return fmt.Errorf("failed to open extract file %s: %w", destName, err)
		}
	}

	// Process symlinks
	for _, f := range links {
		s := strings.SplitN(f.Name, "/", 2) // Strip off Ollama.app/
		if len(s) < 2 || s[1] == "" {
			slog.Debug("skipping link", "file", f.Name)
			continue
		}
		name := s[1]
		src, err := f.Open()
		if err != nil {
			anyFailures = true
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			anyFailures = true
			return err
		}
		link := string(buf)
		if link[0] == '/' {
			anyFailures = true
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		// Don't allow links outside of Ollama.app
		if strings.HasPrefix(filepath.Join(filepath.Dir(name), link), "..") {
			anyFailures = true
			return fmt.Errorf("bundle contains link outside of contents %s -> %s", f.Name, link)
		}
		if err = os.Symlink(link, filepath.Join(BundlePath, name)); err != nil {
			anyFailures = true
			return err
		}
	}

	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	}
	f.Close()
	// Make sure to remove the staged download now that we succeeded so we don't inadvertently try again.
	cleanupOldDownloads(UpdateStageDir)

	return nil
}
