//go:build windows || darwin

// Webview-Styles-Modul: JavaScript-Initialisierung und CSS-Styles
// Dieses Modul enthaelt die JS-Initialisierung und Windows-spezifische Scrollbar-Styles.
// Aufgeteilt aus der urspruenglichen webview.go (528 LOC)

package main

import (
	"runtime"

	"github.com/ollama/ollama/app/webview"
)

// initWebviewJS initialisiert die JavaScript-Injektionen fuer die WebView
func initWebviewJS(wv webview.WebView, w *Webview) {
	init := getBaseJSInit(w.token)

	// Windows-specific scrollbar styling
	if runtime.GOOS == "windows" {
		init += getWindowsScrollbarStyles()
	}

	// on windows make ctrl+n open new chat
	if runtime.GOOS == "windows" {
		init += getWindowsKeyboardShortcuts()
	}

	init += `
		window.OLLAMA_WEBSEARCH = true;
	`

	wv.Init(init)

	// Add keyboard handler for zoom
	wv.Init(getZoomKeyboardHandler())
}

// getBaseJSInit liefert die grundlegende JavaScript-Initialisierung
func getBaseJSInit(token string) string {
	return `
		// Disable reload
		document.addEventListener('keydown', function(e) {
			if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
				e.preventDefault();
				return false;
			}
		});

		// Prevent back/forward navigation
		window.addEventListener('popstate', function(e) {
			e.preventDefault();
			history.pushState(null, '', window.location.pathname);
			return false;
		});

		// Clear history on load
		window.addEventListener('load', function() {
			history.pushState(null, '', window.location.pathname);
			window.history.replaceState(null, '', window.location.pathname);
		});

		// Set token cookie
		document.cookie = "token=` + token + `; path=/";
	`
}

// getWindowsKeyboardShortcuts liefert Windows-spezifische Tastaturkuerzel
func getWindowsKeyboardShortcuts() string {
	return `
		document.addEventListener('keydown', function(e) {
			if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
				e.preventDefault();
				// Use the existing navigation method
				history.pushState({}, '', '/c/new');
				window.dispatchEvent(new PopStateEvent('popstate'));
				return false;
			}
		});
	`
}

// getZoomKeyboardHandler liefert den Keyboard-Handler fuer Zoom-Funktionen
func getZoomKeyboardHandler() string {
	return `
		window.addEventListener('keydown', function(e) {
			// CMD/Ctrl + Plus/Equals (zoom in)
			if ((e.metaKey || e.ctrlKey) && (e.key === '+' || e.key === '=')) {
				e.preventDefault();
				window.zoomIn && window.zoomIn();
				return false;
			}

			// CMD/Ctrl + Minus (zoom out)
			if ((e.metaKey || e.ctrlKey) && e.key === '-') {
				e.preventDefault();
				window.zoomOut && window.zoomOut();
				return false;
			}

			// CMD/Ctrl + 0 (reset zoom)
			if ((e.metaKey || e.ctrlKey) && e.key === '0') {
				e.preventDefault();
				window.zoomReset && window.zoomReset();
				return false;
			}
		}, true);
	`
}

// getWindowsScrollbarStyles liefert die CSS-Styles fuer Windows-Scrollbars
func getWindowsScrollbarStyles() string {
	return `
		// Fix scrollbar styling for Edge WebView2 on Windows only
		function updateScrollbarStyles() {
			const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
			const existingStyle = document.getElementById('scrollbar-style');
			if (existingStyle) existingStyle.remove();

			const style = document.createElement('style');
			style.id = 'scrollbar-style';

			if (isDark) {
				style.textContent = ` + "`" + `
					::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
					::-webkit-scrollbar-track { background: #1a1a1a !important; }
					::-webkit-scrollbar-thumb { background: #404040 !important; border-radius: 6px !important; }
					::-webkit-scrollbar-thumb:hover { background: #505050 !important; }
					::-webkit-scrollbar-corner { background: #1a1a1a !important; }
					::-webkit-scrollbar-button {
						background: transparent !important;
						border: none !important;
						width: 0px !important;
						height: 0px !important;
						margin: 0 !important;
						padding: 0 !important;
					}
					::-webkit-scrollbar-button:vertical:start:decrement {
						background: transparent !important;
						height: 0px !important;
					}
					::-webkit-scrollbar-button:vertical:end:increment {
						background: transparent !important;
						height: 0px !important;
					}
					::-webkit-scrollbar-button:horizontal:start:decrement {
						background: transparent !important;
						width: 0px !important;
					}
					::-webkit-scrollbar-button:horizontal:end:increment {
						background: transparent !important;
						width: 0px !important;
					}
				` + "`" + `;
			} else {
				style.textContent = ` + "`" + `
					::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
					::-webkit-scrollbar-track { background: #f0f0f0 !important; }
					::-webkit-scrollbar-thumb { background: #c0c0c0 !important; border-radius: 6px !important; }
					::-webkit-scrollbar-thumb:hover { background: #a0a0a0 !important; }
					::-webkit-scrollbar-corner { background: #f0f0f0 !important; }
					::-webkit-scrollbar-button {
						background: transparent !important;
						border: none !important;
						width: 0px !important;
						height: 0px !important;
						margin: 0 !important;
						padding: 0 !important;
					}
					::-webkit-scrollbar-button:vertical:start:decrement {
						background: transparent !important;
						height: 0px !important;
					}
					::-webkit-scrollbar-button:vertical:end:increment {
						background: transparent !important;
						height: 0px !important;
					}
					::-webkit-scrollbar-button:horizontal:start:decrement {
						background: transparent !important;
						width: 0px !important;
					}
					::-webkit-scrollbar-button:horizontal:end:increment {
						background: transparent !important;
						width: 0px !important;
					}
				` + "`" + `;
			}
			document.head.appendChild(style);
		}

		window.addEventListener('load', updateScrollbarStyles);
		window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateScrollbarStyles);
	`
}
